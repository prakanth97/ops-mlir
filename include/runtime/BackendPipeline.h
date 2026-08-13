#ifndef OPS_BACKEND
#define OPS_BACKEND
#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Error.h"
#include "passes/UseMonoCuStream.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "runtime/GpuReductionWriteback.h"

namespace ops_mlir {

class MakeFunctionOpPublicPass : public mlir::PassWrapper<MakeFunctionOpPublicPass, mlir::OperationPass<mlir::ModuleOp>> {

public:
	void runOnOperation() override {
		getOperation().walk([&](mlir::func::FuncOp funcOp) {
			if (funcOp.getSymName().starts_with("ops_par_loop"))
				funcOp.setVisibility(mlir::SymbolTable::Visibility::Public);
		});
	}
};

class MapParallelToGpuLaunchPass
    : public mlir::PassWrapper<MapParallelToGpuLaunchPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  explicit MapParallelToGpuLaunchPass(llvm::ArrayRef<int64_t> blockSizes)
      : blockSizes_(blockSizes.begin(), blockSizes.end()) {}

  llvm::StringRef getArgument() const override {
    return "ops-map-parallel-to-gpu-launch";
  }

  void runOnOperation() override {
    // TODO: Check Reduction handle here.
    // Avoid mapping if any parallel loop has reductions -- OPS's reduction
    bool hasReduction = false;
    getOperation().walk([&](mlir::scf::ParallelOp op) {
      if (op.getNumReductions() > 0)
        hasReduction = true;
    });
    if (hasReduction)
      return;

    llvm::SmallVector<mlir::scf::ParallelOp> targets;
    getOperation().walk([&](mlir::scf::ParallelOp op) {
      if (!op->getParentOfType<mlir::scf::ParallelOp>())
        targets.push_back(op);
    });
    for (mlir::scf::ParallelOp op : targets)
      convert(op);
  }

private:
  void convert(mlir::scf::ParallelOp op) const {
    mlir::OpBuilder builder(op);
    mlir::Location loc = op.getLoc();
    mlir::Block *origBody = op.getBody();
    unsigned numLoops = op.getNumLoops();

    llvm::SmallVector<mlir::Value> lbs(op.getLowerBound());
    llvm::SmallVector<mlir::Value> ubs(op.getUpperBound());
    llvm::SmallVector<mlir::Value> steps(op.getStep());

    mlir::Value one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
    llvm::SmallVector<mlir::Value, 3> blockSizeVals(3, one);
    llvm::SmallVector<mlir::Value, 3> gridSizeVals(3, one);
    for (unsigned i = 0; i < 3; ++i) {
      if (i >= numLoops)
        continue;
      unsigned tileIdx = numLoops - 1 - i;
      int64_t tile = tileIdx < blockSizes_.size() ? blockSizes_[tileIdx] : 1;
      mlir::Value tripCount =
          mlir::arith::SubIOp::create(builder, loc, ubs[i], lbs[i]);
      tripCount =
          mlir::arith::CeilDivSIOp::create(builder, loc, tripCount, steps[i]);
      blockSizeVals[i] = mlir::arith::ConstantIndexOp::create(builder, loc, tile);
      gridSizeVals[i] = mlir::arith::CeilDivSIOp::create(builder, loc, tripCount,
                                                        blockSizeVals[i]);
    }

    auto launchOp = mlir::gpu::LaunchOp::create(
        builder, loc, gridSizeVals[0], gridSizeVals[1], gridSizeVals[2],
        blockSizeVals[0], blockSizeVals[1], blockSizeVals[2]);

    builder.setInsertionPointToStart(&launchOp.getBody().front());
    mlir::gpu::KernelDim3 blockIds = launchOp.getBlockIds();
    mlir::gpu::KernelDim3 threadIds = launchOp.getThreadIds();
    std::array<mlir::Value, 3> bIds = {blockIds.x, blockIds.y, blockIds.z};
    std::array<mlir::Value, 3> tIds = {threadIds.x, threadIds.y, threadIds.z};

    llvm::SmallVector<mlir::Value> globalIdx(numLoops);
    for (unsigned i = 0; i < numLoops; ++i) {
      mlir::Value withinTile =
          mlir::arith::MulIOp::create(builder, loc, bIds[i], blockSizeVals[i]);
      withinTile = mlir::arith::AddIOp::create(builder, loc, withinTile, tIds[i]);
      mlir::Value scaled =
          mlir::arith::MulIOp::create(builder, loc, withinTile, steps[i]);
      mlir::Value idx = mlir::arith::AddIOp::create(builder, loc, lbs[i], scaled);
      mlir::Value lastValid =
          mlir::arith::SubIOp::create(builder, loc, ubs[i], steps[i]);
      globalIdx[i] = mlir::arith::MinSIOp::create(builder, loc, idx, lastValid);
    }

    mlir::IRMapping mapping;
    for (unsigned i = 0; i < numLoops; ++i)
      mapping.map(origBody->getArgument(i), globalIdx[i]);
    for (mlir::Operation &bodyOp : origBody->without_terminator())
      builder.clone(bodyOp, mapping);

    mlir::gpu::TerminatorOp::create(builder, loc);

    op.erase();
  }

  llvm::SmallVector<int64_t, 3> blockSizes_;
};

class BackendPipeline {
public:
  virtual ~BackendPipeline() = default;
  virtual void build(mlir::PassManager &pm) const = 0;

  mlir::LogicalResult run(mlir::ModuleOp module, mlir::MLIRContext &ctx) const {
    mlir::PassManager pm(&ctx);
    build(pm);
    if (std::getenv("OPS_DEBUG_PASS_IR")) {
      ctx.disableMultithreading();
      pm.enableIRPrinting();
    }
    return pm.run(module);
  }

  virtual llvm::Error transformLLVMModule(llvm::Module *) const {
    return llvm::Error::success();
  }
};

class CPUSequentialPipeline : public BackendPipeline {
public:
  void build(mlir::PassManager &pm) const override {
    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
		pm.addPass(std::make_unique<MakeFunctionOpPublicPass>());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());

    mlir::ConvertFuncToLLVMPassOptions funcToLLVMOpts;
    funcToLLVMOpts.useBarePtrCallConv = true;
    pm.addPass(mlir::createConvertFuncToLLVMPass(funcToLLVMOpts));

    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }
};

class OpenMPPipeline : public BackendPipeline {
public:
  void build(mlir::PassManager &pm) const override {
    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
		pm.addPass(std::make_unique<MakeFunctionOpPublicPass>());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());

    mlir::ConvertFuncToLLVMPassOptions funcToLLVMOpts;
    funcToLLVMOpts.useBarePtrCallConv = true;
    pm.addPass(mlir::createConvertFuncToLLVMPass(funcToLLVMOpts));

    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

    pm.addPass(mlir::createConvertOpenMPToLLVMPass());

    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }
};

class CudaPipeline : public BackendPipeline {
public:
  explicit CudaPipeline(std::string nvgpuSm) : nvgpuSm_(std::move(nvgpuSm)) {}

  void build(mlir::PassManager &pm) const override {
    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    
    pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<StripGpuReductionWritebackPass>());

    // Note: Tunable parameters
    pm.addPass(std::make_unique<MapParallelToGpuLaunchPass>(
        llvm::ArrayRef<int64_t>{32, 4, 1}));

    pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertParallelLoopToGpuPass());
    pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<InsertGpuReductionWritebackPass>());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());

    pm.addPass(mlir::createGpuKernelOutliningPass());

    // Inline the kernel function into the gpu.func region, 
    // so that NVVM math redirection can be applied to it. 
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createInlinerPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());

    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuAsyncRegionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps());
    mlir::GpuNVVMAttachTargetOptions gputargetOptions;
    gputargetOptions.chip = nvgpuSm_;
    gputargetOptions.optLevel = 3;
    // Experiment hook: e.g. OPS_PTXAS_OPTS="-maxrregcount=64" to trade
    // registers/thread for occupancy on register-bound kernels -- see
    // ncu's "Block Limit Registers" for whether a given kernel is even
    // register-limited before reaching for this.
    if (const char *ptxasOpts = std::getenv("OPS_PTXAS_OPTS"))
      gputargetOptions.cmdOptions = ptxasOpts;
    // gputargetOptions.fastFlag = true;
    pm.addPass(mlir::createGpuNVVMAttachTarget(gputargetOptions));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // Arith, Math to llvm conversion should be done after GPU to NVVM conversion.
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    mlir::ConvertFuncToLLVMPassOptions funcToLLVMOpts;
    funcToLLVMOpts.useBarePtrCallConv = true;
    pm.addPass(mlir::createConvertFuncToLLVMPass(funcToLLVMOpts));

    pm.addPass(mlir::createGpuToLLVMConversionPass());
    pm.addPass(mlir::createGpuModuleToBinaryPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  llvm::Error transformLLVMModule(llvm::Module *m) const override {
    return useMonoCudaStream(m);
  }

private:
  std::string nvgpuSm_;
};

} // namespace ops_mlir


#endif // OPS_BACKEND
