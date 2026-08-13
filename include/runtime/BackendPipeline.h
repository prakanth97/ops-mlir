#ifndef OPS_BACKEND
#define OPS_BACKEND
#include <array>
#include <string>
#include <vector>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
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

class BackendPipeline {
public:
  virtual ~BackendPipeline() = default;
  virtual void build(mlir::PassManager &pm) const = 0;

  mlir::LogicalResult run(mlir::ModuleOp module, mlir::MLIRContext &ctx) const {
    mlir::PassManager pm(&ctx);
    build(pm);
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

class RocmPipeline : public BackendPipeline {
public:
  explicit RocmPipeline(std::string amdgpuChip) : amdgpuChip_(std::move(amdgpuChip)) {}

  void build(mlir::PassManager &pm) const override {
    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<StripGpuReductionWritebackPass>());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertParallelLoopToGpuPass());
    pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<InsertGpuReductionWritebackPass>());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());

    pm.addPass(mlir::createGpuKernelOutliningPass());

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

    // Vendor-specific passes ---
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToROCDLOpsPass());

    mlir::GpuROCDLAttachTargetOptions rocdlTargetOptions;
    rocdlTargetOptions.chip = amdgpuChip_;
    pm.addPass(mlir::createGpuROCDLAttachTarget(rocdlTargetOptions));
    // End of vendor-specific passes ---

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

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
    return llvm::Error::success();  // TODO: not sure if something specific needs to go here
  }

private:
  std::string amdgpuChip_;
};

} // namespace ops_mlir


#endif // OPS_BACKEND
