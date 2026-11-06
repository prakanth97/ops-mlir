#include "ops-mlir/Pipelines/Pipelines.h"

#include "ops-mlir/Conversion/Passes.h"
#include "ops-mlir/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace ops_mlir::ops {

using opdsl::OptionMap;
using opdsl::PipelineRegistry;

namespace {

llvm::Error pipelineError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument, message);
}

llvm::Error buildFrontend(mlir::OpPassManager &pm, const OptionMap &) {
  pm.addPass(createMaterializeKernels());
  pm.addPass(createSpecializeConsts());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  return llvm::Error::success();
}

llvm::Error buildPlan(mlir::OpPassManager &pm, const OptionMap &o) {
  FusePointLocalOptions fuse;
  fuse.maxLoops = o.getInt("max-loops", 8);
  fuse.reorder = o.getBool("reorder", true);
  fuse.placement = o.getString("placement", "earliest");
  if (fuse.placement != "earliest" && fuse.placement != "latest")
    return pipelineError("option 'placement' must be earliest or latest, got '" + fuse.placement + "'");
  fuse.guarded = o.getBool("guarded", false);
  pm.addPass(createFusePointLocal(fuse));
  pm.addPass(createForwardStores());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(createOutlineGroups());
  return llvm::Error::success();
}

/// The affine stage of the CPU pipeline: the passes the user asked for (a textual pass pipeline, written with `+` for `,` and `;`
/// for a blank because the option syntax separates options with those: `func.func(affine-loop-tile{tile-sizes=32+32})`), then
/// lower-affine back to scf. The passes of MLIR's affine dialect are registered here; a pass of another origin has to be linked
/// into this library and registered with MLIR's pass registry before the pipeline is built.
llvm::Error addAffineStage(mlir::OpPassManager &pm, std::string passes) {
  if (!passes.empty()) {
    static std::once_flag registered;
    std::call_once(registered, [] { mlir::affine::registerAffinePasses(); });
    std::replace(passes.begin(), passes.end(), '+', ',');
    std::replace(passes.begin(), passes.end(), ';', ' ');
    std::string error;
    llvm::raw_string_ostream os(error);
    if (mlir::failed(mlir::parsePassPipeline(passes, pm, os))) return pipelineError("option 'affine-passes': " + error);
  }
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  return llvm::Error::success();
}

llvm::Error buildCPU(mlir::OpPassManager &pm, const OptionMap &o) {
  OpsSpecializeOptions specialize;
  specialize.mode = o.getString("specialize", "none");
  if (specialize.mode != "none" && specialize.mode != "ranges" && specialize.mode != "shapes")
    return pipelineError("option 'specialize' must be none, ranges or shapes, got '" + specialize.mode + "'");
  ConvertOpsToSCFOptions lower;
  lower.soa = o.getBool("soa", false);
  lower.loopStyle = o.getString("loops", "scf");
  if (lower.loopStyle != "scf" && lower.loopStyle != "affine")
    return pipelineError("option 'loops' must be scf or affine, got '" + lower.loopStyle + "'");
  bool omp = o.getBool("omp", true);
  std::string affinePasses = o.getString("affine-passes", "");
  if (!affinePasses.empty() && lower.loopStyle != "affine") return pipelineError("option 'affine-passes' needs loops=affine");

  pm.addPass(createOpsSpecialize(specialize));
  pm.addPass(createConvertOpsToSCF(lower));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (lower.loopStyle == "affine") {
    if (llvm::Error e = addAffineStage(pm, affinePasses)) return e;
  }
  if (o.getBool("fma", false)) pm.addPass(createOpsFpContract());
  // Arithmetic on half precision values is done in single precision (rounded back after every operation): the CPU has
  // no half precision arithmetic, only conversions.
  mlir::arith::ArithEmulateUnsupportedFloatsOptions half;
  half.sourceTypeStrs = {"f16"};
  half.targetTypeStr = "f32";
  pm.addPass(mlir::arith::createArithEmulateUnsupportedFloats(half));
  if (omp) pm.addPass(mlir::createConvertSCFToOpenMPPass());
  // The accumulators of reductions go to the entry of the parallel region (or the function). The OpenMP conversion
  // wraps the loop body in a memref.alloca_scope: with nothing left to allocate, canonicalisation removes it (a scope
  // whose body is lowered to several blocks, inner loops, would be invalid).
  pm.addPass(createOpsHoistAccumulators());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  // expand-strided-metadata runs its patterns greedily, canonicalisation of branches included: after the control flow
  // is in the LLVM dialect that merges the two identical successors of a `cond_br` (the loads of a conditional
  // expression) into one block with memref arguments, which the LLVM dialect cannot have. Before it, they are cf ops.
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(createOpsMarkNoalias());
  if (omp) pm.addPass(mlir::createConvertOpenMPToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  return llvm::Error::success();
}

/// Lowers a group for NVIDIA GPUs: one thread per point (convert-ops-to-scf target=gpu), the kernel to NVVM and a
/// fatbin through MLIR's gpu-lower-to-nvvm-pipeline (ptxas from the CUDA toolkit: CUDA_ROOT), the host side with
/// calls to the mgpu* runtime functions (opdsl GpuShim).
llvm::Error buildCUDA(mlir::OpPassManager &pm, const OptionMap &o) {
  OpsSpecializeOptions specialize;
  specialize.mode = o.getString("specialize", "none");
  if (specialize.mode != "none" && specialize.mode != "ranges" && specialize.mode != "shapes")
    return pipelineError("option 'specialize' must be none, ranges or shapes, got '" + specialize.mode + "'");
  // The pipeline option separates the block sizes with `x` (the option syntax uses commas to separate options).
  std::string block = o.getString("block", "32x4x1");
  std::replace(block.begin(), block.end(), 'x', ',');
  ConvertOpsToSCFOptions lower;
  lower.soa = o.getBool("soa", false);
  lower.target = "gpu";
  lower.block = block;

  pm.addPass(createOpsSpecialize(specialize));
  pm.addPass(createConvertOpsToSCF(lower));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (o.getBool("fma", false)) pm.addPass(createOpsFpContract());

  mlir::gpu::GPUToNVVMPipelineOptions nvvm;
  nvvm.cubinChip = o.getString("chip", "sm_80");
  nvvm.cubinFeatures = o.getString("features", "");
  std::string ptxas = o.getString("ptxas", "");
  std::replace(ptxas.begin(), ptxas.end(), ':', ' '); // options are separated by ":" in the option string
  nvvm.cmdOptions = ptxas;
  nvvm.optLevel = 3;
  // The result is a module in the LLVM dialect with a gpu.binary and gpu.launch_func: the translation to LLVM IR
  // (JITSession) turns the launches into calls of the mgpu* runtime functions (opdsl GpuShim).
  // The stages of mlir::gpu::buildLowerToNVVMPassPipeline (GPUToNVVMPipeline.cpp), with ops-gpu-kernel-attrs between the
  // conversion of the kernels and the serialisation.
  pm.addPass(mlir::createConvertNVGPUToNVVMPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertNVVMToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  mlir::GpuNVVMAttachTargetOptions target;
  target.triple = nvvm.cubinTriple;
  target.chip = nvvm.cubinChip;
  target.features = nvvm.cubinFeatures;
  target.optLevel = nvvm.optLevel;
  target.cmdOptions = nvvm.cmdOptions;
  pm.addPass(mlir::createGpuNVVMAttachTarget(target));
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  mlir::ConvertIndexToLLVMPassOptions index;
  index.indexBitwidth = nvvm.indexBitWidth;
  pm.addPass(mlir::createConvertIndexToLLVMPass(index));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  mlir::ConvertGpuOpsToNVVMOpsOptions kernels;
  kernels.useBarePtrCallConv = nvvm.kernelUseBarePtrCallConv;
  kernels.indexBitwidth = nvvm.indexBitWidth;
  kernels.allowPatternRollback = nvvm.allowPatternRollback;
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps(kernels));
  if (o.getBool("restrict", true)) pm.addNestedPass<mlir::gpu::GPUModuleOp>(createOpsGpuKernelAttrs());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createReconcileUnrealizedCastsPass());
  mlir::GpuToLLVMConversionPassOptions host;
  host.hostBarePtrCallConv = nvvm.hostUseBarePtrCallConv;
  host.kernelBarePtrCallConv = nvvm.kernelUseBarePtrCallConv;
  pm.addPass(mlir::createGpuToLLVMConversionPass(host));
  mlir::GpuModuleToBinaryPassOptions binary;
  binary.compilationTarget = nvvm.cubinFormat;
  pm.addPass(mlir::createGpuModuleToBinaryPass(binary));
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  return llvm::Error::success();
}

} // namespace

void registerOPSPipelines(PipelineRegistry &registry) {
  registry.add({"ops-jit-frontend",
                "materialise kernels into loops, fold bound constants",
                {},
                buildFrontend});
  registry.add({"ops-jit-plan",
                "fuse loops, forward stores, outline the groups",
                {{"max-loops", "int", "most loops in a group"},
                 {"reorder", "bool", "let independent loops move past each other"},
                 {"placement", "string", "earliest or latest"},
                 {"guarded", "bool", "fuse loops of different ranges (not implemented)"}},
                buildPlan});
  registry.add({"ops-jit-cpu",
                "lower a group to the LLVM dialect for the host CPU",
                {{"omp", "bool", "parallelise with OpenMP"},
                 {"specialize", "string", "none, ranges or shapes"},
                 {"soa", "bool", "structure-of-arrays layout of multi-component dats"},
                 {"fma", "bool", "allow fused multiply-adds (contract fast-math flags)"},
                 {"loops", "string", "scf (default) or affine: the loops and accesses of the group are affine ops until lower-affine"},
                 {"affine-passes", "string", "with loops=affine: a pass pipeline run on the affine form (+ for comma, ; for blank)"}},
                buildCPU});
  registry.add({"ops-jit-cuda",
                "lower a group to a CUDA kernel launch (host code in the LLVM dialect, kernel as a fatbin)",
                {{"chip", "string", "GPU architecture, e.g. sm_80"},
                 {"features", "string", "NVVM target features, e.g. +ptx80"},
                 {"block", "string", "thread block size, e.g. 32x4x1 (x fastest)"},
                 {"ptxas", "string", "options for ptxas"},
                 {"specialize", "string", "none, ranges or shapes"},
                 {"soa", "bool", "structure-of-arrays layout of multi-component dats"},
                 {"fma", "bool", "allow fused multiply-adds (contract fast-math flags)"},
                 {"restrict", "bool", "mark the dat pointers of the kernels noalias (default true)"}},
                buildCUDA});
}

void registerOPSPipelinesWithMLIR() {
  for (const PipelineRegistry::Entry *entry : PipelineRegistry::global().entries()) {
    std::string name = entry->name;
    mlir::registerPassPipeline(
        name, entry->description,
        [name](mlir::OpPassManager &pm, llvm::StringRef options,
               llvm::function_ref<mlir::LogicalResult(const llvm::Twine &)> reportError) {
          if (llvm::Error e = PipelineRegistry::global().build(name, options, pm))
            return reportError(llvm::toString(std::move(e)));
          return mlir::success();
        },
        [](llvm::function_ref<void(const mlir::detail::PassOptions &)>) {});
  }
}

std::string planPipelineOptions(const PipelineSettings &s) {
  if (!s.fusion) return "max-loops=1";
  return "max-loops=" + std::to_string(s.fusionMax) + " reorder=" + (s.fusionReorder ? "true" : "false") + " placement=" + s.fusionPlacement +
         " guarded=" + (s.fusionGuarded ? "true" : "false");
}

std::array<int64_t, 3> gpuBlockFor(const std::string &option, int rank) {
  std::array<int64_t, 3> b{32, 4, 1};
  std::sscanf(option.c_str(), "%ld,%ld,%ld", &b[0], &b[1], &b[2]);
  for (int d = rank; d < 3; ++d) {
    b[0] *= b[d];
    b[d] = 1;
  }
  return b;
}

GroupPipeline groupPipeline(bool device, int rank, const PipelineSettings &s) {
  GroupPipeline p;
  const std::string soa = s.soa ? "true" : "false";
  if (device) {
    // block sizes with `x` between them, ptxas options with `:` between them (the option syntax uses blanks and commas)
    std::array<int64_t, 3> bs = gpuBlockFor(s.gpuBlock, rank);
    std::string block = std::to_string(bs[0]) + "x" + std::to_string(bs[1]) + "x" + std::to_string(bs[2]), ptxas = s.ptxasOptions;
    std::replace(ptxas.begin(), ptxas.end(), ' ', ':');
    p.name = "ops-jit-cuda";
    p.options = "chip=" + s.gpuArch + " features=" + s.gpuFeatures + " block=" + block + " specialize=" + s.specialize + " soa=" + soa;
    if (!ptxas.empty()) p.options += " ptxas=" + ptxas;
    if (s.fma) p.options += " fma=true";
    return p;
  }
  p.name = "ops-jit-cpu";
  p.options = std::string("omp=") + (s.omp ? "true" : "false") + " specialize=" + s.specialize + " soa=" + soa;
  if (s.fma) p.options += " fma=true";
  if (s.loops == "affine" || !s.affinePasses.empty()) p.options += " loops=affine";
  if (!s.affinePasses.empty()) p.options += " affine-passes=" + s.affinePasses;
  return p;
}

} // namespace ops_mlir::ops
