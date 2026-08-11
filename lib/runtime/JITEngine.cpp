#include "runtime/JITEngine.h"
#include "runtime/BackendPipeline.h"
#include "runtime/KernelIRBuilder.h"
#include "runtime/KernelProfiler.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Python.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Host.h"

#include <algorithm>
#include <memory>

#ifdef OPS_ENABLE_CUDA
#include <cuda.h>
#endif

namespace ops_mlir {

JITEngine &JITEngine::instance() {
  static JITEngine rt;
  return rt;
}

JITEngine::JITEngine() {
  if (!Py_IsInitialized()) {
    Py_Initialize();
  }

  // Make xdsl_impl/ops_to_xdsl.py importable. OPS_XDSL_DIR is injected by
  // lib/runtime/CMakeLists.txt as the absolute path to xdsl_impl/.
  std::string setup = "import sys\nsys.path.insert(0, '" OPS_XDSL_DIR "')\n";
  PyRun_SimpleString(setup.c_str());

  // Register needed dialects - will need to add more later
  mlir::registerAllPasses();
  // TODO: add all required dialects
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  ctx.getOrLoadDialect<mlir::gpu::GPUDialect>();
  ctx.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();

  // Register Interfaces
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerAllExtensions(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  ctx.appendDialectRegistry(registry);

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeNativeTargetAsmPrinter();

  // Resolve backend (without working CLI flags for now)
  backend_ = resolveBackend(0, nullptr);
}

JITEngine::~JITEngine() {
  // Safe here specifically because profiler_ is a member subobject, not a
  // separate singleton -- see its declaration's comment (JITEngine.h).
  profiler_.report();

  if (Py_IsInitialized()) {
    Py_FinalizeEx();
  }
}

void JITEngine::setFlushCallback(FlushCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  flushCallback_ = std::move(callback);
}

void JITEngine::enqueueParLoop(std::uintptr_t kernelToken,
                                    const char *kernelName, ops_block block,
                                    int dims, const int *range,
                                    const ops_arg *args, std::size_t nargs) {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.push_back(
      buildLoopDesc(kernelToken, kernelName, block, dims, range, args, nargs));
}

void JITEngine::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (flushCallback_ && !queue_.empty()) {
    flushCallback_(queue_);
  }
  queue_.clear();
}

std::string JITEngine::detectNVGpuSm() {
  if (const char *env = std::getenv("OPS_GPU_SM"))
    return env;

#ifdef OPS_ENABLE_CUDA
  if (cuInit(0) != CUDA_SUCCESS) {
    throw std::runtime_error(
        "cuInit failed -- no NVIDIA driver found. Set OPS_GPU_SM manually (e.g. \"sm_86\").");
  }
  CUdevice device;
  cuDeviceGet(&device, 0);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  return "sm_" + std::to_string(major) + std::to_string(minor);
#else
  throw std::runtime_error(
      "This build was compiled without CUDA support (OPS_ENABLE_CUDA=OFF). "
      "Reconfigure with -DOPS_ENABLE_CUDA=ON to use the CUDA backend, or "
      "set OPS_GPU_SM manually (e.g. \"sm_86\") if targeting a remote/"
      "precompiled binary.");
#endif
}

LoopDesc JITEngine::buildLoopDesc(std::uintptr_t kernelToken,
                                       const char *kernelName, ops_block block,
                                       int dims, const int *range,
                                       const ops_arg *args, std::size_t nargs) {
  LoopDesc loop;
  loop.kernel_name = kernelName;
  loop.kernel_token = kernelToken;
  loop.block = reinterpret_cast<std::uintptr_t>(block);
  loop.dims = dims;

  loop.range.assign(range, range + 2 * dims);

  loop.args.reserve(nargs);
  for (std::size_t i = 0; i < nargs; ++i) {
    loop.args.push_back(buildArgDesc(args[i]));
  }

  return loop;
}

ArgDesc JITEngine::buildArgDesc(const ops_arg &arg) {
  ArgDesc desc;
  desc.dim = arg.dim;
  desc.elem_size = arg.elem_size;
  desc.data = reinterpret_cast<std::uintptr_t>(arg.data);
  desc.data_d = reinterpret_cast<std::uintptr_t>(arg.data_d);
  desc.acc = arg.acc;
  desc.argtype = arg.argtype;
  desc.opt = arg.opt;

  if (arg.argtype == OPS_ARG_DAT) {
    if (arg.dat)
      desc.dat = describeDat(arg.dat);
    if (arg.stencil)
      desc.stencil = describeStencil(arg.stencil);
  }

  return desc;
}

DatDesc JITEngine::describeDat(ops_dat dat) {
  DatDesc d;
  d.handle = reinterpret_cast<std::uintptr_t>(dat); // TODO: Casting pointers to uintptr_t is not portable. We should use a better way to represent pointers in MLIR attributes.
  d.index = dat->index;
  d.block = reinterpret_cast<std::uintptr_t>(dat->block); // TODO: Casting pointers to uintptr_t is not portable. We should use a better way to represent pointers in MLIR attributes.
  
  d.dim = dat->dim;
  d.type_size = dat->type_size;
  d.elem_size = dat->elem_size;

  int ndim = dat->block ? dat->block->dims : 1;
  for (int i = 0; i < ndim; ++i) {
    d.size.push_back(dat->size[i]);
    d.base.push_back(dat->base[i]);
    d.d_m.push_back(dat->d_m[i]);
    d.d_p.push_back(dat->d_p[i]);
    d.stride.push_back(dat->stride[i]);
  }

  d.name = dat->name ? dat->name : "";
  d.type = dat->type ? dat->type : "";

  d.data = reinterpret_cast<std::uintptr_t>(dat->data);
  d.data_d = reinterpret_cast<std::uintptr_t>(dat->data_d);

  return d;
}

StencilDesc JITEngine::describeStencil(ops_stencil stencil) {
  StencilDesc s;
  s.index = stencil->index;
  s.dims = stencil->dims;
  s.points = stencil->points;
  s.name = stencil->name ? stencil->name : "";

  s.stencil = reinterpret_cast<std::uintptr_t>(stencil->stencil);
  s.stride = reinterpret_cast<std::uintptr_t>(stencil->stride);
  s.mgrid_stride = reinterpret_cast<std::uintptr_t>(stencil->mgrid_stride);
  s.type = stencil->type;

  return s;
}

static std::string fetchPyError() {
  if (!PyErr_Occurred())
    return "unknown Python error";
  PyObject *type, *value, *tb;
  PyErr_Fetch(&type, &value, &tb);
  PyErr_NormalizeException(&type, &value, &tb);
  std::string msg = "unknown Python error";
  if (value) {
    PyObject *str = PyObject_Str(value);
    if (str) {
      if (const char *s = PyUnicode_AsUTF8(str)) msg = s;
      Py_DECREF(str);
    }
  }
  Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(tb);
  return msg;
}

XdslResult JITEngine::runXdslLowering(const std::string &ir) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  XdslResult result;

  PyObject *mod = PyImport_ImportModule("ops_to_xdsl");
  if (!mod) {
    result.error = fetchPyError();
    PyGILState_Release(gstate);
    return result;
  }

  PyObject *func = PyObject_GetAttrString(mod, "convert_ir_text");
  Py_DECREF(mod);
  if (!func || !PyCallable_Check(func)) {
    result.error = fetchPyError();
    Py_XDECREF(func);
    PyGILState_Release(gstate);
    return result;
  }

  PyObject *args = Py_BuildValue("(s)", ir.c_str());
  if (!args) {
    result.error = fetchPyError();
    Py_DECREF(func);
    PyGILState_Release(gstate);
    return result;
  }
  PyObject *pyResult = PyObject_CallObject(func, args);
  Py_DECREF(args);
  Py_DECREF(func);

  if (!pyResult) {
    result.error = fetchPyError();
    PyGILState_Release(gstate);
    return result;
  }

  if (const char *text = PyUnicode_AsUTF8(pyResult)) {
    result.ir = text;
    result.success = true;
  } else {
    result.error = fetchPyError();
  }
  Py_DECREF(pyResult);

  PyGILState_Release(gstate);
  return result;
}

void JITEngine::runBackendLowering(mlir::ModuleOp module, Backend backend) {
  std::unique_ptr<BackendPipeline> pipeline;

  switch (backend) {
  case Backend::Sequential:
    pipeline = std::make_unique<CPUSequentialPipeline>();
    break;
  case Backend::OpenMP:
    pipeline = std::make_unique<OpenMPPipeline>();
    break;
  case Backend::CUDA:
#ifdef OPS_ENABLE_CUDA
    pipeline = std::make_unique<CudaPipeline>(detectNVGpuSm());
#else
    throw std::runtime_error(
        "CUDA backend requested but this build was compiled without "
        "CUDA support (OPS_ENABLE_CUDA=OFF).");
#endif
    break;
  }

  if (!pipeline) {
    llvm::errs() << "no lowering pipeline for requested backend\n";
    return;
  }

  if (mlir::failed(pipeline->run(module, ctx))) {
    llvm::errs() << "backend lowering failed for module\n";
    return;
  }

  // Kept alive for compile_and_execute() -- see currentPipeline_'s
  // comment (JITEngine.h).
  currentPipeline_ = std::move(pipeline);

#ifdef OPS_ENABLE_DEBUG
  llvm::outs() << "=== BACKEND-LOWERED MLIR IR ===\n\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs().flush();
#endif
}

void JITEngine::compile() {
  module = builder.buildModule(queue_);

  std::string ir = builder.moduleToString(module);

#ifdef OPS_ENABLE_DEBUG
  llvm::outs() << "=== OPS.PAR_LOOP MLIR IR ===\n\n" << ir << "\n";
  llvm::outs().flush();
#endif

  XdslResult lowered = runXdslLowering(ir);
  if (!lowered.success) {
    llvm::errs() << "xDSL lowering failed: " <<  lowered.error << "\n";
    return;
  }
  
#ifdef OPS_ENABLE_DEBUG
  llvm::outs() << "=== LOWERED STENCIL IR (xDSL, in-process) ===\n\n"
            << lowered.ir << "\n";
  llvm::outs().flush();
#endif

  loweredModule_ =
    mlir::parseSourceString<mlir::ModuleOp>(lowered.ir, &ctx);

  if (!loweredModule_) {
    llvm::errs() << "Failed to parse xDSL output as MLIR\n";
    return;
  }

  std::vector<std::pair<std::string, int>> kernels;
  for (const LoopDesc &loop : queue_) {
    bool seen = std::any_of(
        kernels.begin(), kernels.end(),
        [&](const auto &k) { return k.first == loop.kernel_name; });
    if (!seen)
      kernels.emplace_back(loop.kernel_name, loop.dims);
  }
  for (const auto &[name, dims] : kernels) {
    materializeKernelBody(name, dims);
  }

  runBackendLowering(*loweredModule_, backend_);
  module = *loweredModule_;
}

bool JITEngine::materializeKernelBody(const std::string &kernelName,
                                      int indexRank) {
  if (kernelSourceFile_.empty()) {
    llvm::errs() << "materializeKernelBody: no kernel source file set "
                    "(call setKernelSourceFile), cannot translate '"
                 << kernelName << "'\n";
    return false;
  }

  mlir::func::FuncOp declOp;
  loweredModule_->walk([&](mlir::func::FuncOp fn) {
    if (fn.getSymName() == kernelName && fn.isDeclaration())
      declOp = fn;
  });
  if (!declOp) {
    // Already materialized (e.g. a prior loop with the same kernel name),
    // or not present in this module.
    return true;
  }

  std::vector<int> constArgDims;
  for (const LoopDesc &loop : queue_) {
    if (loop.kernel_name != kernelName)
      continue;
    for (const ArgDesc &arg : loop.args) {
      if (arg.argtype == OPS_ARG_GBL && arg.acc == OPS_READ) {
        // NOTE: still relies on matching by declared C++ parameter name --
        // see caveat below.
        constArgDims.push_back(arg.dim);
      }
    }
    break; // first matching loop's arg shapes are sufficient
  }


  KernelIRBuilder kernelBuilder(ctx);
  mlir::func::FuncOp translatedFn = kernelBuilder.generate(
      kernelSourceFile_, kernelName, indexRank, kernelConstants_, constArgDims, llvm::errs());
  if (!translatedFn) {
    llvm::errs() << "materializeKernelBody: could not translate '"
                 << kernelName << "' from " << kernelSourceFile_ << "\n";
    return false;
  }

  declOp.erase();
  loweredModule_->push_back(translatedFn);
  return true;
}

static std::size_t datByteSize(const DatDesc &dat) {
  std::size_t total = static_cast<std::size_t>(dat.elem_size);
  for (int64_t dim : dat.size)
    total *= static_cast<std::size_t>(dim);
  return total;
}

std::uintptr_t JITEngine::ensureDeviceBuffer(std::uintptr_t hostPtr,
                                             std::size_t bytes) {
#ifdef OPS_ENABLE_CUDA
  auto it = deviceBuffers_.find(hostPtr);
  if (it != deviceBuffers_.end()) {
    // A host-side mutation we don't observe as an ops_par_loop (e.g. an
    // intervening ops_halo_transfer) may have changed the host buffer
    // since this device copy was made -- see invalidateDeviceBuffers().
    if (it->second.dirty) {
      cuMemcpyHtoD(static_cast<CUdeviceptr>(it->second.devPtr),
                  reinterpret_cast<const void *>(hostPtr), bytes);
      it->second.dirty = false;
    }
    return it->second.devPtr;
  }

  // Use the same (primary) CUDA context mlir_cuda_runtime's wrappers use
  // (see CudaRuntimeWrappers.cpp's ScopedContext), so buffers allocated
  // here are valid in the kernels that runtime launches.
  static bool contextReady = [] {
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext ctx;
    cuDevicePrimaryCtxRetain(&ctx, device);
    cuCtxSetCurrent(ctx);
    return true;
  }();
  (void)contextReady;

  CUdeviceptr devPtr = 0;
  if (CUresult rc = cuMemAlloc(&devPtr, bytes); rc != CUDA_SUCCESS) {
    llvm::errs() << "ensureDeviceBuffer: cuMemAlloc failed (CUresult " << rc
                 << ")\n";
    return 0;
  }
  deviceBuffers_[hostPtr] = {static_cast<std::uintptr_t>(devPtr), false};
  cuMemcpyHtoD(devPtr, reinterpret_cast<const void *>(hostPtr), bytes);
  return devPtr;
#else
  (void)hostPtr;
  (void)bytes;
  return 0;
#endif
}

std::uintptr_t JITEngine::ensurePersistentCudaStream() {
#ifdef OPS_ENABLE_CUDA
  static CUstream stream = [] {
    CUstream s = nullptr;
    if (CUresult rc = cuStreamCreate(&s, CU_STREAM_DEFAULT); rc != CUDA_SUCCESS) {
      llvm::errs() << "ensurePersistentCudaStream: cuStreamCreate failed "
                      "(CUresult "
                   << rc << ")\n";
      return static_cast<CUstream>(nullptr);
    }
    return s;
  }();
  return reinterpret_cast<std::uintptr_t>(stream);
#else
  return 0;
#endif
}

// Free function so it has a stable, unmangled name to declare/call from
// generated LLVM IR and to hand to registerSymbols -- matches how
// KernelProfiler etc. get exposed, just at the raw-address level instead of
// a mlir_ciface wrapper.
extern "C" void *ops_mlir_get_persistent_cuda_stream() {
  return reinterpret_cast<void *>(JITEngine::instance().ensurePersistentCudaStream());
}

// TODO: Use dat.data_d instead of deviceBuffers_.
void JITEngine::invalidateDeviceBuffer(std::uintptr_t hostPtr) {
  auto it = deviceBuffers_.find(hostPtr);
  if (it != deviceBuffers_.end())
    it->second.dirty = true;
}

void haloTransferIntercepted(ops_halo_group group) {
  ::ops_halo_transfer(group);
  // Only the dats this group actually writes (`to`) need invalidating --
  // `from` isn't mutated by the copy, and every other cached dat is
  // untouched by this call. See invalidateDeviceBuffer's comment for why
  // that distinction matters.
  for (int i = 0; i < group->nhalos; ++i) {
    ops_dat to = group->halos[i]->to;
    JITEngine::instance().invalidateDeviceBuffer(
        reinterpret_cast<std::uintptr_t>(to->data));
  }
}

void JITEngine::shutdown() {
  engineCache_.clear();
}

void exitIntercepted() {
  JITEngine::instance().shutdown();
  ::ops_exit();
}

void JITEngine::synchronizeBackend(Backend backend) {
  switch (backend) {
  case Backend::Sequential:
  case Backend::OpenMP:
    return;
  case Backend::CUDA:
#ifdef OPS_ENABLE_CUDA
    cuCtxSynchronize();
#endif
    return;
  }
}

void JITEngine::execute(mlir::ExecutionEngine &engine) {
  // Each ops.par_loop was lowered to a standalone function named
  // "ops_par_loop_<kernel_name>_<queue_index>" taking one bare pointer per
  // ops_dat argument, in the order the loops were enqueued. We invoke them
  // one by one with the live data pointers.
  //
  // For the CUDA backend, the compiled kernel operates on device memory,
  // not the host `ops_dat` buffer: each dat arg is mirrored into a device
  // buffer (allocated/cached by ensureDeviceBuffer), copied host->device
  // before the launch, and copied back device->host afterwards for any
  // dat the kernel writes, so host code and later CPU-side loops always
  // see up-to-date contents.
  for (std::size_t i = 0; i < queue_.size(); ++i) {
    const LoopDesc &loop = queue_[i];
    std::string funcName =
        "ops_par_loop_" + loop.kernel_name + "_" + std::to_string(i);

    std::vector<void *> argPtrs;
    std::vector<std::pair<std::uintptr_t, std::size_t>> writebacks;
    for (const ArgDesc &arg : loop.args) {
      if (arg.argtype != OPS_ARG_DAT && arg.argtype != OPS_ARG_GBL)
        continue;

      if (arg.argtype == OPS_ARG_GBL) {
        if (arg.acc == OPS_READ) {
          // Broadcast read-only constant
          argPtrs.push_back(reinterpret_cast<void *>(arg.data));
        } else {
          ops_reduction handle = reinterpret_cast<ops_reduction>(arg.data);
          argPtrs.push_back(reinterpret_cast<void *>(handle->data));

        }
        continue;
      }

      if (backend_ == Backend::CUDA) {
        std::size_t bytes = datByteSize(arg.dat);
        std::uintptr_t devPtr = ensureDeviceBuffer(arg.data, bytes);
        if (!devPtr) {
          llvm::errs() << "Failed to allocate device buffer for '"
                       << funcName << "'\n";
          this->flush();
          return;
        }
        argPtrs.push_back(reinterpret_cast<void *>(devPtr));
        if (arg.acc == OPS_WRITE || arg.acc == OPS_RW || arg.acc == OPS_INC)
          writebacks.emplace_back(arg.data, bytes);
      } else {
        argPtrs.push_back(reinterpret_cast<void *>(arg.data));
      }
    }

    llvm::SmallVector<void *> packedArgs;
    packedArgs.reserve(argPtrs.size());
    for (void *&ptr : argPtrs) {
      packedArgs.push_back(&ptr);
    }

#ifdef OPS_ENABLE_DEBUG
    llvm::outs() << "About to invoke '" << funcName << "'\n";
    llvm::outs().flush();
#endif
    auto kernelStart = profiler_.start();
    if (auto err = engine.invokePacked(funcName, packedArgs)) {
      llvm::errs() << "Failed to invoke '" << funcName
                  << "': " << llvm::toString(std::move(err)) << "\n";
      this->flush();
      return;
    }
    synchronizeBackend(backend_);
    profiler_.end(loop.kernel_name, kernelStart);
#ifdef OPS_ENABLE_CUDA
    for (const auto &[hostPtr, bytes] : writebacks) {
      std::uintptr_t devPtr = ensureDeviceBuffer(hostPtr, bytes);
      cuMemcpyDtoH(reinterpret_cast<void *>(hostPtr), devPtr, bytes);
    }
#endif
  }
  // Clear the queue
  this->flush();
}

void JITEngine::compile_and_execute() {
  if (queue_.empty())
    return;

  ModuleKey key(queue_);

  auto cached = engineCache_.find(key);
  if (cached != engineCache_.end()) {
    execute(*cached->second);
    return;
  }

  compile();

  llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());
  std::string targetLookupError;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, targetLookupError);
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  if (!target) {
    llvm::errs() << "Warning: could not look up target for '"
                 << targetTriple.str() << "': " << targetLookupError
                 << "; falling back to no target machine (no "
                    "auto-vectorization)\n";
  } else {
    llvm::SubtargetFeatures features;
    for (auto &[feature, enabled] : llvm::sys::getHostCPUFeatures())
      features.AddFeature(feature, enabled);

    llvm::TargetOptions targetOptions;
    targetMachine.reset(target->createTargetMachine(
        targetTriple, llvm::sys::getHostCPUName(), features.getString(),
        targetOptions, /*RM=*/std::nullopt, /*CM=*/std::nullopt,
        llvm::CodeGenOptLevel::Aggressive));
  }

  mlir::ExecutionEngineOptions engineOptions;
  auto optTransformer = mlir::makeOptimizingTransformer(
    /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/targetMachine.get());

  std::function<llvm::Error(llvm::Module *)> transformer =
      [this, optTransformer](llvm::Module *m) -> llvm::Error {
    if (currentPipeline_)
      if (auto err = currentPipeline_->transformLLVMModule(m))
        return err;
    return optTransformer(m);
  };
  engineOptions.transformer = transformer;

  // gpu.launch_func lowers to calls into MLIR's CUDA driver-API wrappers
  // (mgpuLaunchKernel, mgpuStreamCreate, ...); the JIT needs
  // libmlir_cuda_runtime.so loaded to resolve them. Path is
  // environment/build-specific (this LLVM tree's MLIR_ENABLE_CUDA_RUNNER
  // build), so it's read from an env var rather than hardcoded.
  std::string cudaRuntimePath;
  if (backend_ == Backend::CUDA) {
    if (const char *path = std::getenv("OPS_MLIR_CUDA_RUNTIME")) {
      cudaRuntimePath = path;
      engineOptions.sharedLibPaths = {cudaRuntimePath};
    } else {
      llvm::errs() << "Warning: OPS_MLIR_CUDA_RUNTIME not set; GPU kernel "
                     "launches will fail to resolve mgpu* symbols. Set it "
                     "to the path of libmlir_cuda_runtime.so.\n";
    }
  }

  auto engineOrErr = mlir::ExecutionEngine::create(module, engineOptions);
  if (!engineOrErr) {
    llvm::errs() << "Failed to create ExecutionEngine: "
                  << llvm::toString(engineOrErr.takeError()) << "\n";
    return;
  }

  auto engine = std::move(*engineOrErr);

  // TODO: [For inlining] Kernel are generate from KernelIRBuilder for all target.
  // switch (backend_) {
  // case Backend::Sequential:
  // case Backend::OpenMP:
  //   registerCpuKernelSymbols(*engine);
  //   break;
  // case Backend::CUDA:
  //   // Kernels should be added to the Host side. CPU side symbols are not valid.
  //   break;
  // }

  if (backend_ == Backend::CUDA) {
    engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
      llvm::orc::SymbolMap symbolMap;
      symbolMap[interner("ops_mlir_get_persistent_cuda_stream")] = {
          llvm::orc::ExecutorAddr::fromPtr(&ops_mlir_get_persistent_cuda_stream),
          llvm::JITSymbolFlags::Exported};
      return symbolMap;
    });
  }

  mlir::ExecutionEngine &engineRef = *engine;
  engineCache_.emplace(std::move(key), std::move(engine));
  execute(engineRef);
}

void JITEngine::registerCpuKernelSymbols(mlir::ExecutionEngine &engine) {
  engine.registerSymbols([this](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    for (const LoopDesc &loop : queue_) {
      void *kernelPtr = reinterpret_cast<void *>(loop.kernel_token);
      symbolMap[interner(loop.kernel_name)] = {
          llvm::orc::ExecutorAddr::fromPtr(kernelPtr),
          llvm::JITSymbolFlags::Exported};
    }
    return symbolMap;
  });
}

const char *accessToString(int access) {
  switch (access) {
  case OPS_READ:
    return "READ";
  case OPS_WRITE:
    return "WRITE";
  case OPS_RW:
    return "RW";
  case OPS_INC:
    return "INC";
  case OPS_MIN:
    return "MIN";
  case OPS_MAX:
    return "MAX";
  default:
    return "UNKNOWN";
  }
}

const char *argKindToString(ArgKind kind) {
  switch (kind) {
  case ArgKind::Dat:
    return "Dat";
  case ArgKind::Gbl:
    return "Gbl";
  case ArgKind::Idx:
    return "Idx";
  case ArgKind::Reduce:
    return "Reduce";
  default:
    return "Unknown";
  }
}

std::optional<Backend> parseBackendName(const std::string &name) {
  if (name == "seq" || name == "sequential") return Backend::Sequential;
  if (name == "openmp" || name == "omp") return Backend::OpenMP;
  if (name == "cuda" || name == "nvgpu") return Backend::CUDA; 
  return std::nullopt;
}


Backend JITEngine::resolveBackend(int argc, char **argv) {
  // Explicit CLI flag takes precendence
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind(kBackendFlagPrefix, 0) == 0) {
      std::string value = arg.substr(std::string(kBackendFlagPrefix).size());
      if (auto b = parseBackendName(value)) return *b;
      throw std::runtime_error("Unknown --backend value: '" + value + "' (expected seq|openmp|cuda)");
    }
  }

  // Fall back to env variable
  if (const char *env = std::getenv(kBackendEnvVar)) {
    if (auto b = parseBackendName(env)) return *b;
    throw std::runtime_error("Unknown " + std::string(kBackendEnvVar) + " value: '" + env + "' (expected seq|openmp|cuda)");
  }

  // Default to sequential if neither cli flag or env var set
  return kDefaultBackend;
}

} // namespace ops_mlir
