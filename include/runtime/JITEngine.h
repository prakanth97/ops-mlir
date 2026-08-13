#ifndef OPS_CAPTURE_H
#define OPS_CAPTURE_H

#include "IRBuilder.h"
#include "Core.h"
#include "runtime/KernelProfiler.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>


namespace ops_mlir {

enum class Backend { Sequential, OpenMP, CUDA, ROCM };

std::optional<Backend> parseBackendName(const std::string &name);
constexpr Backend kDefaultBackend = Backend::CUDA;

static constexpr const char *kBackendFlagPrefix = "--backend=";
static constexpr const char *kBackendEnvVar = "OPS_BACKEND";

struct XdslResult {
  bool success = false;
  std::string ir;      // populated on success
  std::string error;   // populated on failure
};

class ModuleKey {
public:
  explicit ModuleKey(const std::vector<LoopDesc> &queue) {
    for (const LoopDesc &loop : queue) {
      digest_ += loop.kernel_name;
      digest_ += '|';
      digest_ += std::to_string(loop.dims);
      digest_ += '|';
      for (int64_t r : loop.range) {
        digest_ += std::to_string(r);
        digest_ += ',';
      }
      for (const ArgDesc &arg : loop.args) {
        digest_ += '[';
        digest_ += std::to_string(arg.argtype);
        digest_ += ';';
        digest_ += std::to_string(arg.acc);
        digest_ += ';';
        digest_ += std::to_string(arg.dim);
        digest_ += ';';
        digest_ += std::to_string(arg.elem_size);
        digest_ += ';';
        digest_ += std::to_string(arg.opt);

        const DatDesc &dat = arg.dat;
        digest_ += ";dat:";
        digest_ += std::to_string(dat.dim);
        digest_ += ',';
        digest_ += std::to_string(dat.type_size);
        digest_ += ',';
        digest_ += std::to_string(dat.elem_size);
        digest_ += ',';
        digest_ += dat.type;
        for (int64_t v : dat.size) { digest_ += ','; digest_ += std::to_string(v); }
        for (int64_t v : dat.base) { digest_ += ','; digest_ += std::to_string(v); }
        for (int64_t v : dat.d_m) { digest_ += ','; digest_ += std::to_string(v); }
        for (int64_t v : dat.d_p) { digest_ += ','; digest_ += std::to_string(v); }
        for (int64_t v : dat.stride) { digest_ += ','; digest_ += std::to_string(v); }

        const StencilDesc &st = arg.stencil;
        digest_ += ";st:";
        digest_ += std::to_string(st.dims);
        digest_ += ',';
        digest_ += std::to_string(st.points);
        digest_ += ',';
        digest_ += std::to_string(st.type);
        if (st.stencil) {
          const int *offsets = reinterpret_cast<const int *>(st.stencil);
          for (int i = 0; i < st.dims * st.points; ++i) {
            digest_ += ',';
            digest_ += std::to_string(offsets[i]);
          }
        }
        digest_ += ']';
      }
      digest_ += '\n';
    }
  }

  [[nodiscard]] const std::string &digest() const { return digest_; }

  bool operator==(const ModuleKey &other) const {
    return digest_ == other.digest_;
  }

  [[nodiscard]] std::size_t hash() const {
    return std::hash<std::string>{}(digest_);
  }

private:
  std::string digest_;
};

} // namespace ops_mlir

namespace std {
template <> struct hash<ops_mlir::ModuleKey> {
  std::size_t operator()(const ops_mlir::ModuleKey &key) const {
    return key.hash();
  }
};
} // namespace std

namespace ops_mlir {

class BackendPipeline;

class JITEngine {
public:
  using FlushCallback = std::function<void(const std::vector<LoopDesc> &)>;

  static JITEngine &instance();

  void setFlushCallback(FlushCallback callback);

  void enqueueParLoop(std::uintptr_t kernelToken, const char *kernelName,
                      ops_block block, int dims, const int *range,
                      const ops_arg *args, std::size_t nargs);

  void flush();

  void compile_and_execute();

  void setBackend(Backend backend) { backend_ = backend; }
  Backend backend() const { return backend_; }

  // Needed for GPU backend kernel translation (via KernelIRBuilder) to materialize real MLIR
  void setKernelSourceFile(std::string path) {
    kernelSourceFile_ = std::move(path);
  }

  // Register extern global kernel constants (e.g. pi, jmax) for translation into MLIR 
  void registerKernelConstant(const std::string &name, const void *ptr) {
    kernelConstants_[name] = ptr;
  }

  const std::map<std::string, const void *> &kernelConstants() const {
    return kernelConstants_;
  }

  Backend resolveBackend(int argc, char **argv);

  const std::vector<LoopDesc> &queue() const { return queue_; }

  KernelProfiler &profiler() { return profiler_; }

private:
  JITEngine();
  ~JITEngine();

  mlir::MLIRContext ctx;
  mlir::ModuleOp module;
  mlir::OwningOpRef<mlir::ModuleOp> loweredModule_;

  IRBuilder builder{&ctx};

  LoopDesc buildLoopDesc(std::uintptr_t kernelToken, const char *kernelName,
                         ops_block block, int dims, const int *range,
                         const ops_arg *args, std::size_t nargs);

  ArgDesc buildArgDesc(const ops_arg &arg);

  DatDesc describeDat(ops_dat dat);
  StencilDesc describeStencil(ops_stencil stencil);

  XdslResult runXdslLowering(const std::string &ir);

  void runBackendLowering(mlir::ModuleOp module, Backend backend);
  std::string detectNVGpuSm();

  void compile();
  void execute(mlir::ExecutionEngine &engine);
  void registerCpuKernelSymbols(mlir::ExecutionEngine &engine);

  // Translates a kernel body from C++ source into MLIR via KernelIRBuilder,
  // splicing it in to replace the `func.func private @kernel(...)`
  // declaration xDSL lowering leaves behind -- required for CUDA (device
  // code can't call back into host object code) and used for the CPU
  // backends too so the kernel body can be inlined into the surrounding
  // loop rather than staying an opaque call through a symbol bound by
  // registerCpuKernelSymbols. Returns false (leaving the declaration in
  // place) if the kernel isn't already materialized and translation fails
  // or is unsupported, so callers can fall back to symbol binding.
  bool materializeKernelBody(const std::string &kernelName, int indexRank);

  // Returns the device buffer mirroring the given host `ops_dat` buffer,
  // allocating it (via cuMemAlloc) on first use. Kernels compiled for the
  // CUDA backend operate on device memory, not the host pointers `dat`
  // args normally carry, so execute() copies host->device before each
  // launch and device->host after for any dat the kernel writes. A no-op
  // returning 0 when built without OPS_ENABLE_CUDA.
  //
  // Note: unconditionally declared (not #ifdef OPS_ENABLE_CUDA) even
  // though only meaningful for CUDA builds -- OPS_ENABLE_CUDA is only
  // defined PRIVATE for the OPSRuntime target, so consumer translation
  // units (e.g. apps linking against it) never see that macro; gating a
  // class member on it here would make JITEngine's layout depend on
  // which TU compiles it, an ODR violation.
  std::uintptr_t ensureDeviceBuffer(std::uintptr_t hostPtr, std::size_t bytes);

  void synchronizeBackend(Backend backend);

public:

  void invalidateDeviceBuffer(std::uintptr_t hostPtr);

  void shutdown();

  // Returns a persistent CUDA stream (CUstream) for kernels to launch on.
  std::uintptr_t ensurePersistentCudaStream();

private:
  Backend backend_ = kDefaultBackend;
  std::mutex mutex_;
  std::vector<LoopDesc> queue_;
  FlushCallback flushCallback_;
  std::string kernelSourceFile_;
  std::map<std::string, const void *> kernelConstants_;

  std::unordered_map<ModuleKey, std::unique_ptr<mlir::ExecutionEngine>>
      engineCache_;

  std::unique_ptr<BackendPipeline> currentPipeline_;

  KernelProfiler profiler_;

  // Host ops_dat pointer -> cached device buffer (stored as uintptr_t to
  // keep this header CUDA-toolkit-header-free; only populated/used when
  // built with OPS_ENABLE_CUDA). `dirty` means the device copy no longer
  // reflects the host buffer and must be re-copied before next use --
  // see invalidateDeviceBuffers().
  struct DeviceBufferEntry {
    std::uintptr_t devPtr;
    bool dirty = false;
  };
  std::map<std::uintptr_t, DeviceBufferEntry> deviceBuffers_;
};

// Forwards to the real ops_halo_transfer (linked in from the OPS host
// library) and then invalidates JITEngine's CUDA device-buffer cache --
// see JITEngine::invalidateDeviceBuffers's comment for why. OPSWrapper.h
// #defines ops_halo_transfer to route call sites through this.
void haloTransferIntercepted(ops_halo_group group);

void exitIntercepted();

const char *accessToString(int access);
const char *argKindToString(ArgKind kind);

} // namespace ops_mlir

#endif // OPS_CAPTURE_H
