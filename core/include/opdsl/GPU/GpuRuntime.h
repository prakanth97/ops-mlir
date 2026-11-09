//===- GpuRuntime.h - a vendor GPU runtime behind the mgpu* symbols ----*- C++ -*-===//
//
// GpuShim.cpp implements the mgpu* symbols once, on this interface; CudaRuntime.cpp (libcuda) and HipRuntime.cpp
// (libamdhip64) implement it. Both are always built (the drivers are loaded with dlopen); get() chooses one per process.
// Calls other than available(), name(), deviceArch() and moduleUnload() require available(): GpuShim checks it.
//
//===----------------------------------------------------------------------===//

#ifndef OPDSL_GPU_GPURUNTIME_H
#define OPDSL_GPU_GPURUNTIME_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace opdsl::gpu {

class GpuRuntime {
public:
  virtual ~GpuRuntime() = default;

  /// "cuda" or "hip".
  virtual const char *name() const = 0;
  /// A driver and at least one device are usable. Loads the driver on first call.
  virtual bool available() = 0;
  /// "sm_89", "gfx90a"; empty if unknown.
  virtual std::string deviceArch() = 0;

  /// Makes the device (and, for CUDA, a context) current on this thread.
  virtual void makeCurrent() = 0;
  virtual void *moduleLoad(void *data, size_t size) = 0;
  virtual void *moduleLoadJIT(void *data, int optLevel) = 0;
  /// Errors are ignored: at exit the driver may already be torn down.
  virtual void moduleUnload(void *module) = 0;
  virtual void *moduleGetFunction(void *module, const char *name) = 0;
  virtual void launch(void *function, intptr_t gridX, intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                      intptr_t blockZ, int32_t sharedMem, void *stream, void **params, void **extra) = 0;
  virtual void streamSynchronize(void *stream) = 0;
  virtual void deviceSynchronize() = 0;
  virtual void *alloc(uint64_t sizeBytes, bool hostShared) = 0;
  virtual void free(void *ptr) = 0;
  /// Complete when it returns (the callers read the host side right after).
  virtual void memcpy(void *dst, void *src, size_t sizeBytes, void *stream) = 0;
  virtual void memset32(void *dst, unsigned value, size_t count, void *stream) = 0;
  virtual void setDefaultDevice(int device) = 0;

  /// The runtime of this process: OPS_JIT_GPU=cuda|hip, else the first available of CUDA and HIP.
  static GpuRuntime &get();
};

std::unique_ptr<GpuRuntime> createCudaRuntime();
std::unique_ptr<GpuRuntime> createHipRuntime();

} // namespace opdsl::gpu

#endif // OPDSL_GPU_GPURUNTIME_H
