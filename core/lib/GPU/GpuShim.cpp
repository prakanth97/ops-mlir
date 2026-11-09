//===- GpuShim.cpp - the mgpu* symbols, on the GpuRuntime of the process ----------===//

#include "opdsl/GPU/GpuShim.h"

#include "opdsl/GPU/GpuRuntime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace opdsl::gpu {

namespace {

bool debugEnabled() {
  static bool on = std::getenv("OPS_JIT_GPU_DEBUG") != nullptr;
  return on;
}

bool syncEnabled() {
  static bool on = [] {
    const char *v = std::getenv("OPS_JIT_SYNC");
    return v && v[0] == '1';
  }();
  return on;
}

/// The runtime, with the device current on this thread; aborts if there is none.
GpuRuntime &ready() {
  GpuRuntime &r = GpuRuntime::get();
  if (!r.available()) {
    std::fprintf(stderr, "ops_jit: no GPU runtime or device available (%s)\n", r.name());
    std::abort();
  }
  r.makeCurrent();
  return r;
}

} // namespace

GpuRuntime &GpuRuntime::get() {
  static std::unique_ptr<GpuRuntime> runtime = [] {
    std::unique_ptr<GpuRuntime> cuda = createCudaRuntime(), hip = createHipRuntime();
    if (const char *want = std::getenv("OPS_JIT_GPU"); want && *want) {
      if (!std::strcmp(want, "cuda")) return cuda;
      if (!std::strcmp(want, "hip")) return hip;
      std::fprintf(stderr, "ops_jit: ignoring OPS_JIT_GPU=%s (cuda or hip)\n", want);
    }
    if (cuda->available() || !hip->available()) return cuda;
    return hip;
  }();
  return *runtime;
}

const char *gpuRuntime() { return GpuRuntime::get().name(); }

bool gpuAvailable() { return GpuRuntime::get().available(); }

const char *gpuDeviceArch() {
  static std::string arch = GpuRuntime::get().deviceArch();
  return arch.c_str();
}

void *gpuShimAnchor() { return reinterpret_cast<void *>(&mgpuLaunchKernel); }

} // namespace opdsl::gpu

using opdsl::gpu::debugEnabled;
using opdsl::gpu::ready;
using opdsl::gpu::syncEnabled;

extern "C" {

void *mgpuModuleLoad(void *data, size_t size) {
  auto &r = ready();
  if (debugEnabled()) std::fprintf(stderr, "ops_jit: loading a GPU module of %zu bytes\n", size);
  return r.moduleLoad(data, size);
}

void *mgpuModuleLoadJIT(void *data, int optLevel, size_t) { return ready().moduleLoadJIT(data, optLevel); }

void mgpuModuleUnload(void *module) {
  if (module) opdsl::gpu::GpuRuntime::get().moduleUnload(module);
}

void *mgpuModuleGetFunction(void *module, const char *name) {
  auto &r = ready();
  if (debugEnabled()) std::fprintf(stderr, "ops_jit: kernel %s of module %p\n", name, module);
  return r.moduleGetFunction(module, name);
}

void mgpuLaunchKernel(void *function, intptr_t gridX, intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
                      int32_t sharedMem, void *stream, void **params, void **extra, size_t) {
  auto &r = ready();
  if (debugEnabled())
    std::fprintf(stderr, "ops_jit: launch grid=%ld,%ld,%ld block=%ld,%ld,%ld smem=%d\n", (long)gridX, (long)gridY, (long)gridZ, (long)blockX,
                 (long)blockY, (long)blockZ, sharedMem);
  // `stream` is null (the legacy default stream: ordered with the application's work).
  r.launch(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, params, extra);
  if (debugEnabled() || syncEnabled()) r.deviceSynchronize();
}

void *mgpuStreamCreate() { return nullptr; }
void mgpuStreamDestroy(void *) {}
void mgpuStreamSynchronize(void *stream) {
  if (syncEnabled()) ready().streamSynchronize(stream);
}
void mgpuStreamWaitEvent(void *, void *) {}
void *mgpuEventCreate() { return nullptr; }
void mgpuEventDestroy(void *) {}
void mgpuEventSynchronize(void *) {}
void mgpuEventRecord(void *, void *) {}

void *mgpuMemAlloc(uint64_t sizeBytes, void *, bool isHostShared) {
  auto &r = ready();
  return sizeBytes == 0 ? nullptr : r.alloc(sizeBytes, isHostShared);
}

void mgpuMemFree(void *ptr, void *) {
  if (ptr) ready().free(ptr);
}

void mgpuMemcpy(void *dst, void *src, size_t sizeBytes, void *stream) { ready().memcpy(dst, src, sizeBytes, stream); }

void mgpuMemset32(void *dst, unsigned int value, size_t count, void *stream) { ready().memset32(dst, value, count, stream); }

void mgpuSetDefaultDevice(int32_t device) { opdsl::gpu::GpuRuntime::get().setDefaultDevice(device); }

} // extern "C"
