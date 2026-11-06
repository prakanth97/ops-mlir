#include "opdsl/GPU/GpuShim.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

// The driver API, as much of it as is used. Types are those of cuda.h (stable ABI), declared here so that building this
// file needs no CUDA installation.
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
constexpr int kJitErrorLogBuffer = 5, kJitErrorLogBufferSize = 6, kJitOptimizationLevel = 7;
constexpr int kAttrComputeMajor = 75, kAttrComputeMinor = 76;

struct Driver {
  void *handle = nullptr;
  CUresult (*cuInit)(unsigned) = nullptr;
  CUresult (*cuCtxGetCurrent)(void **) = nullptr;
  CUresult (*cuCtxSetCurrent)(void *) = nullptr;
  CUresult (*cuCtxSynchronize)() = nullptr;
  CUresult (*cuDeviceGet)(CUdevice *, int) = nullptr;
  CUresult (*cuDeviceGetCount)(int *) = nullptr;
  CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRetain)(void **, CUdevice) = nullptr;
  CUresult (*cuModuleLoadData)(void **, const void *) = nullptr;
  CUresult (*cuModuleLoadDataEx)(void **, const void *, unsigned, int *, void **) = nullptr;
  CUresult (*cuModuleUnload)(void *) = nullptr;
  CUresult (*cuModuleGetFunction)(void **, void *, const char *) = nullptr;
  CUresult (*cuLaunchKernel)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) = nullptr;
  CUresult (*cuMemAlloc)(CUdeviceptr *, size_t) = nullptr;
  CUresult (*cuMemAllocManaged)(CUdeviceptr *, size_t, unsigned) = nullptr;
  CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
  CUresult (*cuMemcpyAsync)(CUdeviceptr, CUdeviceptr, size_t, void *) = nullptr;
  CUresult (*cuMemsetD32Async)(CUdeviceptr, unsigned, size_t, void *) = nullptr;
  CUresult (*cuStreamSynchronize)(void *) = nullptr;
  CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;
  bool ok = false;
};

Driver driver;
std::once_flag driverOnce;
thread_local int defaultDevice = 0;

template <typename F> bool bind(F &fn, const char *name) {
  fn = reinterpret_cast<F>(dlsym(driver.handle, name));
  return fn != nullptr;
}

void loadDriver() {
  driver.handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!driver.handle) driver.handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  if (!driver.handle) return;
  bool ok = true;
  ok &= bind(driver.cuInit, "cuInit");
  ok &= bind(driver.cuCtxGetCurrent, "cuCtxGetCurrent");
  ok &= bind(driver.cuCtxSetCurrent, "cuCtxSetCurrent");
  ok &= bind(driver.cuCtxSynchronize, "cuCtxSynchronize");
  ok &= bind(driver.cuDeviceGet, "cuDeviceGet");
  ok &= bind(driver.cuDeviceGetCount, "cuDeviceGetCount");
  ok &= bind(driver.cuDeviceGetAttribute, "cuDeviceGetAttribute");
  ok &= bind(driver.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
  ok &= bind(driver.cuModuleLoadData, "cuModuleLoadData");
  ok &= bind(driver.cuModuleLoadDataEx, "cuModuleLoadDataEx");
  ok &= bind(driver.cuModuleUnload, "cuModuleUnload");
  ok &= bind(driver.cuModuleGetFunction, "cuModuleGetFunction");
  ok &= bind(driver.cuLaunchKernel, "cuLaunchKernel");
  ok &= bind(driver.cuMemAlloc, "cuMemAlloc_v2");
  ok &= bind(driver.cuMemAllocManaged, "cuMemAllocManaged");
  ok &= bind(driver.cuMemFree, "cuMemFree_v2");
  ok &= bind(driver.cuMemcpyAsync, "cuMemcpyAsync");
  ok &= bind(driver.cuMemsetD32Async, "cuMemsetD32Async");
  ok &= bind(driver.cuStreamSynchronize, "cuStreamSynchronize");
  bind(driver.cuGetErrorString, "cuGetErrorString");
  if (!ok) return;
  if (driver.cuInit(0) != 0) return;
  int count = 0;
  if (driver.cuDeviceGetCount(&count) != 0 || count == 0) return;
  driver.ok = true;
}

Driver &cuda() {
  std::call_once(driverOnce, loadDriver);
  return driver;
}

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

void check(CUresult result, const char *what) {
  if (result == 0) return;
  const char *text = "unknown error";
  if (driver.cuGetErrorString) driver.cuGetErrorString(result, &text);
  std::fprintf(stderr, "ops_jit: CUDA error %d (%s) in %s\n", result, text, what);
  std::abort();
}

/// Makes a context current on this thread: whatever the application has (its runtime-API calls made the primary
/// context current), else the primary context of the default device.
void ensureContext() {
  Driver &d = cuda();
  if (!d.ok) {
    std::fprintf(stderr, "ops_jit: no CUDA driver or device available\n");
    std::abort();
  }
  void *current = nullptr;
  check(d.cuCtxGetCurrent(&current), "cuCtxGetCurrent");
  if (current) return;
  CUdevice device;
  check(d.cuDeviceGet(&device, defaultDevice), "cuDeviceGet");
  void *context = nullptr;
  check(d.cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
  check(d.cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

} // namespace

namespace opdsl::gpu {

bool cudaAvailable() { return cuda().ok; }

const char *cudaDeviceArch() {
  static std::string arch = [] {
    Driver &d = cuda();
    if (!d.ok) return std::string();
    CUdevice device;
    if (d.cuDeviceGet(&device, defaultDevice) != 0) return std::string();
    int major = 0, minor = 0;
    d.cuDeviceGetAttribute(&major, kAttrComputeMajor, device);
    d.cuDeviceGetAttribute(&minor, kAttrComputeMinor, device);
    return "sm_" + std::to_string(major) + std::to_string(minor);
  }();
  return arch.c_str();
}

void *cudaShimAnchor() {
  return reinterpret_cast<void *>(&mgpuLaunchKernel);
}

} // namespace opdsl::gpu

extern "C" {

void *mgpuModuleLoad(void *data, size_t size) {
  ensureContext();
  if (debugEnabled()) std::fprintf(stderr, "ops_jit: loading a GPU module of %zu bytes\n", size);
  void *module = nullptr;
  check(driver.cuModuleLoadData(&module, data), "cuModuleLoadData");
  return module;
}

void *mgpuModuleLoadJIT(void *data, int optLevel, size_t) {
  ensureContext();
  void *module = nullptr;
  char log[4096] = {0};
  int options[] = {kJitErrorLogBuffer, kJitErrorLogBufferSize, kJitOptimizationLevel};
  void *values[] = {log, reinterpret_cast<void *>(sizeof(log)), reinterpret_cast<void *>(static_cast<intptr_t>(optLevel))};
  CUresult r = driver.cuModuleLoadDataEx(&module, data, 3, options, values);
  if (r != 0) std::fprintf(stderr, "ops_jit: JIT compilation of the GPU module failed: '%s'\n", log);
  check(r, "cuModuleLoadDataEx");
  return module;
}

void mgpuModuleUnload(void *module) {
  if (!driver.ok || !module) return;
  // At exit the context may already be gone: ignore the error.
  driver.cuModuleUnload(module);
}

void *mgpuModuleGetFunction(void *module, const char *name) {
  ensureContext();
  if (debugEnabled()) std::fprintf(stderr, "ops_jit: kernel %s of module %p\n", name, module);
  void *function = nullptr;
  check(driver.cuModuleGetFunction(&function, module, name), "cuModuleGetFunction");
  return function;
}

void mgpuLaunchKernel(void *function, intptr_t gridX, intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
                      int32_t sharedMem, void *stream, void **params, void **extra, size_t) {
  ensureContext();
  if (debugEnabled())
    std::fprintf(stderr, "ops_jit: launch grid=%ld,%ld,%ld block=%ld,%ld,%ld smem=%d\n", (long)gridX, (long)gridY, (long)gridZ, (long)blockX,
                 (long)blockY, (long)blockZ, sharedMem);
  // `stream` is null (the legacy default stream: ordered with the application's work).
  check(driver.cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, params, extra), "cuLaunchKernel");
  if (debugEnabled() || syncEnabled()) check(driver.cuCtxSynchronize(), "cuCtxSynchronize after a launch");
}

void *mgpuStreamCreate() { return nullptr; }
void mgpuStreamDestroy(void *) {}
void mgpuStreamSynchronize(void *stream) {
  if (!syncEnabled()) return;
  ensureContext();
  check(driver.cuStreamSynchronize(stream), "cuStreamSynchronize");
}
void mgpuStreamWaitEvent(void *, void *) {}
void *mgpuEventCreate() { return nullptr; }
void mgpuEventDestroy(void *) {}
void mgpuEventSynchronize(void *) {}
void mgpuEventRecord(void *, void *) {}

void *mgpuMemAlloc(uint64_t sizeBytes, void *, bool isHostShared) {
  ensureContext();
  if (sizeBytes == 0) return nullptr;
  CUdeviceptr ptr = 0;
  if (isHostShared) check(driver.cuMemAllocManaged(&ptr, sizeBytes, 1 /*CU_MEM_ATTACH_GLOBAL*/), "cuMemAllocManaged");
  else check(driver.cuMemAlloc(&ptr, sizeBytes), "cuMemAlloc");
  return reinterpret_cast<void *>(ptr);
}

void mgpuMemFree(void *ptr, void *) {
  if (!ptr) return;
  ensureContext();
  check(driver.cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)), "cuMemFree");
}

void mgpuMemcpy(void *dst, void *src, size_t sizeBytes, void *stream) {
  ensureContext();
  check(driver.cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), sizeBytes, stream), "cuMemcpyAsync");
}

void mgpuMemset32(void *dst, unsigned int value, size_t count, void *stream) {
  ensureContext();
  check(driver.cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst), value, count, stream), "cuMemsetD32Async");
}

void mgpuSetDefaultDevice(int32_t device) { defaultDevice = device; }

} // extern "C"
