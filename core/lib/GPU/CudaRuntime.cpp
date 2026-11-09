//===- CudaRuntime.cpp - GpuRuntime on the CUDA driver API (NVIDIA) ---------------===//

#include "opdsl/GPU/GpuRuntime.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace opdsl::gpu {

namespace {

// The driver API, as much of it as is used. Types are those of cuda.h (stable ABI), declared here so that building this
// file needs no CUDA installation.
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
constexpr int kJitErrorLogBuffer = 5, kJitErrorLogBufferSize = 6, kJitOptimizationLevel = 7;
constexpr int kAttrComputeMajor = 75, kAttrComputeMinor = 76;
constexpr unsigned kMemAttachGlobal = 1;

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

thread_local int defaultDevice = 0;

class CudaRuntime final : public GpuRuntime {
public:
  const char *name() const override { return "cuda"; }

  bool available() override { return driver().ok; }

  std::string deviceArch() override {
    Driver &d = driver();
    if (!d.ok) return {};
    CUdevice device;
    if (d.cuDeviceGet(&device, defaultDevice) != 0) return {};
    int major = 0, minor = 0;
    d.cuDeviceGetAttribute(&major, kAttrComputeMajor, device);
    d.cuDeviceGetAttribute(&minor, kAttrComputeMinor, device);
    return "sm_" + std::to_string(major) + std::to_string(minor);
  }

  /// Whatever context the application has (its runtime-API calls made the primary context current), else the primary
  /// context of the default device.
  void makeCurrent() override {
    void *current = nullptr;
    check(d_.cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current) return;
    CUdevice device;
    check(d_.cuDeviceGet(&device, defaultDevice), "cuDeviceGet");
    void *context = nullptr;
    check(d_.cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check(d_.cuCtxSetCurrent(context), "cuCtxSetCurrent");
  }

  void *moduleLoad(void *data, size_t) override {
    void *module = nullptr;
    check(d_.cuModuleLoadData(&module, data), "cuModuleLoadData");
    return module;
  }

  void *moduleLoadJIT(void *data, int optLevel) override {
    void *module = nullptr;
    char log[4096] = {0};
    int options[] = {kJitErrorLogBuffer, kJitErrorLogBufferSize, kJitOptimizationLevel};
    void *values[] = {log, reinterpret_cast<void *>(sizeof(log)), reinterpret_cast<void *>(static_cast<intptr_t>(optLevel))};
    CUresult r = d_.cuModuleLoadDataEx(&module, data, 3, options, values);
    if (r != 0) std::fprintf(stderr, "ops_jit: JIT compilation of the GPU module failed: '%s'\n", log);
    check(r, "cuModuleLoadDataEx");
    return module;
  }

  void moduleUnload(void *module) override {
    if (d_.ok) d_.cuModuleUnload(module);
  }

  void *moduleGetFunction(void *module, const char *name) override {
    void *function = nullptr;
    check(d_.cuModuleGetFunction(&function, module, name), "cuModuleGetFunction");
    return function;
  }

  void launch(void *function, intptr_t gridX, intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
              int32_t sharedMem, void *stream, void **params, void **extra) override {
    check(d_.cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, params, extra), "cuLaunchKernel");
  }

  void streamSynchronize(void *stream) override { check(d_.cuStreamSynchronize(stream), "cuStreamSynchronize"); }
  void deviceSynchronize() override { check(d_.cuCtxSynchronize(), "cuCtxSynchronize"); }

  void *alloc(uint64_t sizeBytes, bool hostShared) override {
    CUdeviceptr ptr = 0;
    if (hostShared) check(d_.cuMemAllocManaged(&ptr, sizeBytes, kMemAttachGlobal), "cuMemAllocManaged");
    else check(d_.cuMemAlloc(&ptr, sizeBytes), "cuMemAlloc");
    return reinterpret_cast<void *>(ptr);
  }

  void free(void *ptr) override { check(d_.cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)), "cuMemFree"); }

  /// Asynchronous on the default stream, but a copy to pageable host memory returns when it is complete.
  void memcpy(void *dst, void *src, size_t sizeBytes, void *stream) override {
    check(d_.cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), sizeBytes, stream), "cuMemcpyAsync");
  }

  void memset32(void *dst, unsigned value, size_t count, void *stream) override {
    check(d_.cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst), value, count, stream), "cuMemsetD32Async");
  }

  void setDefaultDevice(int device) override { defaultDevice = device; }

private:
  Driver d_;
  std::once_flag once_;

  Driver &driver() {
    std::call_once(once_, [this] { load(); });
    return d_;
  }

  template <typename F> bool bind(F &fn, const char *name) {
    fn = reinterpret_cast<F>(dlsym(d_.handle, name));
    return fn != nullptr;
  }

  void load() {
    d_.handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!d_.handle) d_.handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!d_.handle) return;
    bool ok = true;
    ok &= bind(d_.cuInit, "cuInit");
    ok &= bind(d_.cuCtxGetCurrent, "cuCtxGetCurrent");
    ok &= bind(d_.cuCtxSetCurrent, "cuCtxSetCurrent");
    ok &= bind(d_.cuCtxSynchronize, "cuCtxSynchronize");
    ok &= bind(d_.cuDeviceGet, "cuDeviceGet");
    ok &= bind(d_.cuDeviceGetCount, "cuDeviceGetCount");
    ok &= bind(d_.cuDeviceGetAttribute, "cuDeviceGetAttribute");
    ok &= bind(d_.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    ok &= bind(d_.cuModuleLoadData, "cuModuleLoadData");
    ok &= bind(d_.cuModuleLoadDataEx, "cuModuleLoadDataEx");
    ok &= bind(d_.cuModuleUnload, "cuModuleUnload");
    ok &= bind(d_.cuModuleGetFunction, "cuModuleGetFunction");
    ok &= bind(d_.cuLaunchKernel, "cuLaunchKernel");
    ok &= bind(d_.cuMemAlloc, "cuMemAlloc_v2");
    ok &= bind(d_.cuMemAllocManaged, "cuMemAllocManaged");
    ok &= bind(d_.cuMemFree, "cuMemFree_v2");
    ok &= bind(d_.cuMemcpyAsync, "cuMemcpyAsync");
    ok &= bind(d_.cuMemsetD32Async, "cuMemsetD32Async");
    ok &= bind(d_.cuStreamSynchronize, "cuStreamSynchronize");
    bind(d_.cuGetErrorString, "cuGetErrorString");
    if (!ok) return;
    if (d_.cuInit(0) != 0) return;
    int count = 0;
    if (d_.cuDeviceGetCount(&count) != 0 || count == 0) return;
    d_.ok = true;
  }

  void check(CUresult result, const char *what) {
    if (result == 0) return;
    const char *text = "unknown error";
    if (d_.cuGetErrorString) d_.cuGetErrorString(result, &text);
    std::fprintf(stderr, "ops_jit: CUDA error %d (%s) in %s\n", result, text, what);
    std::abort();
  }
};

} // namespace

std::unique_ptr<GpuRuntime> createCudaRuntime() { return std::make_unique<CudaRuntime>(); }

} // namespace opdsl::gpu
