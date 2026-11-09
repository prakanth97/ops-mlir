//===- HipRuntime.cpp - GpuRuntime on the HIP runtime API (AMD) -------------------===//

#include "opdsl/GPU/GpuRuntime.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace opdsl::gpu {

namespace {

// The HIP runtime API, as much of it as is used. Types and constants are those of hip_runtime_api.h (ROCm 6/7),
// declared here so that building this file needs no ROCm installation.
using hipError_t = int;
constexpr int kMemcpyDefault = 4; // hipMemcpyDefault
constexpr unsigned kMemAttachGlobal = 1;
// hipDeviceProp_tR0600: size and offset of gcnArchName ("gfx90a:sramecc+:xnack-").
constexpr size_t kDevicePropSize = 1472, kGcnArchNameOffset = 1160;

struct Library {
  void *handle = nullptr;
  hipError_t (*hipInit)(unsigned) = nullptr;
  hipError_t (*hipGetDeviceCount)(int *) = nullptr;
  hipError_t (*hipGetDevice)(int *) = nullptr;
  hipError_t (*hipSetDevice)(int) = nullptr;
  hipError_t (*hipDeviceSynchronize)() = nullptr;
  hipError_t (*hipGetDevicePropertiesR0600)(void *, int) = nullptr;
  hipError_t (*hipModuleLoadData)(void **, const void *) = nullptr;
  hipError_t (*hipModuleUnload)(void *) = nullptr;
  hipError_t (*hipModuleGetFunction)(void **, void *, const char *) = nullptr;
  hipError_t (*hipModuleLaunchKernel)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                                      void **) = nullptr;
  hipError_t (*hipMalloc)(void **, size_t) = nullptr;
  hipError_t (*hipMallocManaged)(void **, size_t, unsigned) = nullptr;
  hipError_t (*hipFree)(void *) = nullptr;
  hipError_t (*hipMemcpy)(void *, const void *, size_t, int) = nullptr;
  hipError_t (*hipMemsetD32Async)(void *, int, size_t, void *) = nullptr;
  hipError_t (*hipStreamSynchronize)(void *) = nullptr;
  const char *(*hipGetErrorString)(hipError_t) = nullptr;
  bool ok = false;
};

thread_local int defaultDevice = -1; // -1: the device the application made current

class HipRuntime final : public GpuRuntime {
public:
  const char *name() const override { return "hip"; }

  bool available() override { return library().ok; }

  std::string deviceArch() override {
    Library &l = library();
    if (!l.ok || !l.hipGetDevicePropertiesR0600) return {};
    int device = defaultDevice;
    if (device < 0 && l.hipGetDevice(&device) != 0) return {};
    alignas(16) char props[kDevicePropSize + 1024] = {0};
    if (l.hipGetDevicePropertiesR0600(props, device) != 0) return {};
    std::string arch(props + kGcnArchNameOffset, strnlen(props + kGcnArchNameOffset, 256));
    arch = arch.substr(0, arch.find(':')); // sramecc, xnack: target features, not the chip
    return arch.rfind("gfx", 0) == 0 ? arch : std::string();
  }

  /// HIP has no contexts to manage: the device is per thread, the one the application set unless setDefaultDevice chose one.
  void makeCurrent() override {
    if (defaultDevice < 0) return;
    int current = -1;
    check(l_.hipGetDevice(&current), "hipGetDevice");
    if (current != defaultDevice) check(l_.hipSetDevice(defaultDevice), "hipSetDevice");
  }

  void *moduleLoad(void *data, size_t) override {
    void *module = nullptr;
    check(l_.hipModuleLoadData(&module, data), "hipModuleLoadData");
    return module;
  }

  void *moduleLoadJIT(void *, int) override {
    // AMD code objects are compiled before loading (gpu-module-to-binary format=fatbin or bin), never JIT-compiled.
    std::fprintf(stderr, "ops_jit: HIP cannot load a module from assembly: compile the kernels to a binary\n");
    std::abort();
  }

  void moduleUnload(void *module) override {
    if (l_.ok) l_.hipModuleUnload(module);
  }

  void *moduleGetFunction(void *module, const char *name) override {
    void *function = nullptr;
    check(l_.hipModuleGetFunction(&function, module, name), "hipModuleGetFunction");
    return function;
  }

  void launch(void *function, intptr_t gridX, intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
              int32_t sharedMem, void *stream, void **params, void **extra) override {
    check(l_.hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, params, extra),
          "hipModuleLaunchKernel");
  }

  void streamSynchronize(void *stream) override { check(l_.hipStreamSynchronize(stream), "hipStreamSynchronize"); }
  void deviceSynchronize() override { check(l_.hipDeviceSynchronize(), "hipDeviceSynchronize"); }

  void *alloc(uint64_t sizeBytes, bool hostShared) override {
    void *ptr = nullptr;
    if (hostShared) check(l_.hipMallocManaged(&ptr, sizeBytes, kMemAttachGlobal), "hipMallocManaged");
    else check(l_.hipMalloc(&ptr, sizeBytes), "hipMalloc");
    return ptr;
  }

  void free(void *ptr) override { check(l_.hipFree(ptr), "hipFree"); }

  /// Synchronous: hipMemcpyAsync to pageable memory is not guaranteed to be complete when it returns.
  void memcpy(void *dst, void *src, size_t sizeBytes, void *) override {
    check(l_.hipMemcpy(dst, src, sizeBytes, kMemcpyDefault), "hipMemcpy");
  }

  void memset32(void *dst, unsigned value, size_t count, void *stream) override {
    check(l_.hipMemsetD32Async(dst, static_cast<int>(value), count, stream), "hipMemsetD32Async");
  }

  void setDefaultDevice(int device) override { defaultDevice = device; }

private:
  Library l_;
  std::once_flag once_;

  Library &library() {
    std::call_once(once_, [this] { load(); });
    return l_;
  }

  template <typename F> bool bind(F &fn, const char *name) {
    fn = reinterpret_cast<F>(dlsym(l_.handle, name));
    return fn != nullptr;
  }

  void load() {
    for (const char *lib : {"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"})
      if ((l_.handle = dlopen(lib, RTLD_NOW | RTLD_GLOBAL))) break;
    if (!l_.handle) return;
    bool ok = true;
    ok &= bind(l_.hipInit, "hipInit");
    ok &= bind(l_.hipGetDeviceCount, "hipGetDeviceCount");
    ok &= bind(l_.hipGetDevice, "hipGetDevice");
    ok &= bind(l_.hipSetDevice, "hipSetDevice");
    ok &= bind(l_.hipDeviceSynchronize, "hipDeviceSynchronize");
    ok &= bind(l_.hipModuleLoadData, "hipModuleLoadData");
    ok &= bind(l_.hipModuleUnload, "hipModuleUnload");
    ok &= bind(l_.hipModuleGetFunction, "hipModuleGetFunction");
    ok &= bind(l_.hipModuleLaunchKernel, "hipModuleLaunchKernel");
    ok &= bind(l_.hipMalloc, "hipMalloc");
    ok &= bind(l_.hipMallocManaged, "hipMallocManaged");
    ok &= bind(l_.hipFree, "hipFree");
    ok &= bind(l_.hipMemcpy, "hipMemcpy");
    ok &= bind(l_.hipMemsetD32Async, "hipMemsetD32Async");
    ok &= bind(l_.hipStreamSynchronize, "hipStreamSynchronize");
    bind(l_.hipGetErrorString, "hipGetErrorString");
    bind(l_.hipGetDevicePropertiesR0600, "hipGetDevicePropertiesR0600"); // ROCm 6+; without it the arch is unknown
    if (!ok) return;
    if (l_.hipInit(0) != 0) return;
    int count = 0;
    if (l_.hipGetDeviceCount(&count) != 0 || count == 0) return;
    l_.ok = true;
  }

  void check(hipError_t result, const char *what) {
    if (result == 0) return;
    const char *text = l_.hipGetErrorString ? l_.hipGetErrorString(result) : "unknown error";
    std::fprintf(stderr, "ops_jit: HIP error %d (%s) in %s\n", result, text, what);
    std::abort();
  }
};

} // namespace

std::unique_ptr<GpuRuntime> createHipRuntime() { return std::make_unique<HipRuntime>(); }

} // namespace opdsl::gpu
