//===- GpuShim.h - the runtime symbols `gpu-to-llvm` code calls (CUDA, HIP) ---*- C++ -*-===//
//
// MLIR's gpu-to-llvm emits calls to `mgpu*` functions: load a module (the binary `gpu-module-to-binary` embedded),
// get a kernel, launch it, create/synchronise streams, allocate memory. The upstream implementations
// (libmlir_cuda_runtime, libmlir_rocm_runtime) create non-blocking streams, which are not ordered with the
// default-stream work of an application such as OPS. This shim differs where that matters:
//
//   * the device is the *current* one (the one the application's runtime-API calls made current), else the default
//     device; for CUDA, its context is never created or destroyed here, so allocations of the application are valid in
//     the kernels;
//   * streams are the legacy default stream: mgpuStreamCreate returns null, destroy does nothing, and
//     mgpuStreamSynchronize only synchronises with OPS_JIT_SYNC=1 (debug). Work is ordered with the application's
//     memcpy/kernels, and with host reads of the results (the memcpy that fetches them);
//   * the driver is loaded with dlopen at first use: libopsjit has no link-time dependency on CUDA or HIP.
//
// The symbols (GpuShim.cpp) call the GpuRuntime of the process (GpuRuntime.h): CUDA (NVIDIA, libcuda) or HIP (AMD,
// libamdhip64), chosen at first use. They have C linkage and are exported by the library that contains them; compiled
// code finds them in the process (JITSession looks symbols up there).
//
//===----------------------------------------------------------------------===//

#ifndef OPDSL_GPU_GPUSHIM_H
#define OPDSL_GPU_GPUSHIM_H

#include <cstddef>
#include <cstdint>

namespace opdsl::gpu {

/// The runtime of the process: "cuda" or "hip" (OPS_JIT_GPU, else the first available of CUDA and HIP).
const char *gpuRuntime();

/// True if a driver and at least one device are usable. Loads the driver on first call.
bool gpuAvailable();

/// Architecture of the current/default device: "sm_<major><minor>" (e.g. "sm_89") for CUDA, the gfx name (e.g.
/// "gfx90a") for HIP; empty if unavailable.
const char *gpuDeviceArch();

/// Keeps the shim's exported symbols referenced so that static linking does not drop them.
void *gpuShimAnchor();

} // namespace opdsl::gpu

extern "C" {
void *mgpuModuleLoad(void *data, size_t blobSize);
void *mgpuModuleLoadJIT(void *data, int optLevel, size_t assemblySize);
void mgpuModuleUnload(void *module);
void *mgpuModuleGetFunction(void *module, const char *name);
void mgpuLaunchKernel(void *function, intptr_t gridX, intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                      intptr_t blockZ, int32_t sharedMem, void *stream, void **params, void **extra, size_t paramsCount);
void *mgpuStreamCreate();
void mgpuStreamDestroy(void *stream);
void mgpuStreamSynchronize(void *stream);
void mgpuStreamWaitEvent(void *stream, void *event);
void *mgpuEventCreate();
void mgpuEventDestroy(void *event);
void mgpuEventSynchronize(void *event);
void mgpuEventRecord(void *event, void *stream);
void *mgpuMemAlloc(uint64_t sizeBytes, void *stream, bool isHostShared);
void mgpuMemFree(void *ptr, void *stream);
void mgpuMemcpy(void *dst, void *src, size_t sizeBytes, void *stream);
void mgpuMemset32(void *dst, unsigned int value, size_t count, void *stream);
void mgpuSetDefaultDevice(int32_t device);
}

#endif // OPDSL_GPU_GPUSHIM_H
