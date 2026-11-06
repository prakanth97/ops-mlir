//===- GpuShim.h - the runtime symbols `gpu-to-llvm` code calls (CUDA) --------*- C++ -*-===//
//
// MLIR's gpu-to-llvm emits calls to `mgpu*` functions: load a module (the binary `gpu-module-to-binary` embedded),
// get a kernel, launch it, create/synchronise streams, allocate memory. The upstream implementation
// (libmlir_cuda_runtime) creates non-blocking streams, which are not ordered with the default-stream work of
// a CUDA application such as OPS. This shim differs where that matters:
//
//   * the context is the *current* one (the one the application's runtime-API calls made current), else the primary
//     context of the default device; it is never created or destroyed here, so allocations of the application are
//     valid in the kernels;
//   * streams are the legacy default stream: mgpuStreamCreate returns null, destroy does nothing, and
//     mgpuStreamSynchronize only synchronises with OPS_JIT_SYNC=1 (debug). Work is ordered with the application's
//     cudaMemcpy/kernels, and with host reads of the results (the cudaMemcpy that fetches them);
//   * the driver API is loaded with dlopen(libcuda.so.1) at first use: libopsjit has no link-time dependency on CUDA.
//
// The symbols have C linkage and are exported by the library that contains this file; compiled code finds them in the
// process (JITSession looks symbols up there).
//
//===----------------------------------------------------------------------===//

#ifndef OPDSL_GPU_GPUSHIM_H
#define OPDSL_GPU_GPUSHIM_H

#include <cstddef>
#include <cstdint>

namespace opdsl::gpu {

/// True if a CUDA driver and at least one device are usable. Loads the driver on first call.
bool cudaAvailable();

/// Compute capability of the current/default device as "sm_<major><minor>" (e.g. "sm_89"), empty if unavailable.
const char *cudaDeviceArch();

/// Keeps the shim's exported symbols referenced so that static linking does not drop them.
void *cudaShimAnchor();

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
