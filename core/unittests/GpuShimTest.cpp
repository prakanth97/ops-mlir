// The CUDA launch shim on a real device: load a PTX module, allocate, set, launch, copy back. Skipped (and reported as
// passed) where there is no CUDA driver or device, so the suite runs everywhere.

#include "opdsl/GPU/GpuShim.h"
#include "opdsl/Testing/TestHarness.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// out[i] = in[i] + 1 for i < n (ptx 7.0, sm_52 and up: the driver compiles it for the device).
const char *kPtx = R"ptx(
.version 7.0
.target sm_52
.address_size 64

.visible .entry add_one(.param .u64 in_p, .param .u64 out_p, .param .u32 n_p)
{
  .reg .pred %p<2>;
  .reg .b32 %r<6>;
  .reg .b64 %rd<8>;
  .reg .f32 %f<3>;
  ld.param.u64 %rd1, [in_p];
  ld.param.u64 %rd2, [out_p];
  ld.param.u32 %r2, [n_p];
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r4, %ntid.x;
  mov.u32 %r5, %tid.x;
  mad.lo.s32 %r1, %r3, %r4, %r5;
  setp.ge.s32 %p1, %r1, %r2;
  @%p1 bra DONE;
  cvta.to.global.u64 %rd3, %rd1;
  cvta.to.global.u64 %rd4, %rd2;
  mul.wide.s32 %rd5, %r1, 4;
  add.s64 %rd6, %rd3, %rd5;
  add.s64 %rd7, %rd4, %rd5;
  ld.global.f32 %f1, [%rd6];
  add.f32 %f2, %f1, 0f3F800000;
  st.global.f32 [%rd7], %f2;
DONE:
  ret;
}
)ptx";

} // namespace

OPDSL_TEST(GpuShim, LaunchesAKernelOnTheDefaultStream) {
  if (!opdsl::gpu::cudaAvailable()) {
    std::printf("no CUDA device: skipped\n");
    return;
  }
  OPDSL_EXPECT(std::strlen(opdsl::gpu::cudaDeviceArch()) >= 5); // "sm_xx"
  constexpr int n = 1000;
  void *module = mgpuModuleLoadJIT(const_cast<char *>(kPtx), 3, std::strlen(kPtx) + 1);
  OPDSL_EXPECT(module != nullptr);
  void *function = mgpuModuleGetFunction(module, "add_one");
  OPDSL_EXPECT(function != nullptr);

  void *in = mgpuMemAlloc(n * sizeof(float), nullptr, false);
  void *out = mgpuMemAlloc(n * sizeof(float), nullptr, false);
  std::vector<float> host(n), result(n, -1.0f);
  for (int i = 0; i < n; ++i) host[i] = float(i);
  mgpuMemcpy(in, host.data(), n * sizeof(float), nullptr);
  mgpuMemset32(out, 0, n, nullptr);

  void *stream = mgpuStreamCreate();
  int count = n;
  void *params[] = {&in, &out, &count};
  mgpuLaunchKernel(function, (n + 127) / 128, 1, 1, 128, 1, 1, 0, stream, params, nullptr, 3);
  mgpuStreamSynchronize(stream);
  mgpuStreamDestroy(stream);
  mgpuMemcpy(result.data(), out, n * sizeof(float), nullptr);
  mgpuStreamSynchronize(nullptr);
  // The copy runs on the default stream, after the launch; host memory is pageable, so it is complete when it returns.

  int wrong = 0;
  for (int i = 0; i < n; ++i)
    if (result[i] != float(i) + 1.0f) ++wrong;
  OPDSL_EXPECT_EQ(wrong, 0);
  mgpuMemFree(in, nullptr);
  mgpuMemFree(out, nullptr);
  mgpuModuleUnload(module);
}
