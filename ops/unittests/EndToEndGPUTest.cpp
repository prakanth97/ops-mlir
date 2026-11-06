// A group through the CUDA pipeline: planned, lowered (ops-jit-cuda: one thread per point, atomics for the reduction),
// compiled by JITSession (the kernel is a fatbin made with ptxas from the CUDA toolkit), and run on device memory
// through the opdsl GpuShim. Skipped where there is no CUDA device or no toolkit.

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Pipelines/GroupModules.h"
#include "ops-mlir/Pipelines/Pipelines.h"
#include "ops-mlir/Runtime/GroupCompiler.h"
#include "ops-mlir/Runtime/PackedArgs.h"

#include "opdsl/GPU/GpuShim.h"
#include "opdsl/Testing/TestHarness.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/FileSystem.h"

#include <cmath>
#include <cstdlib>
#include <vector>

using namespace ops_mlir;
using namespace ops_mlir::ops;
using namespace ops_mlir::runtime;

namespace {

// y(i,j) = x(i,j) + x(i+1,j); the maximum of x over the range is reduced.
const char *kProgram = R"mlir(
ops.kernel @pair_sum(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>, %m: !ops.red<f64>) {
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %w = ops.read %x[1, 0] : !ops.acc<f64, 2, 1> -> f64
  %s = arith.addf %v, %w : f64
  ops.write %y[0, 0], %s : !ops.acc<f64, 2, 1>
  ops.reduce %m, %v : !ops.red<f64>
  ops.return
}
ops.chain @chain(%x: !ops.dat<f64, 2, 1>, %y: !ops.dat<f64, 2, 1>,
                 %lo0: index, %hi0: index, %lo1: index, %hi1: index)
    -> f64 attributes {ops.range_values = [array<i64: 0, 70, 0, 9>]} {
  %m = ops.par_loop @pair_sum range(%lo0, %hi0, %lo1, %hi1)
      dats(%x : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1, 0]]>, %y : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() attributes {reduction_kinds = [#ops.red_kind<max>]} -> f64 {
  ^bb0(%a: !ops.acc<f64, 2, 1>, %b: !ops.acc<f64, 2, 1>, %r: !ops.red<f64>):
  }
  ops.return %m : f64
}
)mlir";

constexpr int kNx = 72, kNy = 10; // 70 x 9 points computed (not a multiple of the 32 x 4 block)

} // namespace

OPDSL_TEST(EndToEndGPU, ThreadPerPointKernelWithAtomicReduction) {
  if (!opdsl::gpu::cudaAvailable()) {
    std::printf("no CUDA device: skipped\n");
    return;
  }
  if (!std::getenv("CUDA_ROOT") && !std::getenv("CUDA_HOME") && !std::getenv("CUDA_PATH")) {
    for (const char *candidate : {"/usr/local/cuda", "/usr/lib/cuda", "/home/shared/software/cuda/12.3"})
      if (llvm::sys::fs::is_directory(candidate)) {
        ::setenv("CUDA_ROOT", candidate, 0);
        break;
      }
  }

  opdsl::PipelineRegistry registry;
  registerOPSPipelines(registry);
  mlir::DialectRegistry dialects;
  registerJITDialects(dialects);
  mlir::MLIRContext ctx(dialects);
  ctx.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kProgram, &ctx);
  OPDSL_EXPECT(static_cast<bool>(module));
  if (!module) return;

  mlir::PassManager planning(&ctx);
  OPDSL_EXPECT(!registry.build("ops-jit-frontend", "", planning));
  OPDSL_EXPECT(!registry.build("ops-jit-plan", "", planning));
  OPDSL_EXPECT(mlir::succeeded(planning.run(*module)));
  std::vector<GroupModule> groups = extractGroupModules(*module);
  OPDSL_EXPECT_EQ(groups.size(), size_t(1));
  if (groups.size() != 1) return;

  CompilerOptions co;
  co.useCache = false;
  GroupCompiler compiler(ctx, registry, co);
  std::string options = std::string("chip=") + opdsl::gpu::cudaDeviceArch() + " block=32x4x1";
  std::string error;
  std::shared_ptr<GroupCode> code = compiler.compile(groups[0].module.get(), "ops-jit-cuda", options, error);
  OPDSL_EXPECT(code != nullptr);
  if (!code) {
    std::printf("compile error: %s\n", error.c_str());
    return;
  }

  // Device memory (the shim's allocation), filled from the host.
  const size_t bytes = size_t(kNx) * kNy * sizeof(double);
  std::vector<double> x(kNx * kNy), y(kNx * kNy, -1.0);
  for (int j = 0; j < kNy; ++j)
    for (int i = 0; i < kNx; ++i) x[j * kNx + i] = std::sin(0.1 * i) + 0.01 * j;
  void *dx = mgpuMemAlloc(bytes, nullptr, false);
  void *dy = mgpuMemAlloc(bytes, nullptr, false);
  // 3 x 3 blocks of 32 x 4 threads = 4 warps each: 36 partial values
  constexpr int kUnits = 3 * 3 * 4;
  void *dm = mgpuMemAlloc(kUnits * sizeof(double), nullptr, false);
  std::vector<double> slots(kUnits);
  double result = -INFINITY;
  mgpuMemcpy(dx, x.data(), bytes, nullptr);
  mgpuMemcpy(dy, y.data(), bytes, nullptr);

  PackedArgs args;
  args.addMemref(dx, {kNy, kNx}, {kNx, 1});
  args.addMemref(dy, {kNy, kNx}, {kNx, 1});
  for (int i = 0; i < 4; ++i) args.addIndex(0); // shifts (dat-major, dimension-minor)
  for (int64_t v : {0, 70, 0, 9}) args.addIndex(v); // lo0 hi0 lo1 hi1
  args.addIndex(0);
  args.addIndex(0); // displacement
  args.addMemref(dm, {kUnits}, {1}); // the reduction buffer: one value per warp
  args.finalize();
  args.call(code->fn);
  mgpuMemcpy(y.data(), dy, bytes, nullptr);
  mgpuMemcpy(slots.data(), dm, kUnits * sizeof(double), nullptr);
  for (double v : slots) result = std::max(result, v);

  int wrong = 0;
  double expectedMax = -INFINITY;
  for (int j = 0; j < kNy; ++j)
    for (int i = 0; i < kNx; ++i) {
      bool inside = i < 70 && j < 9;
      double e = inside ? x[j * kNx + i] + x[j * kNx + i + 1] : -1.0;
      if (inside) expectedMax = std::max(expectedMax, x[j * kNx + i]);
      if (std::fabs(y[j * kNx + i] - e) > 1e-14) ++wrong;
    }
  OPDSL_EXPECT_EQ(wrong, 0);
  OPDSL_EXPECT_EQ(result, expectedMax);
  mgpuMemFree(dx, nullptr);
  mgpuMemFree(dy, nullptr);
  mgpuMemFree(dm, nullptr);
}
