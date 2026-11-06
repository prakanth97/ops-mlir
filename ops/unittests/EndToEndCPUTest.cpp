// Phase 1 exit test: a hand-written chain of unmaterialised loops is planned, split into groups, lowered, compiled
// to a host function and run on real arrays; then the same group is loaded from the disk cache in a second
// session without lowering or compiling again.

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Pipelines/GroupModules.h"
#include "ops-mlir/Pipelines/Pipelines.h"

#include "opdsl/Cache/PersistentCache.h"
#include "opdsl/JIT/JITSession.h"
#include "opdsl/Support/Hash.h"
#include "opdsl/Testing/TestHarness.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/FileSystem.h"

#include <vector>

using namespace ops_mlir::ops;

namespace {

// b = a * factor (factor is bound: its value, 2.0, is folded); c = b + bias (bias is an argument of the group).
const char *kProgram = R"mlir(
ops.kernel @scale(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>) {
  %factor = ops.const_ref @factor : f64
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %w = arith.mulf %v, %factor : f64
  ops.write %y[0, 0], %w : !ops.acc<f64, 2, 1>
  ops.return
}
ops.kernel @add_bias(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>) {
  %bias = ops.const_ref @bias : f64
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %w = arith.addf %v, %bias : f64
  ops.write %y[0, 0], %w : !ops.acc<f64, 2, 1>
  ops.return
}
ops.chain @chain(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %c: !ops.dat<f64, 2, 1>,
                 %lo0: index, %hi0: index, %lo1: index, %hi1: index,
                 %lo2: index, %hi2: index, %lo3: index, %hi3: index)
    attributes {ops.const_roles = {factor = "bound"},
                ops.const_values = {factor = array<i8: 0, 0, 0, 0, 0, 0, 0, 64>},
                ops.range_values = [array<i64: 1, 7, 1, 7>, array<i64: 1, 7, 1, 7>]} {
  ops.par_loop @scale range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop @add_bias range(%lo2, %hi2, %lo3, %hi3)
      dats(%b : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %c : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.return
}
)mlir";

constexpr int kN = 8;

struct Descriptor { // memref<?x?xf64, strided<[?, 1]>>
  double *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};

Descriptor describe(std::vector<double> &data) {
  return {data.data(), data.data(), 0, {kN, kN}, {kN, 1}};
}

/// Calls the group with the ABI of plan 4.6: memrefs (a, b, c), shifts, range, displacement, bias.
void run(opdsl::JITSession::PackedFn fn, std::vector<double> &a, std::vector<double> &b, std::vector<double> &c, double bias) {
  Descriptor da = describe(a), db = describe(b), dc = describe(c);
  Descriptor *pa = &da, *pb = &db, *pc = &dc;
  int64_t shifts[6] = {0, 0, 0, 0, 0, 0};
  int64_t range[4] = {1, 7, 1, 7};
  int64_t disp[2] = {0, 0};
  void *args[3 + 6 + 4 + 2 + 1];
  int n = 0;
  args[n++] = &pa;
  args[n++] = &pb;
  args[n++] = &pc;
  for (int64_t &v : shifts) args[n++] = &v;
  for (int64_t &v : range) args[n++] = &v;
  for (int64_t &v : disp) args[n++] = &v;
  args[n++] = &bias;
  fn(args);
}

std::vector<double> input() {
  std::vector<double> a(kN * kN);
  for (int j = 0; j < kN; ++j)
    for (int i = 0; i < kN; ++i) a[j * kN + i] = 10 * j + i;
  return a;
}

void expectResults(const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &c, double bias) {
  int wrong = 0;
  for (int j = 0; j < kN; ++j)
    for (int i = 0; i < kN; ++i) {
      bool inside = j >= 1 && j < 7 && i >= 1 && i < 7;
      double eb = inside ? 2.0 * a[j * kN + i] : 0.0;
      double ec = inside ? eb + bias : 0.0;
      if (b[j * kN + i] != eb || c[j * kN + i] != ec) ++wrong;
    }
  OPDSL_EXPECT_EQ(wrong, 0);
}

} // namespace

OPDSL_TEST(EndToEndCPU, PlanLowerCompileRunAndReloadFromTheDiskCache) {
  opdsl::PipelineRegistry registry;
  registerOPSPipelines(registry);

  mlir::DialectRegistry dialects;
  registerJITDialects(dialects);
  mlir::MLIRContext ctx(dialects);
  ctx.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kProgram, &ctx);
  OPDSL_EXPECT(static_cast<bool>(module));
  if (!module) return;

  // Planning: materialise, fold `factor`, fuse the two loops, outline.
  mlir::PassManager planning(&ctx);
  OPDSL_EXPECT(!registry.build("ops-jit-frontend", "", planning));
  OPDSL_EXPECT(!registry.build("ops-jit-plan", "", planning));
  OPDSL_EXPECT(mlir::succeeded(planning.run(*module)));
  std::vector<GroupModule> groups = extractGroupModules(*module);
  OPDSL_EXPECT_EQ(groups.size(), size_t(1));
  if (groups.size() != 1) return;
  const std::string entry = groups[0].name;

  // The cache key: group bytecode, pipeline and options, target, versions.
  auto session = opdsl::JITSession::create();
  OPDSL_EXPECT(static_cast<bool>(session));
  if (!session) return;
  const std::string pipelineOptions = "omp=false";
  std::string key = opdsl::Hasher()
                        .add(moduleBytecode(groups[0].module.get()))
                        .add("ops-jit-cpu")
                        .add(pipelineOptions)
                        .add((*session)->targetDescription())
                        .addInt(1)
                        .hex();

  llvm::SmallString<128> dirPath;
  llvm::sys::fs::createUniqueDirectory("opdsl-e2e", dirPath);
  opdsl::CacheOptions cacheOptions;
  cacheOptions.directory = std::string(dirPath);

  {
    // First run: a miss; lower, compile, store, run.
    opdsl::PersistentCache cache(cacheOptions);
    OPDSL_EXPECT(!cache.lookup(key).has_value());
    mlir::PassManager lowering(&ctx);
    OPDSL_EXPECT(!registry.build("ops-jit-cpu", pipelineOptions, lowering));
    OPDSL_EXPECT(mlir::succeeded(lowering.run(groups[0].module.get())));
    auto object = (*session)->compile(groups[0].module.get(), entry);
    OPDSL_EXPECT(static_cast<bool>(object));
    if (!object) return;
    OPDSL_EXPECT(!cache.store(key, "{\"entry\": \"" + entry + "\"}", (*object)->getBuffer()));
    auto fn = (*session)->load(std::move(*object), entry);
    OPDSL_EXPECT(static_cast<bool>(fn));
    if (!fn) return;
    std::vector<double> a = input(), b(kN * kN, 0.0), c(kN * kN, 0.0);
    run(*fn, a, b, c, 0.5);
    expectResults(a, b, c, 0.5);
  }
  {
    // Second run, a new session and cache object: a hit; no lowering, no compile.
    auto second = opdsl::JITSession::create();
    OPDSL_EXPECT(static_cast<bool>(second));
    if (!second) return;
    opdsl::PersistentCache cache(cacheOptions);
    auto hit = cache.lookup(key);
    OPDSL_EXPECT(hit.has_value());
    if (!hit) return;
    auto fn = (*second)->load(llvm::MemoryBuffer::getMemBufferCopy(hit->payload, "cached"), entry);
    OPDSL_EXPECT(static_cast<bool>(fn));
    if (!fn) return;
    std::vector<double> a = input(), b(kN * kN, 0.0), c(kN * kN, 0.0);
    run(*fn, a, b, c, 2.25);
    expectResults(a, b, c, 2.25);
  }
  llvm::sys::fs::remove_directories(dirPath);
}

namespace {

/// Plans the program, lowers the group with `pipelineOptions`, compiles it and runs it on real arrays.
void planLowerAndRun(const std::string &pipelineOptions) {
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
  auto session = opdsl::JITSession::create();
  OPDSL_EXPECT(static_cast<bool>(session));
  if (!session) return;
  mlir::PassManager lowering(&ctx);
  OPDSL_EXPECT(!registry.build("ops-jit-cpu", pipelineOptions, lowering));
  OPDSL_EXPECT(mlir::succeeded(lowering.run(groups[0].module.get())));
  auto object = (*session)->compile(groups[0].module.get(), groups[0].name);
  OPDSL_EXPECT(static_cast<bool>(object));
  if (!object) return;
  auto fn = (*session)->load(std::move(*object), groups[0].name);
  OPDSL_EXPECT(static_cast<bool>(fn));
  if (!fn) return;
  std::vector<double> a = input(), b(kN * kN, 0.0), c(kN * kN, 0.0);
  run(*fn, a, b, c, 0.5);
  expectResults(a, b, c, 0.5);
}

} // namespace

// loops=affine: the group is affine.parallel / affine.for with affine accesses until lower-affine; the results are those of scf.
OPDSL_TEST(EndToEndCPU, AffineLoopsGiveTheResultsOfTheScfForm) { planLowerAndRun("omp=false loops=affine"); }

// An affine transformation in between (unroll the x loop by 4: the range has 6 points, so the epilogue of 2 is exercised).
OPDSL_TEST(EndToEndCPU, AffineLoopsWithAnAffinePass) {
  planLowerAndRun("omp=false loops=affine affine-passes=func.func(affine-loop-unroll{unroll-factor=4})");
}

