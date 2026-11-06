// libopsjit on a real (small, sequential) OPS world: a block, dats, stencils, constants and a reduction handle, with
// kernel artifacts written by hand (what ops-kernel-import would embed). The executor is called as OPS calls it: the
// loops are queued with create_kerneldesc_and_enque_jit and run at ops_execute.

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Pipelines/GroupModules.h"
#include "ops-mlir/Runtime/Artifact.h"
#include "ops-mlir/Runtime/Claim.h"
#include "ops-mlir/Runtime/Executor.h"
#include "ops-mlir/Runtime/Options.h"

#include "opdsl/Testing/TestHarness.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "ops_jit.h"
#include "ops_lazy_executor.h"

#include <cmath>
#include <cstring>
#include <deque>
#include <vector>

using namespace ops_mlir;
using namespace ops_mlir::runtime;

namespace {

// ---- kernels -------------------------------------------------------------------------------------------------------

const char *kKernels = R"mlir(
ops.kernel @axpy(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>, %g: !ops.gbl<f64, 1>) {
  %offset = ops.const_ref @offset : f64
  %a = ops.gbl_read %g[0] : !ops.gbl<f64, 1> -> f64
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %w = arith.mulf %v, %a : f64
  %r = arith.addf %w, %offset : f64
  ops.write %y[0, 0], %r : !ops.acc<f64, 2, 1>
  ops.return
}
ops.kernel @cross(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>) {
  %c = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %e = ops.read %x[1, 0] : !ops.acc<f64, 2, 1> -> f64
  %w = ops.read %x[-1, 0] : !ops.acc<f64, 2, 1> -> f64
  %n = ops.read %x[0, 1] : !ops.acc<f64, 2, 1> -> f64
  %s = ops.read %x[0, -1] : !ops.acc<f64, 2, 1> -> f64
  %t1 = arith.addf %e, %w : f64
  %t2 = arith.addf %n, %s : f64
  %t3 = arith.addf %t1, %t2 : f64
  %t4 = arith.addf %t3, %c : f64
  ops.write %y[0, 0], %t4 : !ops.acc<f64, 2, 1>
  ops.return
}
ops.kernel @total(%x: !ops.acc<f64, 2, 1>, %r: !ops.red<f64>) {
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  ops.reduce %r, %v : !ops.red<f64>
  ops.return
}
ops.kernel @largest(%x: !ops.acc<f64, 2, 1>, %r: !ops.red<f64>) {
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  ops.reduce %r, %v : !ops.red<f64>
  ops.return
}
ops.kernel @halve(%x: !ops.acc<f16, 2, 1>, %y: !ops.acc<f16, 2, 1>) {
  %v = ops.read %x[0, 0] : !ops.acc<f16, 2, 1> -> f16
  %half = arith.constant 0.5 : f16
  %w = arith.mulf %v, %half : f16
  ops.write %y[0, 0], %w : !ops.acc<f16, 2, 1>
  ops.return
}
ops.kernel @repeat(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>) {
  %n = ops.const_ref @nreps : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %nn = arith.index_cast %n : i32 to index
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %s = scf.for %i = %c0 to %nn step %c1 iter_args(%acc = %v) -> (f64) {
    %t = arith.addf %acc, %v : f64
    scf.yield %t : f64
  }
  ops.write %y[0, 0], %s : !ops.acc<f64, 2, 1>
  ops.return
}
)mlir";

struct KernelDef {
  const char *name;
  const char *manifest; // JSON
};

const KernelDef kDefs[] = {
    {"axpy", R"({"manifest_version":1,"kernel":"axpy","status":"ok","importer":"test","digest":"t-axpy","ndim":2,
      "params":[{"kind":"acc","elem":"f64","ncomp":1,"const":true},{"kind":"acc","elem":"f64","ncomp":1,"const":false},{"kind":"gbl","elem":"f64","dim":1}],
      "consts":[{"name":"offset","elem":"f64","offset":0}],
      "traits":{"reads":{"0":[[0,0]],"1":[]},"writes":{"0":[],"1":[[0,0]]},"dynamic_reads":[],"idx_axes_read":[],"dropped_printf":0}})"},
    {"cross", R"({"manifest_version":1,"kernel":"cross","status":"ok","importer":"test","digest":"t-cross","ndim":2,
      "params":[{"kind":"acc","elem":"f64","ncomp":1,"const":true},{"kind":"acc","elem":"f64","ncomp":1,"const":false}],
      "consts":[],
      "traits":{"reads":{"0":[[0,0],[1,0],[-1,0],[0,1],[0,-1]],"1":[]},"writes":{"0":[],"1":[[0,0]]},"dynamic_reads":[],"idx_axes_read":[],"dropped_printf":0}})"},
    {"total", R"({"manifest_version":1,"kernel":"total","status":"ok","importer":"test","digest":"t-total","ndim":2,
      "params":[{"kind":"acc","elem":"f64","ncomp":1,"const":true},{"kind":"reduce","elem":"f64","dim":1,"style":"accumulate"}],
      "consts":[],"traits":{"reads":{"0":[[0,0]]},"writes":{"0":[]},"dynamic_reads":[],"idx_axes_read":[],"dropped_printf":0}})"},
    {"largest", R"({"manifest_version":1,"kernel":"largest","status":"ok","importer":"test","digest":"t-largest","ndim":2,
      "params":[{"kind":"acc","elem":"f64","ncomp":1,"const":true},{"kind":"reduce","elem":"f64","dim":1,"style":"accumulate"}],
      "consts":[],"traits":{"reads":{"0":[[0,0]]},"writes":{"0":[]},"dynamic_reads":[],"idx_axes_read":[],"dropped_printf":0}})"},
    {"halve", R"({"manifest_version":1,"kernel":"halve","status":"ok","importer":"test","digest":"t-halve","ndim":2,
      "params":[{"kind":"acc","elem":"f16","ncomp":1,"const":true},{"kind":"acc","elem":"f16","ncomp":1,"const":false}],
      "consts":[],"traits":{"reads":{"0":[[0,0]],"1":[]},"writes":{"0":[],"1":[[0,0]]},"dynamic_reads":[],"idx_axes_read":[],"dropped_printf":0}})"},
    {"repeat", R"({"manifest_version":1,"kernel":"repeat","status":"ok","importer":"test","digest":"t-repeat","ndim":2,
      "params":[{"kind":"acc","elem":"f64","ncomp":1,"const":true},{"kind":"acc","elem":"f64","ncomp":1,"const":false}],
      "consts":[{"name":"nreps","elem":"i32","offset":0,"role":"bound"}],
      "traits":{"reads":{"0":[[0,0]],"1":[]},"writes":{"0":[],"1":[[0,0]]},"dynamic_reads":[],"idx_axes_read":[],"dropped_printf":0}})"},
};

/// The embedded artifacts, as the translator would emit them.
struct Artifacts {
  std::deque<ops_jit_artifact> all;
  std::deque<std::string> strings;
  std::map<std::string, const ops_jit_artifact *> byName;

  Artifacts() {
    mlir::DialectRegistry registry;
    ops::registerJITDialects(registry);
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(kKernels, &ctx);
    if (!module) return;
    for (const KernelDef &def : kDefs) {
      // One module per kernel, as the importer writes them.
      mlir::OwningOpRef<mlir::ModuleOp> one(mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx)));
      for (ops::KernelOp k : module->getOps<ops::KernelOp>())
        if (k.getSymName() == def.name) {
          mlir::OpBuilder b(one->getBody(), one->getBody()->begin());
          b.insert(k->clone());
        }
      std::string bytes;
      llvm::raw_string_ostream os(bytes);
      if (mlir::failed(mlir::writeBytecodeToFile(one.get(), os))) continue;
      strings.push_back(bytes);
      const std::string &code = strings.back();
      strings.push_back(def.manifest);
      const std::string &manifest = strings.back();
      strings.push_back(def.name);
      ops_jit_artifact a{};
      a.version = OPS_JIT_ARTIFACT_VERSION;
      a.status = 0;
      a.kernel_name = strings.back().c_str();
      a.bytecode = reinterpret_cast<const unsigned char *>(code.data());
      a.bytecode_size = code.size();
      a.manifest_json = manifest.c_str();
      a.digest = "";
      all.push_back(a);
      byName[def.name] = &all.back();
    }
  }
};

Artifacts &artifacts() {
  static Artifacts a;
  return a;
}

// ---- the OPS world ---------------------------------------------------------------------------------------------------

constexpr int kNx = 10, kNy = 8; // interior; the dats have a halo of 1

struct World {
  ops_block block;
  ops_dat a, b, c;
  ops_stencil s00, s5;
  ops_reduction sum, maxh;
  ops_dat ha, hb;
  double offset = 0.5, scale = 2.0;
  int nreps = 3;

  World() {
    const char *argv[] = {"runtime_test"};
    ops_init(1, argv, 0);
    block = ops_decl_block(2, "block");
    int size[2] = {kNx, kNy}, base[2] = {0, 0}, dm[2] = {-1, -1}, dp[2] = {1, 1};
    a = ops_decl_dat(block, 1, size, base, dm, dp, (double *)NULL, "double", "a");
    b = ops_decl_dat(block, 1, size, base, dm, dp, (double *)NULL, "double", "b");
    c = ops_decl_dat(block, 1, size, base, dm, dp, (double *)NULL, "double", "c");
    ha = ops_decl_dat(block, 1, size, base, dm, dp, (_Float16 *)NULL, "half", "ha");
    hb = ops_decl_dat(block, 1, size, base, dm, dp, (_Float16 *)NULL, "half", "hb");
    int p00[2] = {0, 0};
    int p5[10] = {0, 0, 1, 0, -1, 0, 0, 1, 0, -1};
    s00 = ops_decl_stencil(2, 1, p00, "00");
    s5 = ops_decl_stencil(2, 5, p5, "cross");
    sum = ops_decl_reduction_handle(sizeof(double), "double", "sum");
    maxh = ops_decl_reduction_handle(sizeof(double), "double", "max");
    ops_decl_const("offset", 1, "double", &offset);
    ops_decl_const("nreps", 1, "int", &nreps);
    ops_partition("");
  }

  /// Element (i, j) in OPS's global indexing (the halo included: i, j >= -1).
  double &at(ops_dat d, int i, int j) {
    int ld = d->size[0];
    return reinterpret_cast<double *>(d->data)[(i - d->base[0] - d->d_m[0]) + (j - d->base[1] - d->d_m[1]) * ld];
  }
};

World &world() {
  static World w;
  return w;
}

int aotCalls = 0;
void aotMarker(ops_kernel_descriptor *) { ++aotCalls; }

void enqueue(const char *name, const char *artifact, ops_arg *args, int nargs, int *range) {
  World &w = world();
  create_kerneldesc_and_enque_jit(name, args, nargs, 0, 2, 0, range, w.block, aotMarker, artifacts().byName.at(artifact));
}

void flush() { ops_execute(OPS_instance::getOPSInstance()); }

void fill(World &w) {
  for (int j = -1; j <= kNy; ++j)
    for (int i = -1; i <= kNx; ++i) {
      w.at(w.a, i, j) = 1.0 + 0.25 * i + 0.5 * j;
      w.at(w.b, i, j) = 0.0;
      w.at(w.c, i, j) = 0.0;
    }
}

/// Registers the executor once, for all tests.
void startExecutor() {
  static bool done = false;
  if (done) return;
  done = true;
  world();
  ops_jit_init();
}

} // namespace

OPDSL_TEST(Runtime, ManifestAndArtifactsParse) {
  mlir::DialectRegistry registry;
  ops::registerJITDialects(registry);
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  ArtifactCache cache(ctx);
  std::string reason;
  const KernelArtifact *k = cache.get(artifacts().byName.at("axpy"), reason);
  OPDSL_EXPECT(k != nullptr);
  if (!k) return;
  OPDSL_EXPECT_EQ(k->manifest.params.size(), size_t(3));
  OPDSL_EXPECT(k->manifest.params[2].kind == ParamInfo::Gbl);
  OPDSL_EXPECT_EQ(k->manifest.consts.size(), size_t(1));
  OPDSL_EXPECT_EQ(k->manifest.consts[0].name, std::string("offset"));
  OPDSL_EXPECT(!k->manifest.consts[0].bound);
  OPDSL_EXPECT(static_cast<bool>(k->kernel));
  const KernelArtifact *r = cache.get(artifacts().byName.at("repeat"), reason);
  OPDSL_EXPECT(r && r->manifest.consts[0].bound);
  OPDSL_EXPECT(cache.get(nullptr, reason) == nullptr);
  OPDSL_EXPECT(!reason.empty());
}

OPDSL_TEST(Runtime, ClaimDeclinesWithReasons) {
  startExecutor();
  World &w = world();
  mlir::DialectRegistry registry;
  ops::registerJITDialects(registry);
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  ArtifactCache cache(ctx);
  std::string reason;
  const KernelArtifact *axpy = cache.get(artifacts().byName.at("axpy"), reason);
  const KernelArtifact *cross = cache.get(artifacts().byName.at("cross"), reason);
  OPDSL_EXPECT(axpy && cross);
  if (!axpy || !cross) return;

  double g = 2.0;
  int range[4] = {0, kNx, 0, kNy};
  auto descriptor = [&](ops_arg *args, int nargs, int dim = 2) {
    ops_kernel_descriptor *d = (ops_kernel_descriptor *)calloc(1, sizeof(ops_kernel_descriptor));
    d->name = (char *)"test";
    d->block = w.block;
    d->dim = dim;
    d->nargs = nargs;
    d->args = args;
    d->range = (int *)calloc(2 * OPS_MAX_DIM, sizeof(int));
    d->orig_range = (int *)calloc(2 * OPS_MAX_DIM, sizeof(int));
    for (int i = 0; i < 4; ++i) d->range[i] = d->orig_range[i] = range[i];
    return d;
  };
  auto freeDescriptor = [](ops_kernel_descriptor *d) {
    free(d->range);
    free(d->orig_range);
    free(d);
  };

  {
    ops_arg args[3] = {ops_arg_dat(w.a, 1, w.s00, "double", OPS_READ), ops_arg_dat(w.b, 1, w.s00, "double", OPS_WRITE),
                       ops_arg_gbl(&g, 1, "double", OPS_READ)};
    ops_kernel_descriptor *d = descriptor(args, 3);
    OPDSL_EXPECT_EQ(claimReason(d, *axpy, false), std::string());
    // Declines: each with its own sentence.
    d->isdevice = 1;
    OPDSL_EXPECT(claimReason(d, *axpy, false).find("device") != std::string::npos);
    d->isdevice = 0;
    OPDSL_EXPECT(claimReason(d, *cross, false).find("parameters") != std::string::npos); // 2 parameters, 3 arguments
    w.a->e_dat = 1; // on one process an edge dat is a dat like any other
    OPDSL_EXPECT_EQ(claimReason(d, *axpy, false), std::string());
    w.a->e_dat = 0;
    w.a->stride[0] = 2;
    OPDSL_EXPECT(claimReason(d, *axpy, false).find("multigrid") != std::string::npos);
    w.a->stride[0] = 1;
    ops_arg written = ops_arg_dat(w.a, 1, w.s00, "double", OPS_WRITE);
    ops_arg savedArg = args[0];
    args[0] = written; // the kernel only reads parameter 0: a write access is allowed, a stencil that is too small is not
    OPDSL_EXPECT_EQ(claimReason(d, *axpy, false), std::string());
    args[0] = savedArg;
    d->dim = 3;
    OPDSL_EXPECT(claimReason(d, *axpy, false).find("2D") != std::string::npos);
    d->dim = 2;
    freeDescriptor(d);
  }
  {
    // The kernel reads at 5 offsets; a loop that passes the point stencil for that argument is declined.
    ops_arg args[2] = {ops_arg_dat(w.a, 1, w.s00, "double", OPS_READ), ops_arg_dat(w.b, 1, w.s00, "double", OPS_WRITE)};
    ops_kernel_descriptor *d = descriptor(args, 2);
    OPDSL_EXPECT(claimReason(d, *cross, false).find("outside the stencil") != std::string::npos);
    args[0] = ops_arg_dat(w.a, 1, w.s5, "double", OPS_READ);
    OPDSL_EXPECT_EQ(claimReason(d, *cross, false), std::string());
    freeDescriptor(d);
  }
}

OPDSL_TEST(Runtime, ChainOfLoopsRunsThroughTheJIT) {
  startExecutor();
  World &w = world();
  Stats &stats = executor().stats();
  fill(w);
  w.sum->initialized = 0;
  aotCalls = 0;

  double g = 2.0;
  int full[4] = {-1, kNx + 1, -1, kNy + 1};
  int inner[4] = {0, kNx, 0, kNy};
  {
    ops_arg args[3] = {ops_arg_dat(w.a, 1, w.s00, "double", OPS_READ), ops_arg_dat(w.b, 1, w.s00, "double", OPS_WRITE),
                       ops_arg_gbl(&g, 1, "double", OPS_READ)};
    enqueue("axpy", "axpy", args, 3, full);
  }
  {
    ops_arg args[2] = {ops_arg_dat(w.b, 1, w.s5, "double", OPS_READ), ops_arg_dat(w.c, 1, w.s00, "double", OPS_WRITE)};
    enqueue("cross", "cross", args, 2, inner);
  }
  {
    ops_arg args[2] = {ops_arg_dat(w.c, 1, w.s00, "double", OPS_READ), ops_arg_reduce(w.sum, 1, "double", OPS_INC)};
    enqueue("total", "total", args, 2, inner);
  }
  g = -1000.0; // changed after the call: the queued loop keeps the value it was given
  w.offset = 0.5;
  flush();

  OPDSL_EXPECT_EQ(aotCalls, 0);
  OPDSL_EXPECT_EQ(stats.loopsDeclined, uint64_t(0));
  OPDSL_EXPECT(stats.groupsExecuted >= 2);

  double expectedSum = 0;
  int wrong = 0;
  auto bval = [&](int i, int j) { return 2.0 * (1.0 + 0.25 * i + 0.5 * j) + 0.5; };
  for (int j = 0; j < kNy; ++j)
    for (int i = 0; i < kNx; ++i) {
      double e = bval(i, j) + bval(i + 1, j) + bval(i - 1, j) + bval(i, j + 1) + bval(i, j - 1);
      expectedSum += e;
      if (std::fabs(w.at(w.c, i, j) - e) > 1e-12) ++wrong;
    }
  OPDSL_EXPECT_EQ(wrong, 0);
  double got;
  std::memcpy(&got, w.sum->data, sizeof(double));
  OPDSL_EXPECT(std::fabs(got - expectedSum) < 1e-9 * std::fabs(expectedSum));
}

OPDSL_TEST(Runtime, SameStructureReusesThePlanAndConstantsAreArguments) {
  startExecutor();
  World &w = world();
  Stats &stats = executor().stats();
  double g = 1.0;
  int inner[4] = {0, kNx, 0, kNy};
  auto run = [&](double offset) {
    fill(w);
    w.offset = offset;
    ops_update_const("offset", 1, "double", &w.offset);
    ops_arg args[3] = {ops_arg_dat(w.a, 1, w.s00, "double", OPS_READ), ops_arg_dat(w.b, 1, w.s00, "double", OPS_WRITE),
                       ops_arg_gbl(&g, 1, "double", OPS_READ)};
    enqueue("axpy", "axpy", args, 3, inner);
    flush();
  };
  run(0.5);
  uint64_t compilesBefore = executor().compileStats().compiles, hitsBefore = stats.planHits;
  run(7.0);
  OPDSL_EXPECT_EQ(executor().compileStats().compiles, compilesBefore);
  OPDSL_EXPECT_EQ(stats.planHits, hitsBefore + 1);
  OPDSL_EXPECT(std::fabs(w.at(w.b, 3, 2) - ((1.0 + 0.25 * 3 + 0.5 * 2) + 7.0)) < 1e-14);
}

OPDSL_TEST(Runtime, BoundConstantsAreFoldedAndPartOfThePlanKey) {
  startExecutor();
  World &w = world();
  Stats &stats = executor().stats();
  int inner[4] = {0, kNx, 0, kNy};
  auto run = [&](int nreps) {
    fill(w);
    w.nreps = nreps;
    ops_update_const("nreps", 1, "int", &w.nreps);
    ops_arg args[2] = {ops_arg_dat(w.a, 1, w.s00, "double", OPS_READ), ops_arg_dat(w.b, 1, w.s00, "double", OPS_WRITE)};
    enqueue("repeat", "repeat", args, 2, inner);
    flush();
  };
  run(3);
  OPDSL_EXPECT(std::fabs(w.at(w.b, 2, 2) - 4.0 * (1.0 + 0.5 + 1.0)) < 1e-14);
  uint64_t compiles = executor().compileStats().compiles;
  run(5); // another value of a bound constant: another plan, and other code
  OPDSL_EXPECT(std::fabs(w.at(w.b, 2, 2) - 6.0 * (1.0 + 0.5 + 1.0)) < 1e-14);
  OPDSL_EXPECT(executor().compileStats().compiles + executor().compileStats().diskHits > compiles);
  uint64_t plans = stats.planBuilds;
  run(3); // back to the first: its plan is still there
  OPDSL_EXPECT_EQ(stats.planBuilds, plans);
  OPDSL_EXPECT(std::fabs(w.at(w.b, 2, 2) - 4.0 * (1.0 + 0.5 + 1.0)) < 1e-14);
}

OPDSL_TEST(Runtime, DeclinedLoopsRunWithTheStockCode) {
  startExecutor();
  World &w = world();
  Stats &stats = executor().stats();
  fill(w);
  aotCalls = 0;
  int inner[4] = {0, kNx, 0, kNy};
  double g = 1.0;
  uint64_t declined = stats.loopsDeclined;
  w.a->stride[0] = 2; // a coarse dat of a multigrid hierarchy is not supported: the loop falls back
  ops_arg args[3] = {ops_arg_dat(w.a, 1, w.s00, "double", OPS_READ), ops_arg_dat(w.b, 1, w.s00, "double", OPS_WRITE),
                     ops_arg_gbl(&g, 1, "double", OPS_READ)};
  enqueue("axpy", "axpy", args, 3, inner);
  flush();
  w.a->stride[0] = 1;
  OPDSL_EXPECT_EQ(aotCalls, 1);
  OPDSL_EXPECT_EQ(stats.loopsDeclined, declined + 1);
}

OPDSL_TEST(Runtime, ReductionsCombineIntoTheHandle) {
  startExecutor();
  World &w = world();
  fill(w);
  int inner[4] = {0, kNx, 0, kNy};
  w.maxh->initialized = 0;
  {
    ops_arg args[2] = {ops_arg_dat(w.a, 1, w.s00, "double", OPS_READ), ops_arg_reduce(w.maxh, 1, "double", OPS_MAX)};
    enqueue("largest", "largest", args, 2, inner);
  }
  flush();
  double got;
  std::memcpy(&got, w.maxh->data, sizeof(double));
  OPDSL_EXPECT_EQ(got, 1.0 + 0.25 * (kNx - 1) + 0.5 * (kNy - 1));
  // A second loop with the same handle continues from the value in it.
  {
    ops_arg args[2] = {ops_arg_dat(w.b, 1, w.s00, "double", OPS_READ), ops_arg_reduce(w.maxh, 1, "double", OPS_MAX)};
    enqueue("largest", "largest", args, 2, inner);
  }
  flush();
  std::memcpy(&got, w.maxh->data, sizeof(double));
  OPDSL_EXPECT_EQ(got, 1.0 + 0.25 * (kNx - 1) + 0.5 * (kNy - 1)); // b holds smaller values: the maximum stays
}

OPDSL_TEST(Runtime, HalfPrecisionDatsAreComputedInSinglePrecisionAndRoundedBack) {
  startExecutor();
  World &w = world();
  int inner[4] = {0, kNx, 0, kNy};
  _Float16 *a = reinterpret_cast<_Float16 *>(w.ha->data), *b = reinterpret_cast<_Float16 *>(w.hb->data);
  const size_t n = w.ha->mem / sizeof(_Float16);
  for (size_t i = 0; i < n; ++i) {
    a[i] = (_Float16)(0.37f * float(i % 29) - 3.0f);
    b[i] = (_Float16)0;
  }
  ops_arg args[2] = {ops_arg_dat(w.ha, 1, w.s00, "half", OPS_READ), ops_arg_dat(w.hb, 1, w.s00, "half", OPS_WRITE)};
  enqueue("halve", "halve", args, 2, inner);
  flush();
  int wrong = 0;
  for (int j = 0; j < kNy; ++j)
    for (int i = 0; i < kNx; ++i) {
      size_t k = (i + 1) + size_t(j + 1) * w.ha->size[0];
      if (b[k] != (_Float16)(0.5f * float(a[k]))) ++wrong;
    }
  OPDSL_EXPECT_EQ(wrong, 0);
}
