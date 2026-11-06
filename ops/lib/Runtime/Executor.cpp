#include "ops-mlir/Runtime/Executor.h"

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Pipelines/GroupModules.h"
#include "ops-mlir/Pipelines/Pipelines.h"
#include "ops-mlir/Runtime/CallDump.h"
#include "ops-mlir/Runtime/Claim.h"
#include "ops-mlir/Runtime/Lifter.h"

#include "opdsl/GPU/GpuShim.h"

#include "opdsl/Support/Hash.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "ops_lazy_executor.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

namespace ops_mlir::runtime {

using namespace mlir;
using namespace ops_mlir::ops;
using Clock = std::chrono::steady_clock;

namespace {

double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct ScopedTime {
  double &sink;
  Clock::time_point t0 = Clock::now();
  explicit ScopedTime(double &s) : sink(s) {}
  ~ScopedTime() { sink += since(t0); }
};

/// Collects the diagnostics emitted while alive.
struct CaptureDiagnostics {
  std::string text;
  ScopedDiagnosticHandler handler;
  explicit CaptureDiagnostics(MLIRContext *ctx)
      : handler(ctx, [this](Diagnostic &d) {
          llvm::raw_string_ostream os(text);
          os << d << "\n";
          return success();
        }) {}
};

std::string errorText(llvm::Error e) { return llvm::toString(std::move(e)); }

/// OPS_JIT_DUMP_DIR: writes `text` (or the module) to <dir>/<name>.
void dumpText(const std::string &name, llvm::StringRef text) {
  const std::string &dir = options().dumpDir;
  if (dir.empty()) return;
  llvm::sys::fs::create_directories(dir);
  std::error_code ec;
  llvm::raw_fd_ostream os(dir + "/" + name, ec);
  if (!ec) os << text;
}

void dumpModule(const std::string &name, ModuleOp module) {
  if (options().dumpDir.empty()) return;
  std::string text;
  llvm::raw_string_ostream os(text);
  module->print(os);
  dumpText(name, text);
}

/// The module of one group without what makes equal code look different: loop numbers, concrete ranges.
OwningOpRef<ModuleOp> canonicalGroup(ModuleOp m, bool keepRanges) {
  OwningOpRef<ModuleOp> copy(cast<ModuleOp>(m->clone()));
  Builder b(m.getContext());
  for (ChainOp chain : copy->getOps<ChainOp>()) {
    chain.setSymName("group");
    if (!keepRanges) {
      chain->removeAttr("ops.range_values");
      chain->removeAttr("ops.shape_values");
    }
    chain->removeAttr("ops.const_values");
    chain->removeAttr("ops.const_roles");
    chain->removeAttr("ops.binding_consts");
    chain.walk([&](ParLoopOp loop) {
      loop->setAttr("loop_id", b.getI64IntegerAttr(-1));
      loop->setAttr("block", b.getI64IntegerAttr(0));
      loop->removeAttr("ops.fused_loops");
    });
  }
  return copy;
}

/// slot = slot (kind) value, with OPS's semantics of MIN/MAX (a < b ? a : b).
template <typename T> void combineInto(char *slot, T value, ReductionKind kind) {
  T cur;
  std::memcpy(&cur, slot, sizeof(T));
  T next;
  if (kind == ReductionKind::Add) next = cur + value;
  else if (kind == ReductionKind::Min) next = cur < value ? cur : value;
  else next = cur > value ? cur : value;
  std::memcpy(slot, &next, sizeof(T));
}


/// Everything OPS_JIT_VERIFY needs to know about one group before and after it ran.
struct Snapshot {
  std::vector<ops_dat> dats;               ///< dats the members write
  std::vector<std::vector<char>> before, jit;
  std::vector<std::pair<char *, size_t>> slots; ///< reduction handles the members update
  std::vector<std::vector<char>> slotBefore, slotJit;
};

std::vector<char> copyBytes(const char *p, size_t n) { return std::vector<char>(p, p + n); }

/// Largest |a-b| and largest |a|, |b| over `n` elements of type T.
template <typename T> void compareElements(const char *a, const char *b, size_t n, double &maxDiff, double &maxAbs, size_t &differing) {
  for (size_t i = 0; i < n; ++i) {
    T x, y;
    std::memcpy(&x, a + i * sizeof(T), sizeof(T));
    std::memcpy(&y, b + i * sizeof(T), sizeof(T));
    if (x == y || (x != x && y != y)) {
      double ax = std::fabs((double)x);
      if (ax == ax && ax > maxAbs) maxAbs = ax;
      continue;
    }
    ++differing;
    double d = std::fabs((double)x - (double)y);
    if (d != d) d = HUGE_VAL;
    if (d > maxDiff) maxDiff = d;
    maxAbs = std::max(maxAbs, std::max(std::fabs((double)x), std::fabs((double)y)));
  }
}

/// Relative difference (to the largest magnitude in the data) of two buffers of `type` elements, and the number of
/// elements that differ.
double relativeDifference(const char *type, const char *a, const char *b, size_t bytes, size_t &differing) {
  double maxDiff = 0, maxAbs = 0;
  differing = 0;
  std::string t = type ? type : "";
  if (t == "double") compareElements<double>(a, b, bytes / 8, maxDiff, maxAbs, differing);
  else if (t == "float") compareElements<float>(a, b, bytes / 4, maxDiff, maxAbs, differing);
#ifdef __FLT16_MAX__
  else if (t == "half") compareElements<_Float16>(a, b, bytes / 2, maxDiff, maxAbs, differing);
#endif
  else compareElements<int32_t>(a, b, bytes / 4, maxDiff, maxAbs, differing);
  if (maxDiff == 0) return 0;
  return maxAbs > 0 ? maxDiff / maxAbs : maxDiff;
}

double toleranceFor(const char *type) {
  std::string t = type ? type : "";
  if (t == "double") return 1e-12;
  if (t == "float") return options().verifyFloatTol;
  if (t == "half") return options().verifyHalfTol; // 11 bits of mantissa; the AOT code rounds a whole expression once, the JIT every operation
  return 0;
}


/// The whole allocation of a dat as bytes, from the host or the device copy.
std::vector<char> readDat(ops_dat d, bool device) {
  std::vector<char> bytes(d->mem);
  if (device) mgpuMemcpy(bytes.data(), d->data_d, d->mem, nullptr);
  else std::memcpy(bytes.data(), d->data, d->mem);
  return bytes;
}

void writeDat(ops_dat d, const std::vector<char> &bytes, bool device) {
  if (device) mgpuMemcpy(d->data_d, const_cast<char *>(bytes.data()), bytes.size(), nullptr);
  else std::memcpy(d->data, bytes.data(), bytes.size());
}

/// Number of elements of a reduction buffer: one per warp (or per thread when the block is not whole warps) of the grid that the
/// lowering launches for the ranges `lo/hi` (see convert-ops-to-scf, target gpu).
int64_t gpuReductionUnits(const std::array<int64_t, 3> &block, const int64_t *extent, int rank) {
  int64_t blocks = 1;
  for (int d = 0; d < rank; ++d) blocks *= std::max<int64_t>(0, (std::max<int64_t>(extent[d], 0) + block[d] - 1) / block[d]);
  const int64_t threads = block[0] * block[1] * block[2];
  const int64_t perBlock = threads % 32 == 0 ? threads / 32 : threads;
  return std::max<int64_t>(1, blocks * perBlock);
}

/// The PTX version the toolkit's ptxas accepts, as an NVVM target feature (+ptx83 for CUDA 12.3): newer PTX allows larger kernel
/// parameter blocks (32 KB from PTX 8.1) which the fused groups of big applications need.
std::string ptxFeature() {
  static std::string feature = [] {
    std::string cmd = "ptxas --version 2>/dev/null";
    const char *root = std::getenv("CUDA_ROOT");
    if (!root) root = std::getenv("CUDA_HOME");
    if (!root) root = std::getenv("CUDA_PATH");
    if (root) cmd = std::string("'") + root + "/bin/ptxas' --version 2>/dev/null";
    std::string out;
    if (FILE *p = popen(cmd.c_str(), "r")) {
      char buf[256];
      while (std::fgets(buf, sizeof buf, p)) out += buf;
      pclose(p);
    }
    size_t at = out.find("release ");
    int major = 0, minor = 0;
    if (at != std::string::npos && std::sscanf(out.c_str() + at, "release %d.%d", &major, &minor) == 2 && major == 12)
      return "+ptx" + std::to_string(80 + std::min(minor, 7));
    return std::string("+ptx80");
  }();
  return feature;
}

/// Verify mode: the dats and reduction handles the members of a group write, and their values before the group ran. Each member is
/// prepared (OPS's bookkeeping before a loop).
Snapshot prepareGroup(const Plan::Group &g, ops_kernel_descriptor **run, bool verify) {
  Snapshot snap;
  if (!verify) {
    for (int k : g.recipe.members) ops_lazy_prepare(run[k]);
    return snap;
  }
  std::set<ops_dat> seen;
  for (int k : g.recipe.members)
    for (int a = 0; a < run[k]->nargs; ++a) {
      const ops_arg &arg = run[k]->args[a];
      if (arg.argtype == OPS_ARG_DAT && arg.acc != OPS_READ && seen.insert(arg.dat).second) snap.dats.push_back(arg.dat);
      if (arg.argtype == OPS_ARG_GBL && arg.acc != OPS_READ) {
        char *slot = ops_lazy_reduction_slot(run[k], a);
        snap.slots.push_back({slot, (size_t)arg.dim * elemBytes(elemName(((ops_reduction)arg.data)->type))});
      }
    }
  for (int k : g.recipe.members) ops_lazy_prepare(run[k]);
  for (ops_dat d : snap.dats) snap.before.push_back(readDat(d, g.device));
  for (auto &s : snap.slots) snap.slotBefore.push_back(copyBytes(s.first, s.second));
  return snap;
}

/// Folds the reductions of a group that has run into the OPS handles of the loops that asked for them. On the device the value is the
/// combination of the partial values of the warps in the staging copy of the device buffer.
void combineReductions(const Plan &plan, const Plan::Group &g, ops_kernel_descriptor **run, const Invocation &inv,
                       const std::vector<char> &staging, int64_t units) {
  for (size_t r = 0; r < g.recipe.reductionOf.size(); ++r) {
    int index = g.recipe.reductionOf[r];
    if (index < 0) continue;
    const ReductionRef &ref = plan.reductions[index];
    char *slot = ops_lazy_reduction_slot(run[ref.loop], ref.arg);
    if (g.device) {
      // The value is the combination of the partial values of the warps in the staging copy of the device buffer.
      for (int64_t k = 0; k < units; ++k) {
        const char *p = &staging[(r * units + k) * 8];
        if (ref.elem == "f64") {
          double v;
          std::memcpy(&v, p, 8);
          combineInto<double>(slot, v, ref.kind);
        } else if (ref.elem == "f32") {
          float v;
          std::memcpy(&v, p, 4);
          combineInto<float>(slot, v, ref.kind);
        } else {
          int32_t v;
          std::memcpy(&v, p, 4);
          combineInto<int32_t>(slot, v, ref.kind);
        }
      }
    } else if (ref.elem == "f64") combineInto<double>(slot, inv.resultAsDouble(r), ref.kind);
    else if (ref.elem == "f32") combineInto<float>(slot, static_cast<float>(inv.resultAsDouble(r)), ref.kind);
    else combineInto<int32_t>(slot, static_cast<int32_t>(inv.resultAsInt(r)), ref.kind);
  }
}

/// OPS_JIT_VERIFY: the group has just run. Read its results, go back to the state before it, run the stock OPS code for each member in
/// order and compare. True when they differ.
bool verifyGroup(OPS_instance *instance, Snapshot &snap, const Plan::Group &g, ops_kernel_descriptor **run) {
  // Back to the state before the group, then the stock code for each member in order.
  for (size_t i = 0; i < snap.dats.size(); ++i) writeDat(snap.dats[i], snap.before[i], g.device);
  for (size_t i = 0; i < snap.slots.size(); ++i) std::memcpy(snap.slots[i].first, snap.slotBefore[i].data(), snap.slotBefore[i].size());
  for (int k : g.recipe.members) ops_lazy_execute_default(instance, &run[k], 1);
  std::string names;
  for (int k : g.recipe.members) names += std::string(names.empty() ? "" : ", ") + run[k]->name;
  bool bad = false;
  for (size_t i = 0; i < snap.dats.size(); ++i) {
    size_t differing = 0;
    std::vector<char> aot = readDat(snap.dats[i], g.device);
    double rel = relativeDifference(snap.dats[i]->type, snap.jit[i].data(), aot.data(), snap.jit[i].size(), differing);
    if (rel > toleranceFor(snap.dats[i]->type)) {
      bad = true;
      std::fprintf(stderr, "ops_jit verify: [%s] dat %s: %zu elements differ, relative difference %.3g (tolerance %.1g)\n", names.c_str(),
                   snap.dats[i]->name ? snap.dats[i]->name : "?", differing, rel, toleranceFor(snap.dats[i]->type));
    }
  }
  for (size_t i = 0; i < snap.slots.size(); ++i) {
    // Reduction values: compared in relative terms; the handles are doubles, floats or ints.
    size_t bytes = snap.slotJit[i].size();
    size_t differing = 0;
    const char *type = bytes == 8 ? "double" : bytes == 4 ? "float" : "int";
    double rel = relativeDifference(type, snap.slotJit[i].data(), snap.slots[i].first, bytes, differing);
    if (rel > std::max(toleranceFor(type), 1e-9)) {
      bad = true;
      auto asDouble = [&](const char *p) {
        if (bytes == 8) { double v; std::memcpy(&v, p, 8); return v; }
        if (bytes == 4 && type[0] == 'f') { float v; std::memcpy(&v, p, 4); return double(v); }
        int v; std::memcpy(&v, p, 4); return double(v);
      };
      std::fprintf(stderr, "ops_jit verify: [%s] reduction %zu differs, relative difference %.3g (before %.17g, JIT %.17g, OPS %.17g)\n",
                   names.c_str(), i, rel, asDouble(snap.slotBefore[i].data()), asDouble(snap.slotJit[i].data()), asDouble(snap.slots[i].first));
    }
  }
  return bad;
}

/// Writes the module after the plan passes to the persistent cache (best effort; `explain` says why not).
void storePlanModule(opdsl::PersistentCache &cache, const std::string &diskKey, const std::string &planKey, ModuleOp module, bool explain) {
  llvm::SmallString<0> bytes;
  llvm::raw_svector_ostream os(bytes);
  if (succeeded(writeBytecodeToFile(module, os))) {
    if (llvm::Error e = cache.store(diskKey, "plan", std::string(bytes.begin(), bytes.end()))) {
      if (explain) std::fprintf(stderr, "ops_jit: plan %.12s not stored: %s\n", planKey.c_str(), errorText(std::move(e)).c_str());
      else llvm::consumeError(std::move(e));
    } else if (explain) {
      std::fprintf(stderr, "ops_jit: plan %.12s stored (%zu bytes)\n", planKey.c_str(), bytes.size());
    }
  } else if (explain) {
    std::fprintf(stderr, "ops_jit: plan %.12s: the module cannot be written as bytecode\n", planKey.c_str());
  }
}


/// The settings of the pipelines for a run whose first loop is `first`: the options of the runtime, the layout of the OPS instance and,
/// for a device, the chip and the PTX version the toolkit accepts.
PipelineSettings pipelineSettingsFor(const Options &o, ops_kernel_descriptor *first) {
  PipelineSettings s;
  s.fusion = o.fusion;
  s.fusionMax = o.fusionMax;
  s.fusionReorder = o.fusionReorder;
  s.fusionPlacement = o.fusionPlacement;
  s.fusionGuarded = o.fusionGuarded;
  s.specialize = o.specialize;
  s.soa = ops_lazy_soa(first->block->instance) != 0;
  s.fma = o.fma;
  s.omp = o.omp;
  s.loops = o.loops;
  s.affinePasses = o.affinePasses;
  s.gpuBlock = o.gpuBlock;
  s.ptxasOptions = o.ptxasOpts;
  if (first->isdevice) {
    s.gpuArch = o.gpuArch.empty() ? opdsl::gpu::cudaDeviceArch() : o.gpuArch;
    s.gpuFeatures = ptxFeature();
  }
  return s;
}

} // namespace

Executor::Executor() {
  registerOPSPipelines(registry_);
  registerJITDialects(dialects_);
  ctx_ = std::make_unique<MLIRContext>(dialects_, MLIRContext::Threading::DISABLED);
  ctx_->loadAllAvailableDialects();
  artifacts_ = std::make_unique<ArtifactCache>(*ctx_);
  CompilerOptions co;
  co.useCache = options().cache;
  co.cacheDir = options().cacheDir;
  co.dumpDir = options().dumpDir;
  co.jit.fpContract = options().fma;
  compiler_ = std::make_unique<GroupCompiler>(*ctx_, registry_, co);
}

Executor &executor() {
  static Executor *instance = new Executor(); // never destroyed: JIT'd code and OPS may outlive static destructors
  return *instance;
}

void Executor::run(OPS_instance *instance, ops_kernel_descriptor **descs, int n, void *ctx) {
  (void)ctx;
  executor().execute(instance, descs, n);
}

bool Executor::deviceReady() {
  if (deviceState_ < 0) {
    deviceState_ = opdsl::gpu::cudaAvailable() ? 1 : 0;
    if (deviceState_) {
      // The NVVM target finds ptxas and libdevice through the toolkit path: the usual variables, else the CUDA used to build OPS.
      if (!std::getenv("CUDA_ROOT") && !std::getenv("CUDA_HOME") && !std::getenv("CUDA_PATH"))
        if (const char *root = std::getenv("CUDA_INSTALL_PATH")) ::setenv("CUDA_ROOT", root, 0);
    }
  }
  return deviceState_ == 1;
}

void *Executor::deviceScratch(size_t bytes) {
  if (bytes > scratchBytes_) {
    if (scratch_) mgpuMemFree(scratch_, nullptr);
    scratchBytes_ = std::max<size_t>(bytes, 256);
    scratch_ = mgpuMemAlloc(scratchBytes_, nullptr, false);
  }
  return scratch_;
}

void Executor::execute(OPS_instance *instance, ops_kernel_descriptor **descs, int n) {
  const Options &opts = options();
  stats_.flushes++;
  std::vector<const KernelArtifact *> artifacts(n, nullptr);
  std::vector<char> claimed(n, 0);
  {
    ScopedTime timer(stats_.tClaim);
    static std::set<std::pair<std::string, std::string>> explained;
    for (int i = 0; i < n; ++i) {
      stats_.loopsSeen++;
      std::string reason;
      if (opts.claim == "none") reason = "OPS_JIT_CLAIM=none";
      else if (descs[i]->startup_func) reason = "the loop has a startup function";
      else {
        artifacts[i] = artifacts_->get(descs[i]->jit_artifact, reason);
        if (artifacts[i]) reason = claimReason(descs[i], *artifacts[i], descs[i]->isdevice ? deviceReady() : false);
      }
      if (reason.empty()) {
        claimed[i] = 1;
        stats_.loopsClaimed++;
      } else {
        stats_.loopsDeclined++;
        stats_.declineReasons[reason]++;
        if (opts.explain && explained.insert({descs[i]->name, reason}).second)
          std::fprintf(stderr, "ops_jit: %s runs with OPS: %s\n", descs[i]->name, reason.c_str());
      }
    }
  }

  int i = 0;
  while (i < n) {
    if (!claimed[i]) {
      int j = i;
      while (j < n && !claimed[j]) ++j;
      ops_lazy_execute_default(instance, descs + i, j - i);
      i = j;
      continue;
    }
    int j = i;
    while (j < n && claimed[j] && j - i < opts.runMax && descs[j]->isdevice == descs[i]->isdevice) ++j; // host and device loops do not share a plan
    std::vector<const KernelArtifact *> runArtifacts(artifacts.begin() + i, artifacts.begin() + j);
    if (!runJit(instance, descs + i, j - i, runArtifacts)) ops_lazy_execute_default(instance, descs + i, j - i);
    i = j;
  }
}

std::string Executor::structuralKey(ops_kernel_descriptor **run, int n, const std::vector<std::vector<int>> &ranges,
                                    const std::vector<const KernelArtifact *> &artifacts, bool soa) {
  opdsl::Hasher h;
  h.add(options().codegenDigest()).addBool(soa).addInt(kCodeFormat);
  std::map<int, int> slotOf;
  std::set<std::string> boundDone;
  for (int k = 0; k < n; ++k) {
    ops_kernel_descriptor *d = run[k];
    h.add(artifacts[k]->manifest.digest).addInt(d->block->index).addInt(d->dim).addInt(d->isdevice);
    for (int a = 0; a < d->nargs; ++a) {
      const ops_arg &arg = d->args[a];
      h.addInt(arg.argtype).addInt(arg.acc);
      if (arg.argtype == OPS_ARG_DAT) {
        int slot = slotOf.emplace(arg.dat->index, (int)slotOf.size()).first->second;
        h.addInt(slot).add(arg.dat->type ? arg.dat->type : "").addInt(arg.dat->dim).addInt(arg.dat->type_size).addInt(arg.dat->e_dat);
        h.addInt(arg.stencil->points).addInt(arg.stencil->type);
        for (int p = 0; p < arg.stencil->points * arg.stencil->dims; ++p) h.addInt(arg.stencil->stencil[p]);
        for (int dd = 0; dd < arg.stencil->dims; ++dd) h.addInt(arg.stencil->stride[dd]);
        if (options().specialize == "shapes")
          for (int dd = 0; dd < d->block->dims; ++dd) h.addInt(arg.dat->size[dd]);
      } else if (arg.argtype == OPS_ARG_GBL) {
        // ops_arg_reduce does not set elem_size (it holds whatever was in memory): a reduction is described by the type of its handle.
        if (arg.acc == OPS_READ) {
          h.addInt(arg.dim).addInt(arg.elem_size);
        } else {
          ops_reduction handle = reinterpret_cast<ops_reduction>(arg.data);
          h.addInt(arg.dim).add(handle && handle->type ? handle->type : "");
        }
      }
    }
    for (int v : ranges[k]) h.addInt(v);
    // The values that are folded into the code: the bytes the kernel reads, not the whole snapshot of the constant (the padding
    // of a struct holds whatever was in memory, which would make the key differ from one process to the next).
    for (const ConstInfo &c : artifacts[k]->manifest.consts) {
      if (!c.bound) continue;
      const ops_const_entry *e = ops_find_const(d->block->instance, c.name.c_str());
      if (!e) continue;
      const size_t total = static_cast<size_t>(e->dim) * e->size, width = elemBytes(c.elem);
      if (!width) {
        if (boundDone.insert(c.name).second) h.add(llvm::StringRef(e->snapshot, total));
        continue;
      }
      if (!boundDone.insert(c.name + "@" + std::to_string(c.offset) + "/" + c.elem).second) continue;
      h.add(c.name).addInt(c.offset);
      const size_t stride = c.dynamicIndex && c.arrayStride > 0 ? static_cast<size_t>(c.arrayStride) : total + 1;
      for (size_t off = static_cast<size_t>(c.offset); off + width <= total; off += stride) h.add(llvm::StringRef(e->snapshot + off, width));
    }
  }
  return h.hex();
}

bool Executor::runJit(OPS_instance *instance, ops_kernel_descriptor **descs, int n, const std::vector<const KernelArtifact *> &artifactsIn) {
  const Options &opts = options();

  // Local ranges. A loop with nothing to do on this process is skipped altogether, as OPS does.
  std::vector<ops_kernel_descriptor *> run;
  std::vector<std::vector<int>> ranges;
  std::vector<const KernelArtifact *> artifacts;
  for (int k = 0; k < n; ++k) {
    int local[2 * OPS_MAX_DIM] = {0};
    if (!ops_lazy_compute_local_range(descs[k], local)) continue;
    run.push_back(descs[k]);
    ranges.emplace_back(local, local + 2 * descs[k]->block->dims);
    artifacts.push_back(artifactsIn[k]);
  }
  const int m = static_cast<int>(run.size());
  if (m == 0) return true;

  const bool soa = ops_lazy_soa(instance) != 0;
  std::string key = structuralKey(run.data(), m, ranges, artifacts, soa);
  std::shared_ptr<Plan> plan;
  auto it = plans_.find(key);
  if (it != plans_.end()) {
    plan = it->second;
    stats_.planHits++;
  } else {
    ScopedTime timer(stats_.tPlan);
    plan = buildPlan(run.data(), m, ranges, artifacts, key);
    plans_[key] = plan;
  }
  if (!plan->declined.empty()) return false;

  std::vector<ops_dat> slots = datSlots(run.data(), m);
  for (Plan::Group &g : plan->groups) {
    if (!g.code) {
      // Not compiled: the loops of the group run with OPS's code, in order. This is the fail-safe: a loop whose kernel the frontend
      // could not capture, a group the pipeline could not lower and a run the planner declined all end up here, and the results are
      // those of OPS (see OPS_JIT_EXPLAIN for the reason).
      for (int k : g.recipe.members) ops_lazy_execute_default(instance, &run[k], 1);
      continue;
    }
    runGroup(instance, *plan, g, run.data(), ranges, slots, soa);
  }
  return true;
}

/// Device groups return their reductions through device buffers (8 bytes each per warp), initialised with the neutral element:
/// `buffers` are the device addresses, `staging` the host copy. Returns the number of partial values per reduction.
int64_t Executor::stageDeviceReductions(const Plan::Group &g, const std::vector<std::vector<int>> &ranges, std::vector<void *> &buffers,
                                        std::vector<char> &staging) {
  const size_t count = g.recipe.reductionOf.size();
  int64_t extent[3] = {1, 1, 1};
  for (int d = 0; d < g.recipe.rank; ++d) {
    const ChainArg &lo = g.recipe.range[2 * d], &hi = g.recipe.range[2 * d + 1];
    extent[d] = ranges[hi.loop][2 * hi.dim + 1] - ranges[lo.loop][2 * lo.dim];
  }
  const int64_t units = gpuReductionUnits(gpuBlockFor(options().gpuBlock, g.recipe.rank), extent, g.recipe.rank);
  staging.assign(std::max<size_t>(count, 1) * units * 8, 0);
  char *base = static_cast<char *>(deviceScratch(staging.size()));
  for (size_t r = 0; r < count; ++r) buffers.push_back(base + r * units * 8);
  return units;
}

/// Runs one compiled group on the current data of its loops: prepare, bind, call, fold the reductions into the OPS handles, finish
/// the loops (and, in verify mode, compare with the stock code).
void Executor::runGroup(OPS_instance *instance, const Plan &plan, const Plan::Group &g, ops_kernel_descriptor **run,
                        const std::vector<std::vector<int>> &ranges, const std::vector<ops_dat> &slots, bool soa) {
  const Options &opts = options();
  Snapshot snap = prepareGroup(g, run, opts.verify);

  std::vector<void *> reductionBuffers;
  std::vector<char> staging;
  const int64_t units = g.device ? stageDeviceReductions(g, ranges, reductionBuffers, staging) : 0;

  Invocation inv;
  std::string error;
  auto tb = Clock::now();
  if (!inv.bind(g.recipe, run, ranges, slots, soa, error, g.device ? &reductionBuffers : nullptr, units)) {
    std::fprintf(stderr, "ops_jit: binding failed: %s\n", error.c_str());
    std::abort(); // loops of the run may already have executed: no way back
  }
  stats_.tBind += since(tb);
  if (opts.trace) {
    std::fprintf(stderr, "ops_jit: running %s:", g.code->entry.c_str());
    for (int k : g.recipe.members) std::fprintf(stderr, " %s", run[k]->name);
    std::fprintf(stderr, "\n");
  }
  std::unique_ptr<CallRecorder> recorder;
  if (opts.dumpData && !opts.dumpDir.empty() && dumpedGroups_.insert(g.code->entry).second) {
    std::vector<std::string> elems;
    for (int slot : g.recipe.datSlots) elems.push_back(elemName(slots[slot]->type ? slots[slot]->type : ""));
    recorder = std::make_unique<CallRecorder>(opts.dumpDir, g.code->entry, g.code->pipeline, g.code->pipelineOptions, inv.args(), elems);
  }
  auto t0 = Clock::now();
  inv.call(g.code->fn);
  if (g.device && !reductionBuffers.empty()) {
    // The copy waits for the kernel (default stream) and for the host to have the values.
    mgpuMemcpy(staging.data(), reductionBuffers.front(), staging.size(), nullptr);
  }
  double seconds = since(t0);
  if (opts.trace) std::fprintf(stderr, "ops_jit:   %.6f s\n", seconds);
  stats_.tExecute += seconds;
  stats_.groupsExecuted++;
  if (recorder) {
    std::string rerror;
    if (!recorder->finish(rerror)) std::fprintf(stderr, "ops_jit: cannot record %s: %s\n", g.code->entry.c_str(), rerror.c_str());
  }

  combineReductions(plan, g, run, inv, staging, units);

  const double share = seconds / std::max<size_t>(1, g.recipe.members.size());
  for (int k : g.recipe.members) {
    ops_lazy_finish(run[k]);
    if (instance->OPS_diags > 1) {
      std::vector<int> local = ranges[k];
      local.resize(2 * OPS_MAX_DIM, 0);
      ops_lazy_account(run[k], share, ops_lazy_transfer_bytes(run[k], local.data()));
    }
  }

  if (opts.verify) {
    for (ops_dat d : snap.dats) snap.jit.push_back(readDat(d, g.device));
    for (auto &s : snap.slots) snap.slotJit.push_back(copyBytes(s.first, s.second));
    stats_.verifyGroups++;
    if (verifyGroup(instance, snap, g, run)) stats_.verifyMismatches++;
  }
}


std::shared_ptr<Plan> Executor::buildPlan(ops_kernel_descriptor **run, int n, const std::vector<std::vector<int>> &ranges,
                                          const std::vector<const KernelArtifact *> &artifacts, const std::string &planKey) {
  const Options &opts = options();
  stats_.planBuilds++;
  auto plan = std::make_shared<Plan>();
  auto fail = [&](const std::string &why) {
    plan->declined = why;
    stats_.planFailures++;
    stats_.declineReasons["plan: " + why]++;
    if (opts.explain) std::fprintf(stderr, "ops_jit: a run of %d loops starting with %s runs with OPS: %s\n", n, run[0]->name, why.c_str());
    return plan;
  };

  CaptureDiagnostics diagnostics(ctx_.get());
  Lifted lifted;
  std::string error;
  {
    ScopedTime timer(stats_.tLift);
    if (!liftRun(*ctx_, run, n, ranges, artifacts, opts.printfPolicy == "keep", lifted, error)) return fail("lifting: " + error);
  }
  // The module after the plan passes is cached on disk by plan key: the key is a digest of everything the passes depend on
  // (the kernels, the structure of the run, the bound constants, the options, the code format), so a later process
  // skips the frontend and plan pipelines (0.4 s of a CloverLeaf 3D run).
  const std::string diskKey = "plan-" + planKey;
  bool fromDisk = false;
  if (opts.cache) {
    if (!planCache_) {
      opdsl::CacheOptions co;
      co.directory = opts.cacheDir;
      planCache_ = std::make_unique<opdsl::PersistentCache>(co);
    }
    if (auto hit = planCache_->lookup(diskKey)) {
      ScopedTime timer(stats_.tPlanPasses);
      OwningOpRef<ModuleOp> cached = parseSourceString<ModuleOp>(hit->payload, ctx_.get());
      ChainOp chain;
      if (cached)
        for (ChainOp c : cached->getOps<ChainOp>())
          if (c.getSymName() == lifted.chain.getSymName()) chain = c;
      if (chain) {
        lifted.module = std::move(cached);
        lifted.chain = chain;
        if (!recordBindingConsts(lifted, error)) return fail(error);
        fromDisk = true;
        stats_.planDiskHits++;
      }
    }
  }
  if (!fromDisk) {
    {
      ScopedTime timer(stats_.tFrontend);
      PassManager pm(ctx_.get());
      if (llvm::Error e = registry_.build("ops-jit-frontend", "", pm)) return fail(errorText(std::move(e)));
      if (failed(pm.run(*lifted.module))) return fail("ops-jit-frontend failed: " + diagnostics.text);
      if (!recordBindingConsts(lifted, error)) return fail(error);
    }
    {
      ScopedTime timer(stats_.tPlanPasses);
      PassManager pm(ctx_.get());
      const std::string planOptions = planPipelineOptions(pipelineSettingsFor(opts, run[0]));
      if (llvm::Error e = registry_.build("ops-jit-plan", planOptions, pm)) return fail(errorText(std::move(e)));
      if (failed(pm.run(*lifted.module))) return fail("ops-jit-plan failed: " + diagnostics.text);
    }
    if (planCache_) storePlanModule(*planCache_, diskKey, planKey, *lifted.module, opts.explain);
  }

  dumpModule("plan_" + planKey.substr(0, 16) + ".mlir", *lifted.module);
  std::vector<GroupModule> groups = extractGroupModules(*lifted.module);
  std::vector<GroupRecipe> recipes;
  if (!buildRecipes(lifted, *lifted.module, recipes, error)) return fail(error);
  if (groups.size() != recipes.size()) return fail("internal: groups and recipes disagree");
  compileGroups(*plan, groups, recipes, run);
  plan->reductions = lifted.reductions;

  if (opts.plan) {
    std::fprintf(stderr, "ops_jit: plan for %d loops: %zu groups\n", n, plan->groups.size());
    for (size_t g = 0; g < plan->groups.size(); ++g) {
      std::fprintf(stderr, "  group %zu [%s]:", g, plan->groups[g].code ? plan->groups[g].code->entry.c_str() : "OPS");
      for (int k : plan->groups[g].recipe.members) std::fprintf(stderr, " %s", run[k]->name);
      std::fprintf(stderr, "\n");
    }
  }
  return plan;
}

/// Compiles each group of the planned module (memory cache, disk cache, or the pipeline); a group that cannot be compiled runs with OPS.
void Executor::compileGroups(Plan &plan, std::vector<GroupModule> &groups, std::vector<GroupRecipe> &recipes, ops_kernel_descriptor **run) {
  const Options &opts = options();
  const bool device = run[0]->isdevice != 0;
  const GroupPipeline pipeline = groupPipeline(device, run[0]->block->dims, pipelineSettingsFor(opts, run[0]));
  for (size_t g = 0; g < groups.size(); ++g) {
    OwningOpRef<ModuleOp> canonical = canonicalGroup(groups[g].module.get(), opts.specialize != "none");
    std::string cerror;
    std::shared_ptr<GroupCode> code = compiler_->compile(canonical.get(), pipeline.name, pipeline.options, cerror);
    Plan::Group pg;
    pg.recipe = std::move(recipes[g]);
    pg.device = device;
    if (!code) {
      // Only this group runs with OPS (e.g. a GPU kernel with more parameters than the driver allows).
      stats_.groupsDeclined++;
      std::string firstLine = cerror.substr(0, cerror.find('\n'));
      stats_.declineReasons["group: " + firstLine]++;
      if (opts.explain) {
        std::fprintf(stderr, "ops_jit: a group of %zu loops starting with %s runs with OPS: compiling failed: %s\n", pg.recipe.members.size(),
                     run[pg.recipe.members.front()]->name, cerror.c_str());
      }
    }
    pg.code = std::move(code);
    plan.groups.push_back(std::move(pg));
  }
}


void Executor::printStats() const {
  const Stats &s = stats_;
  const CompileStats &c = compiler_->stats();
  std::fprintf(stderr, "ops_jit counters: flushes=%llu loops=%llu claimed=%llu declined=%llu groups=%llu plans_built=%llu plans_reused=%llu plans_failed=%llu groups_declined=%llu compiled=%llu memory_hits=%llu disk_hits=%llu verify_groups=%llu verify_mismatches=%llu\n",
               (unsigned long long)s.flushes, (unsigned long long)s.loopsSeen, (unsigned long long)s.loopsClaimed,
               (unsigned long long)s.loopsDeclined, (unsigned long long)s.groupsExecuted, (unsigned long long)s.planBuilds,
               (unsigned long long)s.planHits, (unsigned long long)s.planFailures, (unsigned long long)s.groupsDeclined, (unsigned long long)c.compiles,
               (unsigned long long)c.memoryHits, (unsigned long long)c.diskHits, (unsigned long long)s.verifyGroups,
               (unsigned long long)s.verifyMismatches);
  std::fprintf(stderr, "ops_jit statistics:\n");
  std::fprintf(stderr, "  flushes %llu, loops %llu: claimed %llu, declined %llu; groups executed %llu\n", (unsigned long long)s.flushes,
               (unsigned long long)s.loopsSeen, (unsigned long long)s.loopsClaimed, (unsigned long long)s.loopsDeclined,
               (unsigned long long)s.groupsExecuted);
  std::fprintf(stderr, "  plans: %llu built (%llu of them from disk), %llu reused, %llu failed; code: %llu compiled, %llu from memory, %llu from disk\n",
               (unsigned long long)s.planBuilds, (unsigned long long)s.planDiskHits, (unsigned long long)s.planHits, (unsigned long long)s.planFailures,
               (unsigned long long)c.compiles, (unsigned long long)c.memoryHits, (unsigned long long)c.diskHits);
  std::fprintf(stderr, "  time [s]: claim %.3f, lift %.3f, frontend %.3f, plan passes %.3f, lowering %.3f, llvm %.3f, cache io %.3f, bind %.3f, execute %.3f\n",
               s.tClaim, s.tLift, s.tFrontend, s.tPlanPasses, c.tLowering, c.tLlvm, c.tCacheIO, s.tBind, s.tExecute);
  if (options().verify)
    std::fprintf(stderr, "  verify: %llu groups checked against OPS, %llu with differences\n", (unsigned long long)s.verifyGroups,
                 (unsigned long long)s.verifyMismatches);
  for (const auto &entry : s.declineReasons)
    std::fprintf(stderr, "  declined %llux: %s\n", (unsigned long long)entry.second, entry.first.c_str());
}

} // namespace ops_mlir::runtime
