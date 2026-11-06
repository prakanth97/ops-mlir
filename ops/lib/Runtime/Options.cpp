#include "ops-mlir/Runtime/Options.h"

#include "llvm/ADT/StringRef.h"

#include <cstdio>
#include <cstdlib>

namespace ops_mlir::runtime {

namespace {

bool parseBool(llvm::StringRef v, bool &out) {
  if (v == "1" || v.equals_insensitive("true") || v.equals_insensitive("on") || v.equals_insensitive("yes")) {
    out = true;
    return true;
  }
  if (v == "0" || v.equals_insensitive("false") || v.equals_insensitive("off") || v.equals_insensitive("no")) {
    out = false;
    return true;
  }
  return false;
}

bool parseInt(llvm::StringRef v, int &out, int min) {
  int n;
  if (v.getAsInteger(10, n) || n < min) return false;
  out = n;
  return true;
}

} // namespace

bool Options::set(const std::string &name, const std::string &value) {
  if (name == "OPS_JIT") return parseBool(value, enabled);
  if (name == "OPS_JIT_CLAIM") {
    if (value != "all" && value != "fused" && value != "none") return false;
    claim = value;
    return true;
  }
  if (name == "OPS_JIT_CPU") {
    if (value != "omp" && value != "seq") return false;
    omp = value == "omp";
    return true;
  }
  if (name == "OPS_JIT_FUSION") return parseBool(value, fusion);
  if (name == "OPS_JIT_FUSION_MAX") return parseInt(value, fusionMax, 1);
  if (name == "OPS_JIT_FUSION_REORDER") return parseBool(value, fusionReorder);
  if (name == "OPS_JIT_FUSION_PLACEMENT") {
    if (value != "earliest" && value != "latest") return false;
    fusionPlacement = value;
    return true;
  }
  if (name == "OPS_JIT_FUSION_GUARDED") {
    bool wanted = false;
    if (!parseBool(value, wanted)) return false;
    if (wanted) std::fprintf(stderr, "ops_jit: OPS_JIT_FUSION_GUARDED (fusing loops of different ranges) is not implemented: ignored\n");
    fusionGuarded = false;
    return true;
  }
  if (name == "OPS_JIT_SPECIALIZE") {
    if (value != "none" && value != "ranges" && value != "shapes") return false;
    specialize = value;
    return true;
  }
  if (name == "OPS_JIT_CACHE") return parseBool(value, cache);
  if (name == "OPS_JIT_CACHE_DIR") {
    cacheDir = value;
    return true;
  }
  if (name == "OPS_JIT_VERIFY") return parseBool(value, verify);
  if (name == "OPS_JIT_EXPLAIN") return parseBool(value, explain);
  if (name == "OPS_JIT_PLAN") return parseBool(value, plan);
  if (name == "OPS_JIT_TRACE") return parseBool(value, trace);
  if (name == "OPS_JIT_STATS") return parseBool(value, stats);
  if (name == "OPS_JIT_DUMP_DIR") {
    dumpDir = value;
    return true;
  }
  if (name == "OPS_JIT_DUMP_DATA") return parseBool(value, dumpData);
  if (name == "OPS_JIT_RUN_MAX") return parseInt(value, runMax, 1);
  if (name == "OPS_JIT_SYNC") return parseBool(value, sync);
  if (name == "OPS_JIT_FMA") return parseBool(value, fma);
  if (name == "OPS_JIT_LOOPS") {
    if (value != "scf" && value != "affine") return false;
    loops = value;
    return true;
  }
  if (name == "OPS_JIT_AFFINE_PASSES") {
    affinePasses = value;
    return true;
  }
  if (name == "OPS_JIT_VERIFY_FLOAT_TOL") {
    char *end = nullptr;
    double v = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end || !(v >= 0)) return false;
    verifyFloatTol = v;
    return true;
  }
  if (name == "OPS_JIT_VERIFY_HALF_TOL") {
    char *end = nullptr;
    double v = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end || !(v >= 0)) return false;
    verifyHalfTol = v;
    return true;
  }
  if (name == "OPS_JIT_PRINTF") {
    if (value != "keep" && value != "drop" && value != "decline") return false;
    printfPolicy = value;
    return true;
  }
  if (name == "OPS_JIT_GPU_BLOCK") {
    // three positive integers separated by commas
    int x, y, z;
    char c1, c2;
    if (std::sscanf(value.c_str(), "%d%c%d%c%d", &x, &c1, &y, &c2, &z) != 5 || c1 != ',' || c2 != ',' || x < 1 || y < 1 || z < 1) return false;
    gpuBlock = value;
    return true;
  }
  if (name == "OPS_JIT_GPU_ARCH") {
    gpuArch = value;
    return true;
  }
  if (name == "OPS_JIT_PTXAS_OPTS") {
    ptxasOpts = value;
    return true;
  }
  return false;
}

Options Options::fromEnvironment() {
  Options o;
  static const char *names[] = {"OPS_JIT",
                                "OPS_JIT_CLAIM",
                                "OPS_JIT_CPU",
                                "OPS_JIT_FUSION",
                                "OPS_JIT_FUSION_MAX",
                                "OPS_JIT_FUSION_REORDER",
                                "OPS_JIT_FUSION_PLACEMENT",
                                "OPS_JIT_FUSION_GUARDED",
                                "OPS_JIT_SPECIALIZE",
                                "OPS_JIT_CACHE",
                                "OPS_JIT_CACHE_DIR",
                                "OPS_JIT_VERIFY",
                                "OPS_JIT_EXPLAIN",
                                "OPS_JIT_PLAN",
                                "OPS_JIT_TRACE",
                                "OPS_JIT_STATS",
                                "OPS_JIT_DUMP_DIR",
                                "OPS_JIT_DUMP_DATA",
                                "OPS_JIT_RUN_MAX",
                                "OPS_JIT_SYNC",
                                "OPS_JIT_FMA",
                                "OPS_JIT_LOOPS",
                                "OPS_JIT_AFFINE_PASSES",
                                "OPS_JIT_VERIFY_HALF_TOL",
                                "OPS_JIT_VERIFY_FLOAT_TOL",
                                "OPS_JIT_PRINTF",
                                "OPS_JIT_GPU_BLOCK",
                                "OPS_JIT_GPU_ARCH",
                                "OPS_JIT_PTXAS_OPTS"};
  for (const char *name : names)
    if (const char *value = std::getenv(name))
      if (!o.set(name, value)) std::fprintf(stderr, "ops_jit: ignoring %s=%s (not a valid value)\n", name, value);
  return o;
}

std::string Options::codegenDigest() const {
  std::string s;
  s += omp ? "omp" : "seq";
  s += fusion ? ";fusion=" + std::to_string(fusionMax) : ";fusion=off";
  s += fusionReorder ? ";reorder" : ";noreorder";
  s += ";placement=" + fusionPlacement;
  s += fusionGuarded ? ";guarded" : ";noguarded";
  s += ";specialize=" + specialize;
  s += ";printf=" + printfPolicy;
  s += fma ? ";fma" : ";nofma";
  s += ";gpu=" + gpuBlock + "," + gpuArch + "," + ptxasOpts;
  return s;
}

Options &options() {
  static Options o = Options::fromEnvironment();
  return o;
}

} // namespace ops_mlir::runtime
