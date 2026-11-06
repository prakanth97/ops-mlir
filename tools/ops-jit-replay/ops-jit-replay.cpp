//===- ops-jit-replay.cpp - re-run a recorded group of libopsjit offline ---------------------===//
//
//   ops-jit-replay <dump dir> <group entry> [--omp=true|false] [--no-cache] [--runs N]
//
// The dump comes from an application run with OPS_JIT_DUMP_DIR=<dir> OPS_JIT_DUMP_DATA=1. The tool compiles
// <entry>.group.mlir with the recorded pipeline, runs it on the recorded inputs and compares the dats (checksums:
// equal bits) and the reduction results with what the application saw. Exit code 0: they match.
//
//===----------------------------------------------------------------------===//

#include "ops-mlir/Pipelines/GroupModules.h"
#include "ops-mlir/Pipelines/Pipelines.h"
#include "ops-mlir/Runtime/CallDump.h"
#include "ops-mlir/Runtime/GroupCompiler.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cmath>

using namespace llvm;
using namespace ops_mlir;
using namespace ops_mlir::runtime;

static cl::opt<std::string> dumpDir(cl::Positional, cl::desc("<dump dir>"), cl::Required);
static cl::opt<std::string> entryName(cl::Positional, cl::desc("<group entry, e.g. group_0123abcd>"), cl::Required);
static cl::opt<std::string> omp("omp", cl::desc("override the pipeline option omp (true|false)"));
static cl::opt<std::string> ompLib("omp-lib", cl::desc("OpenMP runtime to load"), cl::init(OPSJIT_LIBOMP));
static cl::opt<bool> noCache("no-cache", cl::desc("do not use the disk cache"));
static cl::opt<int> runs("runs", cl::desc("timed repetitions after the checked run"), cl::init(0));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "re-run a group recorded by libopsjit\n");

  LoadedCall call;
  std::string error;
  if (!loadCall(dumpDir, entryName, call, error)) {
    errs() << "ops-jit-replay: " << error << "\n";
    return 2;
  }

  opdsl::PipelineRegistry registry;
  ops::registerOPSPipelines(registry);
  mlir::DialectRegistry dialects;
  ops::registerJITDialects(dialects);
  mlir::MLIRContext ctx(dialects);
  ctx.loadAllAvailableDialects();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(dumpDir + "/" + entryName + ".group.mlir", &ctx);
  if (!module) {
    errs() << "ops-jit-replay: cannot parse " << dumpDir << "/" << entryName << ".group.mlir\n";
    return 2;
  }

  std::string options = call.pipelineOptions;
  if (!omp.empty()) {
    size_t at = options.find("omp=");
    if (at != std::string::npos) options.replace(at, options.find(' ', at) - at, "omp=" + omp);
    else options += " omp=" + std::string(omp);
  }
  CompilerOptions co;
  co.useCache = !noCache;
  if (!ompLib.empty()) co.jit.sharedLibraries.push_back(ompLib);
  GroupCompiler compiler(ctx, registry, co);
  std::shared_ptr<GroupCode> code = compiler.compile(*module, call.pipeline, options, error);
  if (!code) {
    errs() << "ops-jit-replay: compiling failed: " << error << "\n";
    return 1;
  }
  outs() << "compiled " << code->entry << " with " << call.pipeline << " " << options << " ("
         << compiler.stats().compiles << " compiled, " << compiler.stats().diskHits << " from the cache)\n";

  call.args.call(code->fn);

  int bad = 0;
  for (size_t i = 0; i < call.buffers.size(); ++i) {
    std::string sum = checksumHex(call.buffers[i].data(), call.buffers[i].size());
    bool ok = sum == call.expectedChecksums[i];
    outs() << "dat " << i << " (" << call.datElems[i] << ", " << call.buffers[i].size() << " bytes): " << (ok ? "same as the application" : "DIFFERENT") << " (" << sum
           << ")\n";
    if (!ok) ++bad;
  }
  for (size_t i = 0; i < call.expectedResults.size(); ++i) {
    double got = call.args.resultAsDouble(i), want = call.expectedResults[i];
    bool ok = std::fabs(got - want) <= 1e-12 * std::max(std::fabs(want), 1e-300);
    outs() << "result " << i << ": " << format("%.17g", got) << " (application: " << format("%.17g", want) << ") " << (ok ? "ok" : "DIFFERENT") << "\n";
    if (!ok) ++bad;
  }
  if ((int)runs > 0) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < (int)runs; ++r) call.args.call(code->fn);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    outs() << format("%d runs: %.6f s each\n", (int)runs, seconds / (int)runs);
  }
  outs() << (bad ? "MISMATCH\n" : "REPLAY OK\n");
  return bad ? 1 : 0;
}
