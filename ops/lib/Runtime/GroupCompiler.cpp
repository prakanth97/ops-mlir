#include "ops-mlir/Runtime/GroupCompiler.h"

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Pipelines/GroupModules.h"

#include "opdsl/Support/Hash.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdio>

namespace ops_mlir::runtime {

using namespace mlir;
using namespace ops_mlir::ops;
using Clock = std::chrono::steady_clock;

namespace {

struct ScopedTime {
  double &sink;
  Clock::time_point t0 = Clock::now();
  explicit ScopedTime(double &s) : sink(s) {}
  ~ScopedTime() { sink += std::chrono::duration<double>(Clock::now() - t0).count(); }
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

} // namespace

GroupCompiler::GroupCompiler(MLIRContext &ctx, const opdsl::PipelineRegistry &registry, CompilerOptions options)
    : ctx(ctx), registry(registry), options(std::move(options)) {}

opdsl::JITSession &GroupCompiler::session() {
  if (!session_) {
    auto s = opdsl::JITSession::create(options.jit);
    if (!s) {
      std::fprintf(stderr, "ops_jit: cannot create the JIT session: %s\n", errorText(s.takeError()).c_str());
      std::abort();
    }
    session_ = std::move(*s);
  }
  return *session_;
}

std::shared_ptr<GroupCode> GroupCompiler::compile(ModuleOp group, const std::string &pipeline, const std::string &pipelineOptions,
                                                  std::string &error) {
  auto dump = [&](const std::string &name, ModuleOp module) {
    if (options.dumpDir.empty()) return;
    llvm::sys::fs::create_directories(options.dumpDir);
    std::error_code ec;
    llvm::raw_fd_ostream os(options.dumpDir + "/" + name, ec);
    if (!ec) module->print(os);
  };

  std::string key = opdsl::Hasher()
                        .add(moduleBytecode(group))
                        .add(pipeline)
                        .add(pipelineOptions)
                        .add(session().targetDescription())
                        .addInt(kCodeFormat)
                        .add(LLVM_VERSION_STRING)
                        .hex();
  auto known = code_.find(key);
  if (known != code_.end()) {
    stats_.memoryHits++;
    return known->second;
  }
  auto code = std::make_shared<GroupCode>();
  code->key = key;
  code->entry = "group_" + key.substr(0, 16);
  code->pipeline = pipeline;
  code->pipelineOptions = pipelineOptions;

  if (options.useCache && !cache_) {
    opdsl::CacheOptions co;
    co.directory = options.cacheDir;
    cache_ = std::make_unique<opdsl::PersistentCache>(co);
  }
  if (options.useCache) {
    ScopedTime timer(stats_.tCacheIO);
    if (auto hit = cache_->lookup(key)) {
      auto fn = session().load(llvm::MemoryBuffer::getMemBufferCopy(hit->payload, code->entry), code->entry);
      if (fn) {
        code->fn = *fn;
        stats_.diskHits++;
        code_[key] = code;
        return code;
      }
      llvm::consumeError(fn.takeError());
    }
  }

  // Lowering and compiling.
  stats_.compiles++;
  dump(code->entry + ".group.mlir", group);
  OwningOpRef<ModuleOp> lowered(cast<ModuleOp>(group->clone()));
  {
    ScopedTime timer(stats_.tLowering);
    for (ChainOp chain : lowered->getOps<ChainOp>()) chain->setAttr("ops.func_name", StringAttr::get(&ctx, code->entry));
    CaptureDiagnostics diagnostics(&ctx);
    PassManager pm(&ctx);
    if (llvm::Error e = registry.build(pipeline, pipelineOptions, pm)) {
      error = errorText(std::move(e));
      return nullptr;
    }
    if (failed(pm.run(*lowered))) {
      error = pipeline + " failed: " + diagnostics.text;
      return nullptr;
    }
  }
  dump(code->entry + ".lowered.mlir", *lowered);
  std::unique_ptr<llvm::MemoryBuffer> object;
  {
    ScopedTime timer(stats_.tLlvm);
    auto o = session().compile(*lowered, code->entry);
    if (!o) {
      error = "LLVM: " + errorText(o.takeError());
      return nullptr;
    }
    object = std::move(*o);
  }
  if (options.useCache) {
    ScopedTime timer(stats_.tCacheIO);
    llvm::json::Object header;
    header["entry"] = code->entry;
    header["pipeline"] = pipeline + " " + pipelineOptions;
    std::string headerText;
    llvm::raw_string_ostream hs(headerText);
    hs << llvm::json::Value(std::move(header));
    if (llvm::Error e = cache_->store(key, headerText, object->getBuffer()))
      std::fprintf(stderr, "ops_jit: cannot write the cache: %s\n", errorText(std::move(e)).c_str());
  }
  {
    ScopedTime timer(stats_.tLlvm);
    auto fn = session().load(std::move(object), code->entry);
    if (!fn) {
      error = "LLVM load: " + errorText(fn.takeError());
      return nullptr;
    }
    code->fn = *fn;
  }
  code_[key] = code;
  return code;
}

} // namespace ops_mlir::runtime
