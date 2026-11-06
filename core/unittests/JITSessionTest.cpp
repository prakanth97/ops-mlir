#include "opdsl/JIT/JITSession.h"
#include "opdsl/Testing/TestHarness.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include <vector>

using opdsl::JITSession;

namespace {

// sum_to(n) = 0 + 1 + ... + (n - 1); fill(p, n, v) stores v to p[0..n); scale(p, n, k) multiplies in place.
const char *kModule = R"mlir(
module {
  llvm.func @sum_to(%n: i64) -> i64 {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^loop(%zero, %zero : i64, i64)
  ^loop(%i: i64, %acc: i64):
    %c = llvm.icmp "slt" %i, %n : i64
    llvm.cond_br %c, ^body, ^exit(%acc : i64)
  ^body:
    %acc2 = llvm.add %acc, %i : i64
    %i2 = llvm.add %i, %one : i64
    llvm.br ^loop(%i2, %acc2 : i64, i64)
  ^exit(%r: i64):
    llvm.return %r : i64
  }

  llvm.func @fill(%p: !llvm.ptr, %n: i64, %v: f64) {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^loop(%zero : i64)
  ^loop(%i: i64):
    %c = llvm.icmp "slt" %i, %n : i64
    llvm.cond_br %c, ^body, ^exit
  ^body:
    %q = llvm.getelementptr %p[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %v, %q : f64, !llvm.ptr
    %i2 = llvm.add %i, %one : i64
    llvm.br ^loop(%i2 : i64)
  ^exit:
    llvm.return
  }
}
)mlir";

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &ctx, const char *text) {
  ctx.loadDialect<mlir::LLVM::LLVMDialect>();
  return mlir::parseSourceString<mlir::ModuleOp>(text, &ctx);
}

} // namespace

OPDSL_TEST(JITSession, CompilesAndRunsPackedFunctions) {
  auto session = JITSession::create();
  OPDSL_EXPECT(static_cast<bool>(session));
  if (!session) return;
  OPDSL_EXPECT(!(*session)->targetDescription().empty());

  mlir::MLIRContext ctx;
  auto module = parse(ctx, kModule);
  OPDSL_EXPECT(static_cast<bool>(module));

  // A function with a result: args[nargs] receives it.
  auto object = (*session)->compile(*module, "sum_to");
  OPDSL_EXPECT(static_cast<bool>(object));
  if (!object) return;
  OPDSL_EXPECT((*object)->getBufferSize() > 0);
  auto fn = (*session)->load(std::move(*object), "sum_to");
  OPDSL_EXPECT(static_cast<bool>(fn));
  if (!fn) return;
  int64_t n = 101, result = -1;
  void *args[] = {&n, &result};
  (*fn)(args);
  OPDSL_EXPECT_EQ(result, int64_t(5050));

  // Pointers and doubles, a second object in the same session (its own dylib).
  auto fillObject = (*session)->compile(*module, "fill");
  OPDSL_EXPECT(static_cast<bool>(fillObject));
  if (!fillObject) return;
  auto fill = (*session)->load(std::move(*fillObject), "fill");
  OPDSL_EXPECT(static_cast<bool>(fill));
  if (!fill) return;
  std::vector<double> data(8, 0.0);
  double *p = data.data();
  int64_t count = 5;
  double value = 2.5;
  void *fillArgs[] = {&p, &count, &value};
  (*fill)(fillArgs);
  for (int i = 0; i < 5; ++i) OPDSL_EXPECT_EQ(data[i], 2.5);
  OPDSL_EXPECT_EQ(data[5], 0.0);

  // The same object loaded again: independent dylibs do not clash.
  auto again = (*session)->compile(*module, "sum_to");
  OPDSL_EXPECT(static_cast<bool>(again));
  if (!again) return;
  auto fn2 = (*session)->load(std::move(*again), "sum_to");
  OPDSL_EXPECT(static_cast<bool>(fn2));
  if (!fn2) return;
  n = 10;
  (*fn2)(args);
  OPDSL_EXPECT_EQ(result, int64_t(45));
}

OPDSL_TEST(JITSession, ObjectsAreReloadable) {
  // What the disk cache does: keep the bytes of an object, load them into another session.
  auto first = JITSession::create();
  auto second = JITSession::create();
  OPDSL_EXPECT(first && second);
  if (!first || !second) return;
  mlir::MLIRContext ctx;
  auto module = parse(ctx, kModule);
  auto object = (*first)->compile(*module, "sum_to");
  OPDSL_EXPECT(static_cast<bool>(object));
  if (!object) return;
  std::string bytes = (*object)->getBuffer().str();
  OPDSL_EXPECT_EQ((*first)->targetDescription(), (*second)->targetDescription());
  auto fn = (*second)->load(llvm::MemoryBuffer::getMemBufferCopy(bytes, "cached"), "sum_to");
  OPDSL_EXPECT(static_cast<bool>(fn));
  if (!fn) return;
  int64_t n = 11, result = 0;
  void *args[] = {&n, &result};
  (*fn)(args);
  OPDSL_EXPECT_EQ(result, int64_t(55));
}

OPDSL_TEST(JITSession, ErrorsAreReported) {
  auto session = JITSession::create();
  OPDSL_EXPECT(static_cast<bool>(session));
  if (!session) return;
  mlir::MLIRContext ctx;
  auto module = parse(ctx, kModule);
  auto missing = (*session)->compile(*module, "no_such_function");
  OPDSL_EXPECT(!missing);
  if (!missing) {
    std::string message = llvm::toString(missing.takeError());
    OPDSL_EXPECT(message.find("no_such_function") != std::string::npos);
  }
}
