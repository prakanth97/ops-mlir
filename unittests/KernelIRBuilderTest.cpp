//===- KernelIRBuilderTest.cpp - Unit tests for KernelIRBuilder -*- C++ -*-===//
//
// Lightweight standalone checks (no external test framework dependency)
// for KernelIRBuilder: writes a small kernel source to a temp file, runs
// it through KernelIRBuilder::generate, and asserts on success/failure
// and on the printed IR containing the ops expected for that construct.
//
// Run directly (not wired through ctest): build the `KernelIRBuilderTest`
// target and execute it; each case prints PASS/FAIL and the process exits
// non-zero if any case failed.
//
//===----------------------------------------------------------------------===//

#include "runtime/KernelIRBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <string>

namespace {

int failures = 0;

/// Writes `source` to a fresh temp file and runs KernelIRBuilder on it,
/// capturing any diagnostics into `diagnostics`.
mlir::func::FuncOp translate(mlir::MLIRContext &ctx, const std::string &source,
                             const std::string &kernelName, int indexRank,
                             const std::map<std::string, const void *> &constants,
                             const std::vector<int> &constArgDims,
                             std::string &diagnostics) {
  llvm::SmallString<128> path;
  llvm::sys::fs::createTemporaryFile("kernel_ir_builder_test", "h", path);
  {
    std::ofstream out(path.c_str());
    out << source;
  }

  llvm::raw_string_ostream errs(diagnostics);
  ops_mlir::KernelIRBuilder builder(ctx);
  mlir::func::FuncOp fn = builder.generate(path.str().str(), kernelName,
                                           indexRank, constants, constArgDims, errs);
  errs.flush();
  llvm::sys::fs::remove(path);
  return fn;
}

std::string printOp(mlir::Operation *op) {
  std::string text;
  llvm::raw_string_ostream os(text);
  op->print(os);
  os.flush();
  return text;
}

void check(bool condition, const std::string &testName,
          const std::string &detail) {
  if (condition) {
    std::cout << "[PASS] " << testName << "\n";
  } else {
    std::cout << "[FAIL] " << testName << ": " << detail << "\n";
    ++failures;
  }
}

void expectSuccessContaining(mlir::MLIRContext &ctx, const std::string &testName,
                             const std::string &source,
                             const std::string &kernelName, int indexRank,
                             const std::map<std::string, const void *> &constants,
                             const std::vector<std::string> &expectedSubstrings,
                             const std::vector<int> &constArgDims = {}) {
  std::string diagnostics;
  mlir::func::FuncOp fn =
      translate(ctx, source, kernelName, indexRank, constants, constArgDims, diagnostics);
  if (!fn) {
    check(false, testName, "expected success but generate() failed: " + diagnostics);
    return;
  }
  std::string ir = printOp(fn);
  for (const std::string &expected : expectedSubstrings) {
    check(ir.find(expected) != std::string::npos, testName,
         "expected IR to contain '" + expected + "' but got:\n" + ir);
  }
  fn.erase();
}

void expectFailureContaining(mlir::MLIRContext &ctx, const std::string &testName,
                             const std::string &source,
                             const std::string &kernelName, int indexRank,
                             const std::map<std::string, const void *> &constants,
                             const std::string &expectedDiagnostic,
                             const std::vector<int> &constArgDims = {}) {
  std::string diagnostics;
  mlir::func::FuncOp fn =
      translate(ctx, source, kernelName, indexRank, constants, constArgDims, diagnostics);
  if (fn) {
    check(false, testName, "expected failure but generate() succeeded:\n" +
                               printOp(fn));
    fn.erase();
    return;
  }
  check(diagnostics.find(expectedDiagnostic) != std::string::npos, testName,
       "expected diagnostic to contain '" + expectedDiagnostic +
           "' but got: " + diagnostics);
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();

  // A no-arg constant-returning kernel (set_zero).
  expectSuccessContaining(
      ctx, "set_zero: literal return",
      "double set_zero() {\n  return 0.0;\n}\n", "set_zero",
      /*indexRank=*/2, {},
      {"arith.constant", "return"});

  // apply_stencil: 5 double params, compound arithmetic, no idx/math.
  expectSuccessContaining(
      ctx, "apply_stencil: compound arithmetic",
      "double apply_stencil(double a, double b, double c, double d, double e) {\n"
      "  return 0.25f * (b + c + d + e);\n"
      "}\n",
      "apply_stencil", /*indexRank=*/2, {},
      {"arith.addf", "arith.addf", "arith.addf", "arith.mulf"});

  // Index-array indexing (left_bndcon-shaped, no extern globals).
  expectSuccessContaining(
      ctx, "idx array: subscript + arithmetic",
      "double idx_kernel(const int *idx) {\n"
      "  return idx[1] + 1;\n"
      "}\n",
      "idx_kernel", /*indexRank=*/2, {},
      {"memref.load", "arith.addi", "arith.sitofp"});

  // Compound math-call case: multiple <math.h> calls combined.
  expectSuccessContaining(
      ctx, "compound math calls",
      "#include <math.h>\n"
      "double combo(double x) {\n"
      "  return sin(x) * cos(x) + sqrt(x);\n"
      "}\n",
      "combo", /*indexRank=*/1, {},
      {"math.sin", "math.cos", "math.sqrt", "arith.mulf", "arith.addf"});

  // Extern global resolved via the registered-constants map (mirrors
  // pi/jmax in left_bndcon/right_bndcon), not dlsym.
  {
    static double kFactor = 3.5;
    std::map<std::string, const void *> constants = {{"factor", &kFactor}};
    expectSuccessContaining(
        ctx, "extern global: registered constant baked in",
        "extern double factor;\n"
        "double scale_by(double x) {\n"
        "  return x * factor;\n"
        "}\n",
        "scale_by", /*indexRank=*/1, constants, {"arith.constant", "arith.mulf"});
  }

  // Negative: unregistered extern global must fail loudly, not silently
  // miscompile.
  expectFailureContaining(
      ctx, "extern global: unregistered symbol fails",
      "extern double unregistered_const;\n"
      "double bad(double x) {\n"
      "  return x * unregistered_const;\n"
      "}\n",
      "bad", /*indexRank=*/1, {}, "reference to unsupported symbol");

  // Negative: unsupported function call (log1p is a real <math.h>
  // function, just outside our supported subset).
  expectFailureContaining(
      ctx, "unsupported call fails",
      "#include <math.h>\n"
      "double bad_call(double x) {\n"
      "  return atan(x) + log1p(x);\n"
      "}\n",
      "bad_call", /*indexRank=*/1, {}, "call to unsupported function");

  // Negative: control flow / multi-statement bodies are out of scope for
  // the single-`return`, double-returning shape.
  expectFailureContaining(
      ctx, "control flow unsupported",
      "double branchy(double x) {\n"
      "  if (x > 0.0) return x;\n"
      "  return -x;\n"
      "}\n",
      "branchy", /*indexRank=*/1, {},
      "must be a single `return <expr>;` statement");

  // Multi-write, void, out-pointer kernel (opensbliblock00Kernel039-shaped):
  // locals threaded across statements, then two `out->field = ...` stores.
  expectSuccessContaining(
      ctx, "out-pointer: locals + multiple field stores",
      "struct Result { double a; double b; };\n"
      "void two_out(double x, Result *out) {\n"
      "  double y = x * 2.0;\n"
      "  out->a = y + 1.0;\n"
      "  out->b = y - 1.0;\n"
      "}\n",
      "two_out", /*indexRank=*/2, {},
      {"memref<2xf64>", "arith.mulf", "arith.addf", "arith.subf",
       "memref.store", "memref.store"});

  // Out-pointer kernel with idx + extern global, mirroring
  // opensbliblock00Kernel039's actual shape (idx-derived locals feeding
  // multiple field stores, plus a registered extern constant).
  {
    static double kScale = 2.0;
    std::map<std::string, const void *> constants = {{"scale", &kScale}};
    expectSuccessContaining(
        ctx, "out-pointer: idx + extern global + locals",
        "extern double scale;\n"
        "struct Result { double p; double q; };\n"
        "void init(const int *idx, Result *out) {\n"
        "  double x0 = scale * idx[0];\n"
        "  out->p = x0;\n"
        "  out->q = x0 * x0;\n"
        "}\n",
        "init", /*indexRank=*/3, constants,
        {"memref<3xi32>", "memref<2xf64>", "memref.load", "arith.mulf",
         "memref.store"});
  }

  // Negative: a kernel can only have one out-pointer parameter -- with two,
  // there's no way to tell which one a bare `->field` store targets, so
  // this is rejected up front rather than guessing.
  expectFailureContaining(
      ctx, "out-pointer: more than one out-pointer parameter rejected",
      "struct Result { double a; };\n"
      "void bad_base(double x, Result *out, Result *other) {\n"
      "  other->a = x;\n"
      "}\n",
      "bad_base", /*indexRank=*/1, {},
      "has more than one out-pointer parameter");

  // Negative: a field name that doesn't exist on the out-struct. Clang's own
  // semantic check catches this while building the AST (before KernelIRBuilder
  // ever sees a proper MemberExpr for it), so translation still correctly
  // fails, just via the generic "must be `out->field`" diagnostic rather
  // than StmtEmitter's more specific unknown-field one.
  expectFailureContaining(
      ctx, "out-pointer: unknown field rejected",
      "struct Result { double a; };\n"
      "void bad_field(double x, Result *out) {\n"
      "  out->b = x;\n"
      "}\n",
      "bad_field", /*indexRank=*/1, {},
      "assignment target must be `out->field`");

  if (failures == 0) {
    std::cout << "All KernelIRBuilder tests passed.\n";
    return 0;
  }
  std::cout << failures << " KernelIRBuilder test(s) failed.\n";
  return 1;
}
