//===- CSubsetWalker.h - C subset of a kernel body -> arith/math/scf --*- C++ -*-===//
//
// The generic half of the kernel importer: walks the clang AST of a kernel body and emits arith, math and scf for
// the part of C that kernels use:
//
//   * scalar locals (SSA values, threaded through scf.if and scf.for), local arrays indexed by constants;
//   * arithmetic, comparisons, casts, the math library, `c ? a : b`;
//   * `if` / `else`, and `for (int v = a; v < b; v++)` with any bounds (scf.for, not unrolled).
//
// Everything specific to a DSL (what an accessor call is, what `*r += x` means, how a global is read) is left to
// the hooks of a subclass, so that op2-mlir can supply its own. Unsupported constructs fail the walk with a
// message that names the source line; the caller then declines to compile the kernel.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_FRONTEND_CSUBSETWALKER_H
#define OPS_MLIR_FRONTEND_CSUBSETWALKER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <utility>

namespace ops_mlir::frontend {

class CSubsetWalker {
public:
  CSubsetWalker(mlir::MLIRContext &ctx, clang::ASTContext &astCtx);
  virtual ~CSubsetWalker() = default;

  /// Emits `body` at the insertion point of `builder()`.
  void emitBody(const clang::Stmt *body);

  bool ok() const { return failureMessage.empty(); }
  const std::string &error() const { return failureMessage; }
  /// Warnings the walk produced (dropped printf...), one per line.
  const std::vector<std::string> &warnings() const { return warningList; }

  mlir::OpBuilder &builder() { return b; }

protected:
  mlir::OpBuilder b;
  mlir::Location loc;
  clang::ASTContext &astCtx;

  //--- reporting ------------------------------------------------------------------------------------------------
  /// Records the first failure (with the line of `at`) and returns a null value.
  mlir::Value fail(const llvm::Twine &message, const clang::Stmt *at = nullptr);
  void warn(const llvm::Twine &message, const clang::Stmt *at = nullptr);
  std::string where(const clang::Stmt *at) const;

  //--- types and constants -------------------------------------------------------------------------------------
  /// The MLIR type of a C scalar type, or null. `half` is recognised and rejected by the subclass's check.
  mlir::Type scalarType(clang::QualType t);
  mlir::Value constFloat(mlir::Type t, double v);
  mlir::Value constInt(mlir::Type t, int64_t v);
  mlir::Value zeroOf(mlir::Type t);
  /// Converts between scalar types the way C does (int<->float, widening, truncation, to/from bool).
  mlir::Value cast(mlir::Value v, mlir::Type to);
  mlir::Value toBool(mlir::Value v) { return cast(v, b.getI1Type()); }
  /// Compile-time integer value of an expression (literals, enumerators, constant folding).
  std::optional<int64_t> evalInt(const clang::Expr *e);

  //--- the walk -------------------------------------------------------------------------------------------------
  mlir::Value emit(const clang::Expr *e);
  void emitStmt(const clang::Stmt *s);

  //--- hooks: what a DSL adds ----------------------------------------------------------------------------------
  /// An expression the generic walker has no rule for, or one it must not handle (accessor reads, a parameter, a
  /// global). nullopt: not special. A null Value: special, and it failed (already reported).
  virtual std::optional<mlir::Value> emitSpecialExpr(const clang::Expr *e) {
    (void)e;
    return std::nullopt;
  }
  /// An assignment (=, +=, ...) whose target is not a local variable. Return true if it was handled.
  virtual bool emitSpecialAssign(const clang::BinaryOperator *op) {
    (void)op;
    return false;
  }
  /// A statement a DSL gives meaning to as a whole (a reduction spelled as an if). Return true if handled.
  virtual bool emitSpecialStmt(const clang::Stmt *s) {
    (void)s;
    return false;
  }
  /// Calls that are dropped (printf). Return true to drop `call` (the subclass counts and warns).
  virtual bool dropCall(const clang::CallExpr *call) {
    (void)call;
    return false;
  }
  /// True if `d` is a local variable of the kernel body the walker owns (not a parameter, not a global).
  static bool isLocal(const clang::VarDecl *d);

  /// Whether a `for` loop with compile-time bounds must be unrolled (its variable selects accessor components, offsets or
  /// global elements, which have to be compile-time constants). Loops are otherwise emitted as scf.for.
  virtual bool shouldUnroll(const clang::ForStmt *s, const clang::VarDecl *var) {
    (void)s;
    (void)var;
    return false;
  }

  /// Evaluating `e` has no side effects and no integer division: it may be evaluated when C would not.
  bool speculatable(const clang::Expr *e);

private:
  struct Env {
    llvm::DenseMap<const clang::VarDecl *, mlir::Value> locals;
    llvm::DenseMap<std::pair<const clang::VarDecl *, int64_t>, mlir::Value> elements; // local array elements
  } env;
  llvm::DenseMap<const clang::VarDecl *, mlir::Type> elementTypes; // of local arrays
  llvm::DenseMap<const clang::VarDecl *, int64_t> constVars;       // variables of loops being unrolled

  std::string failureMessage;
  std::vector<std::string> warningList;

  /// A local variable (or element of a local array) the statements of a subtree assign.
  using LocalTarget = std::pair<const clang::VarDecl *, int64_t>; // element -1: the variable itself
  struct AssignedLocals;
  bool collectAssigned(const clang::Stmt *s, std::vector<LocalTarget> &out, const Env &visible);
  static void sortTargets(std::vector<LocalTarget> &targets);
  mlir::Value current(const LocalTarget &t);
  void assignLocal(const LocalTarget &t, mlir::Value v);
  mlir::Type typeOf(const LocalTarget &t);

  mlir::Value emitCall(const clang::CallExpr *call);
  /// Inlines a call of a function defined in the translation unit: scalar by-value parameters, a body that ends in
  /// `return value;` and has no other return.
  mlir::Value inlineCall(const clang::CallExpr *call, const clang::FunctionDecl *definition);
  std::vector<const clang::FunctionDecl *> inlineStack;
  mlir::Value emitBinary(const clang::BinaryOperator *op);
  mlir::Value emitConditional(const clang::ConditionalOperator *op);
  mlir::Value emitShortCircuit(const clang::BinaryOperator *op);
  void emitAssignment(const clang::BinaryOperator *op);
  void emitIncDec(const clang::UnaryOperator *op);
  void emitDecl(const clang::DeclStmt *s);
  void emitIf(const clang::IfStmt *s);
  void emitFor(const clang::ForStmt *s);

  /// Positions the builder inside `block`, before its terminator if it has one.
  void enter(mlir::Block *block);
};

} // namespace ops_mlir::frontend

#endif // OPS_MLIR_FRONTEND_CSUBSETWALKER_H
