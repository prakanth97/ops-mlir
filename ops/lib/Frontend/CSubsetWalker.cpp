#include "ops-mlir/Frontend/CSubsetWalker.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <set>
#include <tuple>

namespace ops_mlir::frontend {

using namespace mlir;

namespace {

/// The math library functions kernels call, by name without a `f`/`l` suffix, and their arity.
int mathArity(llvm::StringRef name) {
  return llvm::StringSwitch<int>(name)
      .Cases({"sqrt", "fabs", "exp", "exp2", "log", "log2", "log10", "sin", "cos", "tan", "asin", "acos", "atan",
              "sinh", "cosh", "tanh", "floor", "ceil", "round", "trunc", "cbrt", "erf", "abs", "labs"}, 1)
      .Cases({"pow", "atan2", "fmin", "fmax", "fmod", "copysign"}, 2)
      .Default(0);
}

/// `sqrtf` -> `sqrt`, `__builtin_sqrt` -> `sqrt`; the name unchanged if it is not a math function.
std::string normalizedMathName(llvm::StringRef name) {
  name.consume_front("__builtin_");
  if (mathArity(name)) return name.str();
  if (name.ends_with("f") || name.ends_with("l")) {
    llvm::StringRef base = name.drop_back();
    if (mathArity(base)) return base.str();
  }
  return "";
}

} // namespace

CSubsetWalker::CSubsetWalker(MLIRContext &ctx, clang::ASTContext &astCtx)
    : b(&ctx), loc(b.getUnknownLoc()), astCtx(astCtx) {}

//===----------------------------------------------------------------------===//
// Reporting
//===----------------------------------------------------------------------===//

std::string CSubsetWalker::where(const clang::Stmt *at) const {
  if (!at) return "";
  clang::PresumedLoc p = astCtx.getSourceManager().getPresumedLoc(at->getBeginLoc());
  if (p.isInvalid()) return "";
  return (llvm::sys::path::filename(p.getFilename()) + ":" + std::to_string(p.getLine()) + ": ").str();
}

Value CSubsetWalker::fail(const llvm::Twine &message, const clang::Stmt *at) {
  if (ok()) failureMessage = where(at) + message.str();
  return {};
}

void CSubsetWalker::warn(const llvm::Twine &message, const clang::Stmt *at) {
  warningList.push_back(where(at) + message.str());
}

//===----------------------------------------------------------------------===//
// Types and constants
//===----------------------------------------------------------------------===//

Type CSubsetWalker::scalarType(clang::QualType t) {
  t = t.getNonReferenceType().getUnqualifiedType().getCanonicalType();
  if (t->isSpecificBuiltinType(clang::BuiltinType::BFloat16)) return {};
  if (t->isHalfType() || t->isSpecificBuiltinType(clang::BuiltinType::Float16)) return b.getF16Type();
  if (t->isSpecificBuiltinType(clang::BuiltinType::Double)) return b.getF64Type();
  if (t->isSpecificBuiltinType(clang::BuiltinType::Float)) return b.getF32Type();
  if (t->isBooleanType()) return b.getI1Type();
  if (t->isIntegerType() || t->isEnumeralType()) return b.getIntegerType(astCtx.getTypeSize(t));
  return {};
}

Value CSubsetWalker::constFloat(Type t, double v) {
  return arith::ConstantOp::create(b, loc, b.getFloatAttr(t, v));
}

Value CSubsetWalker::constInt(Type t, int64_t v) {
  return arith::ConstantOp::create(b, loc, b.getIntegerAttr(t, v));
}

Value CSubsetWalker::zeroOf(Type t) { return isa<FloatType>(t) ? constFloat(t, 0.0) : constInt(t, 0); }

Value CSubsetWalker::cast(Value v, Type to) {
  if (!v || v.getType() == to) return v;
  Type from = v.getType();
  bool fromF = isa<FloatType>(from), toF = isa<FloatType>(to);
  if (fromF && toF)
    return from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth() ? Value(arith::ExtFOp::create(b, loc, to, v))
                                                                    : Value(arith::TruncFOp::create(b, loc, to, v));
  if (!fromF && toF) return from.isInteger(1) ? Value(arith::UIToFPOp::create(b, loc, to, v)) : Value(arith::SIToFPOp::create(b, loc, to, v));
  if (fromF && !toF) {
    if (to.isInteger(1)) return arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UNE, v, constFloat(from, 0.0));
    return arith::FPToSIOp::create(b, loc, to, v);
  }
  if (to.isInteger(1)) return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, v, constInt(from, 0));
  if (from.isInteger(1)) return arith::ExtUIOp::create(b, loc, to, v);
  if (isa<IndexType>(from) || isa<IndexType>(to)) return arith::IndexCastOp::create(b, loc, to, v);
  return from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth() ? Value(arith::ExtSIOp::create(b, loc, to, v))
                                                                  : Value(arith::TruncIOp::create(b, loc, to, v));
}

std::optional<int64_t> CSubsetWalker::evalInt(const clang::Expr *e) {
  e = e->IgnoreParenImpCasts();
  if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(e))
    if (const auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl())) {
      auto cv = constVars.find(vd);
      if (cv != constVars.end()) return cv->second;
      if (env.locals.count(vd) && !vd->getType().isConstQualified()) return std::nullopt; // a mutable local
    }
  // Expressions of constants (loop variables of unrolled loops included) the compiler cannot fold alone.
  if (const auto *un = llvm::dyn_cast<clang::UnaryOperator>(e); un && (un->getOpcode() == clang::UO_Minus || un->getOpcode() == clang::UO_Plus)) {
    if (auto v = evalInt(un->getSubExpr())) return un->getOpcode() == clang::UO_Minus ? -*v : *v;
  }
  if (const auto *bin = llvm::dyn_cast<clang::BinaryOperator>(e)) {
    auto l = evalInt(bin->getLHS()), r = evalInt(bin->getRHS());
    if (l && r) {
      switch (bin->getOpcode()) {
      case clang::BO_Add: return *l + *r;
      case clang::BO_Sub: return *l - *r;
      case clang::BO_Mul: return *l * *r;
      case clang::BO_Div: if (*r != 0) return *l / *r; break;
      case clang::BO_Rem: if (*r != 0) return *l % *r; break;
      default: break;
      }
    }
  }
  clang::Expr::EvalResult res;
  if (!e->isValueDependent() && e->EvaluateAsInt(res, astCtx)) return res.Val.getInt().getSExtValue();
  return std::nullopt;
}

bool CSubsetWalker::isLocal(const clang::VarDecl *d) {
  return d && !llvm::isa<clang::ParmVarDecl>(d) && !d->hasGlobalStorage();
}

//===----------------------------------------------------------------------===//
// Speculation
//===----------------------------------------------------------------------===//

bool CSubsetWalker::speculatable(const clang::Expr *e) {
  struct Visitor : clang::RecursiveASTVisitor<Visitor> {
    bool safe = true;
    bool VisitBinaryOperator(clang::BinaryOperator *op) {
      switch (op->getOpcode()) {
      case clang::BO_Div:
      case clang::BO_Rem:
        if (op->getType()->isIntegerType()) safe = false;
        break;
      default:
        if (op->isAssignmentOp()) safe = false;
      }
      return safe;
    }
    bool VisitUnaryOperator(clang::UnaryOperator *op) {
      if (op->isIncrementDecrementOp()) safe = false;
      return safe;
    }
    bool VisitCallExpr(clang::CallExpr *call) {
      if (llvm::isa<clang::CXXOperatorCallExpr>(call)) return true; // accessor reads
      const clang::FunctionDecl *f = call->getDirectCallee();
      if (!f || !f->getDeclName().isIdentifier() || normalizedMathName(f->getName()).empty()) safe = false;
      return safe;
    }
  } v;
  v.TraverseStmt(const_cast<clang::Expr *>(e));
  return v.safe;
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

Value CSubsetWalker::emitCall(const clang::CallExpr *call) {
  if (dropCall(call)) return constInt(b.getI32Type(), 0);
  const clang::FunctionDecl *callee = call->getDirectCallee();
  if (!callee || !callee->getDeclName().isIdentifier()) return fail("call to an unknown function", call);
  std::string name = normalizedMathName(callee->getName());
  if (name.empty()) {
    if (const clang::FunctionDecl *definition = callee->getDefinition()) return inlineCall(call, definition);
    return fail("call to unsupported function '" + callee->getNameAsString() + "'", call);
  }

  SmallVector<Value, 2> a;
  for (const clang::Expr *arg : call->arguments()) {
    Value v = emit(arg);
    if (!v) return {};
    a.push_back(v);
  }
  Type resultType = scalarType(call->getType());
  if (!resultType) return fail("call result of unsupported type", call);
  // `abs` of an integer; every other function works on floating point.
  if (name == "abs" || name == "labs") {
    Value v = cast(a[0], resultType);
    return math::AbsIOp::create(b, loc, v);
  }
  if (!isa<FloatType>(resultType)) return fail("math function with a non-floating-point result", call);
  for (Value &v : a) v = cast(v, resultType);

  auto un = [&](auto tag) -> Value {
    using Op = decltype(tag);
    return Op::create(b, loc, a[0]);
  };
  if (a.size() == 1) {
    if (name == "sqrt") return un(math::SqrtOp());
    if (name == "fabs") return un(math::AbsFOp());
    if (name == "exp") return un(math::ExpOp());
    if (name == "exp2") return un(math::Exp2Op());
    if (name == "log") return un(math::LogOp());
    if (name == "log2") return un(math::Log2Op());
    if (name == "log10") return un(math::Log10Op());
    if (name == "sin") return un(math::SinOp());
    if (name == "cos") return un(math::CosOp());
    if (name == "tan") return un(math::TanOp());
    if (name == "asin") return un(math::AsinOp());
    if (name == "acos") return un(math::AcosOp());
    if (name == "atan") return un(math::AtanOp());
    if (name == "sinh") return un(math::SinhOp());
    if (name == "cosh") return un(math::CoshOp());
    if (name == "tanh") return un(math::TanhOp());
    if (name == "floor") return un(math::FloorOp());
    if (name == "ceil") return un(math::CeilOp());
    if (name == "round") return un(math::RoundOp());
    if (name == "trunc") return un(math::TruncOp());
    if (name == "cbrt") return un(math::CbrtOp());
    if (name == "erf") return un(math::ErfOp());
  } else if (a.size() == 2) {
    if (name == "pow") return math::PowFOp::create(b, loc, a[0], a[1]);
    if (name == "atan2") return math::Atan2Op::create(b, loc, a[0], a[1]);
    if (name == "fmin") return arith::MinNumFOp::create(b, loc, a[0], a[1]);
    if (name == "fmax") return arith::MaxNumFOp::create(b, loc, a[0], a[1]);
    if (name == "fmod") return arith::RemFOp::create(b, loc, a[0], a[1]);
    if (name == "copysign") return math::CopySignOp::create(b, loc, a[0], a[1]);
  }
  return fail("call to unsupported function '" + callee->getNameAsString() + "'", call);
}

namespace {
/// Counts the return statements of a function body.
struct ReturnCounter : clang::RecursiveASTVisitor<ReturnCounter> {
  int count = 0;
  bool VisitReturnStmt(clang::ReturnStmt *) {
    ++count;
    return true;
  }
};
} // namespace

Value CSubsetWalker::inlineCall(const clang::CallExpr *call, const clang::FunctionDecl *def) {
  const std::string name = def->getNameAsString();
  if (inlineStack.size() >= 8 || llvm::is_contained(inlineStack, def)) return fail("recursive or deeply nested call of '" + name + "'", call);
  const auto *body = llvm::dyn_cast_or_null<clang::CompoundStmt>(def->getBody());
  if (!body || body->body_empty() || !llvm::isa<clang::ReturnStmt>(body->body_back()) ||
      !llvm::cast<clang::ReturnStmt>(body->body_back())->getRetValue())
    return fail("function '" + name + "' cannot be inlined: its body must end in `return value;`", call);
  ReturnCounter returns;
  returns.TraverseStmt(const_cast<clang::CompoundStmt *>(body));
  if (returns.count != 1) return fail("function '" + name + "' cannot be inlined: it has more than one return", call);
  Type resultType = scalarType(def->getReturnType());
  if (!resultType) return fail("function '" + name + "' returns a type that is not supported", call);

  // The arguments first (they may call the same function), then bind the parameters.
  SmallVector<Value> args;
  for (auto [i, arg] : llvm::enumerate(call->arguments())) {
    const clang::ParmVarDecl *param = def->getParamDecl(i);
    clang::QualType t = param->getType();
    Type paramType = (t->isReferenceType() || t->isPointerType()) ? Type() : scalarType(t);
    if (!paramType) return fail("function '" + name + "' cannot be inlined: parameter '" + param->getNameAsString() + "' is not a scalar passed by value", call);
    Value v = emit(arg);
    if (!v) return {};
    args.push_back(cast(v, paramType));
  }
  for (auto [i, v] : llvm::enumerate(args)) env.locals[def->getParamDecl(i)] = v;

  inlineStack.push_back(def);
  for (const clang::Stmt *s : body->body()) {
    if (s == body->body_back()) break;
    emitStmt(s);
    if (!ok()) break;
  }
  Value result;
  if (ok()) result = emit(llvm::cast<clang::ReturnStmt>(body->body_back())->getRetValue());
  inlineStack.pop_back();
  if (!result) return {};
  return cast(result, resultType);
}

Value CSubsetWalker::emitShortCircuit(const clang::BinaryOperator *op) {
  bool isAnd = op->getOpcode() == clang::BO_LAnd;
  Value l = toBool(emit(op->getLHS()));
  if (!l) return {};
  if (speculatable(op->getRHS())) {
    Value r = toBool(emit(op->getRHS()));
    if (!r) return {};
    return isAnd ? Value(arith::AndIOp::create(b, loc, l, r)) : Value(arith::OrIOp::create(b, loc, l, r));
  }
  // The right operand must only be evaluated when the left does not decide the result.
  auto ifOp = scf::IfOp::create(b, loc, TypeRange{b.getI1Type()}, l, /*addThenBlock=*/true, /*addElseBlock=*/true);
  OpBuilder::InsertionGuard guard(b);
  enter(&ifOp.getThenRegion().front());
  Value rhs = isAnd ? toBool(emit(op->getRHS())) : constInt(b.getI1Type(), 1);
  if (!rhs) return {};
  scf::YieldOp::create(b, loc, rhs);
  enter(&ifOp.getElseRegion().front());
  rhs = isAnd ? constInt(b.getI1Type(), 0) : toBool(emit(op->getRHS()));
  if (!rhs) return {};
  scf::YieldOp::create(b, loc, rhs);
  return ifOp.getResult(0);
}

Value CSubsetWalker::emitBinary(const clang::BinaryOperator *op) {
  using P = arith::CmpFPredicate;
  using I = arith::CmpIPredicate;
  if (op->getOpcode() == clang::BO_LAnd || op->getOpcode() == clang::BO_LOr) return emitShortCircuit(op);
  if (op->getOpcode() == clang::BO_Comma) {
    emit(op->getLHS());
    return emit(op->getRHS());
  }
  Value l = emit(op->getLHS()), r = emit(op->getRHS());
  if (!l || !r) return {};
  if (l.getType() != r.getType()) r = cast(r, l.getType()); // e.g. an i1 compared with an int
  bool f = isa<FloatType>(l.getType());
  auto cmp = [&](P fp, I ip) -> Value {
    return f ? Value(arith::CmpFOp::create(b, loc, fp, l, r)) : Value(arith::CmpIOp::create(b, loc, ip, l, r));
  };
  switch (op->getOpcode()) {
  case clang::BO_Add: return f ? Value(arith::AddFOp::create(b, loc, l, r)) : Value(arith::AddIOp::create(b, loc, l, r));
  case clang::BO_Sub: return f ? Value(arith::SubFOp::create(b, loc, l, r)) : Value(arith::SubIOp::create(b, loc, l, r));
  case clang::BO_Mul: return f ? Value(arith::MulFOp::create(b, loc, l, r)) : Value(arith::MulIOp::create(b, loc, l, r));
  case clang::BO_Div: return f ? Value(arith::DivFOp::create(b, loc, l, r)) : Value(arith::DivSIOp::create(b, loc, l, r));
  case clang::BO_Rem: return f ? fail("floating-point %", op) : Value(arith::RemSIOp::create(b, loc, l, r));
  case clang::BO_And: return f ? fail("bitwise operator on floating point", op) : Value(arith::AndIOp::create(b, loc, l, r));
  case clang::BO_Or: return f ? fail("bitwise operator on floating point", op) : Value(arith::OrIOp::create(b, loc, l, r));
  case clang::BO_Xor: return f ? fail("bitwise operator on floating point", op) : Value(arith::XOrIOp::create(b, loc, l, r));
  case clang::BO_Shl: return f ? fail("shift of floating point", op) : Value(arith::ShLIOp::create(b, loc, l, r));
  case clang::BO_Shr: return f ? fail("shift of floating point", op) : Value(arith::ShRSIOp::create(b, loc, l, r));
  case clang::BO_LT: return cmp(P::OLT, I::slt);
  case clang::BO_LE: return cmp(P::OLE, I::sle);
  case clang::BO_GT: return cmp(P::OGT, I::sgt);
  case clang::BO_GE: return cmp(P::OGE, I::sge);
  case clang::BO_EQ: return cmp(P::OEQ, I::eq);
  case clang::BO_NE: return cmp(P::UNE, I::ne);
  default: return fail("unsupported binary operator", op);
  }
}

Value CSubsetWalker::emitConditional(const clang::ConditionalOperator *op) {
  Value cond = toBool(emit(op->getCond()));
  if (!cond) return {};
  Type type = scalarType(op->getType());
  if (!type) return fail("conditional expression of unsupported type", op);
  // Cheap, side-effect free branches are both evaluated and selected (what the C compiler does).
  if (speculatable(op->getTrueExpr()) && speculatable(op->getFalseExpr())) {
    Value t = cast(emit(op->getTrueExpr()), type), f = cast(emit(op->getFalseExpr()), type);
    if (!t || !f) return {};
    return arith::SelectOp::create(b, loc, cond, t, f);
  }
  auto ifOp = scf::IfOp::create(b, loc, TypeRange{type}, cond, /*addThenBlock=*/true, /*addElseBlock=*/true);
  OpBuilder::InsertionGuard guard(b);
  enter(&ifOp.getThenRegion().front());
  Value t = cast(emit(op->getTrueExpr()), type);
  if (!t) return {};
  scf::YieldOp::create(b, loc, t);
  enter(&ifOp.getElseRegion().front());
  Value f = cast(emit(op->getFalseExpr()), type);
  if (!f) return {};
  scf::YieldOp::create(b, loc, f);
  return ifOp.getResult(0);
}

Value CSubsetWalker::emit(const clang::Expr *e) {
  if (!ok() || !e) return {};
  if (std::optional<Value> special = emitSpecialExpr(e)) return *special;

  if (const auto *x = llvm::dyn_cast<clang::ParenExpr>(e)) return emit(x->getSubExpr());
  if (const auto *x = llvm::dyn_cast<clang::ExprWithCleanups>(e)) return emit(x->getSubExpr());
  if (const auto *x = llvm::dyn_cast<clang::ConstantExpr>(e)) return emit(x->getSubExpr());
  if (const auto *x = llvm::dyn_cast<clang::ImplicitCastExpr>(e)) {
    Value v = emit(x->getSubExpr());
    if (!v) return {};
    switch (x->getCastKind()) {
    case clang::CK_LValueToRValue:
    case clang::CK_NoOp:
    case clang::CK_ArrayToPointerDecay:
    case clang::CK_FunctionToPointerDecay:
    case clang::CK_BuiltinFnToFnPtr:
      return v;
    case clang::CK_IntegralToBoolean:
    case clang::CK_FloatingToBoolean:
      return cast(v, b.getI1Type());
    default: {
      Type to = scalarType(x->getType());
      return to ? cast(v, to) : fail("unsupported implicit conversion", e);
    }
    }
  }
  if (llvm::isa<clang::CStyleCastExpr>(e) || llvm::isa<clang::CXXStaticCastExpr>(e) ||
      llvm::isa<clang::CXXFunctionalCastExpr>(e)) {
    const auto *x = llvm::cast<clang::ExplicitCastExpr>(e);
    Type to = scalarType(x->getType());
    Value v = emit(x->getSubExpr());
    if (!v) return {};
    return to ? cast(v, to) : fail("unsupported cast", e);
  }
  if (const auto *x = llvm::dyn_cast<clang::IntegerLiteral>(e)) {
    Type t = scalarType(x->getType());
    return t ? constInt(t, x->getValue().getSExtValue()) : fail("integer literal of unsupported type", e);
  }
  if (const auto *x = llvm::dyn_cast<clang::CharacterLiteral>(e)) return constInt(b.getI32Type(), x->getValue());
  if (const auto *x = llvm::dyn_cast<clang::FloatingLiteral>(e)) {
    Type t = scalarType(x->getType());
    return t ? constFloat(t, x->getValueAsApproximateDouble()) : fail("floating literal of unsupported type", e);
  }
  if (const auto *x = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(e)) return constInt(b.getI1Type(), x->getValue());
  if (const auto *x = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
    const clang::ValueDecl *d = x->getDecl();
    if (const auto *vd0 = llvm::dyn_cast<clang::VarDecl>(d))
      if (auto cv = constVars.find(vd0); cv != constVars.end()) return constInt(scalarType(vd0->getType()) ? scalarType(vd0->getType()) : b.getI32Type(), cv->second);
    if (const auto *en = llvm::dyn_cast<clang::EnumConstantDecl>(d)) return constInt(b.getI32Type(), en->getInitVal().getSExtValue());
    if (const auto *vd = llvm::dyn_cast<clang::VarDecl>(d)) {
      auto it = env.locals.find(vd);
      if (it != env.locals.end()) return it->second;
    }
    // A constant the compiler can evaluate (a const global with an initialiser).
    clang::Expr::EvalResult res;
    if (!e->isValueDependent() && e->getType()->isScalarType() && e->EvaluateAsRValue(res, astCtx)) {
      Type t = scalarType(e->getType());
      if (t && res.Val.isInt()) return constInt(t, res.Val.getInt().getSExtValue());
      if (t && res.Val.isFloat()) return constFloat(t, res.Val.getFloat().convertToDouble());
    }
    return fail("reference to '" + d->getNameAsString() + "' is not supported here", e);
  }
  if (const auto *x = llvm::dyn_cast<clang::UnaryOperator>(e)) {
    Value v = emit(x->getSubExpr());
    if (!v) return {};
    switch (x->getOpcode()) {
    case clang::UO_Minus:
      return isa<FloatType>(v.getType()) ? Value(arith::NegFOp::create(b, loc, v))
                                         : Value(arith::SubIOp::create(b, loc, zeroOf(v.getType()), v));
    case clang::UO_Plus: return v;
    case clang::UO_LNot: return arith::XOrIOp::create(b, loc, toBool(v), constInt(b.getI1Type(), 1));
    case clang::UO_Not: return arith::XOrIOp::create(b, loc, v, constInt(v.getType(), -1));
    default: return fail("unsupported unary operator", e);
    }
  }
  if (const auto *x = llvm::dyn_cast<clang::BinaryOperator>(e)) {
    if (x->isAssignmentOp()) return fail("an assignment used as a value is not supported", e);
    return emitBinary(x);
  }
  if (const auto *x = llvm::dyn_cast<clang::ConditionalOperator>(e)) return emitConditional(x);
  if (const auto *x = llvm::dyn_cast<clang::ArraySubscriptExpr>(e)) {
    const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(x->getBase()->IgnoreParenImpCasts());
    const auto *vd = dre ? llvm::dyn_cast<clang::VarDecl>(dre->getDecl()) : nullptr;
    if (vd && elementTypes.count(vd)) {
      auto i = evalInt(x->getIdx());
      if (!i) return fail("local array indexed by a value that is not a compile-time constant", e);
      auto it = env.elements.find({vd, *i});
      return it != env.elements.end() ? it->second : fail("index out of range of a local array", e);
    }
  }
  if (const auto *x = llvm::dyn_cast<clang::CallExpr>(e)) return emitCall(x);

  // Last resort: anything the compiler can fold.
  clang::Expr::EvalResult res;
  if (!e->isValueDependent() && e->getType()->isScalarType() && e->EvaluateAsRValue(res, astCtx)) {
    Type t = scalarType(e->getType());
    if (t && res.Val.isInt()) return constInt(t, res.Val.getInt().getSExtValue());
    if (t && res.Val.isFloat()) return constFloat(t, res.Val.getFloat().convertToDouble());
  }
  return fail(llvm::Twine("unsupported expression (") + e->getStmtClassName() + ")", e);
}

//===----------------------------------------------------------------------===//
// Local variables across control flow
//===----------------------------------------------------------------------===//

bool CSubsetWalker::collectAssigned(const clang::Stmt *s, std::vector<LocalTarget> &out, const Env &visible) {
  struct Scan : clang::RecursiveASTVisitor<Scan> {
    CSubsetWalker &w;
    const Env &visible;
    std::set<LocalTarget> found;
    bool bad = false;
    Scan(CSubsetWalker &w, const Env &visible) : w(w), visible(visible) {}
    void note(const clang::Expr *lhs) {
      lhs = lhs->IgnoreParenImpCasts();
      if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(lhs)) {
        if (const auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl()))
          if (visible.locals.count(vd)) found.insert({vd, -1});
        return;
      }
      if (const auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(lhs)) {
        const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(sub->getBase()->IgnoreParenImpCasts());
        const auto *vd = dre ? llvm::dyn_cast<clang::VarDecl>(dre->getDecl()) : nullptr;
        if (!vd || !w.elementTypes.count(vd)) return;
        auto i = w.evalInt(sub->getIdx());
        if (!i) {
          bad = true;
          return;
        }
        found.insert({vd, *i});
      }
    }
    bool VisitBinaryOperator(clang::BinaryOperator *op) {
      if (op->isAssignmentOp()) note(op->getLHS());
      return true;
    }
    bool VisitUnaryOperator(clang::UnaryOperator *op) {
      if (op->isIncrementDecrementOp()) note(op->getSubExpr());
      return true;
    }
  } scan(*this, visible);
  scan.TraverseStmt(const_cast<clang::Stmt *>(s));
  if (scan.bad) {
    fail("write to an element of a local array whose index is not a compile-time constant", s);
    return false;
  }
  out.assign(scan.found.begin(), scan.found.end());
  sortTargets(out);
  return true;
}

/// The order in which locals become results and iter_args: the order of their declarations, so that the IR does not
/// depend on where the compiler allocated the declarations.
void CSubsetWalker::sortTargets(std::vector<LocalTarget> &targets) {
  std::sort(targets.begin(), targets.end(), [](const LocalTarget &a, const LocalTarget &b) {
    unsigned la = a.first->getLocation().getRawEncoding(), lb = b.first->getLocation().getRawEncoding();
    return std::tie(la, a.second) < std::tie(lb, b.second);
  });
}

Value CSubsetWalker::current(const LocalTarget &t) {
  if (t.second < 0) {
    auto it = env.locals.find(t.first);
    return it == env.locals.end() ? Value() : it->second;
  }
  auto it = env.elements.find({t.first, t.second});
  return it == env.elements.end() ? Value() : it->second;
}

Type CSubsetWalker::typeOf(const LocalTarget &t) {
  if (t.second >= 0) return elementTypes.lookup(t.first);
  return scalarType(t.first->getType());
}

void CSubsetWalker::assignLocal(const LocalTarget &t, Value v) {
  if (!v) return;
  Type type = typeOf(t);
  v = type ? cast(v, type) : v;
  if (t.second < 0) env.locals[t.first] = v;
  else env.elements[{t.first, t.second}] = v;
}

void CSubsetWalker::enter(Block *block) {
  if (!block->empty() && block->back().hasTrait<OpTrait::IsTerminator>()) b.setInsertionPoint(&block->back());
  else b.setInsertionPointToEnd(block);
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

void CSubsetWalker::emitDecl(const clang::DeclStmt *s) {
  for (const clang::Decl *decl : s->decls()) {
    const auto *vd = llvm::dyn_cast<clang::VarDecl>(decl);
    if (!vd) {
      if (llvm::isa<clang::TypedefNameDecl>(decl) || llvm::isa<clang::StaticAssertDecl>(decl)) continue;
      fail("unsupported declaration", s);
      return;
    }
    if (const clang::ArrayType *array = astCtx.getAsArrayType(vd->getType())) {
      const auto *constant = llvm::dyn_cast<clang::ConstantArrayType>(array);
      Type elem = scalarType(array->getElementType());
      if (!constant || !elem || array->getElementType()->isArrayType() || constant->getSize().getZExtValue() > 64) {
        fail("local array '" + vd->getNameAsString() + "' is not a one-dimensional array of at most 64 scalars", s);
        return;
      }
      elementTypes[vd] = elem;
      const auto *list = vd->hasInit() ? llvm::dyn_cast<clang::InitListExpr>(vd->getInit()->IgnoreParenImpCasts()) : nullptr;
      if (vd->hasInit() && !list) {
        fail("unsupported initialiser of local array '" + vd->getNameAsString() + "'", s);
        return;
      }
      for (uint64_t i = 0; i < constant->getSize().getZExtValue(); ++i) {
        Value v = list && i < list->getNumInits() ? cast(emit(list->getInit(i)), elem) : zeroOf(elem);
        if (!v) return;
        env.elements[{vd, static_cast<int64_t>(i)}] = v;
      }
      continue;
    }
    Type ty = scalarType(vd->getType());
    if (!ty || vd->getType()->isPointerType() || vd->getType()->isReferenceType()) {
      fail("local '" + vd->getNameAsString() + "' has an unsupported type (" + vd->getType().getAsString() + ")", s);
      return;
    }
    Value v = vd->hasInit() ? cast(emit(vd->getInit()), ty) : zeroOf(ty);
    if (!v) return;
    env.locals[vd] = v;
  }
}

void CSubsetWalker::emitAssignment(const clang::BinaryOperator *op) {
  const clang::Expr *lhs = op->getLHS()->IgnoreParenImpCasts();
  LocalTarget target{nullptr, -1};
  if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(lhs)) {
    if (const auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl()); vd && isLocal(vd)) target = {vd, -1};
  } else if (const auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(lhs)) {
    const auto *base = llvm::dyn_cast<clang::DeclRefExpr>(sub->getBase()->IgnoreParenImpCasts());
    const auto *vd = base ? llvm::dyn_cast<clang::VarDecl>(base->getDecl()) : nullptr;
    if (vd && elementTypes.count(vd)) {
      auto i = evalInt(sub->getIdx());
      if (!i) {
        fail("write to a local array element with a non-constant index", op);
        return;
      }
      target = {vd, *i};
    }
  }
  if (!target.first) {
    if (!emitSpecialAssign(op) && ok()) fail("assignment to something that is neither a local variable nor an output", op);
    return;
  }

  Value rhs = emit(op->getRHS());
  if (!rhs) return;
  if (op->getOpcode() != clang::BO_Assign) {
    Value cur = current(target);
    if (!cur) {
      fail("compound assignment to a variable with no value", op);
      return;
    }
    rhs = cast(rhs, cur.getType());
    bool f = isa<FloatType>(cur.getType());
    switch (op->getOpcode()) {
    case clang::BO_AddAssign: rhs = f ? Value(arith::AddFOp::create(b, loc, cur, rhs)) : Value(arith::AddIOp::create(b, loc, cur, rhs)); break;
    case clang::BO_SubAssign: rhs = f ? Value(arith::SubFOp::create(b, loc, cur, rhs)) : Value(arith::SubIOp::create(b, loc, cur, rhs)); break;
    case clang::BO_MulAssign: rhs = f ? Value(arith::MulFOp::create(b, loc, cur, rhs)) : Value(arith::MulIOp::create(b, loc, cur, rhs)); break;
    case clang::BO_DivAssign: rhs = f ? Value(arith::DivFOp::create(b, loc, cur, rhs)) : Value(arith::DivSIOp::create(b, loc, cur, rhs)); break;
    case clang::BO_RemAssign:
      if (f) {
        fail("floating-point %=", op);
        return;
      }
      rhs = arith::RemSIOp::create(b, loc, cur, rhs);
      break;
    default:
      fail("unsupported compound assignment", op);
      return;
    }
  }
  assignLocal(target, rhs);
}

void CSubsetWalker::emitIncDec(const clang::UnaryOperator *op) {
  const clang::Expr *sub = op->getSubExpr()->IgnoreParenImpCasts();
  LocalTarget target{nullptr, -1};
  if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(sub))
    if (const auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl()); vd && isLocal(vd)) target = {vd, -1};
  Value cur = target.first ? current(target) : Value();
  if (!cur) {
    fail("++/-- on something other than a local scalar", op);
    return;
  }
  bool f = isa<FloatType>(cur.getType());
  Value one = f ? constFloat(cur.getType(), 1.0) : constInt(cur.getType(), 1);
  Value r = op->isIncrementOp() ? (f ? Value(arith::AddFOp::create(b, loc, cur, one)) : Value(arith::AddIOp::create(b, loc, cur, one)))
                                : (f ? Value(arith::SubFOp::create(b, loc, cur, one)) : Value(arith::SubIOp::create(b, loc, cur, one)));
  assignLocal(target, r);
}

void CSubsetWalker::emitIf(const clang::IfStmt *s) {
  if (s->getInit() || s->getConditionVariable() || s->isConstexpr()) {
    fail("unsupported form of if", s);
    return;
  }
  Value cond = toBool(emit(s->getCond()));
  if (!cond) return;

  std::vector<LocalTarget> targets;
  {
    std::vector<LocalTarget> inThen, inElse;
    if (!collectAssigned(s->getThen(), inThen, env)) return;
    if (s->getElse() && !collectAssigned(s->getElse(), inElse, env)) return;
    std::set<LocalTarget> all(inThen.begin(), inThen.end());
    all.insert(inElse.begin(), inElse.end());
    targets.assign(all.begin(), all.end());
    sortTargets(targets);
  }
  SmallVector<Type> types;
  for (const LocalTarget &t : targets) types.push_back(current(t).getType());

  Env saved = env;
  auto ifOp = scf::IfOp::create(b, loc, TypeRange(types), cond, /*addThenBlock=*/true, /*addElseBlock=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    auto branch = [&](Region &region, const clang::Stmt *body) {
      env = saved;
      enter(&region.front());
      if (body) emitStmt(body);
      if (!ok()) return;
      SmallVector<Value> values;
      for (auto [i, t] : llvm::enumerate(targets)) values.push_back(cast(current(t), types[i]));
      // The builder does not always create the terminator of a region without results: make it explicit.
      Block *block = &region.front();
      if (!block->empty() && block->back().hasTrait<OpTrait::IsTerminator>()) block->back().erase();
      scf::YieldOp::create(b, loc, values);
    };
    branch(ifOp.getThenRegion(), s->getThen());
    if (ok()) branch(ifOp.getElseRegion(), s->getElse());
  }
  env = saved;
  if (!ok()) return;
  b.setInsertionPointAfter(ifOp);
  for (auto [i, t] : llvm::enumerate(targets)) assignLocal(t, ifOp.getResult(i));
}

void CSubsetWalker::emitFor(const clang::ForStmt *s) {
  const auto *init = llvm::dyn_cast_or_null<clang::DeclStmt>(s->getInit());
  const auto *cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(s->getCond() ? s->getCond()->IgnoreParenImpCasts() : nullptr);
  const clang::VarDecl *var = nullptr;
  if (init && init->isSingleDecl()) var = llvm::dyn_cast<clang::VarDecl>(init->getSingleDecl());
  if (!var || !var->hasInit() || !cond || !s->getInc() || s->getConditionVariable()) {
    fail("loop is not of the form for (int v = a; v < b; v++)", s);
    return;
  }
  Type varType = scalarType(var->getType());
  if (!varType || !varType.isInteger() || varType.isInteger(1)) {
    fail("loop variable is not an integer", s);
    return;
  }
  auto isVar = [&](const clang::Expr *x) {
    const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(x->IgnoreParenImpCasts());
    return dre && dre->getDecl() == var;
  };
  // v < b, v <= b
  if (!isVar(cond->getLHS()) || (cond->getOpcode() != clang::BO_LT && cond->getOpcode() != clang::BO_LE)) {
    fail("loop condition is not of the form v < b or v <= b", s);
    return;
  }
  // v++, ++v, v += c, v = v + c with a positive constant c
  int64_t step = 0;
  const clang::Expr *inc = s->getInc()->IgnoreParenImpCasts();
  if (const auto *un = llvm::dyn_cast<clang::UnaryOperator>(inc); un && un->isIncrementOp() && isVar(un->getSubExpr())) {
    step = 1;
  } else if (const auto *bin = llvm::dyn_cast<clang::BinaryOperator>(inc); bin && isVar(bin->getLHS())) {
    if (bin->getOpcode() == clang::BO_AddAssign) {
      if (auto c = evalInt(bin->getRHS())) step = *c;
    } else if (bin->getOpcode() == clang::BO_Assign) {
      if (const auto *add = llvm::dyn_cast<clang::BinaryOperator>(bin->getRHS()->IgnoreParenImpCasts());
          add && add->getOpcode() == clang::BO_Add && isVar(add->getLHS()))
        if (auto c = evalInt(add->getRHS())) step = *c;
    }
  }
  if (step <= 0) {
    fail("loop increment is not v++ or v += c with a positive constant c", s);
    return;
  }

  // Constant bounds and a variable that must be known at compile time: unroll.
  if (auto lo = evalInt(var->getInit()); lo && shouldUnroll(s, var)) {
    auto hi = evalInt(cond->getRHS());
    if (hi) {
      int64_t end = cond->getOpcode() == clang::BO_LE ? *hi + 1 : *hi;
      if ((end - *lo + step - 1) / step > 64) {
        fail("loop would unroll to more than 64 iterations", s);
        return;
      }
      for (int64_t i = *lo; i < end && ok(); i += step) {
        constVars[var] = i;
        emitStmt(s->getBody());
      }
      constVars.erase(var);
      return;
    }
  }

  Type index = b.getIndexType();
  Value lb = cast(emit(var->getInit()), varType), ub = cast(emit(cond->getRHS()), varType);
  if (!lb || !ub) return;
  if (cond->getOpcode() == clang::BO_LE) ub = arith::AddIOp::create(b, loc, ub, constInt(varType, 1));
  lb = arith::IndexCastOp::create(b, loc, index, lb);
  ub = arith::IndexCastOp::create(b, loc, index, ub);
  Value stepValue = arith::ConstantIndexOp::create(b, loc, step);

  std::vector<LocalTarget> targets;
  if (!collectAssigned(s->getBody(), targets, env)) return;
  for (const LocalTarget &t : targets)
    if (t.first == var) {
      fail("the loop body assigns the loop variable", s);
      return;
    }
  SmallVector<Value> inits;
  for (const LocalTarget &t : targets) inits.push_back(current(t));

  Env saved = env;
  auto forOp = scf::ForOp::create(b, loc, lb, ub, stepValue, inits);
  {
    OpBuilder::InsertionGuard guard(b);
    Block *body = forOp.getBody();
    enter(body);
    env = saved;
    env.locals[var] = arith::IndexCastOp::create(b, loc, varType, forOp.getInductionVar());
    for (auto [i, t] : llvm::enumerate(targets)) {
      if (t.second < 0) env.locals[t.first] = forOp.getRegionIterArg(i);
      else env.elements[{t.first, t.second}] = forOp.getRegionIterArg(i);
    }
    emitStmt(s->getBody());
    if (ok()) {
      SmallVector<Value> values;
      for (auto [i, t] : llvm::enumerate(targets)) values.push_back(cast(current(t), inits[i].getType()));
      // The terminator the builder created yields the iter args unchanged: replace it.
      if (!body->empty() && body->back().hasTrait<OpTrait::IsTerminator>()) body->back().erase();
      scf::YieldOp::create(b, loc, values);
    }
  }
  env = saved;
  if (!ok()) return;
  b.setInsertionPointAfter(forOp);
  for (auto [i, t] : llvm::enumerate(targets)) assignLocal(t, forOp.getResult(i));
}

void CSubsetWalker::emitStmt(const clang::Stmt *s) {
  if (!ok() || !s) return;
  if (emitSpecialStmt(s)) return;
  if (const auto *c = llvm::dyn_cast<clang::CompoundStmt>(s)) {
    for (const clang::Stmt *x : c->body()) emitStmt(x);
  } else if (const auto *d = llvm::dyn_cast<clang::DeclStmt>(s)) {
    emitDecl(d);
  } else if (const auto *x = llvm::dyn_cast<clang::IfStmt>(s)) {
    emitIf(x);
  } else if (const auto *x = llvm::dyn_cast<clang::ForStmt>(s)) {
    emitFor(x);
  } else if (const auto *x = llvm::dyn_cast<clang::BinaryOperator>(s)) {
    if (x->isAssignmentOp()) emitAssignment(x);
    else if (x->getOpcode() == clang::BO_Comma) {
      emitStmt(x->getLHS());
      emitStmt(x->getRHS());
    } else fail("expression statement without effect", s);
  } else if (const auto *x = llvm::dyn_cast<clang::UnaryOperator>(s)) {
    if (x->isIncrementDecrementOp()) emitIncDec(x);
    else fail("expression statement without effect", s);
  } else if (const auto *x = llvm::dyn_cast<clang::CallExpr>(s)) {
    if (!dropCall(x)) fail("a call used as a statement is not supported (only dropped ones like printf)", s);
  } else if (const auto *x = llvm::dyn_cast<clang::ReturnStmt>(s)) {
    if (x->getRetValue()) fail("kernel returns a value", s);
    // a trailing `return;` is fine; an early one is not handled
  } else if (llvm::isa<clang::NullStmt>(s)) {
  } else if (const auto *x = llvm::dyn_cast<clang::ExprWithCleanups>(s)) {
    emitStmt(x->getSubExpr());
  } else {
    fail(llvm::Twine("unsupported statement (") + s->getStmtClassName() + ")", s);
  }
}

void CSubsetWalker::emitBody(const clang::Stmt *body) { emitStmt(body); }

} // namespace ops_mlir::frontend
