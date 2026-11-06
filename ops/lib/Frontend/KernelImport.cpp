#include "ops-mlir/Frontend/KernelImport.h"

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Frontend/CSubsetWalker.h"

#include "opdsl/Support/Hash.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>

namespace ops_mlir::frontend {

using namespace mlir;
using namespace ops_mlir::ops;

const char *importerId() { return "A/1.0"; }

namespace {

#ifndef OPS_CLANG_RESOURCE_DIR
#define OPS_CLANG_RESOURCE_DIR ""
#endif

/// OPS_ACCn(x, y, z) -- how pointer-style kernels index their nth dat -- as a call that carries n, so the importer
/// can tell which argument an index belongs to (the generated code defines these macros itself, per backend; the
/// translation unit is parsed with OPS_ACC_IGNORE, as OPS's own translator does, so that ops_macros.h leaves them out).
std::string accessMacros() {
  std::string text = "int __ops_acc(int, ...);\nint __ops_acc_md(int, ...);\n";
  for (int n = 0; n < 128; ++n) {
    text += "#define OPS_ACC" + std::to_string(n) + "(...) __ops_acc(" + std::to_string(n) + ", __VA_ARGS__)\n";
    text += "#define OPS_ACC_MD" + std::to_string(n) + "(...) __ops_acc_md(" + std::to_string(n) + ", __VA_ARGS__)\n";
  }
  return text;
}

std::string typeName(Type t) {
  if (t.isF64()) return "f64";
  if (t.isF32()) return "f32";
  if (t.isF16()) return "f16";
  if (t.isInteger(32)) return "i32";
  if (t.isInteger(64)) return "i64";
  std::string s;
  llvm::raw_string_ostream(s) << t;
  return s;
}

struct Param {
  enum Kind { Acc, Gbl, Red, Idx } kind = Acc;
  const clang::ParmVarDecl *decl = nullptr;
  Type elem;
  bool isConst = false;
  bool legacyMacro = false;
  bool byValue = false;
  int64_t ncomp = 1;  // accessors: components used
  int64_t gblDim = 1; // globals: elements used
  Value arg;          // the ops.kernel block argument (not for Idx)
  std::set<std::vector<int64_t>> reads, writes;
  bool dynamicRead = false;
  std::string style = "accumulate"; // reductions
  std::string kindHint;             // reductions: add, min, max
};

struct ConstUse {
  std::string name;
  Type elem;
  int64_t offset = 0, stride = 0;
  bool dynamicIndex = false;
  bool operator<(const ConstUse &o) const {
    return std::tie(name, offset, stride, dynamicIndex) < std::tie(o.name, o.offset, o.stride, o.dynamicIndex);
  }
};

const clang::CXXRecordDecl *accRecord(clang::QualType t) {
  t = t.getNonReferenceType().getUnqualifiedType();
  return t->getAsCXXRecordDecl();
}

bool isAccType(clang::QualType t) {
  const clang::CXXRecordDecl *r = accRecord(t);
  return r && r->getName() == "ACC";
}

/// The element type `T` of an `ACC<T>`.
clang::QualType accElement(clang::QualType t) {
  if (const auto *spec = llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(accRecord(t)))
    if (spec->getTemplateArgs().size() >= 1 && spec->getTemplateArgs()[0].getKind() == clang::TemplateArgument::Type)
      return spec->getTemplateArgs()[0].getAsType();
  return {};
}

class OPSKernelWalker : public CSubsetWalker {
public:
  OPSKernelWalker(MLIRContext &ctx, clang::ASTContext &astCtx, std::vector<Param> &params, int rank)
      : CSubsetWalker(ctx, astCtx), params(params), rank(rank) {}

  const std::set<ConstUse> &constUses() const { return consts; }
  unsigned droppedPrintf() const { return dropped; }
  unsigned printfs() const { return printed; }
  unsigned idxAxesRead() const { return idxAxes; }

protected:
  std::vector<Param> &params;
  int rank;
  std::set<ConstUse> consts;
  unsigned dropped = 0;
  unsigned printed = 0;
  unsigned idxAxes = 0;

  int paramIndex(const clang::Expr *e) {
    e = e->IgnoreParenImpCasts();
    if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(e))
      if (const auto *pv = llvm::dyn_cast<clang::ParmVarDecl>(dre->getDecl()))
        for (size_t i = 0; i < params.size(); ++i)
          if (params[i].decl == pv) return static_cast<int>(i);
    return -1;
  }

  //--- accessors --------------------------------------------------------------------------------------------------
  struct AccessRef {
    int param = -1;
    int64_t comp = 0;
    SmallVector<const clang::Expr *> offsets; // one per dimension
    bool ok = false;
  };

  /// `a(dx, dy)` / `a(c, dx, dy)` on an ACC parameter, or `a[OPS_ACCn(dx, dy)]` / `a[OPS_ACC_MDn(c, dx, dy)]`.
  AccessRef matchAccess(const clang::Expr *e) {
    AccessRef out;
    e = e->IgnoreParenImpCasts();
    SmallVector<const clang::Expr *> args;
    bool multiComp = false;
    if (const auto *call = llvm::dyn_cast<clang::CXXOperatorCallExpr>(e)) {
      if (call->getOperator() != clang::OO_Call || call->getNumArgs() < 1) return out;
      int p = paramIndex(call->getArg(0));
      if (p < 0 || params[p].kind != Param::Acc) return out;
      out.param = p;
      for (unsigned i = 1; i < call->getNumArgs(); ++i) args.push_back(call->getArg(i));
      multiComp = args.size() == static_cast<size_t>(rank) + 1;
    } else if (const auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(e)) {
      int p = paramIndex(sub->getBase());
      if (p < 0 || params[p].kind != Param::Acc) return out;
      const auto *acc = llvm::dyn_cast<clang::CallExpr>(sub->getIdx()->IgnoreParenImpCasts());
      const clang::FunctionDecl *callee = acc ? acc->getDirectCallee() : nullptr;
      if (!callee || (callee->getName() != "__ops_acc" && callee->getName() != "__ops_acc_md") || acc->getNumArgs() < 1) return out;
      auto n = evalInt(acc->getArg(0));
      if (!n || *n != p) {
        fail("OPS_ACCn must index the argument it is applied to", e);
        return out;
      }
      out.param = p;
      for (unsigned i = 1; i < acc->getNumArgs(); ++i) args.push_back(acc->getArg(i));
      multiComp = callee->getName() == "__ops_acc_md";
    } else {
      return out;
    }
    if (multiComp) { // the component comes first
      auto c = evalInt(args[0]);
      if (!c || *c < 0) {
        fail("the component of a multi-component accessor is not a compile-time constant", e);
        out.param = -1;
        return out;
      }
      out.comp = *c;
      args.erase(args.begin());
    }
    if (args.size() != static_cast<size_t>(rank)) {
      fail("accessor with " + llvm::Twine(args.size()) + " indices in a " + llvm::Twine(rank) + "D kernel", e);
      out.param = -1;
      return out;
    }
    out.offsets = args;
    out.ok = true;
    return out;
  }

  /// Constant offsets of an access, or nullopt if one depends on data.
  std::optional<SmallVector<int64_t>> constantOffsets(const AccessRef &ref) {
    SmallVector<int64_t> off;
    for (const clang::Expr *e : ref.offsets) {
      auto v = evalInt(e);
      if (!v) return std::nullopt;
      off.push_back(*v);
    }
    return off;
  }

  Value emitRead(const AccessRef &ref) {
    Param &p = params[ref.param];
    p.ncomp = std::max(p.ncomp, ref.comp + 1);
    if (auto off = constantOffsets(ref)) {
      p.reads.insert(std::vector<int64_t>(off->begin(), off->end()));
      return ReadOp::create(b, loc, p.elem, p.arg, b.getDenseI64ArrayAttr(*off), b.getI64IntegerAttr(ref.comp));
    }
    p.dynamicRead = true;
    SmallVector<Value> offsets;
    for (const clang::Expr *e : ref.offsets) {
      Value v = emit(e);
      if (!v) return {};
      offsets.push_back(cast(v, b.getI32Type()));
    }
    return ReadDynOp::create(b, loc, p.elem, p.arg, offsets, b.getI64IntegerAttr(ref.comp));
  }

  //--- reductions ----------------------------------------------------------------------------------------------
  /// `*r` or `r[0]` on a reduction parameter: its parameter index, or -1.
  int matchReduction(const clang::Expr *e) {
    e = e->IgnoreParenImpCasts();
    const clang::Expr *base = nullptr;
    if (const auto *un = llvm::dyn_cast<clang::UnaryOperator>(e); un && un->getOpcode() == clang::UO_Deref) {
      base = un->getSubExpr();
    } else if (const auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(e)) {
      base = sub->getBase();
      auto k = evalInt(sub->getIdx());
      if (!k || *k != 0) return -1;
    } else {
      return -1;
    }
    int p = paramIndex(base);
    return p >= 0 && params[p].kind == Param::Red ? p : -1;
  }

  static bool sameExpr(const clang::ASTContext &ctx, const clang::Expr *a, const clang::Expr *b) {
    llvm::FoldingSetNodeID ia, ib;
    a->IgnoreParenImpCasts()->Profile(ia, ctx, /*Canonical=*/true);
    b->IgnoreParenImpCasts()->Profile(ib, ctx, /*Canonical=*/true);
    return ia == ib;
  }

  struct ReduceUpdate {
    const clang::Expr *contribution = nullptr;
    bool negate = false;
    std::string kind; // add, min, max, or "" (plain assignment)
  };

  /// The contribution of `*r = <rhs>` when the right side spells an accumulation, a min or a max.
  ReduceUpdate classifyAssignment(int param, const clang::Expr *rhs) {
    rhs = rhs->IgnoreParenImpCasts();
    auto isR = [&](const clang::Expr *x) { return matchReduction(x) == param; };
    ReduceUpdate none;
    if (const auto *add = llvm::dyn_cast<clang::BinaryOperator>(rhs); add && add->getOpcode() == clang::BO_Add) {
      if (isR(add->getLHS())) return {add->getRHS(), false, "add"};
      if (isR(add->getRHS())) return {add->getLHS(), false, "add"};
    }
    if (const auto *sub = llvm::dyn_cast<clang::BinaryOperator>(rhs); sub && sub->getOpcode() == clang::BO_Sub && isR(sub->getLHS()))
      return {sub->getRHS(), true, "add"};
    if (const auto *call = llvm::dyn_cast<clang::CallExpr>(rhs); call && call->getNumArgs() == 2 && call->getDirectCallee()) {
      llvm::StringRef name = call->getDirectCallee()->getName();
      name.consume_front("__builtin_");
      if (name == "fmin" || name == "fmax" || name == "fminf" || name == "fmaxf") {
        bool isMin = name.starts_with("fmin");
        if (isR(call->getArg(0))) return {call->getArg(1), false, isMin ? "min" : "max"};
        if (isR(call->getArg(1))) return {call->getArg(0), false, isMin ? "min" : "max"};
      }
    }
    if (const auto *cond = llvm::dyn_cast<clang::ConditionalOperator>(rhs)) {
      const auto *cmp = llvm::dyn_cast<clang::BinaryOperator>(cond->getCond()->IgnoreParenImpCasts());
      const clang::Expr *t = cond->getTrueExpr(), *f = cond->getFalseExpr();
      if (cmp && cmp->isComparisonOp() && cmp->getOpcode() != clang::BO_EQ && cmp->getOpcode() != clang::BO_NE) {
        const clang::Expr *x = isR(t) ? f : (isR(f) ? t : nullptr);
        bool picksR = isR(t);
        if (x && !isR(x)) {
          // The comparison must be between r and x, in either order.
          const clang::Expr *l = cmp->getLHS(), *r = cmp->getRHS();
          bool lr = isR(l) && sameExpr(astCtx, r, x), rl = isR(r) && sameExpr(astCtx, l, x);
          if (lr || rl) {
            bool less = cmp->getOpcode() == clang::BO_LT || cmp->getOpcode() == clang::BO_LE;
            // The branch taken when the condition holds is `t`; the smaller operand wins if the comparison is a
            // less-than and the true branch is its left operand (or a greater-than and its right operand).
            bool trueIsLeft = lr ? picksR : !picksR;
            bool isMin = less ? trueIsLeft : !trueIsLeft;
            return {x, false, isMin ? "min" : "max"};
          }
        }
      }
    }
    return none;
  }

  /// `ops.reduce` for the update; returns false (after reporting) if the statement is not supported.
  bool emitReduce(int param, const ReduceUpdate &u, const clang::Stmt *at) {
    Param &p = params[param];
    Value v = emit(u.contribution);
    if (!v) return false;
    v = cast(v, p.elem);
    if (u.negate)
      v = isa<FloatType>(v.getType()) ? Value(arith::NegFOp::create(b, loc, v))
                                      : Value(arith::SubIOp::create(b, loc, zeroOf(v.getType()), v));
    if (!u.kind.empty()) {
      if (!p.kindHint.empty() && p.kindHint != u.kind) {
        fail("a reduction that is updated as both " + llvm::Twine(p.kindHint) + " and " + u.kind, at);
        return false;
      }
      p.kindHint = u.kind;
    } else {
      p.style = "assign";
    }
    ReduceOp::create(b, loc, p.arg, v);
    return true;
  }

  //--- globals -------------------------------------------------------------------------------------------------
  struct ConstPath {
    const clang::VarDecl *var = nullptr;
    int64_t offset = 0;
    Value dynIndex;
    int64_t stride = 0;
    clang::QualType type;
    bool ok = false;
  };

  ConstPath constPath(const clang::Expr *e) {
    e = e->IgnoreParenImpCasts();
    ConstPath out;
    if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
      const auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl());
      if (!vd || llvm::isa<clang::ParmVarDecl>(vd) || !vd->hasGlobalStorage()) return out;
      out.var = vd;
      out.type = vd->getType();
      out.ok = true;
      return out;
    }
    if (const auto *me = llvm::dyn_cast<clang::MemberExpr>(e)) {
      if (me->isArrow()) return out;
      const auto *fd = llvm::dyn_cast<clang::FieldDecl>(me->getMemberDecl());
      if (!fd) return out;
      ConstPath base = constPath(me->getBase());
      if (!base.ok || fd->getParent()->isInvalidDecl()) return out;
      base.offset += astCtx.getASTRecordLayout(fd->getParent()).getFieldOffset(fd->getFieldIndex()) / 8;
      base.type = fd->getType();
      return base;
    }
    if (const auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(e)) {
      ConstPath base = constPath(sub->getBase());
      if (!base.ok) return out;
      clang::QualType elem;
      if (const clang::ArrayType *array = astCtx.getAsArrayType(base.type)) elem = array->getElementType();
      else if (base.type->isPointerType()) elem = base.type->getPointeeType();
      if (elem.isNull() || elem->isIncompleteType()) return out;
      int64_t size = astCtx.getTypeSizeInChars(elem).getQuantity();
      if (auto i = evalInt(sub->getIdx())) {
        base.offset += *i * size;
      } else {
        if (base.dynIndex) return out; // one dynamic index level
        Value iv = emit(sub->getIdx());
        if (!iv) return out;
        base.dynIndex = cast(iv, b.getI32Type());
        base.stride = size;
      }
      base.type = elem;
      return base;
    }
    return out;
  }

  Value emitGlobal(const clang::Expr *e) {
    ConstPath path = constPath(e);
    if (!path.ok) return ok() ? fail("unsupported reference to a global or a member of one", e) : Value();
    Type t = scalarType(path.type);
    if (!t || t.isInteger(1) || !isSupportedElementType(t))
      return fail("global '" + path.var->getNameAsString() + "' has an unsupported type (" + path.type.getAsString() + ")", e);
    ConstUse use{path.var->getNameAsString(), t, path.offset, path.stride, static_cast<bool>(path.dynIndex)};
    consts.insert(use);
    return ConstRefOp::create(b, loc, t, FlatSymbolRefAttr::get(b.getContext(), use.name), b.getI64IntegerAttr(path.offset),
                              path.dynIndex, b.getI64IntegerAttr(path.stride));
  }

  /// A global that the compiler can evaluate (a const with an initialiser) is a constant; the others are
  /// registered constants read from the application at run time.
  Value emitGlobalOrConstant(const clang::Expr *x) {
    clang::Expr::EvalResult res;
    if (!x->isValueDependent() && x->getType()->isScalarType() && x->EvaluateAsRValue(res, astCtx)) {
      Type t = scalarType(x->getType());
      if (t && res.Val.isInt()) return constInt(t, res.Val.getInt().getSExtValue());
      if (t && res.Val.isFloat()) return constFloat(t, res.Val.getFloat().convertToDouble());
    }
    return emitGlobal(x);
  }

  Value indexValue(int64_t dim) {
    idxAxes |= 1u << dim;
    return IndexOp::create(b, loc, b.getI32Type(), b.getI64IntegerAttr(dim));
  }

  Value gblRead(Param &param, int64_t k) {
    param.gblDim = std::max(param.gblDim, k + 1);
    return GblReadOp::create(b, loc, param.elem, param.arg, b.getI64IntegerAttr(k)).getResult();
  }

  //--- hooks ------------------------------------------------------------------------------------------------------
  std::optional<Value> emitSpecialExpr(const clang::Expr *e) override {
    const clang::Expr *x = e->IgnoreParenImpCasts();
    if (llvm::isa<clang::CXXOperatorCallExpr>(x)) {
      AccessRef ref = matchAccess(x);
      if (!ok()) return Value();
      if (ref.ok) return emitRead(ref);
      return std::nullopt;
    }
    if (const auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(x)) {
      int p = paramIndex(sub->getBase());
      if (p >= 0) {
        Param &param = params[p];
        if (param.kind == Param::Acc) {
          AccessRef ref = matchAccess(x);
          if (!ok()) return Value();
          if (!ref.ok) return fail("unsupported use of an accessor parameter", x);
          return emitRead(ref);
        }
        auto k = evalInt(sub->getIdx());
        if (param.kind == Param::Gbl && k && *k >= 0) return gblRead(param, *k);
        if (param.kind == Param::Idx && k && *k >= 0 && *k < rank) return indexValue(*k);
        if (param.kind == Param::Red) return fail("a reduction variable is read", x);
        return fail("a global or index parameter is subscripted by a value that is not a compile-time constant", x);
      }
      if (constPath(x).ok) return emitGlobalOrConstant(x);
      return std::nullopt;
    }
    if (const auto *un = llvm::dyn_cast<clang::UnaryOperator>(x); un && un->getOpcode() == clang::UO_Deref) {
      int p = paramIndex(un->getSubExpr());
      if (p >= 0) {
        Param &param = params[p];
        if (param.kind == Param::Gbl) return gblRead(param, 0);
        if (param.kind == Param::Idx) return indexValue(0);
        if (param.kind == Param::Red) return fail("a reduction variable is read", x);
      }
      return std::nullopt;
    }
    if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(x)) {
      if (llvm::isa<clang::ParmVarDecl>(dre->getDecl())) {
        int p = paramIndex(x);
        if (p < 0) return std::nullopt; // a parameter of an inlined helper function: bound by the generic walker
        if (params[p].kind == Param::Gbl && params[p].byValue) return gblRead(params[p], 0);
        return fail("kernel parameter '" + dre->getDecl()->getNameAsString() + "' is used in an unsupported way", x);
      }
      if (const auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl()); vd && vd->hasGlobalStorage()) return emitGlobalOrConstant(x);
      return std::nullopt;
    }
    if (llvm::isa<clang::MemberExpr>(x)) return emitGlobalOrConstant(x);
    return std::nullopt;
  }

  bool emitSpecialAssign(const clang::BinaryOperator *op) override {
    const clang::Expr *lhs = op->getLHS()->IgnoreParenImpCasts();
    // reduction
    if (int r = matchReduction(lhs); r >= 0) {
      ReduceUpdate u;
      switch (op->getOpcode()) {
      case clang::BO_AddAssign: u = {op->getRHS(), false, "add"}; break;
      case clang::BO_SubAssign: u = {op->getRHS(), true, "add"}; break;
      case clang::BO_Assign: {
        u = classifyAssignment(r, op->getRHS());
        if (!u.contribution) u = {op->getRHS(), false, ""};
        break;
      }
      default:
        fail("unsupported compound assignment to a reduction", op);
        return true;
      }
      emitReduce(r, u, op);
      return true;
    }
    // writes to a global array through a parameter
    if (const auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(lhs)) {
      int p = paramIndex(sub->getBase());
      if (p >= 0 && params[p].kind != Param::Acc) {
        fail("write to the global or reduction argument '" + params[p].decl->getNameAsString() +
                 "' (only accumulations into a reduction are supported)",
             op);
        return true;
      }
    }
    // accessor
    AccessRef ref = matchAccess(lhs);
    if (!ok()) return true;
    if (!ref.ok) return false;
    Param &p = params[ref.param];
    p.ncomp = std::max(p.ncomp, ref.comp + 1);
    if (p.isConst) {
      fail("write to the const accessor '" + p.decl->getNameAsString() + "'", op);
      return true;
    }
    auto off = constantOffsets(ref);
    if (!off) {
      fail("write to an accessor at a data-dependent offset", op);
      return true;
    }
    Value rhs = emit(op->getRHS());
    if (!rhs) return true;
    rhs = cast(rhs, p.elem);
    if (op->getOpcode() != clang::BO_Assign) {
      Value cur = emitRead(ref);
      if (!cur) return true;
      bool f = isa<FloatType>(p.elem);
      switch (op->getOpcode()) {
      case clang::BO_AddAssign: rhs = f ? Value(arith::AddFOp::create(b, loc, cur, rhs)) : Value(arith::AddIOp::create(b, loc, cur, rhs)); break;
      case clang::BO_SubAssign: rhs = f ? Value(arith::SubFOp::create(b, loc, cur, rhs)) : Value(arith::SubIOp::create(b, loc, cur, rhs)); break;
      case clang::BO_MulAssign: rhs = f ? Value(arith::MulFOp::create(b, loc, cur, rhs)) : Value(arith::MulIOp::create(b, loc, cur, rhs)); break;
      case clang::BO_DivAssign: rhs = f ? Value(arith::DivFOp::create(b, loc, cur, rhs)) : Value(arith::DivSIOp::create(b, loc, cur, rhs)); break;
      default:
        fail("unsupported compound assignment to an accessor", op);
        return true;
      }
    }
    p.writes.insert(std::vector<int64_t>(off->begin(), off->end()));
    WriteOp::create(b, loc, p.arg, b.getDenseI64ArrayAttr(*off), rhs, b.getI64IntegerAttr(ref.comp));
    return true;
  }

  bool emitSpecialStmt(const clang::Stmt *s) override {
    // if (x < *r) *r = x;   (a minimum or maximum spelled as a test)
    const auto *ifs = llvm::dyn_cast<clang::IfStmt>(s);
    if (!ifs || ifs->getElse() || ifs->getInit() || ifs->getConditionVariable()) return false;
    const auto *cmp = llvm::dyn_cast<clang::BinaryOperator>(ifs->getCond()->IgnoreParenImpCasts());
    if (!cmp || !cmp->isComparisonOp() || cmp->getOpcode() == clang::BO_EQ || cmp->getOpcode() == clang::BO_NE) return false;
    const clang::Stmt *thenStmt = ifs->getThen();
    if (const auto *comp = llvm::dyn_cast<clang::CompoundStmt>(thenStmt)) {
      if (comp->size() != 1) return false;
      thenStmt = comp->body_front();
    }
    const auto *assign = llvm::dyn_cast<clang::BinaryOperator>(thenStmt);
    if (!assign || assign->getOpcode() != clang::BO_Assign) return false;
    int r = matchReduction(assign->getLHS());
    if (r < 0) return false;
    const clang::Expr *x = assign->getRHS();
    const clang::Expr *l = cmp->getLHS(), *rr = cmp->getRHS();
    bool lr = matchReduction(l) == r && sameExpr(astCtx, rr, x), rl = matchReduction(rr) == r && sameExpr(astCtx, l, x);
    if (!lr && !rl) return false;
    bool less = cmp->getOpcode() == clang::BO_LT || cmp->getOpcode() == clang::BO_LE;
    // x < r -> r = x: minimum. r < x -> r = x: maximum.
    bool isMin = rl ? less : !less;
    emitReduce(r, {x, false, isMin ? "min" : "max"}, s);
    return true;
  }

  bool shouldUnroll(const clang::ForStmt *s, const clang::VarDecl *var) override {
    struct Mentions : clang::RecursiveASTVisitor<Mentions> {
      const clang::VarDecl *var = nullptr;
      bool hit = false;
      bool VisitDeclRefExpr(clang::DeclRefExpr *d) {
        if (d->getDecl() == var) hit = true;
        return true;
      }
    };
    struct Scan : clang::RecursiveASTVisitor<Scan> {
      OPSKernelWalker &w;
      const clang::VarDecl *var;
      bool found = false;
      Scan(OPSKernelWalker &w, const clang::VarDecl *var) : w(w), var(var) {}
      bool mentions(const clang::Expr *e) {
        Mentions m;
        m.var = var;
        m.TraverseStmt(const_cast<clang::Expr *>(e));
        return m.hit;
      }
      bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *c) {
        if (c->getOperator() != clang::OO_Call || c->getNumArgs() < 1) return true;
        int p = w.paramIndex(c->getArg(0));
        if (p < 0 || w.params[p].kind != Param::Acc) return true;
        for (unsigned i = 1; i < c->getNumArgs(); ++i)
          if (mentions(c->getArg(i))) found = true;
        return true;
      }
      bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *a) {
        int p = w.paramIndex(a->getBase());
        if (p >= 0 && (w.params[p].kind == Param::Gbl || w.params[p].kind == Param::Acc) && mentions(a->getIdx())) found = true;
        return true;
      }
    } scan(*this, var);
    scan.TraverseStmt(const_cast<clang::Stmt *>(s->getBody()));
    return scan.found;
  }

  bool dropCall(const clang::CallExpr *call) override {
    const clang::FunctionDecl *f = call->getDirectCallee();
    if (!f || !f->getDeclName().isIdentifier()) return false;
    llvm::StringRef name = f->getName();
    if (name != "printf" && name != "fprintf" && name != "puts" && name != "putchar" && name != "fflush" && name != "OPS_printf") return false;
    if (name == "printf" && call->getNumArgs() >= 1) {
      // A printf with a literal format string becomes ops.print (the runtime decides what to do with it).
      const auto *format = llvm::dyn_cast<clang::StringLiteral>(call->getArg(0)->IgnoreParenImpCasts());
      if (format && format->isOrdinary()) {
        SmallVector<Value> values;
        bool ok = true;
        for (unsigned i = 1; i < call->getNumArgs() && ok; ++i) {
          Value v = emit(call->getArg(i));
          if (!v) return true; // reported
          // Default argument promotions of C varargs.
          if (auto ft = dyn_cast<FloatType>(v.getType())) {
            if (ft.getWidth() < 64) v = cast(v, b.getF64Type());
          } else if (v.getType().isInteger(1) || (v.getType().isInteger() && v.getType().getIntOrFloatBitWidth() < 32)) {
            v = cast(v, b.getI32Type());
          } else if (!v.getType().isInteger(32) && !v.getType().isInteger(64)) {
            ok = false;
          }
          values.push_back(v);
        }
        if (ok) {
          ops::PrintOp::create(b, loc, b.getStringAttr(format->getString()), values);
          ++printed;
          return true;
        }
      }
    }
    ++dropped;
    warn("call to " + name.str() + " is dropped", call);
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Parameters
//===----------------------------------------------------------------------===//

/// What the body does with each pointer parameter, before anything is emitted.
struct UsageScan : clang::RecursiveASTVisitor<UsageScan> {
  clang::ASTContext &ctx;
  std::vector<Param> &params;
  int rank;
  std::set<const clang::ParmVarDecl *> accStyle; // indexed with __ops_acc
  UsageScan(clang::ASTContext &ctx, std::vector<Param> &params, int rank) : ctx(ctx), params(params), rank(rank) {}

  Param *find(const clang::Expr *e) {
    e = e->IgnoreParenImpCasts();
    if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(e))
      if (const auto *pv = llvm::dyn_cast<clang::ParmVarDecl>(dre->getDecl()))
        for (Param &p : params)
          if (p.decl == pv) return &p;
    return nullptr;
  }
  bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *sub) {
    Param *p = find(sub->getBase());
    if (!p) return true;
    if (const auto *call = llvm::dyn_cast<clang::CallExpr>(sub->getIdx()->IgnoreParenImpCasts())) {
      const clang::FunctionDecl *f = call->getDirectCallee();
      if (f && (f->getName() == "__ops_acc" || f->getName() == "__ops_acc_md")) {
        p->legacyMacro = true;
        accStyle.insert(p->decl);
      }
    }
    return true;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// The import
//===----------------------------------------------------------------------===//

std::string failedManifest(const ImportOptions &options, const std::string &reason) {
  llvm::json::Object o;
  o["manifest_version"] = 1;
  o["kernel"] = options.kernelName;
  o["status"] = "failed";
  o["reason"] = reason;
  o["importer"] = importerId();
  o["ndim"] = options.ndim;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(o))) << "\n";
  return text;
}

ImportResult importKernel(MLIRContext &ctx, const ImportOptions &options) {
  ImportResult result;
  auto failWith = [&](const std::string &reason) {
    result.ok = false;
    result.reason = reason;
    result.manifest = failedManifest(options, reason);
    return std::move(result);
  };
  if (options.ndim < 1 || options.ndim > 3) return failWith("the block dimension must be 1, 2 or 3");

  // --- the translation unit -------------------------------------------------------------------------------------
  std::string code, fileName;
  if (!options.sourceFile.empty()) {
    auto buffer = llvm::MemoryBuffer::getFile(options.sourceFile);
    if (!buffer) return failWith("cannot read " + options.sourceFile);
    code = (*buffer)->getBuffer().str();
    llvm::SmallString<256> abs(options.sourceFile);
    llvm::sys::fs::make_absolute(abs);
    fileName = std::string(abs);
  } else if (!options.headerFile.empty()) {
    llvm::SmallString<256> abs(options.headerFile);
    llvm::sys::fs::make_absolute(abs);
    code = "#include <ops_lib_core.h>\n#include \"" + std::string(abs) + "\"\n";
    fileName = "kernel_tu.cpp";
  } else {
    return failWith("neither a source file nor a header was given");
  }

  llvm::SmallString<128> macroPath;
  int fd;
  if (llvm::sys::fs::createTemporaryFile("ops-import-macros", "h", fd, macroPath)) return failWith("cannot create a temporary file");
  {
    llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
    os << accessMacros();
  }
  llvm::scope_exit removeMacros([&] { llvm::sys::fs::remove(macroPath); });

  std::string resourceDir = options.resourceDir.empty() ? OPS_CLANG_RESOURCE_DIR : options.resourceDir;
  std::vector<std::string> args = {"-x", "c++", "-std=c++17", "-ferror-limit=0", "-Wno-everything",
                                   "-DOPS_" + std::to_string(options.ndim) + "D=", "-DOPS_ACC_IGNORE", "-include",
                                   std::string(macroPath)};
  if (!resourceDir.empty()) args.push_back("-resource-dir=" + resourceDir);
  args.insert(args.end(), options.clangArgs.begin(), options.clangArgs.end());

  std::unique_ptr<clang::ASTUnit> unit = clang::tooling::buildASTFromCodeWithArgs(code, args, fileName);
  if (!unit) return failWith("clang could not parse " + fileName);
  {
    const clang::SourceManager &sm = unit->getSourceManager();
    for (auto it = sm.fileinfo_begin(); it != sm.fileinfo_end(); ++it) {
      std::string name = it->first.getName().str();
      if (!name.empty() && name != macroPath.str().str() && llvm::sys::fs::exists(name)) result.dependencies.push_back(name);
    }
  }

  struct Finder : clang::RecursiveASTVisitor<Finder> {
    llvm::StringRef name;
    const clang::FunctionDecl *found = nullptr;
    bool VisitFunctionDecl(clang::FunctionDecl *d) {
      if (d->getDeclName().isIdentifier() && d->doesThisDeclarationHaveABody() && d->getName() == name) found = d;
      return true;
    }
  } finder;
  finder.name = options.kernelName;
  finder.TraverseDecl(unit->getASTContext().getTranslationUnitDecl());
  if (!finder.found) return failWith("the definition of kernel '" + options.kernelName + "' was not found in " + fileName);
  const clang::FunctionDecl *decl = finder.found;
  clang::ASTContext &astCtx = unit->getASTContext();

  // A body that did not parse cleanly (an undeclared name...) has expressions we would misread: refuse it.
  struct Unresolved : clang::RecursiveASTVisitor<Unresolved> {
    bool found = false;
    bool VisitRecoveryExpr(clang::RecoveryExpr *) { return found = true, false; }
  } unresolved;
  unresolved.TraverseStmt(const_cast<clang::Stmt *>(decl->getBody()));
  if (unresolved.found || decl->isInvalidDecl()) return failWith("the body of the kernel has unresolved names (see the compiler errors)");

  // --- parameters ----------------------------------------------------------------------------------------------------
  if (!options.kinds.empty() && options.kinds.size() != decl->getNumParams())
    return failWith("the kernel has " + std::to_string(decl->getNumParams()) + " parameters, " + std::to_string(options.kinds.size()) + " kinds were given");
  Builder bld(&ctx);
  std::vector<Param> params(decl->getNumParams());
  for (unsigned i = 0; i < decl->getNumParams(); ++i) params[i].decl = decl->getParamDecl(i);

  UsageScan scan(astCtx, params, options.ndim);
  for (Param &p : params)
    if (isAccType(p.decl->getType())) p.kind = Param::Acc;
  if (!options.kinds.empty()) {
    for (size_t i = 0; i < params.size(); ++i) {
      const std::string &k = options.kinds[i];
      if (k == "acc") params[i].kind = Param::Acc;
      else if (k == "gbl") params[i].kind = Param::Gbl;
      else if (k == "red" || k == "reduce") params[i].kind = Param::Red;
      else if (k == "idx") params[i].kind = Param::Idx;
      else return failWith("unknown parameter kind '" + k + "'");
    }
    scan.TraverseStmt(const_cast<clang::Stmt *>(decl->getBody()));
  } else {
    // Pointer parameters: pre-scan for OPS_ACCn-style indexing to tell dats from globals and reductions.
    for (Param &p : params)
      if (!isAccType(p.decl->getType())) p.kind = Param::Gbl; // provisional
    scan.TraverseStmt(const_cast<clang::Stmt *>(decl->getBody()));
    for (Param &p : params) {
      clang::QualType t = p.decl->getType();
      if (isAccType(t)) continue;
      bool pointer = t->isPointerType();
      if (scan.accStyle.count(p.decl)) p.kind = Param::Acc;
      else if (pointer && p.decl->getName() == "idx") p.kind = Param::Idx;
      else if (pointer && !t->getPointeeType().isConstQualified()) p.kind = Param::Red;
      else p.kind = Param::Gbl;
    }
  }

  for (Param &p : params) {
    clang::QualType t = p.decl->getType();
    clang::QualType elem;
    if (isAccType(t)) {
      elem = accElement(t);
      p.isConst = t.getNonReferenceType().isConstQualified();
    } else if (t->isPointerType()) {
      elem = t->getPointeeType();
      p.isConst = elem.isConstQualified();
    } else {
      elem = t;
      p.byValue = true;
      p.isConst = true;
    }
    if (p.kind == Param::Idx) continue;
    if (elem.isNull()) return failWith("cannot determine the element type of parameter '" + p.decl->getNameAsString() + "'");
    clang::QualType canon = elem.getNonReferenceType().getUnqualifiedType().getCanonicalType();
    if (canon->isSpecificBuiltinType(clang::BuiltinType::BFloat16)) return failWith("bfloat16 is not supported (parameter '" + p.decl->getNameAsString() + "')");
    Type mlirType;
    if (canon->isHalfType() || canon->isSpecificBuiltinType(clang::BuiltinType::Float16)) mlirType = bld.getF16Type();
    else if (canon->isSpecificBuiltinType(clang::BuiltinType::Double)) mlirType = bld.getF64Type();
    else if (canon->isSpecificBuiltinType(clang::BuiltinType::Float)) mlirType = bld.getF32Type();
    else if (canon->isIntegerType() && !canon->isBooleanType() && !canon->isEnumeralType()) mlirType = bld.getIntegerType(astCtx.getTypeSize(canon));
    if (!mlirType || !isSupportedElementType(mlirType))
      return failWith("parameter '" + p.decl->getNameAsString() + "' has an element type that is not supported (" + elem.getAsString() + ")");
    p.elem = mlirType;
    if (p.kind == Param::Acc && p.byValue) return failWith("accessor parameter '" + p.decl->getNameAsString() + "' is passed by value");
    if (p.kind == Param::Red && p.byValue) return failWith("reduction parameter '" + p.decl->getNameAsString() + "' is passed by value");
  }

  // --- the kernel -------------------------------------------------------------------------------------------------
  OwningOpRef<ModuleOp> module = ModuleOp::create(bld.getUnknownLoc());
  auto signature = [&] {
    SmallVector<Type> inputTypes;
    for (Param &p : params) {
      switch (p.kind) {
      case Param::Acc: inputTypes.push_back(AccType::get(&ctx, p.elem, options.ndim, p.ncomp)); break;
      case Param::Gbl: inputTypes.push_back(GblType::get(&ctx, p.elem, p.gblDim)); break;
      case Param::Red: inputTypes.push_back(RedType::get(&ctx, p.elem)); break;
      case Param::Idx: break;
      }
    }
    return inputTypes;
  };
  OpBuilder mb(module->getBody(), module->getBody()->begin());
  auto kernel = KernelOp::create(mb, bld.getUnknownLoc(), options.kernelName, bld.getFunctionType(signature(), {}));
  Block *entry = kernel.addEntryBlock();
  unsigned arg = 0;
  for (Param &p : params)
    if (p.kind != Param::Idx) p.arg = entry->getArgument(arg++);

  OPSKernelWalker walker(ctx, astCtx, params, options.ndim);
  walker.builder().setInsertionPointToStart(entry);
  walker.emitBody(decl->getBody());
  for (const std::string &w : walker.warnings()) result.warnings.push_back(w);
  if (!walker.ok()) return failWith(walker.error());

  // The component counts and global sizes are known once the body has been walked (loops over components are
  // unrolled): give the arguments their final types.
  {
    SmallVector<Type> types = signature();
    for (auto [i, t] : llvm::enumerate(types)) entry->getArgument(i).setType(t);
    kernel.setFunctionType(bld.getFunctionType(types, {}));
  }
  walker.builder().setInsertionPointToEnd(entry);
  ReturnOp::create(walker.builder(), bld.getUnknownLoc(), ValueRange{});
  if (failed(verify(*module))) return failWith("the imported kernel does not verify");

  // Constants whose value shapes the code and must therefore be folded (role "bound"): those that reach a loop bound
  // or the index of an array constant or of a data-dependent read, and the arrays that are indexed.
  std::set<std::string> boundConsts;
  kernel.walk([&](ConstRefOp ref) {
    if (ref.getIndex()) boundConsts.insert(ref.getName().str());
    SmallVector<Value> work{ref.getResult()};
    DenseSet<Value> seen;
    while (!work.empty()) {
      Value v = work.pop_back_val();
      if (!seen.insert(v).second) continue;
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (auto loop = dyn_cast<scf::ForOp>(user)) {
          if (use.getOperandNumber() < 3) boundConsts.insert(ref.getName().str());
        } else if (auto other = dyn_cast<ConstRefOp>(user)) {
          if (other.getIndex() && use.get() == other.getIndex()) boundConsts.insert(ref.getName().str());
        } else if (isa<ReadDynOp>(user)) {
          boundConsts.insert(ref.getName().str());
        } else if (user->getDialect() && user->getDialect()->getNamespace() == "arith" && user->getNumResults() == 1) {
          work.push_back(user->getResult(0));
        }
      }
    }
  });

  // --- the manifest -------------------------------------------------------------------------------------------------
  auto offsetsJson = [](const std::set<std::vector<int64_t>> &set) {
    llvm::json::Array a;
    for (const auto &off : set) {
      llvm::json::Array o;
      for (int64_t v : off) o.push_back(v);
      a.push_back(std::move(o));
    }
    return a;
  };
  llvm::json::Array paramsJson;
  llvm::json::Object reads, writes;
  llvm::json::Array dynamicReads;
  for (auto [i, p] : llvm::enumerate(params)) {
    llvm::json::Object o;
    switch (p.kind) {
    case Param::Acc:
      o["kind"] = "acc";
      o["elem"] = typeName(p.elem);
      o["ncomp"] = p.ncomp;
      o["const"] = p.isConst;
      o["legacy_macro"] = p.legacyMacro;
      reads[std::to_string(i)] = offsetsJson(p.reads);
      writes[std::to_string(i)] = offsetsJson(p.writes);
      if (p.dynamicRead) dynamicReads.push_back(std::to_string(i));
      break;
    case Param::Gbl:
      o["kind"] = "gbl";
      o["elem"] = typeName(p.elem);
      o["dim"] = p.gblDim;
      break;
    case Param::Red:
      o["kind"] = "reduce";
      o["elem"] = typeName(p.elem);
      o["dim"] = 1;
      o["style"] = p.style;
      o["kind_hint"] = p.kindHint.empty() ? "none" : p.kindHint;
      break;
    case Param::Idx: o["kind"] = "idx"; break;
    }
    paramsJson.push_back(std::move(o));
  }
  llvm::json::Array constsJson;
  for (const ConstUse &c : walker.constUses()) {
    llvm::json::Object o;
    o["name"] = c.name;
    o["elem"] = typeName(c.elem);
    o["offset"] = c.offset;
    if (c.dynamicIndex) {
      o["array_stride"] = c.stride;
      o["dynamic_index"] = true;
    }
    if (boundConsts.count(c.name)) o["role"] = "bound";
    constsJson.push_back(std::move(o));
  }
  llvm::json::Array idxAxes;
  for (int d = 0; d < 3; ++d)
    if (walker.idxAxesRead() & (1u << d)) idxAxes.push_back(d);
  llvm::json::Object traits;
  traits["idx_axes_read"] = std::move(idxAxes);
  traits["dynamic_reads"] = std::move(dynamicReads);
  traits["dropped_printf"] = static_cast<int64_t>(walker.droppedPrintf());
  traits["prints"] = static_cast<int64_t>(walker.printfs());
  traits["reads"] = std::move(reads);
  traits["writes"] = std::move(writes);

  llvm::json::Object manifest;
  manifest["manifest_version"] = 1;
  manifest["kernel"] = options.kernelName;
  manifest["status"] = "ok";
  manifest["importer"] = importerId();
  manifest["ndim"] = options.ndim;
  manifest["params"] = std::move(paramsJson);
  manifest["consts"] = std::move(constsJson);
  manifest["traits"] = std::move(traits);

  // The digest identifies what the JIT will use: the module and everything in the manifest.
  std::string moduleText;
  {
    llvm::raw_string_ostream ts(moduleText);
    module->print(ts);
  }
  std::string body;
  llvm::raw_string_ostream bs(body);
  bs << llvm::formatv("{0}", llvm::json::Value(llvm::json::Object(manifest)));
  manifest["digest"] = opdsl::Hasher().add(moduleText).add(body).add(importerId()).hex();

  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest))) << "\n";
  result.ok = true;
  result.module = std::move(module);
  result.manifest = std::move(text);
  return result;
}

} // namespace ops_mlir::frontend
