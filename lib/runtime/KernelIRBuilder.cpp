//===- KernelIRBuilder.cpp - C++ kernel body -> MLIR ------------*- C++ -*-===//
// Tree walks a kernel's C++ AST to translate it into MLIR for the GPU backend, where
// the kernel body must become actual device code rather than a symbol resolved
// against a compiled host binary.
//
//===----------------------------------------------------------------------===//

// TODO: Support more than just `double` kernel parameters and locals.

#include "runtime/KernelIRBuilder.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/MemoryBuffer.h"

#include <map>

namespace ops_mlir {

namespace {

/// Shared empty placeholders for ExprEmitter's outFields_/outFieldValues_
/// parameters, used at call sites with no out-pointer context (the
/// single-`return`-expression kernel shape has no out-pointer at all)
static const llvm::SmallVector<const clang::FieldDecl *> kNoOutFields;
static const llvm::SmallVector<mlir::Value> kNoOutFieldValues;

class KernelFunctionFinder
    : public clang::RecursiveASTVisitor<KernelFunctionFinder> {
public:
  explicit KernelFunctionFinder(llvm::StringRef kernelName)
      : kernelName_(kernelName) {}

  bool VisitFunctionDecl(clang::FunctionDecl *decl) {
    // getName() asserts on non-identifier names (operators, constructors,
    // deduction guides, ...) -- <math.h> pulls in plenty of those via
    // <cmath> once parsed in C++ mode, so skip them explicitly.
    if (decl->getDeclName().isIdentifier() && decl->hasBody() &&
        decl->getName() == kernelName_)
      found_ = decl;
    // Keep walking in case of duplicate/shadowed declarations; the last
    // one with a body wins, matching normal C++ redeclaration semantics.
    return true;
  }

  clang::FunctionDecl *result() const { return found_; }

private:
  llvm::StringRef kernelName_;
  clang::FunctionDecl *found_ = nullptr;
};


/// Maps a <math.h> function name to the math dialect op that computes it.
/// Returns null for names outside the supported set.
using MathOpBuilder = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                      llvm::ArrayRef<mlir::Value>);

template <typename OpTy>
static mlir::Value buildUnaryMathOp(mlir::OpBuilder &b, mlir::Location loc,
                                    llvm::ArrayRef<mlir::Value> args) {
  return b.create<OpTy>(loc, args[0]);
}

template <typename OpTy>
static mlir::Value buildBinaryMathOp(mlir::OpBuilder &b, mlir::Location loc,
                                     llvm::ArrayRef<mlir::Value> args) {
  return b.create<OpTy>(loc, args[0], args[1]);
}

static MathOpBuilder lookupMathFunction(llvm::StringRef name) {
  static const llvm::StringMap<MathOpBuilder> kTable = {
      {"sin", &buildUnaryMathOp<mlir::math::SinOp>},
      {"cos", &buildUnaryMathOp<mlir::math::CosOp>},
      {"tan", &buildUnaryMathOp<mlir::math::TanOp>},
      {"asin", &buildUnaryMathOp<mlir::math::AsinOp>},
      {"acos", &buildUnaryMathOp<mlir::math::AcosOp>},
      {"atan", &buildUnaryMathOp<mlir::math::AtanOp>},
      {"sinh", &buildUnaryMathOp<mlir::math::SinhOp>},
      {"cosh", &buildUnaryMathOp<mlir::math::CoshOp>},
      {"tanh", &buildUnaryMathOp<mlir::math::TanhOp>},
      {"exp", &buildUnaryMathOp<mlir::math::ExpOp>},
      {"exp2", &buildUnaryMathOp<mlir::math::Exp2Op>},
      {"log", &buildUnaryMathOp<mlir::math::LogOp>},
      {"log2", &buildUnaryMathOp<mlir::math::Log2Op>},
      {"log10", &buildUnaryMathOp<mlir::math::Log10Op>},
      {"sqrt", &buildUnaryMathOp<mlir::math::SqrtOp>},
      {"fabs", &buildUnaryMathOp<mlir::math::AbsFOp>},
      {"floor", &buildUnaryMathOp<mlir::math::FloorOp>},
      {"ceil", &buildUnaryMathOp<mlir::math::CeilOp>},
      {"atan2", &buildBinaryMathOp<mlir::math::Atan2Op>},
      {"pow", &buildBinaryMathOp<mlir::math::PowFOp>},
  };
  auto it = kTable.find(name);
  return it == kTable.end() ? nullptr : it->second;
}

/// Recognises `*reduceParam = combiner(a, b)` where exactly one of a/b is
/// a dereference of reduceParam, and returns the other (non-self-
/// referential) argument.(I.e. '*error = fabs(*error, anew - a)')
/// The combiner's identity/kind (max/min/add) is already known from the OPS
/// dialect's access mode, so it is deliberately not matched by name here.
static const clang::Expr *extractReductionContribution(
    const clang::Expr *rhs, const clang::ParmVarDecl *reduceParam,
    llvm::raw_ostream &errs) {
  const auto *call = llvm::dyn_cast<clang::CallExpr>(rhs->IgnoreParenImpCasts());
  if (!call || call->getNumArgs() != 2) {
    errs << "KernelIRBuilder: reduction write to '"
        << reduceParam->getNameAsString()
        << "' must be `*param = combiner(a, b)` with exactly 2 arguments\n";
    return nullptr;
  }
  auto isDerefOfParam = [&](const clang::Expr *e) {
    const auto *unary =
        llvm::dyn_cast<clang::UnaryOperator>(e->IgnoreParenImpCasts());
    if (!unary || unary->getOpcode() != clang::UO_Deref)
      return false;
    const auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(
        unary->getSubExpr()->IgnoreParenImpCasts());
    return ref && ref->getDecl() == reduceParam;
  };
  bool arg0IsSelf = isDerefOfParam(call->getArg(0));
  bool arg1IsSelf = isDerefOfParam(call->getArg(1));
  if (arg0IsSelf == arg1IsSelf) {
    errs << "KernelIRBuilder: could not identify the self-referential "
            "operand in the reduction combiner for '"
        << reduceParam->getNameAsString() << "'\n";
    return nullptr;
  }
  return arg0IsSelf ? call->getArg(1) : call->getArg(0);
}

// TODO: Support more than just `double` kernel parameters and locals.
static bool isDoubleStructPointer(
    clang::QualType type,
    llvm::SmallVectorImpl<const clang::FieldDecl *> &outFieldOrder) {
  if (!type->isPointerType())
    return false;
  const clang::RecordType *rt = type->getPointeeType()->getAs<clang::RecordType>();
  if (!rt)
    return false;
  const clang::RecordDecl *rd = rt->getDecl();
  outFieldOrder.clear();
  for (const clang::FieldDecl *f : rd->fields()) {
    if (!f->getType()->isSpecificBuiltinType(clang::BuiltinType::Double))
      return false;
    outFieldOrder.push_back(f);
  }
  return !outFieldOrder.empty();
}

class ExprEmitter : public clang::ConstStmtVisitor<ExprEmitter, mlir::Value> {
public:
  ExprEmitter(mlir::OpBuilder &builder, mlir::Location loc,
              const std::map<const clang::ValueDecl *, mlir::Value> &values,
              const std::map<std::string, const void *> &constants,
              const clang::ParmVarDecl *outParam,
              const llvm::SmallVectorImpl<const clang::FieldDecl *> &outFields,
              const llvm::SmallVectorImpl<mlir::Value> &outFieldValues,
              llvm::raw_ostream &errs)
      : builder_(builder), loc_(loc), values_(values),
        constants_(constants), outParam_(outParam), outFields_(outFields),
        outFieldValues_(outFieldValues), errs_(errs) {}

  bool ok() const { return ok_; }

  mlir::Value VisitParenExpr(const clang::ParenExpr *expr) {
    return Visit(expr->getSubExpr());
  }

  mlir::Value VisitImplicitCastExpr(const clang::ImplicitCastExpr *expr) {
    mlir::Value sub = Visit(expr->getSubExpr());
    if (!ok_)
      return {};
    switch (expr->getCastKind()) {
    case clang::CK_LValueToRValue:
    case clang::CK_NoOp:
      return sub;
    case clang::CK_IntegralToFloating:
      return builder_.create<mlir::arith::SIToFPOp>(
          loc_, builder_.getF64Type(), sub);
    case clang::CK_FloatingCast:
      // KernelIRBuilder always materializes floating literals/results
      // as f64 directly (see VisitFloatingLiteral), so a float->double
      // promotion node has nothing left to do here.
      return sub;
    default:
      return fail(expr, "unsupported implicit cast");
    }
  }

  mlir::Value VisitDeclRefExpr(const clang::DeclRefExpr *expr) {
    // Covers both kernel parameters and local `double` variables declared
    // earlier in the body -- both get seeded into `values_` (params up
    // front, locals as their DeclStmt is walked; see StmtEmitter).
    auto it = values_.find(expr->getDecl());
    if (it != values_.end())
      return it->second;

    // If the kernel body references an extern global constant (e.g. `pi` or
    // `jmax`), look it up in the map of live addresses registered by the
    // app via JITEngine::registerKernelConstant. This mirrors enqueueParLoop
    // capturing a kernel's function pointer directly rather than resolving it
    // by symbol name later.
    if (const auto *var = llvm::dyn_cast<clang::VarDecl>(expr->getDecl())) {
      if (mlir::Value baked = bakeGlobalConstant(var))
        return baked;
    }

    return fail(expr, "reference to unsupported symbol '" +
                          expr->getDecl()->getNameAsString() +
                          "' (kernel bodies may only use their own "
                          "parameters, locals, <math.h> calls, and extern "
                          "globals registered via "
                          "JITEngine::registerKernelConstant)");
  }

  mlir::Value VisitFloatingLiteral(const clang::FloatingLiteral *expr) {
    double value = expr->getValueAsApproximateDouble();
    return builder_.create<mlir::arith::ConstantOp>(
        loc_, builder_.getF64FloatAttr(value));
  }

  mlir::Value VisitIntegerLiteral(const clang::IntegerLiteral *expr) {
    int64_t value = expr->getValue().getSExtValue();
    return builder_.create<mlir::arith::ConstantOp>(
        loc_, builder_.getI32IntegerAttr(static_cast<int32_t>(value)));
  }

  mlir::Value VisitUnaryMinus(const clang::UnaryOperator *expr) {
    mlir::Value sub = Visit(expr->getSubExpr());
    if (!ok_)
      return {};
    if (mlir::isa<mlir::FloatType>(sub.getType()))
      return builder_.create<mlir::arith::NegFOp>(loc_, sub);
    mlir::Value zero = builder_.create<mlir::arith::ConstantOp>(
        loc_, builder_.getI32IntegerAttr(0));
    return builder_.create<mlir::arith::SubIOp>(loc_, zero, sub);
  }

  mlir::Value VisitBinAdd(const clang::BinaryOperator *e) { return binOp(e); }
  mlir::Value VisitBinSub(const clang::BinaryOperator *e) { return binOp(e); }
  mlir::Value VisitBinMul(const clang::BinaryOperator *e) { return binOp(e); }
  mlir::Value VisitBinDiv(const clang::BinaryOperator *e) { return binOp(e); }

  mlir::Value VisitArraySubscriptExpr(const clang::ArraySubscriptExpr *expr) {
    const auto *base = llvm::dyn_cast<clang::DeclRefExpr>(
        expr->getBase()->IgnoreParenImpCasts());
    if (!base || !values_.count(base->getDecl()))
      return fail(expr, "array subscript base must be a kernel parameter");

    mlir::Value indexVal = Visit(expr->getIdx());
    if (!ok_)
      return {};
    mlir::Value index = builder_.create<mlir::arith::IndexCastOp>(
        loc_, builder_.getIndexType(), indexVal);
    return builder_.create<mlir::memref::LoadOp>(
        loc_, values_.at(base->getDecl()), mlir::ValueRange{index});
  }

  mlir::Value VisitCallExpr(const clang::CallExpr *expr) {
    const clang::FunctionDecl *callee = expr->getDirectCallee();
    bool hasSimpleName =
        callee && callee->getDeclName().isIdentifier();
    MathOpBuilder mathOp =
        hasSimpleName ? lookupMathFunction(callee->getName()) : nullptr;
    if (!mathOp) {
      return fail(expr, "call to unsupported function '" +
                            (callee ? callee->getNameAsString() : "<unknown>") +
                            "' (only <math.h> calls are supported)");
    }
    llvm::SmallVector<mlir::Value, 2> args;
    for (const clang::Expr *arg : expr->arguments()) {
      mlir::Value v = Visit(arg);
      if (!ok_)
        return {};
      args.push_back(v);
    }
    return mathOp(builder_, loc_, args);
  }

  mlir::Value VisitStmt(const clang::Stmt *stmt) {
    return fail(stmt, "unsupported expression construct");
  }

  mlir::Value VisitMemberExpr(const clang::MemberExpr *expr) {
    if (!expr->isArrow() || !outParam_)
      return fail(expr, "member access is only supported for the "
                        "out-pointer parameter's fields");
    const auto *base = llvm::dyn_cast<clang::DeclRefExpr>(
        expr->getBase()->IgnoreParenImpCasts());
    if (!base || base->getDecl() != outParam_)
      return fail(expr, "member access base must be the kernel's "
                        "out-pointer parameter");
    const auto *field = llvm::dyn_cast<clang::FieldDecl>(expr->getMemberDecl());
    auto it = field ? llvm::find(outFields_, field) : outFields_.end();
    if (it == outFields_.end())
      return fail(expr, "'" + expr->getMemberDecl()->getNameAsString() +
                            "' is read before being written in this "
                            "kernel (out-pointer fields can only be read "
                            "after they've been assigned earlier in the "
                            "same kernel body)");
    std::size_t idx = std::distance(outFields_.begin(), it);
    if (!outFieldValues_[idx])
      return fail(expr, "'" + expr->getMemberDecl()->getNameAsString() +
                            "' is read before being written in this "
                            "kernel (out-pointer fields can only be read "
                            "after they've been assigned earlier in the "
                            "same kernel body)");
    return outFieldValues_[idx];
  }

  mlir::Value fail(const clang::Stmt *at, const llvm::Twine &message) {
    if (ok_) {
      ok_ = false;
      errs_ << "KernelIRBuilder: " << message << "\n";
    }
    (void)at;
    return {};
  }

private:
  mlir::Value binOp(const clang::BinaryOperator *expr) {
    mlir::Value lhs = Visit(expr->getLHS());
    if (!ok_)
      return {};
    mlir::Value rhs = Visit(expr->getRHS());
    if (!ok_)
      return {};

    bool isFloat = mlir::isa<mlir::FloatType>(lhs.getType());
    switch (expr->getOpcode()) {
    case clang::BO_Add:
      return isFloat ? builder_.create<mlir::arith::AddFOp>(loc_, lhs, rhs)
                             .getResult()
                     : builder_.create<mlir::arith::AddIOp>(loc_, lhs, rhs)
                             .getResult();
    case clang::BO_Sub:
      return isFloat ? builder_.create<mlir::arith::SubFOp>(loc_, lhs, rhs)
                             .getResult()
                     : builder_.create<mlir::arith::SubIOp>(loc_, lhs, rhs)
                             .getResult();
    case clang::BO_Mul:
      return isFloat ? builder_.create<mlir::arith::MulFOp>(loc_, lhs, rhs)
                             .getResult()
                     : builder_.create<mlir::arith::MulIOp>(loc_, lhs, rhs)
                             .getResult();
    case clang::BO_Div:
      return isFloat ? builder_.create<mlir::arith::DivFOp>(loc_, lhs, rhs)
                             .getResult()
                     : builder_.create<mlir::arith::DivSIOp>(loc_, lhs, rhs)
                             .getResult();
    default:
      return fail(expr, "unsupported binary operator");
    }
  }

  // If the kernel body references an extern global constant (e.g. `pi` or
  // `jmax`), look it up in the map of live addresses registered by the
  // app via JITEngine::registerKernelConstant. This mirrors enqueueParLoop
  // capturing a kernel's function pointer directly rather than resolving it
  // by symbol name later.
  mlir::Value bakeGlobalConstant(const clang::VarDecl *var) {
    if (!var->hasGlobalStorage())
      return {};

    auto it = constants_.find(var->getNameAsString());
    if (it == constants_.end() || !it->second)
      return {};
    const void *addr = it->second;

    clang::QualType type = var->getType();
    if (type->isSpecificBuiltinType(clang::BuiltinType::Double)) {
      double value = *reinterpret_cast<const double *>(addr);
      return builder_.create<mlir::arith::ConstantOp>(
          loc_, builder_.getF64FloatAttr(value));
    }
    if (type->isSpecificBuiltinType(clang::BuiltinType::Int)) {
      int32_t value = *reinterpret_cast<const int32_t *>(addr);
      return builder_.create<mlir::arith::ConstantOp>(
          loc_, builder_.getI32IntegerAttr(value));
    }
    return fail(nullptr, "global '" + var->getNameAsString() +
                             "' has an unsupported type (only extern "
                             "double/int globals can be baked in)");
  }

  mlir::OpBuilder &builder_;
  mlir::Location loc_;
  const std::map<const clang::ValueDecl *, mlir::Value> &values_;
  const std::map<std::string, const void *> &constants_;
  llvm::raw_ostream &errs_;
  const clang::ParmVarDecl *outParam_;
  const llvm::SmallVectorImpl<const clang::FieldDecl *> &outFields_;
  const llvm::SmallVectorImpl<mlir::Value> &outFieldValues_;
  bool ok_ = true;
};

// TODO: Support more than just `double` kernel parameters and locals.
/// Walks the body of a `void` out-pointer kernel: a sequence of
///   double <name> = <expr>;             (locals, evaluated once and cached)
///   <out>-><field> = <expr>;             (stores into the out-pointer's memref)
/// in source order. Unlike ExprEmitter's single-`return <expr>;` kernels,
/// these have multiple statements and must thread newly-declared locals
/// forward to later statements that reference them.
class StmtEmitter {
public:
  StmtEmitter(mlir::OpBuilder &builder, mlir::Location loc,
              std::map<const clang::ValueDecl *, mlir::Value> &values,
              const std::map<std::string, const void *> &constants,
              const clang::ParmVarDecl *outParam, mlir::Value outMemref,
              const llvm::SmallVectorImpl<const clang::FieldDecl *> &outFields,
              llvm::raw_ostream &errs)
      : builder_(builder), loc_(loc), values_(values), constants_(constants),
        outParam_(outParam), outMemref_(outMemref), outFields_(outFields),
        outFieldValues_(outFields.size()), errs_(errs) {}

  bool ok() const { return ok_; }

  void run(const clang::CompoundStmt *body) {
    for (const clang::Stmt *stmt : body->body()) {
      if (!ok_)
        return;
      if (const auto *declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
        visitDeclStmt(declStmt);
      } else if (const auto *ret = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
        if (ret->getRetValue())
          fail(ret, "a void out-pointer kernel's `return;` must not carry a value");
      } else if (const auto *bin =
                     llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
        visitAssign(bin);
      } else {
        fail(stmt, "unsupported statement in a multi-statement kernel body "
                    "(only `double x = <expr>;` locals and `out->field = "
                    "<expr>;` stores are supported)");
      }
    }
  }

private:
  void visitDeclStmt(const clang::DeclStmt *declStmt) {
    for (const clang::Decl *d : declStmt->decls()) {
      if (!ok_)
        return;
      const auto *var = llvm::dyn_cast<clang::VarDecl>(d);
      if (!var || !var->getType()->isSpecificBuiltinType(clang::BuiltinType::Double)) {
        fail(declStmt, "local variable declarations must be `double`");
        return;
      }
      if (!var->hasInit()) {
        fail(declStmt, "local '" + var->getNameAsString() +
                            "' must be initialized at declaration");
        return;
      }
      ExprEmitter emitter(builder_, loc_, values_, constants_, outParam_, outFields_, outFieldValues_, errs_);
      mlir::Value v = emitter.Visit(var->getInit());
      if (!emitter.ok()) {
        ok_ = false;
        return;
      }
      values_[var] = v;
    }
  }

  void visitAssign(const clang::BinaryOperator *bin) {
    if (bin->getOpcode() != clang::BO_Assign) {
      fail(bin, "only assignment statements are supported at statement level");
      return;
    }
    const auto *member =
        llvm::dyn_cast<clang::MemberExpr>(bin->getLHS()->IgnoreParenImpCasts());
    if (!member || !member->isArrow()) {
      fail(bin, "assignment target must be `out->field`");
      return;
    }
    const auto *base = llvm::dyn_cast<clang::DeclRefExpr>(
        member->getBase()->IgnoreParenImpCasts());
    if (!base || base->getDecl() != outParam_) {
      fail(bin, "assignment target's base must be the kernel's out-pointer "
                "parameter");
      return;
    }
    const auto *field = llvm::dyn_cast<clang::FieldDecl>(member->getMemberDecl());
    auto it = field ? llvm::find(outFields_, field) : outFields_.end();
    if (!field || it == outFields_.end()) {
      fail(bin, "'" + member->getMemberDecl()->getNameAsString() +
                    "' is not a field of the out-pointer's struct");
      return;
    }
    std::size_t fieldIndex = std::distance(outFields_.begin(), it);

    ExprEmitter emitter(builder_, loc_, values_, constants_, outParam_, outFields_, outFieldValues_, errs_);
    mlir::Value rhs = emitter.Visit(bin->getRHS());
    if (!emitter.ok()) {
      ok_ = false;
      return;
    }

    mlir::Value index = builder_.create<mlir::arith::ConstantOp>(
        loc_, builder_.getIndexAttr(static_cast<int64_t>(fieldIndex)));
    builder_.create<mlir::memref::StoreOp>(loc_, rhs, outMemref_,
                                           mlir::ValueRange{index});
    outFieldValues_[fieldIndex] = rhs;  
  }

  void fail(const clang::Stmt *, const llvm::Twine &message) {
    if (ok_) {
      ok_ = false;
      errs_ << "KernelIRBuilder: " << message << "\n";
    }
  }

  mlir::OpBuilder &builder_;
  mlir::Location loc_;
  std::map<const clang::ValueDecl *, mlir::Value> &values_;
  const std::map<std::string, const void *> &constants_;
  const clang::ParmVarDecl *outParam_;
  mlir::Value outMemref_;
  const llvm::SmallVectorImpl<const clang::FieldDecl *> &outFields_;
  llvm::SmallVector<mlir::Value> outFieldValues_;
  llvm::raw_ostream &errs_;
  bool ok_ = true;
};

} // namespace

//===----------------------------------------------------------------------===//
// KernelIRBuilder
//===----------------------------------------------------------------------===//

mlir::func::FuncOp KernelIRBuilder::generate(
    const std::string &sourceFile, const std::string &kernelName,
    int indexRank, const std::map<std::string, const void *> &constants,
    const std::vector<int> &constArgDims,
    llvm::raw_ostream &errs) {
  auto codeOrErr = llvm::MemoryBuffer::getFile(sourceFile);
  if (!codeOrErr) {
    errs << "KernelIRBuilder: could not read '" << sourceFile
        << "': " << codeOrErr.getError().message() << "\n";
    return nullptr;
  }

  // -x c++: force C++ mode explicitly (buildASTFromCodeWithArgs infers the
  // language from sourceFile's extension -- .h means C -- which conflicts
  // with -std=c++17). -resource-dir: needed since building the AST via
  // this library API bypasses the driver logic that normally locates
  // Clang's own bundled headers (stddef.h etc.) relative to itself.
  std::unique_ptr<clang::ASTUnit> unit = clang::tooling::buildASTFromCodeWithArgs(
      (*codeOrErr)->getBuffer(),
      /*Args=*/{"-x", "c++", "-std=c++17",
               "-resource-dir=" OPS_CLANG_RESOURCE_DIR},
      sourceFile);
  if (!unit) {
    errs << "KernelIRBuilder: failed to parse '" << sourceFile << "'\n";
    return nullptr;
  }

  KernelFunctionFinder finder(kernelName);
  finder.TraverseDecl(unit->getASTContext().getTranslationUnitDecl());
  clang::FunctionDecl *decl = finder.result();
  if (!decl) {
    errs << "KernelIRBuilder: no definition of '" << kernelName << "' in "
        << sourceFile << "\n";
    return nullptr;
  }

  // Two supported kernel shapes:
  //  - `double kernel(...)`: a single `return <expr>;` (one write).
  //  - `void kernel(..., Result *out)`: a sequence of `double x = <expr>;`
  //     locals and `out->field = <expr>;` stores (more than one write) --
  //     see opensbliblock00Kernel039's out-pointer comment for why.
  bool isVoidReturn = decl->getReturnType()->isVoidType();
  bool isDoubleReturn =
      decl->getReturnType()->isSpecificBuiltinType(clang::BuiltinType::Double);
  if (!isVoidReturn && !isDoubleReturn) {
    errs << "KernelIRBuilder: '" << kernelName
        << "' must return double (single write) or void (multi-write, via "
           "an out-pointer parameter)\n";
    return nullptr;
  }

  mlir::OpBuilder builder(&context_);
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type, 4> paramTypes;
  llvm::SmallVector<const clang::ParmVarDecl *, 2> reduceParams;
  llvm::SmallVector<const clang::ParmVarDecl *, 2> constParams;
  const clang::ParmVarDecl *outParam = nullptr;
  llvm::SmallVector<const clang::FieldDecl *, 8> outFields;
  for (const clang::ParmVarDecl *param : decl->parameters()) {
    clang::QualType type = param->getType();
    llvm::SmallVector<const clang::FieldDecl *, 8> fields;
    if (type->isSpecificBuiltinType(clang::BuiltinType::Double)) {
      paramTypes.push_back(builder.getF64Type());
    } else if (type->isPointerType() && type->getPointeeType()->isSpecificBuiltinType(clang::BuiltinType::Double) && type->getPointeeType().isConstQualified()) {
      // Read-only broadcast constant (const double *name)
      // Indexed via array subscript and never written through
      std::size_t constIndex = constParams.size();
      if (constIndex >= constArgDims.size()) {
        errs << "KernelIRBuilder: no dimension registered for const "
                "parameter #" << constIndex << " ('" << param->getNameAsString()
            << "') of '" << kernelName << "' -- expected " << constArgDims.size()
            << " const parameter(s) worth of dims\n";
        return nullptr;
      }
      constParams.push_back(param);
      paramTypes.push_back(mlir::MemRefType::get({constArgDims[constIndex]}, builder.getF64Type()));

    } else if (type->isPointerType() && type->getPointeeType()->isSpecificBuiltinType(clang::BuiltinType::Double)) {
      // Reduction handle (double *name) - non-const
      reduceParams.push_back(param);
      paramTypes.push_back(mlir::MemRefType::get({}, builder.getF64Type()));
    } else if (type->isPointerType() &&
              type->getPointeeType()->isSpecificBuiltinType(
                  clang::BuiltinType::Int)) {
      paramTypes.push_back(
          mlir::MemRefType::get({indexRank}, builder.getI32Type()));
    } else if (isDoubleStructPointer(type, fields)) {
      if (outParam) {
        errs << "KernelIRBuilder: '" << kernelName
            << "' has more than one out-pointer parameter\n";
        return nullptr;
      }
      outParam = param;
      outFields = std::move(fields);
      paramTypes.push_back(mlir::MemRefType::get(
          {static_cast<int64_t>(outFields.size())}, builder.getF64Type()));
    } else {
      errs << "KernelIRBuilder: unsupported parameter type for '"
          << kernelName << "' (only double, const int*, and a pointer to a "
                            "struct of doubles are supported)\n";
      return nullptr;
    }
  }

  if (isVoidReturn && !outParam) {
    errs << "KernelIRBuilder: '" << kernelName
        << "' returns void but has no out-pointer parameter to write "
           "results through\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Type, 2> resultTypeVec;
  if (isDoubleReturn) {
    resultTypeVec.push_back(builder.getF64Type());
    resultTypeVec.append(reduceParams.size(), builder.getF64Type());
  }
  mlir::TypeRange resultTypes =
      isDoubleReturn ? mlir::TypeRange{resultTypeVec} : mlir::TypeRange{};
  auto funcType = builder.getFunctionType(paramTypes, resultTypes);
  auto funcOp = mlir::func::FuncOp::create(loc, kernelName, funcType);
  funcOp.setPrivate();
  mlir::Block *entry = funcOp.addEntryBlock();

  std::map<const clang::ValueDecl *, mlir::Value> values;
  for (auto [param, arg] : llvm::zip(decl->parameters(), entry->getArguments()))
    values[param] = arg;

  builder.setInsertionPointToStart(entry);

  const auto *body = llvm::dyn_cast<clang::CompoundStmt>(decl->getBody());
  if (!body) {
    errs << "KernelIRBuilder: '" << kernelName << "' has no body\n";
    funcOp.erase();
    return nullptr;
  }

  if (isDoubleReturn) {
    const clang::ReturnStmt *ret = nullptr;
    std::map<const clang::ParmVarDecl *, mlir::Value> contributions;

    for (const clang::Stmt *stmt : body->body()) {
      if (const auto *declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
        for (const clang::Decl *d : declStmt->decls()) {
          const auto *var = llvm::dyn_cast<clang::VarDecl>(d);
          if (!var || !var->hasInit() ||
              !var->getType()->isSpecificBuiltinType(clang::BuiltinType::Double)) {
            errs << "KernelIRBuilder: local declarations must be initialized "
                    "`double`\n";
            funcOp.erase();
            return nullptr;
          }
          ExprEmitter emitter(builder, loc, values, constants, /*outParam=*/nullptr, kNoOutFields, kNoOutFieldValues, errs);
          mlir::Value v = emitter.Visit(var->getInit());
          if (!emitter.ok()) { funcOp.erase(); return nullptr; }
          values[var] = v;
        }
      } else if (const auto *bin = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
        if (bin->getOpcode() != clang::BO_Assign) {
          errs << "KernelIRBuilder: unsupported statement (only locals, "
                  "reduction writes, and a final return are supported)\n";
          funcOp.erase();
          return nullptr;
        }
        const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(
            bin->getLHS()->IgnoreParenImpCasts());
        const clang::ParmVarDecl *target = nullptr;
        if (unary && unary->getOpcode() == clang::UO_Deref) {
          if (const auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(
                  unary->getSubExpr()->IgnoreParenImpCasts()))
            target = llvm::dyn_cast<clang::ParmVarDecl>(ref->getDecl());
        }
        if (!target || !llvm::is_contained(reduceParams, target)) {
          errs << "KernelIRBuilder: assignment target must be a reduction "
                  "parameter dereference (`*error = ...`)\n";
          funcOp.erase();
          return nullptr;
        }
        const clang::Expr *contribExpr =
            extractReductionContribution(bin->getRHS(), target, errs);
        if (!contribExpr) { funcOp.erase(); return nullptr; }

        ExprEmitter emitter(builder, loc, values, constants, /*outParam=*/nullptr, kNoOutFields, kNoOutFieldValues, errs);
        mlir::Value v = emitter.Visit(contribExpr);
        if (!emitter.ok()) { funcOp.erase(); return nullptr; }
        contributions[target] = v;
      } else if (const auto *r = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
        ret = r;
      } else {
        errs << "KernelIRBuilder: unsupported statement in kernel body\n";
        funcOp.erase();
        return nullptr;
      }
    }

    if (!ret || !ret->getRetValue()) {
      errs << "KernelIRBuilder: '" << kernelName
          << "' must end with `return <expr>;`\n";
      funcOp.erase();
      return nullptr;
    }
    ExprEmitter retEmitter(builder, loc, values, constants, /*outParam=*/nullptr, kNoOutFields, kNoOutFieldValues, errs);
    mlir::Value mainResult = retEmitter.Visit(ret->getRetValue());
    if (!retEmitter.ok()) { funcOp.erase(); return nullptr; }

    llvm::SmallVector<mlir::Value, 2> results{mainResult};
    for (const clang::ParmVarDecl *rp : reduceParams) {
      auto it = contributions.find(rp);
      if (it == contributions.end()) {
        errs << "KernelIRBuilder: reduction parameter '" << rp->getNameAsString()
            << "' was never written\n";
        funcOp.erase();
        return nullptr;
      }
      results.push_back(it->second);
    }

    builder.create<mlir::func::ReturnOp>(loc, results);
    return funcOp;
  }

  StmtEmitter stmtEmitter(builder, loc, values, constants, outParam,
                          values.at(outParam), outFields, errs);
  stmtEmitter.run(body);
  if (!stmtEmitter.ok()) {
    funcOp.erase();
    return nullptr;
  }

  builder.create<mlir::func::ReturnOp>(loc);
  return funcOp;
}

} // namespace ops_mlir
