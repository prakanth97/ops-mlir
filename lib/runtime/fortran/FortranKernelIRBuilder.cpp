//===- FortranKernelIRBuilder.cpp - Fortran kernel body -> MLIR -----------===//
//
// Part of OPS-MLIR Project
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
// Lowers an OPS Fortran kernel module to FIR with flang's frontend, run
// in-process the way KernelIRBuilder runs clang, then rewrites the kernel
// procedure into the `func.func` shape the stencil lowering expects.
//
// The supported subset mirrors KernelIRBuilder's: a straight-line body with
// no control flow, made of scalar arithmetic, intrinsic math calls, reads of
// the kernel's own dummy arguments, and reads of module constants registered
// through JITEngine::registerKernelConstant. That is enough for OPS stencil
// kernels, which are pure per-point expressions by construction.
//
// The translation is a linear walk rather than a dialect conversion because
// Fortran's pass-by-reference ABI has to disappear entirely: FIR passes
// scalars as `!fir.ref<f64>` and returns results through an `fir.alloca`
// slot, while the stencil lowering calls the kernel with plain `f64` values.
// Modelling each alloca as an SSA value ("slot") collapses that indirection
// without needing a memory dialect on the device side.
//
//===----------------------------------------------------------------------===//

#include "runtime/fortran/FortranKernelIRBuilder.h"

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/TextDiagnosticBuffer.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdint>
#include <cstring>
#include <mutex>

namespace ops_mlir {

bool fortranFIRImportAvailable() { return true; }

namespace {

//===----------------------------------------------------------------------===//
// Symbol matching
//===----------------------------------------------------------------------===//

/// Flang mangles a module procedure as `_QM<module>P<name>` and a free
/// procedure as `_QP<name>`; module variables become `_QM<module>E<name>`.
/// Rather than reproduce the mangling scheme, match on the trailing
/// `<tag><lowercased name>` to map a kernel onto its FIR symbol.
bool symbolMatches(llvm::StringRef symbol, llvm::StringRef name, char tag) {
  const std::string folded = name.lower();
  const std::string lowered = symbol.lower();
  if (lowered == folded)
    return true;
  // The tag is written as it appears in the mangled name ('P', 'E'), but the
  // comparison is case-insensitive, so fold it too.
  const std::string lowTag(1, static_cast<char>(llvm::toLower(tag)));
  // Flang spells a module `parameter` with a `C` before the name tag
  // (`_QM<module>ECpi`), so accept an optional one.
  return llvm::StringRef(lowered).ends_with(lowTag + folded) ||
         llvm::StringRef(lowered).ends_with(lowTag + "c" + folded);
}

/// Maps a Fortran intrinsic (as flang spells it in an emitted call) to the
/// math dialect op computing it. Modern flang lowers most real intrinsics
/// straight to the math dialect, but some still arrive as calls to libm or
/// to a Fortran runtime entry point, so both spellings are accepted.
using MathOpBuilder = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                      llvm::ArrayRef<mlir::Value>);

template <typename OpTy>
mlir::Value buildUnaryMathOp(mlir::OpBuilder &b, mlir::Location loc,
                             llvm::ArrayRef<mlir::Value> args) {
  return b.create<OpTy>(loc, args[0]);
}

template <typename OpTy>
mlir::Value buildBinaryMathOp(mlir::OpBuilder &b, mlir::Location loc,
                              llvm::ArrayRef<mlir::Value> args) {
  return b.create<OpTy>(loc, args[0], args[1]);
}

MathOpBuilder lookupMathFunction(llvm::StringRef name) {
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
      {"log", &buildUnaryMathOp<mlir::math::LogOp>},
      {"log10", &buildUnaryMathOp<mlir::math::Log10Op>},
      {"sqrt", &buildUnaryMathOp<mlir::math::SqrtOp>},
      {"abs", &buildUnaryMathOp<mlir::math::AbsFOp>},
      {"floor", &buildUnaryMathOp<mlir::math::FloorOp>},
      {"ceiling", &buildUnaryMathOp<mlir::math::CeilOp>},
      {"atan2", &buildBinaryMathOp<mlir::math::Atan2Op>},
      {"pow", &buildBinaryMathOp<mlir::math::PowFOp>},
  };

  // Strip the decorations flang puts on emitted intrinsic calls: a libm
  // precision suffix (`sinf`), or a Fortran runtime prefix
  // (`_FortranASin`). What is left is the bare intrinsic name.
  llvm::StringRef bare = name;
  bare.consume_front("_Fortran");
  if (bare.starts_with("A") && bare.size() > 1 && llvm::isUpper(bare[1]))
    bare = bare.drop_front();
  std::string folded = bare.lower();
  if (auto it = kTable.find(folded); it != kTable.end())
    return it->second;
  // `sinf`/`sinl` and friends: drop a trailing precision marker.
  if (folded.size() > 1 && (folded.back() == 'f' || folded.back() == 'l')) {
    if (auto it = kTable.find(folded.substr(0, folded.size() - 1));
        it != kTable.end())
      return it->second;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// FIR body importer
//===----------------------------------------------------------------------===//

class FIRBodyImporter {
public:
  FIRBodyImporter(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::ModuleOp artifactModule,
                  const std::map<std::string, const void *> &constants,
                  llvm::raw_ostream &errs)
      : builder_(builder), loc_(loc), artifact_(artifactModule),
        constants_(constants), errs_(errs) {}

  bool ok() const { return ok_; }

  /// Binds a FIR dummy argument to the value the OPS ABI supplies for it.
  /// Scalars arrive by value, so a later `fir.load` of the reference simply
  /// yields this value; an index array arrives as a memref and is loaded
  /// from properly.
  void bindArgument(mlir::Value firArg, mlir::Value opsValue, bool isMemRef) {
    if (isMemRef) {
      memrefs_[firArg] = opsValue;
    } else {
      slots_[firArg] = opsValue;
      // Fortran dummies are by-reference, so assigning to one would be
      // visible to the caller -- but the stencil lowering passes scalars by
      // value, so such a write has nowhere to go. Record the binding to
      // reject it rather than silently drop it.
      argumentSlots_.insert(firArg);
    }
  }

  void run(mlir::Block &body) {
    for (mlir::Operation &op : body) {
      if (!ok_)
        return;
      translate(&op);
    }
  }

  /// The value the translated `func.return` should produce, if any.
  mlir::Value result() const { return result_; }
  bool sawReturn() const { return sawReturn_; }

private:
  //===--------------------------------------------------------------------===//

  void translate(mlir::Operation *op) {
    if (auto alloca = mlir::dyn_cast<fir::AllocaOp>(op)) {
      // A local variable, including the implicit slot holding a function's
      // result. It starts undefined; a `fir.store` gives it a value.
      slots_[alloca.getResult()] = mlir::Value();
      return;
    }
    if (auto declare = mlir::dyn_cast<fir::DeclareOp>(op)) {
      // `fir.declare` only attaches source-level information; alias its
      // result to whatever it is describing.
      aliasTo(declare.getResult(), declare.getMemref());
      return;
    }
    if (auto load = mlir::dyn_cast<fir::LoadOp>(op)) {
      translateLoad(load);
      return;
    }
    if (auto store = mlir::dyn_cast<fir::StoreOp>(op)) {
      translateStore(store);
      return;
    }
    if (auto addrOf = mlir::dyn_cast<fir::AddrOfOp>(op)) {
      globals_[addrOf.getResult()] = addrOf.getSymbol().getRootReference().getValue();
      return;
    }
    if (auto coord = mlir::dyn_cast<fir::CoordinateOp>(op)) {
      translateCoordinate(coord.getResult(), coord.getRef(),
                          coord.getCoor(), op);
      return;
    }
    if (auto arrayCoor = mlir::dyn_cast<fir::ArrayCoorOp>(op)) {
      // `fir.array_coor` indices are one-based Fortran subscripts.
      translateCoordinate(arrayCoor.getResult(), arrayCoor.getMemref(),
                          arrayCoor.getIndices(), op, /*oneBased=*/true);
      return;
    }
    if (auto convert = mlir::dyn_cast<fir::ConvertOp>(op)) {
      translateConvert(convert);
      return;
    }
    if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
      translateCall(call);
      return;
    }
    if (auto noReassoc = mlir::dyn_cast<fir::NoReassocOp>(op)) {
      // Marks a parenthesised subexpression so later passes will not
      // reassociate it. It computes nothing, so forward its operand.
      mlir::Value value = mapped(noReassoc.getVal(), op);
      if (ok_)
        values_[noReassoc.getResult()] = value;
      return;
    }
    if (mlir::isa<fir::ShapeOp, fir::ShapeShiftOp, fir::UndefOp,
                  fir::DummyScopeOp>(op)) {
      // Shape/scope bookkeeping carries no computation for the explicit-shape
      // dummies this subset allows.
      return;
    }
    if (mlir::isa<mlir::func::ReturnOp>(op)) {
      sawReturn_ = true;
      if (op->getNumOperands() == 1)
        result_ = mapped(op->getOperand(0), op);
      return;
    }
    if (op->getDialect() &&
        (op->getDialect()->getNamespace() == "arith" ||
         op->getDialect()->getNamespace() == "math")) {
      cloneWithMappedOperands(op);
      return;
    }
    fail(op, "unsupported FIR operation '" +
                 op->getName().getStringRef().str() + "'");
  }

  //===--------------------------------------------------------------------===//

  void translateLoad(fir::LoadOp load) {
    mlir::Value ref = resolve(load.getMemref());

    if (auto slot = slots_.find(ref); slot != slots_.end()) {
      if (!slot->second) {
        fail(load, "read of an uninitialized local variable");
        return;
      }
      values_[load.getResult()] = slot->second;
      return;
    }
    if (auto element = elements_.find(ref); element != elements_.end()) {
      values_[load.getResult()] = element->second;
      return;
    }
    if (auto global = globals_.find(ref); global != globals_.end()) {
      if (mlir::Value baked = bakeGlobal(global->second, load.getType(), load))
        values_[load.getResult()] = baked;
      return;
    }
    fail(load, "load from an unsupported location (kernel bodies may only "
               "read their own arguments, locals, and registered module "
               "constants)");
  }

  void translateStore(fir::StoreOp store) {
    mlir::Value ref = resolve(store.getMemref());
    auto slot = slots_.find(ref);
    if (slot == slots_.end()) {
      fail(store, "store to an unsupported location (kernel bodies may only "
                  "write their own locals and result variable)");
      return;
    }
    if (argumentSlots_.contains(ref)) {
      fail(store, "assignment to a kernel argument is not supported -- an OPS "
                  "kernel must return its result, not write it back through a "
                  "dummy argument");
      return;
    }
    mlir::Value value = mapped(store.getValue(), store);
    if (!ok_)
      return;
    slot->second = value;
  }

  /// Resolves `array(i)` into a load from the memref the OPS ABI supplied.
  /// Only the index-array argument reaches here, so a single subscript is
  /// enough.
  void translateCoordinate(mlir::Value result, mlir::Value base,
                           mlir::ValueRange indices, mlir::Operation *op,
                           bool oneBased = false) {
    auto memref = memrefs_.find(resolve(base));
    if (memref == memrefs_.end()) {
      fail(op, "array access on an unsupported object (only the index "
               "argument may be subscripted)");
      return;
    }
    if (indices.size() != 1) {
      fail(op, "only rank-one array access is supported");
      return;
    }

    mlir::Value index = mapped(indices.front(), op);
    if (!ok_)
      return;
    if (!mlir::isa<mlir::IndexType>(index.getType()))
      index = builder_.create<mlir::arith::IndexCastOp>(
          loc_, builder_.getIndexType(), index);
    if (oneBased) {
      // Fortran subscripts start at 1; the memref the OPS ABI hands over is
      // zero-based, exactly as the C++ kernel's `idx[]` parameter is.
      mlir::Value one =
          builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
      index = builder_.create<mlir::arith::SubIOp>(loc_, index, one);
    }
    elements_[result] = builder_.create<mlir::memref::LoadOp>(
        loc_, memref->second, mlir::ValueRange{index});
  }

  void translateConvert(fir::ConvertOp convert) {
    // A conversion whose operand is a reference is pure bookkeeping (FIR
    // uses it to reinterpret pointers); forward the aliasing instead of
    // emitting an op.
    if (mlir::isa<fir::ReferenceType>(convert.getValue().getType())) {
      aliasTo(convert.getResult(), convert.getValue());
      return;
    }

    mlir::Value operand = mapped(convert.getValue(), convert);
    if (!ok_)
      return;
    mlir::Type target = convert.getType();
    mlir::Value converted = castValue(operand, target, convert);
    if (converted)
      values_[convert.getResult()] = converted;
  }

  void translateCall(fir::CallOp call) {
    llvm::StringRef callee;
    if (auto symbol = call.getCallee())
      callee = symbol->getRootReference().getValue();
    MathOpBuilder mathOp = callee.empty() ? nullptr : lookupMathFunction(callee);
    if (!mathOp) {
      fail(call, "call to unsupported procedure '" + callee.str() +
                     "' (only intrinsic math functions are supported)");
      return;
    }

    llvm::SmallVector<mlir::Value, 2> args;
    for (mlir::Value operand : call.getArgs()) {
      mlir::Value value = mapped(operand, call);
      if (!ok_)
        return;
      args.push_back(value);
    }
    if (call.getNumResults() != 1) {
      fail(call, "intrinsic call must produce exactly one result");
      return;
    }
    values_[call.getResult(0)] = mathOp(builder_, loc_, args);
  }

  void cloneWithMappedOperands(mlir::Operation *op) {
    mlir::IRMapping mapping;
    for (mlir::Value operand : op->getOperands()) {
      mlir::Value value = mapped(operand, op);
      if (!ok_)
        return;
      mapping.map(operand, value);
    }
    mlir::Operation *cloned = builder_.clone(*op, mapping);
    for (auto [from, to] : llvm::zip(op->getResults(), cloned->getResults()))
      values_[from] = to;
  }

  //===--------------------------------------------------------------------===//

  /// Reads the current value of a registered module constant and materializes
  /// it as a literal, mirroring how KernelIRBuilder bakes extern globals in.
  mlir::Value bakeGlobal(llvm::StringRef symbol, mlir::Type type,
                         mlir::Operation *op) {
    const void *address = nullptr;
    for (const auto &[name, pointer] : constants_) {
      if (pointer && symbolMatches(symbol, name, 'E')) {
        address = pointer;
        break;
      }
    }
    if (!address) {
      // Not registered by the application -- but the FIR module may still
      // carry a compile-time initializer (a Fortran `parameter`).
      if (mlir::Value literal = bakeGlobalInitializer(symbol, type))
        return literal;
      fail(op, "module constant '" + symbol.str() +
                   "' is neither a Fortran parameter nor registered via "
                   "ops_mlir_register_kernel_constant");
      return {};
    }

    if (type.isF64()) {
      double value;
      std::memcpy(&value, address, sizeof(value));
      return builder_.create<mlir::arith::ConstantOp>(
          loc_, builder_.getF64FloatAttr(value));
    }
    if (type.isF32()) {
      float value;
      std::memcpy(&value, address, sizeof(value));
      return builder_.create<mlir::arith::ConstantOp>(
          loc_, builder_.getF32FloatAttr(value));
    }
    if (type.isInteger(32)) {
      std::int32_t value;
      std::memcpy(&value, address, sizeof(value));
      return builder_.create<mlir::arith::ConstantOp>(
          loc_, builder_.getI32IntegerAttr(value));
    }
    if (type.isInteger(64)) {
      std::int64_t value;
      std::memcpy(&value, address, sizeof(value));
      return builder_.create<mlir::arith::ConstantOp>(
          loc_, builder_.getI64IntegerAttr(value));
    }
    fail(op, "module constant '" + symbol.str() +
                 "' has an unsupported type (only real(4)/real(8) and "
                 "integer(4)/integer(8) can be baked in)");
    return {};
  }

  /// A Fortran `parameter` keeps its value in the FIR module as a
  /// `fir.global` with a constant initializer, so it needs no registration.
  mlir::Value bakeGlobalInitializer(llvm::StringRef symbol, mlir::Type type) {
    auto global = artifact_.lookupSymbol<fir::GlobalOp>(symbol);
    if (!global || global.getRegion().empty())
      return {};
    for (mlir::Operation &op : global.getRegion().front()) {
      if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
        if (constant.getType() != type)
          continue;
        return builder_.create<mlir::arith::ConstantOp>(loc_,
                                                        constant.getValue());
      }
    }
    return {};
  }

  mlir::Value castValue(mlir::Value value, mlir::Type target,
                        mlir::Operation *op) {
    mlir::Type source = value.getType();
    if (source == target)
      return value;

    auto sourceFloat = mlir::dyn_cast<mlir::FloatType>(source);
    auto targetFloat = mlir::dyn_cast<mlir::FloatType>(target);
    auto sourceInt = mlir::dyn_cast<mlir::IntegerType>(source);
    auto targetInt = mlir::dyn_cast<mlir::IntegerType>(target);

    if (sourceInt && targetFloat)
      return builder_.create<mlir::arith::SIToFPOp>(loc_, target, value);
    if (sourceFloat && targetInt)
      return builder_.create<mlir::arith::FPToSIOp>(loc_, target, value);
    if (sourceFloat && targetFloat)
      return sourceFloat.getWidth() < targetFloat.getWidth()
                 ? builder_.create<mlir::arith::ExtFOp>(loc_, target, value)
                       .getResult()
                 : builder_.create<mlir::arith::TruncFOp>(loc_, target, value)
                       .getResult();
    if (sourceInt && targetInt) {
      if (mlir::isa<mlir::IndexType>(target) ||
          mlir::isa<mlir::IndexType>(source))
        return builder_.create<mlir::arith::IndexCastOp>(loc_, target, value);
      return sourceInt.getWidth() < targetInt.getWidth()
                 ? builder_.create<mlir::arith::ExtSIOp>(loc_, target, value)
                       .getResult()
                 : builder_.create<mlir::arith::TruncIOp>(loc_, target, value)
                       .getResult();
    }
    if (mlir::isa<mlir::IndexType>(target) || mlir::isa<mlir::IndexType>(source))
      return builder_.create<mlir::arith::IndexCastOp>(loc_, target, value);

    fail(op, "unsupported conversion in kernel body");
    return {};
  }

  /// An alias records that two FIR references denote the same storage.
  /// `fir.declare` introduces one for every variable, so essentially every
  /// load and store in real flang output goes through an alias rather than
  /// naming the `fir.alloca` (or block argument) directly.
  void aliasTo(mlir::Value from, mlir::Value to) {
    mlir::Value target = resolve(to);
    if (auto memref = memrefs_.find(target); memref != memrefs_.end()) {
      memrefs_[from] = memref->second;
      return;
    }
    if (auto global = globals_.find(target); global != globals_.end()) {
      globals_[from] = global->second;
      return;
    }
    // Scalar storage stays a single slot with one mutable value, so alias by
    // reference rather than copying: a store through either name must be
    // visible through the other.
    aliases_[from] = target;
  }

  /// Follows a chain of aliases to the value that actually owns the storage.
  mlir::Value resolve(mlir::Value value) const {
    // The chain is built as the body is walked, so it is acyclic and short;
    // the bound still keeps malformed FIR from hanging the JIT.
    for (unsigned step = 0; step < 64; ++step) {
      auto alias = aliases_.find(value);
      if (alias == aliases_.end())
        return value;
      value = alias->second;
    }
    return value;
  }

  mlir::Value mapped(mlir::Value value, mlir::Operation *op) {
    if (auto it = values_.find(value); it != values_.end())
      return it->second;
    if (auto it = elements_.find(value); it != elements_.end())
      return it->second;

    // A constant defined outside the walked body (FIR hoists some) can be
    // cloned on demand.
    if (mlir::Operation *definer = value.getDefiningOp()) {
      if (definer->hasTrait<mlir::OpTrait::ConstantLike>()) {
        mlir::Operation *cloned = builder_.clone(*definer);
        values_[value] = cloned->getResult(0);
        return cloned->getResult(0);
      }
    }
    fail(op, "value has no translation (unsupported kernel construct)");
    return {};
  }

  void fail(mlir::Operation *, const llvm::Twine &message) {
    if (ok_) {
      ok_ = false;
      errs_ << "FortranKernelIRBuilder: " << message << "\n";
    }
  }

  mlir::OpBuilder &builder_;
  mlir::Location loc_;
  mlir::ModuleOp artifact_;
  const std::map<std::string, const void *> &constants_;
  llvm::raw_ostream &errs_;

  /// Translated SSA values, keyed by the FIR value they came from.
  llvm::DenseMap<mlir::Value, mlir::Value> values_;
  /// Scalar storage modelled as a single SSA value (allocas, by-reference
  /// dummies). A null mapped value means "declared but not yet stored".
  llvm::DenseMap<mlir::Value, mlir::Value> slots_;
  /// The subset of `slots_` bound to a kernel argument rather than a local.
  llvm::DenseSet<mlir::Value> argumentSlots_;
  /// Array storage backed by a memref the OPS ABI supplied.
  llvm::DenseMap<mlir::Value, mlir::Value> memrefs_;
  /// Loaded array elements, keyed by the FIR reference denoting them.
  llvm::DenseMap<mlir::Value, mlir::Value> elements_;
  /// References to module globals, keyed by FIR value.
  llvm::DenseMap<mlir::Value, llvm::StringRef> globals_;
  llvm::DenseMap<mlir::Value, mlir::Value> aliases_;

  mlir::Value result_;
  bool sawReturn_ = false;
  bool ok_ = true;
};

//===----------------------------------------------------------------------===//
// Fortran -> FIR
//===----------------------------------------------------------------------===//

/// Runs one flang frontend action, as `flang -fc1 <args>` would, sending
/// the action's output (if any) to `output`. Diagnostics go to stderr, the
/// same place clang's land for a C++ kernel header.
bool runFlang(Fortran::frontend::FrontendAction &action,
              const std::vector<std::string> &args,
              std::unique_ptr<llvm::raw_pwrite_stream> output) {
  Fortran::frontend::CompilerInstance flang;
  flang.createDiagnostics();
  if (!flang.hasDiagnostics())
    return false;

  std::vector<const char *> argv;
  for (const std::string &arg : args)
    argv.push_back(arg.c_str());

  // Argument errors are buffered and replayed through the instance's own
  // diagnostics engine, as fc1_main does.
  auto *argDiags = new Fortran::frontend::TextDiagnosticBuffer;
  clang::DiagnosticOptions diagOpts;
  clang::DiagnosticsEngine diags(clang::DiagnosticIDs::create(), diagOpts,
                                 argDiags);
  bool ok = Fortran::frontend::CompilerInvocation::createFromArgs(
      flang.getInvocation(), argv, diags);
  argDiags->flushDiagnostics(flang.getDiagnostics());
  if (!ok)
    return false;

  if (output)
    flang.setOutputStream(std::move(output));
  ok = flang.executeAction(action) &&
       !flang.getDiagnostics().hasErrorOccurred();
  flang.clearOutputFiles(/*EraseFiles=*/false);
  return ok;
}

/// Lowers `source` to FIR text. Every kernel of an application comes from
/// the same module, so the result is cached per source: flang runs once per
/// program, not once per kernel.
bool lowerToFIR(const FortranKernelSource &source, std::string &fir,
                llvm::raw_ostream &errs) {
  static std::mutex cacheMutex;
  static std::map<std::string, std::string> cache;
  std::lock_guard<std::mutex> lock(cacheMutex);

  std::string key = source.file;
  for (const std::string &dependency : source.dependencies)
    key += "\n" + dependency;
  auto cached = cache.find(key);
  if (cached != cache.end()) {
    fir = cached->second;
    return true;
  }

  // Flang looks the target up to fill in the module's data layout.
  llvm::InitializeNativeTarget();

  // The `.mod` files the dependencies produce are only needed while the
  // kernel module compiles, so they live in a scratch directory.
  llvm::SmallString<128> moduleDir;
  if (std::error_code ec = llvm::sys::fs::createUniqueDirectory(
          "ops-mlir-fortran", moduleDir)) {
    errs << "FortranKernelIRBuilder: could not create a module directory: "
         << ec.message() << "\n";
    return false;
  }

  std::vector<std::string> common = {"-module-dir", moduleDir.str().str()};
  // fc1 derives the intrinsic module path (iso_c_binding.mod and friends)
  // from the running executable -- here the application, not flang -- so
  // point it at the flang build explicitly.
  if (std::strlen(OPS_FLANG_INTRINSIC_MODULE_DIR) != 0) {
    common.push_back("-fintrinsic-modules-path");
    common.push_back(OPS_FLANG_INTRINSIC_MODULE_DIR);
  }

  bool ok = true;
  for (const std::string &dependency : source.dependencies) {
    std::vector<std::string> args = common;
    args.push_back("-fsyntax-only");
    args.push_back(dependency);
    Fortran::frontend::ParseSyntaxOnlyAction action;
    if (!runFlang(action, args, nullptr)) {
      errs << "FortranKernelIRBuilder: flang could not compile '"
           << dependency << "'\n";
      ok = false;
      break;
    }
  }

  llvm::SmallString<0> buffer;
  if (ok) {
    std::vector<std::string> args = common;
    args.push_back("-emit-fir");
    args.push_back(source.file);
    Fortran::frontend::EmitFIRAction action;
    if (!runFlang(action, args,
                  std::make_unique<llvm::raw_svector_ostream>(buffer))) {
      errs << "FortranKernelIRBuilder: flang could not compile '"
           << source.file << "'\n";
      ok = false;
    }
  }

  llvm::sys::fs::remove_directories(moduleDir);
  if (!ok)
    return false;

  fir = buffer.str().str();
  cache[key] = fir;
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// FortranKernelIRBuilder
//===----------------------------------------------------------------------===//

mlir::func::FuncOp FortranKernelIRBuilder::generate(
    const FortranKernelSource &source, const std::string &kernelName, int indexRank,
    mlir::FunctionType opsABI,
    const std::map<std::string, const void *> &constants,
    llvm::raw_ostream &errs) {
  // Flang's output carries module-level `dlti.dl_spec` and `llvm.*`
  // attributes describing the target, so those dialects must be loaded for
  // the parse to succeed even though the kernel bodies never use them.
  context_.loadDialect<fir::FIROpsDialect, mlir::arith::ArithDialect,
                       mlir::math::MathDialect, mlir::memref::MemRefDialect,
                       mlir::func::FuncDialect, mlir::DLTIDialect,
                       mlir::LLVM::LLVMDialect>();

  std::string fir;
  if (!lowerToFIR(source, fir, errs))
    return nullptr;

  mlir::OwningOpRef<mlir::ModuleOp> artifactModule =
      mlir::parseSourceString<mlir::ModuleOp>(fir, &context_);
  if (!artifactModule) {
    errs << "FortranKernelIRBuilder: failed to parse the FIR flang produced "
            "for '"
         << source.file << "'\n";
    return nullptr;
  }

  // Locate the procedure. Flang mangles a module procedure's name, so match
  // on the `P<name>` suffix rather than reproducing the mangling.
  mlir::func::FuncOp firFn;
  artifactModule->walk([&](mlir::func::FuncOp fn) {
    if (!fn.isDeclaration() &&
        symbolMatches(fn.getSymName(), kernelName, 'P'))
      firFn = fn;
  });
  if (!firFn) {
    errs << "FortranKernelIRBuilder: no definition of '" << kernelName
         << "' in " << source.file << "\n";
    return nullptr;
  }

  if (firFn.getNumArguments() != opsABI.getNumInputs()) {
    errs << "FortranKernelIRBuilder: '" << kernelName << "' takes "
         << firFn.getNumArguments() << " argument(s) but the loop passes "
         << opsABI.getNumInputs() << "\n";
    return nullptr;
  }

  mlir::OpBuilder builder(&context_);
  mlir::Location loc = builder.getUnknownLoc();

  auto funcOp = mlir::func::FuncOp::create(loc, kernelName, opsABI);
  funcOp.setPrivate();
  mlir::Block *entry = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  FIRBodyImporter importer(builder, loc, *artifactModule, constants, errs);

  // Bind each Fortran dummy argument to the value the OPS ABI supplies.
  // Fortran passes everything by reference, so a `!fir.ref<f64>` dummy is
  // what an `f64` kernel argument arrives as, and `!fir.ref<!fir.array<..>>`
  // is what the index argument's memref stands in for.
  //
  // Binding is by kind, not position: the stencil lowering emits the scalar
  // arguments first and moves every `ops_arg_idx` memref to the end, so a
  // kernel that interleaves them in its Fortran signature would mis-bind
  // under a positional match.
  llvm::SmallVector<unsigned, 8> scalarSlots, arraySlots;
  for (unsigned index = 0; index < opsABI.getNumInputs(); ++index) {
    if (mlir::isa<mlir::MemRefType>(opsABI.getInput(index)))
      arraySlots.push_back(index);
    else
      scalarSlots.push_back(index);
  }

  unsigned nextScalar = 0, nextArray = 0;
  for (unsigned index = 0; index < firFn.getNumArguments(); ++index) {
    mlir::Type firType = firFn.getArgument(index).getType();
    auto reference = mlir::dyn_cast<fir::ReferenceType>(firType);
    if (!reference) {
      errs << "FortranKernelIRBuilder: argument " << index << " of '"
           << kernelName
           << "' is not a by-reference dummy (only ordinary scalar and "
              "explicit-shape array dummies are supported)\n";
      funcOp.erase();
      return nullptr;
    }

    mlir::Type element = reference.getEleTy();
    auto sequence = mlir::dyn_cast<fir::SequenceType>(element);
    const bool isArray = static_cast<bool>(sequence);

    llvm::SmallVectorImpl<unsigned> &slots = isArray ? arraySlots : scalarSlots;
    unsigned &next = isArray ? nextArray : nextScalar;
    if (next >= slots.size()) {
      errs << "FortranKernelIRBuilder: '" << kernelName << "' declares more "
           << (isArray ? "array" : "scalar")
           << " arguments than the loop passes\n";
      funcOp.erase();
      return nullptr;
    }
    unsigned slot = slots[next++];
    mlir::Type opsType = opsABI.getInput(slot);

    if (!isArray) {
      if (element != opsType) {
        errs << "FortranKernelIRBuilder: argument " << index << " of '"
             << kernelName << "' has type " << element
             << " but the loop passes " << opsType << "\n";
        funcOp.erase();
        return nullptr;
      }
    } else {
      auto opsMemRef = mlir::cast<mlir::MemRefType>(opsType);
      if (opsMemRef.getRank() != 1 ||
          sequence.getEleTy() != opsMemRef.getElementType()) {
        errs << "FortranKernelIRBuilder: argument " << index << " of '"
             << kernelName << "' is " << element
             << " but the loop passes " << opsType << "\n";
        funcOp.erase();
        return nullptr;
      }
      // The index argument's extent is the block dimensionality; a Fortran
      // kernel that declares it otherwise is reading past what OPS fills in.
      if (opsMemRef.getDimSize(0) != indexRank ||
          sequence.getConstantArraySize() != indexRank) {
        errs << "FortranKernelIRBuilder: the index argument of '" << kernelName
             << "' must be declared with extent " << indexRank
             << " to match the block's dimensionality\n";
        funcOp.erase();
        return nullptr;
      }
    }

    importer.bindArgument(firFn.getArgument(index), entry->getArgument(slot),
                          isArray);
  }

  if (firFn.getBlocks().size() != 1) {
    errs << "FortranKernelIRBuilder: '" << kernelName
         << "' has control flow, which is not supported (kernel bodies must "
            "be straight-line expressions)\n";
    funcOp.erase();
    return nullptr;
  }

  importer.run(firFn.getBlocks().front());
  if (!importer.ok()) {
    funcOp.erase();
    return nullptr;
  }

  const bool expectsResult = opsABI.getNumResults() == 1;
  if (expectsResult) {
    mlir::Value result = importer.result();
    if (!result) {
      errs << "FortranKernelIRBuilder: '" << kernelName
           << "' must assign its function result\n";
      funcOp.erase();
      return nullptr;
    }
    if (result.getType() != opsABI.getResult(0)) {
      errs << "FortranKernelIRBuilder: '" << kernelName << "' returns "
           << result.getType() << " but the loop expects "
           << opsABI.getResult(0) << "\n";
      funcOp.erase();
      return nullptr;
    }
    builder.create<mlir::func::ReturnOp>(loc, result);
  } else {
    builder.create<mlir::func::ReturnOp>(loc);
  }

  return funcOp;
}

} // namespace ops_mlir
