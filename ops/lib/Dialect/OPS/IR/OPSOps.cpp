#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace ops_mlir::ops;

//===----------------------------------------------------------------------===//
// Custom assembly: constant offsets "[1, 0]" and the dat operands of a loop
//===----------------------------------------------------------------------===//

static ParseResult parseOffsets(OpAsmParser &parser, DenseI64ArrayAttr &offsets) {
  SmallVector<int64_t> values;
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() -> ParseResult {
        int64_t v;
        if (parser.parseInteger(v)) return failure();
        values.push_back(v);
        return success();
      }))
    return failure();
  offsets = parser.getBuilder().getDenseI64ArrayAttr(values);
  return success();
}

static void printOffsets(OpAsmPrinter &printer, Operation *, DenseI64ArrayAttr offsets) {
  printer << '[';
  llvm::interleaveComma(offsets.asArrayRef(), printer);
  printer << ']';
}

// dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : ... write #ops.stencil<[[0, 0]]>)
static ParseResult parseDatOperands(OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dats,
                                    SmallVectorImpl<Type> &types, ArrayAttr &accesses, ArrayAttr &stencils) {
  MLIRContext *ctx = parser.getContext();
  SmallVector<Attribute> accessAttrs, stencilAttrs;

  auto parseTail = [&](OpAsmParser::UnresolvedOperand operand) -> ParseResult {
    Type type;
    StringRef keyword;
    SMLoc keywordLoc;
    StencilAttr stencil;
    if (parser.parseColonType(type)) return failure();
    keywordLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&keyword)) return failure();
    std::optional<AccessMode> mode = symbolizeAccessMode(keyword);
    if (!mode) return parser.emitError(keywordLoc, "unknown access mode '") << keyword << "'";
    if (parser.parseAttribute(stencil)) return failure();
    dats.push_back(operand);
    types.push_back(type);
    accessAttrs.push_back(AccessModeAttr::get(ctx, *mode));
    stencilAttrs.push_back(stencil);
    return success();
  };

  OpAsmParser::UnresolvedOperand first;
  OptionalParseResult head = parser.parseOptionalOperand(first);
  if (head.has_value()) {
    if (failed(*head) || failed(parseTail(first))) return failure();
    while (succeeded(parser.parseOptionalComma())) {
      OpAsmParser::UnresolvedOperand next;
      if (parser.parseOperand(next) || parseTail(next)) return failure();
    }
  }
  accesses = ArrayAttr::get(ctx, accessAttrs);
  stencils = ArrayAttr::get(ctx, stencilAttrs);
  return success();
}

static void printDatOperands(OpAsmPrinter &printer, Operation *, OperandRange dats, TypeRange types,
                             ArrayAttr accesses, ArrayAttr stencils) {
  for (unsigned i = 0; i < dats.size(); ++i) {
    if (i) printer << ", ";
    printer << dats[i] << " : " << types[i];
    if (i < accesses.size()) printer << ' ' << stringifyAccessMode(cast<AccessModeAttr>(accesses[i]).getValue());
    if (i < stencils.size()) printer << ' ' << stencils[i];
  }
}

//===----------------------------------------------------------------------===//
// Operations of kernel bodies
//===----------------------------------------------------------------------===//

/// Checks an access to a dat of an enclosing ops.par_loop against the loop's access mode and stencil.
/// `offsets` is empty for data-dependent accesses (only the mode is checked then).
static LogicalResult verifyAgainstLoop(Operation *op, Value acc, ArrayRef<int64_t> offsets, bool isWrite,
                                       bool dynamic) {
  auto arg = dyn_cast<BlockArgument>(acc);
  if (!arg) return success();
  auto loop = dyn_cast_or_null<ParLoopOp>(arg.getOwner()->getParentOp());
  if (!loop || arg.getOwner() != &loop.getBodyBlock() || arg.getArgNumber() >= loop.getNumDats()) return success();

  unsigned i = arg.getArgNumber();
  AccessMode mode = loop.getAccess(i);
  if (isWrite && mode == AccessMode::Read)
    return op->emitOpError("writes dat operand #") << i << " of the enclosing ops.par_loop, which is only read";
  if (!dynamic && !loop.getStencil(i).contains(offsets)) {
    InFlightDiagnostic diag = op->emitOpError("offset [");
    llvm::interleaveComma(offsets, diag);
    diag << "] is not a point of the stencil of dat operand #" << i << " of the enclosing ops.par_loop";
    return diag;
  }
  return success();
}

/// I64Attr accessors return uint64_t; these attributes are signed.
static int64_t asSigned(uint64_t v) { return static_cast<int64_t>(v); }

static LogicalResult verifyComponent(Operation *op, AccType type, int64_t comp) {
  if (comp < 0 || comp >= type.getNcomp())
    return op->emitOpError("component ") << comp << " is out of range for an accessor with " << type.getNcomp()
                                         << " components";
  return success();
}

LogicalResult ReadOp::verify() {
  AccType type = cast<AccType>(getAcc().getType());
  if ((int64_t)getOffsets().size() != type.getRank())
    return emitOpError("expects ") << type.getRank() << " offsets, got " << getOffsets().size();
  if (failed(verifyComponent(*this, type, asSigned(getComp())))) return failure();
  return verifyAgainstLoop(*this, getAcc(), getOffsets(), /*isWrite=*/false, /*dynamic=*/false);
}

LogicalResult ReadDynOp::verify() {
  AccType type = cast<AccType>(getAcc().getType());
  if ((int64_t)getOffsets().size() != type.getRank())
    return emitOpError("expects ") << type.getRank() << " offsets, got " << getOffsets().size();
  if (failed(verifyComponent(*this, type, asSigned(getComp())))) return failure();
  return verifyAgainstLoop(*this, getAcc(), {}, /*isWrite=*/false, /*dynamic=*/true);
}

LogicalResult WriteOp::verify() {
  AccType type = cast<AccType>(getAcc().getType());
  if ((int64_t)getOffsets().size() != type.getRank())
    return emitOpError("expects ") << type.getRank() << " offsets, got " << getOffsets().size();
  if (failed(verifyComponent(*this, type, asSigned(getComp())))) return failure();
  return verifyAgainstLoop(*this, getAcc(), getOffsets(), /*isWrite=*/true, /*dynamic=*/false);
}

LogicalResult GblReadOp::verify() {
  GblType type = cast<GblType>(getGbl().getType());
  int64_t index = asSigned(getIndex());
  if (index < 0 || index >= type.getSize())
    return emitOpError("index ") << index << " is out of range for a global of size " << type.getSize();
  return success();
}

LogicalResult IndexOp::verify() {
  int64_t dim = asSigned(getDim());
  if (dim < 0 || dim >= kMaxRank) return emitOpError("dimension must be between 0 and ") << kMaxRank - 1;
  if (auto loop = (*this)->getParentOfType<ParLoopOp>())
    if (dim >= loop.getRank())
      return emitOpError("dimension ") << dim << " does not exist in a loop of rank " << loop.getRank();
  return success();
}

LogicalResult ConstRefOp::verify() {
  if (asSigned(getOffset()) < 0) return emitOpError("offset must not be negative");
  if (getIndex() && asSigned(getArrayStride()) <= 0)
    return emitOpError("an indexed constant needs a positive array_stride");
  return success();
}

//===----------------------------------------------------------------------===//
// ops.kernel and ops.chain
//===----------------------------------------------------------------------===//

static ParseResult parseFunctionLike(OpAsmParser &parser, OperationState &result, StringAttr typeAttrName,
                                     StringAttr argAttrsName, StringAttr resAttrsName) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, results);
  };
  return function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false, typeAttrName,
                                                  buildFuncType, argAttrsName, resAttrsName);
}

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFunctionLike(parser, result, getFunctionTypeAttrName(result.name), getArgAttrsAttrName(result.name),
                           getResAttrsAttrName(result.name));
}

void KernelOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
                                           getArgAttrsAttrName(), getResAttrsAttrName());
}

ParseResult ChainOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFunctionLike(parser, result, getFunctionTypeAttrName(result.name), getArgAttrsAttrName(result.name),
                           getResAttrsAttrName(result.name));
}

void ChainOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
                                           getArgAttrsAttrName(), getResAttrsAttrName());
}

LogicalResult KernelOp::verify() {
  if (!getResultTypes().empty()) return emitOpError("must not have results");
  for (auto [i, type] : llvm::enumerate(getArgumentTypes()))
    if (!isa<AccType, GblType, RedType>(type))
      return emitOpError("parameter #") << i << " must be !ops.acc, !ops.gbl or !ops.red, got " << type;
  return success();
}

LogicalResult KernelOp::verifyRegions() {
  if (isExternal()) return emitOpError("must have a body");
  if (!getBody().hasOneBlock()) return emitOpError("body must be a single block");
  auto ret = dyn_cast<ReturnOp>(getBody().front().getTerminator());
  if (!ret) return emitOpError("body must end with ops.return");
  if (ret.getNumOperands() != 0) return ret.emitOpError("of a kernel must not have operands");
  return success();
}

LogicalResult ChainOp::verify() {
  for (auto [i, type] : llvm::enumerate(getArgumentTypes()))
    if (!isa<DatType>(type) && !type.isIntOrIndexOrFloat())
      return emitOpError("argument #") << i << " must be a dat or a scalar, got " << type;
  for (auto [i, type] : llvm::enumerate(getResultTypes()))
    if (!isSupportedElementType(type))
      return emitOpError("result #") << i << " must be a reduction value (f16, f32, f64, i32, i64), got " << type;
  return success();
}

LogicalResult ChainOp::verifyRegions() {
  if (isExternal()) return emitOpError("must have a body");
  if (!getBody().hasOneBlock()) return emitOpError("body must be a single block");
  auto ret = dyn_cast<ReturnOp>(getBody().front().getTerminator());
  if (!ret) return emitOpError("body must end with ops.return");
  if (ret.getOperandTypes() != getResultTypes())
    return ret.emitOpError("operand types do not match the results of the chain");
  return success();
}

//===----------------------------------------------------------------------===//
// ops.par_loop
//===----------------------------------------------------------------------===//

AccessMode ParLoopOp::getAccess(unsigned i) { return cast<AccessModeAttr>(getAccesses()[i]).getValue(); }

StencilAttr ParLoopOp::getStencil(unsigned i) { return cast<StencilAttr>(getStencils()[i]); }

ReductionKind ParLoopOp::getReductionKind(unsigned i) {
  return cast<ReductionKindAttr>((*getReductionKinds())[i]).getValue();
}

LogicalResult ParLoopOp::verify() {
  if (getRange().empty() || getRange().size() % 2 != 0)
    return emitOpError("range must hold lower and upper bound of each dimension, got ") << getRange().size()
                                                                                        << " values";
  unsigned rank = getRank();
  if (rank > kMaxRank) return emitOpError("rank ") << rank << " is not supported (at most " << kMaxRank << ")";

  if (getAccesses().size() != getNumDats() || getStencils().size() != getNumDats())
    return emitOpError("expects one access mode and one stencil per dat operand");
  for (unsigned i = 0; i < getNumDats(); ++i) {
    auto dat = cast<DatType>(getDats()[i].getType());
    if (dat.getRank() != rank)
      return emitOpError("dat operand #") << i << " has rank " << dat.getRank() << ", the loop has rank " << rank;
    StencilAttr stencil = getStencil(i);
    if (stencil.getRank() != rank)
      return emitOpError("stencil of dat operand #") << i << " has rank " << stencil.getRank() << ", the loop has rank "
                                                     << rank;
  }

  unsigned numKinds = getReductionKinds() ? getReductionKinds()->size() : 0;
  if (numKinds != getReductions().size())
    return emitOpError("expects one entry of reduction_kinds per result, got ") << numKinds << " for "
                                                                               << getReductions().size()
                                                                               << " results";
  return success();
}

LogicalResult ParLoopOp::verifyRegions() {
  Block &body = getBodyBlock();
  unsigned numDats = getNumDats(), numScalars = getScalars().size(), numReds = getReductions().size();
  if (body.getNumArguments() != numDats + numScalars + numReds)
    return emitOpError("body must have ") << numDats + numScalars + numReds << " arguments (" << numDats
                                          << " accessors, " << numScalars << " scalars, " << numReds
                                          << " reductions), got " << body.getNumArguments();
  for (unsigned i = 0; i < numDats; ++i) {
    auto dat = cast<DatType>(getDats()[i].getType());
    AccType expected = AccType::get(getContext(), dat.getElementType(), dat.getRank(), dat.getNcomp());
    if (body.getArgument(i).getType() != expected)
      return emitOpError("body argument #") << i << " must be " << expected << " to match dat operand #" << i;
  }
  for (unsigned i = 0; i < numScalars; ++i)
    if (body.getArgument(numDats + i).getType() != getScalars()[i].getType())
      return emitOpError("body argument #") << numDats + i << " must have the type of scalar operand #" << i;
  for (unsigned i = 0; i < numReds; ++i) {
    RedType expected = RedType::get(getContext(), getReductions()[i].getType());
    if (body.getArgument(numDats + numScalars + i).getType() != expected)
      return emitOpError("body argument #") << numDats + numScalars + i << " must be " << expected
                                            << " to match result #" << i;
  }
  return success();
}

void ParLoopOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  MutableOperandRange dats = getDatsMutable();
  for (unsigned i = 0; i < dats.size(); ++i) {
    AccessMode mode = getAccess(i);
    OpOperand *operand = &dats[i];
    if (mode != AccessMode::Write)
      effects.emplace_back(MemoryEffects::Read::get(), operand, SideEffects::DefaultResource::get());
    if (mode != AccessMode::Read)
      effects.emplace_back(MemoryEffects::Write::get(), operand, SideEffects::DefaultResource::get());
  }
}


LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto chain = symbolTable.lookupNearestSymbolFrom<ChainOp>(*this, getCalleeAttr());
  if (!chain) return emitOpError("'") << getCallee() << "' does not reference a valid ops.chain";
  FunctionType type = chain.getFunctionType();
  if (type.getNumInputs() != getOperands().size())
    return emitOpError("has ") << getOperands().size() << " operands, the callee takes " << type.getNumInputs();
  for (unsigned i = 0; i < type.getNumInputs(); ++i)
    if (getOperands()[i].getType() != type.getInput(i))
      return emitOpError("operand #") << i << " must have type " << type.getInput(i) << ", got "
                                      << getOperands()[i].getType();
  if (type.getNumResults() != getNumResults())
    return emitOpError("has ") << getNumResults() << " results, the callee returns " << type.getNumResults();
  for (unsigned i = 0; i < type.getNumResults(); ++i)
    if (getResult(i).getType() != type.getResult(i))
      return emitOpError("result #") << i << " must have type " << type.getResult(i) << ", got "
                                     << getResult(i).getType();
  return success();
}


namespace {
/// A read at offsets that turn out to be constants is an ordinary read (after unrolling a loop that computed them).
struct ReadDynToRead : public OpRewritePattern<ReadDynOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReadDynOp op, PatternRewriter &rewriter) const override {
    SmallVector<int64_t> offsets;
    for (Value v : op.getOffsets()) {
      APInt value;
      if (!matchPattern(v, m_ConstantInt(&value))) return failure();
      offsets.push_back(value.getSExtValue());
    }
    rewriter.replaceOpWithNewOp<ReadOp>(op, op.getResult().getType(), op.getAcc(), rewriter.getDenseI64ArrayAttr(offsets),
                                        rewriter.getI64IntegerAttr(static_cast<int64_t>(op.getComp())));
    return success();
  }
};
} // namespace

void ReadDynOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<ReadDynToRead>(ctx);
}

#define GET_OP_CLASSES
#include "ops-mlir/Dialect/OPS/IR/OPSOps.cpp.inc"
