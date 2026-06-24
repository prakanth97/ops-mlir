#include "Dialect/stencil/StencilDialect.h"
#include "Dialect/stencil/StencilOps.h"
#include "Dialect/stencil/StencilAttr.h"
#include "Dialect/stencil/StencilTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::stencil;

#include "Dialect/stencil/StencilDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/stencil/StencilOpsAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/stencil/StencilOpsTypes.cpp.inc"

namespace mlir::stencil {

LogicalResult IndexAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 DenseI64ArrayAttr array) {
  if (array.size() < 1 || array.size() > 3)
    return emitError() << "expected 1 to 3 indices, got " << array.size();
  return success();
}

LogicalResult
StencilBoundsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           DenseI64ArrayAttr lb, DenseI64ArrayAttr ub) {
  if (lb.size() != ub.size())
    return emitError() << "lb (" << lb.size() << " dims) and ub ("
                        << ub.size() << " dims) rank mismatch";
  for (auto [l, u] : llvm::zip(lb.asArrayRef(), ub.asArrayRef()))
    if (l > u)
      return emitError() << "lb must be <= ub in every dimension";
  return success();
}

namespace {

/// Parses the xDSL-style bounded-type body (no leading `<`/trailing `>`,
/// those are handled by the caller): per-dim `[lb,ub]` pairs joined by
/// `x`, followed directly by the element type, e.g.
/// `[-1,4097]x[-1,4097]xf64` -- see xdsl_impl/laplace_stencil.mlir. The
/// repeated bracket-pair-by-rank shape isn't expressible as a plain
/// declarative assemblyFormat (rank is dynamic), and the trailing `x` is
/// juxtaposed with the element type token (e.g. "xf64"), which needs
/// parseXInDimensionList()'s special handling -- hence hand-written.
ParseResult parseBoundedType(AsmParser &parser, SmallVectorImpl<int64_t> &lb,
                              SmallVectorImpl<int64_t> &ub,
                              Type &elementType) {
  while (succeeded(parser.parseOptionalLSquare())) {
    int64_t l, u;
    if (parser.parseInteger(l) || parser.parseComma() ||
        parser.parseInteger(u) || parser.parseRSquare())
      return failure();
    lb.push_back(l);
    ub.push_back(u);
    if (parser.parseXInDimensionList())
      return failure();
  }
  return parser.parseType(elementType);
}

void printBoundedType(AsmPrinter &printer, StencilBoundsAttr bounds,
                      Type elementType) {
  for (auto [l, u] : llvm::zip(bounds.getLb().asArrayRef(),
                               bounds.getUb().asArrayRef()))
    printer << "[" << l << "," << u << "]x";
  printer << elementType;
}

} // namespace

Type FieldType::parse(AsmParser &parser) {
  SmallVector<int64_t> lb, ub;
  Type elementType;
  if (parser.parseLess() || parseBoundedType(parser, lb, ub, elementType) ||
      parser.parseGreater())
    return {};
  auto bounds = StencilBoundsAttr::get(
      parser.getContext(), DenseI64ArrayAttr::get(parser.getContext(), lb),
      DenseI64ArrayAttr::get(parser.getContext(), ub));
  return FieldType::get(parser.getContext(), bounds, elementType);
}

void FieldType::print(AsmPrinter &printer) const {
  printer << "<";
  printBoundedType(printer, getBounds(), getElementType());
  printer << ">";
}

Type TempType::parse(AsmParser &parser) {
  SmallVector<int64_t> lb, ub;
  Type elementType;
  if (parser.parseLess() || parseBoundedType(parser, lb, ub, elementType) ||
      parser.parseGreater())
    return {};
  auto bounds = StencilBoundsAttr::get(
      parser.getContext(), DenseI64ArrayAttr::get(parser.getContext(), lb),
      DenseI64ArrayAttr::get(parser.getContext(), ub));
  return TempType::get(parser.getContext(), bounds, elementType);
}

void TempType::print(AsmPrinter &printer) const {
  printer << "<";
  printBoundedType(printer, getBounds(), getElementType());
  printer << ">";
}

void StencilDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/stencil/StencilOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/stencil/StencilOpsAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/stencil/StencilOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::stencil
