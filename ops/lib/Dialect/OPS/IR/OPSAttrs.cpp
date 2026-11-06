#include "ops-mlir/Dialect/OPS/IR/OPSAttrs.h"

#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"
#include "ops-mlir/Dialect/OPS/IR/OPSTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace ops_mlir::ops;

#include "ops-mlir/Dialect/OPS/IR/OPSEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// StencilAttr
//===----------------------------------------------------------------------===//

StencilAttr StencilAttr::get(MLIRContext *ctx, int64_t rank, ArrayRef<int64_t> points) {
  return get(ctx, rank, points, SmallVector<int64_t>(rank, 1));
}

LogicalResult StencilAttr::verify(function_ref<InFlightDiagnostic()> emitError, int64_t rank,
                                  ArrayRef<int64_t> points, ArrayRef<int64_t> strides) {
  if (rank < 1 || rank > kMaxRank) return emitError() << "rank must be between 1 and " << kMaxRank << ", got " << rank;
  if (points.empty()) return emitError() << "a stencil needs at least one point";
  if (points.size() % rank != 0)
    return emitError() << "the number of offsets (" << points.size() << ") is not a multiple of the rank " << rank;
  if ((int64_t)strides.size() != rank)
    return emitError() << "expected " << rank << " strides, got " << strides.size();
  for (int64_t s : strides)
    if (s != 0 && s != 1) return emitError() << "strides must be 0 or 1, got " << s;
  return success();
}

bool StencilAttr::contains(ArrayRef<int64_t> offsets) const {
  if ((int64_t)offsets.size() != getRank()) return false;
  for (int64_t i = 0, n = getNumPoints(); i < n; ++i)
    if (getPoint(i) == offsets) return true;
  return false;
}

bool StencilAttr::isPointLocal() const {
  if (getNumPoints() != 1) return false;
  for (int64_t o : getPoint(0))
    if (o != 0) return false;
  return llvm::all_of(getStrides(), [](int64_t s) { return s == 1; });
}

std::pair<int64_t, int64_t> StencilAttr::getExtent(int64_t d) const {
  int64_t lo = 0, hi = 0;
  for (int64_t i = 0, n = getNumPoints(); i < n; ++i) {
    int64_t o = getPoint(i)[d];
    lo = std::min(lo, o);
    hi = std::max(hi, o);
  }
  return {lo, hi};
}

Attribute StencilAttr::parse(AsmParser &parser, Type) {
  SmallVector<int64_t> points, strides;
  int64_t rank = -1;
  if (parser.parseLess()) return {};

  auto parsePoint = [&]() -> ParseResult {
    SmallVector<int64_t> point;
    SMLoc loc = parser.getCurrentLocation();
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() -> ParseResult {
          int64_t value;
          if (parser.parseInteger(value)) return failure();
          point.push_back(value);
          return success();
        }))
      return failure();
    if (rank < 0)
      rank = point.size();
    else if ((int64_t)point.size() != rank)
      return parser.emitError(loc, "all stencil points must have the same number of offsets");
    points.append(point.begin(), point.end());
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parsePoint)) return {};

  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("strides") || parser.parseEqual() ||
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() -> ParseResult {
          int64_t value;
          if (parser.parseInteger(value)) return failure();
          strides.push_back(value);
          return success();
        }))
      return {};
  }
  if (parser.parseGreater()) return {};
  if (rank < 1) {
    parser.emitError(parser.getCurrentLocation(), "a stencil needs at least one point");
    return {};
  }
  if (strides.empty()) strides.assign(rank, 1);
  return parser.getChecked<StencilAttr>(parser.getContext(), rank, points, strides);
}

void StencilAttr::print(AsmPrinter &printer) const {
  printer << '<' << '[';
  llvm::interleaveComma(llvm::seq<int64_t>(0, getNumPoints()), printer, [&](int64_t i) {
    printer << '[';
    llvm::interleaveComma(getPoint(i), printer);
    printer << ']';
  });
  printer << ']';
  if (llvm::any_of(getStrides(), [](int64_t s) { return s != 1; })) {
    printer << ", strides = [";
    llvm::interleaveComma(getStrides(), printer);
    printer << ']';
  }
  printer << '>';
}

#define GET_ATTRDEF_CLASSES
#include "ops-mlir/Dialect/OPS/IR/OPSAttrs.cpp.inc"

void OPSDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ops-mlir/Dialect/OPS/IR/OPSAttrs.cpp.inc"
      >();
}
