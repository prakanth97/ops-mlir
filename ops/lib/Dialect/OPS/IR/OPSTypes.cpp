#include "ops-mlir/Dialect/OPS/IR/OPSTypes.h"

#include "ops-mlir/Dialect/OPS/IR/OPSAttrs.h"
#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace ops_mlir::ops;

bool ops_mlir::ops::isSupportedElementType(Type type) {
  return type.isF16() || type.isF32() || type.isF64() || type.isInteger(32) || type.isInteger(64);
}

static LogicalResult verifyElementType(function_ref<InFlightDiagnostic()> emitError, Type elementType) {
  if (!isSupportedElementType(elementType))
    return emitError() << "element type must be f16, f32, f64, i32 or i64, got " << elementType;
  return success();
}

static LogicalResult verifyFieldLike(function_ref<InFlightDiagnostic()> emitError, Type elementType, int64_t rank,
                                     int64_t ncomp) {
  if (failed(verifyElementType(emitError, elementType))) return failure();
  if (rank < 1 || rank > kMaxRank) return emitError() << "rank must be between 1 and " << kMaxRank << ", got " << rank;
  if (ncomp < 1) return emitError() << "number of components must be at least 1, got " << ncomp;
  return success();
}

LogicalResult DatType::verify(function_ref<InFlightDiagnostic()> emitError, Type elementType, int64_t rank,
                              int64_t ncomp) {
  return verifyFieldLike(emitError, elementType, rank, ncomp);
}

LogicalResult AccType::verify(function_ref<InFlightDiagnostic()> emitError, Type elementType, int64_t rank,
                              int64_t ncomp) {
  return verifyFieldLike(emitError, elementType, rank, ncomp);
}

LogicalResult GblType::verify(function_ref<InFlightDiagnostic()> emitError, Type elementType, int64_t size) {
  if (failed(verifyElementType(emitError, elementType))) return failure();
  if (size < 1) return emitError() << "size must be at least 1, got " << size;
  return success();
}

LogicalResult RedType::verify(function_ref<InFlightDiagnostic()> emitError, Type elementType) {
  return verifyElementType(emitError, elementType);
}

#define GET_TYPEDEF_CLASSES
#include "ops-mlir/Dialect/OPS/IR/OPSTypes.cpp.inc"

void OPSDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ops-mlir/Dialect/OPS/IR/OPSTypes.cpp.inc"
      >();
}
