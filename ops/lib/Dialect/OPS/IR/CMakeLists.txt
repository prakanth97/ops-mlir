add_mlir_dialect_library(OPSMLIRDialect
  OPSDialect.cpp
  OPSTypes.cpp
  OPSAttrs.cpp
  OPSOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/ops-mlir/Dialect/OPS/IR

  DEPENDS
  OPSMLIRDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFunctionInterfaces
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRBytecodeOpInterface
)
# Users of the dialect find its headers (sources and generated files) through the library.
target_include_directories(OPSMLIRDialect PUBLIC
  ${CMAKE_SOURCE_DIR}/ops/include
  ${CMAKE_BINARY_DIR}/ops/include)
