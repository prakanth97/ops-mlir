add_mlir_dialect_library(
  MLIROPSPasses
  OPSToStencil.cpp
  OPSPtrToMemref.cpp

  LINK_LIBS
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIROPSDialect
  MLIRStencilDialect
)

target_include_directories(
  MLIROPSPasses
  PUBLIC
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
)
