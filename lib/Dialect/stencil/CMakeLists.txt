#===============================================================================
# Stencil Dialect TableGen Setup
#===============================================================================

include_directories(${CMAKE_SOURCE_DIR}/include/Dialect/stencil)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/Dialect/stencil/StencilOps.td)

mlir_tablegen(StencilDialect.h.inc -gen-dialect-decls -dialect=stencil)
mlir_tablegen(StencilDialect.cpp.inc -gen-dialect-defs -dialect=stencil)

mlir_tablegen(StencilOps.h.inc -gen-op-decls)
mlir_tablegen(StencilOps.cpp.inc -gen-op-defs)

mlir_tablegen(StencilOpsAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=stencil)
mlir_tablegen(StencilOpsAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=stencil)

mlir_tablegen(StencilOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=stencil)
mlir_tablegen(StencilOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=stencil)

add_public_tablegen_target(StencilDialectTdGen)

#===============================================================================
# Stencil Dialect Library
#===============================================================================

add_mlir_dialect_library(
  MLIRStencilDialect
  StencilDialect.cpp
  StencilOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Dialect/stencil
  ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
  StencilDialectTdGen

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS
  MLIRIR
  MLIRDialectUtils
  MLIRInferTypeOpInterface
  MLIRPass
  MLIRSupport
)

target_include_directories(
  MLIRStencilDialect
  PUBLIC
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
)
