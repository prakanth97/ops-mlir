add_mlir_library(OPSMLIRFrontend
  CSubsetWalker.cpp
  KernelImport.cpp

  LINK_LIBS PUBLIC
  OPSMLIRDialect
  OPDSLSupport
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRBytecodeWriter
  clangTooling
  clangFrontend
  clangAST
  clangBasic
  clangSema
  clangParse
  clangLex
  clangSerialization
  clangDriver
)
target_compile_definitions(obj.OPSMLIRFrontend PRIVATE OPS_CLANG_RESOURCE_DIR="${OPSJIT_CLANG_RESOURCE_DIR}")
