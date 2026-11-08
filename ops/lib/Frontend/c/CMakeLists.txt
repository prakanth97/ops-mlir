add_mlir_library(OPSMLIRFrontendC
  CSubsetWalker.cpp
  KernelImport.cpp

  LINK_LIBS PUBLIC
  OPSMLIRFrontendCommon
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
target_compile_definitions(obj.OPSMLIRFrontendC PRIVATE OPS_CLANG_RESOURCE_DIR="${OPSJIT_CLANG_RESOURCE_DIR}")
