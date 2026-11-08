add_mlir_library(OPSMLIRFrontendCommon
  Manifest.cpp

  LINK_LIBS PUBLIC
  OPSMLIRDialect
  OPDSLSupport
  MLIRIR
  MLIRSCFDialect
)
