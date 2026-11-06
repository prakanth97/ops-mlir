add_mlir_library(OPSMLIRAnalysis
  Footprint.cpp
  ChainDependence.cpp
  FusionPlanner.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  OPSMLIRDialect
  MLIRIR
  MLIRPass
)
