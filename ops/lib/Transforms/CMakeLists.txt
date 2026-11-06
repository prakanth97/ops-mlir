add_mlir_library(OPSMLIRTransforms
  MaterializeKernels.cpp
  SpecializeConsts.cpp
  FusePointLocal.cpp
  ForwardStores.cpp
  StoreForwarding.cpp
  OutlineGroups.cpp
  Specialize.cpp

  DEPENDS
  OPSMLIRTransformsIncGen

  LINK_LIBS PUBLIC
  OPSMLIRDialect
  OPSMLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRTransformUtils
  MLIRFunctionInterfaces
)
