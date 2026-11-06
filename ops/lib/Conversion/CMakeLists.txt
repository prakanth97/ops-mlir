add_mlir_library(OPSMLIRConversion
  OpsToSCF.cpp
  MarkNoalias.cpp
  HoistAccumulators.cpp
  FpContract.cpp
  GpuKernelAttrs.cpp

  DEPENDS
  OPSMLIRConversionIncGen

  LINK_LIBS PUBLIC
  OPSMLIRDialect
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRTransformUtils
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRGPUDialect
)
