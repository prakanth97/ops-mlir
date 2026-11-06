get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_opsjit_tool(ops-opt ops-opt.cpp
  LINK
    OPSMLIRDialect
    OPSMLIRAnalysis
    OPSMLIRTransforms
    OPSMLIRConversion
    OPSMLIRPipelines
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${mlir_extension_libs}
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
    MLIROptLib
    MLIRPass)
llvm_update_compile_flags(ops-opt)
