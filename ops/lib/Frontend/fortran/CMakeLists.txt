if(OPSJIT_ENABLE_FORTRAN)
  add_mlir_library(OPSMLIRFrontendFortran
    FortranKernelImport.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    OPSMLIRFrontendCommon
    OPSMLIRDialect
    MLIRIR
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
    MLIRParser
    MLIRDLTIDialect
    MLIRLLVMDialect
    # CUFAttrs: core FIR ops carry a cuf.data_attr attribute.
    flangFrontend
    FIRDialect
    FIRDialectSupport
    FIRSupport
    CUFAttrs
    FortranSupport
  )
  target_include_directories(obj.OPSMLIRFrontendFortran PRIVATE ${FLANG_INCLUDE_DIRS})
  target_compile_definitions(obj.OPSMLIRFrontendFortran PRIVATE
    OPS_FLANG_INTRINSIC_MODULE_DIR="${OPSJIT_FLANG_INTRINSIC_MODULE_DIR}")
else()
  add_mlir_library(OPSMLIRFrontendFortran
    FortranKernelImportUnavailable.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    OPSMLIRFrontendCommon
    MLIRIR
  )
endif()
