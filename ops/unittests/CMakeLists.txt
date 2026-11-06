# Unit tests of the OPS-specific libraries. They build ops.chain with ChainBuilder.h and run the analyses.
add_opsjit_unittest(ops_footprint_test FootprintTest.cpp LINK OPSMLIRAnalysis MLIRArithDialect)
add_opsjit_unittest(ops_chain_dependence_test ChainDependenceTest.cpp LINK OPSMLIRAnalysis MLIRArithDialect)
add_opsjit_unittest(ops_fusion_planner_test FusionPlannerTest.cpp LINK OPSMLIRAnalysis MLIRArithDialect)
add_opsjit_unittest(ops_end_to_end_cpu_test EndToEndCPUTest.cpp
  LINK OPSMLIRPipelines OPSMLIRDialect OPDSLJIT OPDSLCache OPDSLSupport MLIRParser)

# The runtime on a real OPS world: needs the sequential OPS library of the install.
if(TARGET OPSMLIRRuntimeObjects)
  find_library(OPSJIT_OPS_SEQ_LIB ops_seq HINTS "${OPS_INSTALL}/lib" NO_DEFAULT_PATH)
  find_package(OpenMP)
  if(OPSJIT_OPS_SEQ_LIB AND OpenMP_CXX_FOUND)
    add_opsjit_unittest(ops_runtime_test RuntimeTest.cpp $<TARGET_OBJECTS:OPDSLGpu> LINK OPSMLIRRuntimeObjects ${CMAKE_DL_LIBS} "${OPSJIT_OPS_SEQ_LIB}" OpenMP::OpenMP_CXX)
    target_compile_definitions(ops_runtime_test PRIVATE OPS_2D)
    target_link_options(ops_runtime_test PRIVATE -Wl,--export-dynamic)
  endif()
endif()

# The CUDA pipeline end to end (skipped without a device/toolkit).
add_opsjit_unittest(ops_end_to_end_gpu_test EndToEndGPUTest.cpp $<TARGET_OBJECTS:OPDSLGpu>
  LINK OPSMLIRRuntimeCore OPSMLIRPipelines OPSMLIRDialect OPDSLJIT OPDSLCache OPDSLSupport MLIRParser ${CMAKE_DL_LIBS})
target_link_options(ops_end_to_end_gpu_test PRIVATE -Wl,--export-dynamic)
