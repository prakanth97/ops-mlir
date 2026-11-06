add_opsjit_unittest(opdsl_hash_test HashTest.cpp LINK OPDSLSupport)
add_opsjit_unittest(opdsl_pipeline_registry_test PipelineRegistryTest.cpp LINK OPDSLPipeline)
add_opsjit_unittest(opdsl_jit_session_test JITSessionTest.cpp LINK OPDSLJIT MLIRLLVMDialect MLIRParser)
add_opsjit_unittest(opdsl_persistent_cache_test PersistentCacheTest.cpp LINK OPDSLCache)

add_opsjit_unittest(opdsl_gpu_shim_test GpuShimTest.cpp $<TARGET_OBJECTS:OPDSLGpu> LINK ${CMAKE_DL_LIBS})
target_link_options(opdsl_gpu_shim_test PRIVATE -Wl,--export-dynamic)
