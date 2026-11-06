add_opsjit_tool(ops-jit-replay ops-jit-replay.cpp
  LINK OPSMLIRRuntimeCore OPSMLIRDialect OPSMLIRPipelines LLVMSupport)
llvm_update_compile_flags(ops-jit-replay)
# The OpenMP runtime the groups were compiled against (the application brings its own; this tool needs one).
find_library(OPSJIT_LIBOMP omp HINTS "${LLVM_LIBRARY_DIR}" "${LLVM_LIBRARY_DIR}/x86_64-unknown-linux-gnu" NO_DEFAULT_PATH)
target_compile_definitions(ops-jit-replay PRIVATE OPSJIT_LIBOMP="${OPSJIT_LIBOMP}")
