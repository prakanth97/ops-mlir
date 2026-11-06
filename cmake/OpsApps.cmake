# Builds an OPS application with the JIT:
#
#   ops_jit_add_app(<name> SOURCE_DIR <dir>
#                   [ODD <regex>...]       sources that other sources #include (no code generation, no compilation)
#                   [OTHERS <regex>...]    sources that need no code generation
#                   [EXTRA <regex>...]     sources the helper must ignore
#                   [HDF5]                 link the HDF5 builds of the OPS libraries
#                   [INPUT_COPY <from> <to>]  copy an input file to the name the app reads (CloverLeaf: clover_bm_short.in -> clover.in)
#                   [NO_SEQ_TEST]          no ctest for the sequential build (too slow: CloverLeaf, like in OPS)
#                   [TRANSLATOR_FLAGS <flags>...] [IMPORTER_FLAGS <flags>...]
#                   [TEST_SCRIPT <cmake_test.sh>])
#
# The helper translates the application with `ops-translator --jit` at configure time (like OPS's own
# build_ops_c_sample), imports its kernels (ops_add_jit_artifacts) and builds, for what the OPS installation and the
# toolchain offer,
#
#   <name>_jit            sequential/OpenMP     OPS::ops_seq
#   <name>_jit_mpi        MPI                   OPS::ops_mpi
#   <name>_jit_cuda       CUDA                  OPS::ops_cuda
#   <name>_jit_mpi_cuda   MPI + CUDA            OPS::ops_mpi_cuda
#
# all with -DOPS_LAZY, linked with libopsjit. With TEST_SCRIPT (OPS's apps/c/<app>/cmake_test.sh) and OPS_TEST=ON the
# same tests as the AOT builds run.
#
# Expects: OPS_INSTALL (find_package(OPS) was done), OPS_PYTHON and OPS_TRANSLATOR, the targets `opsjit` and the
# variable OPSJIT_IMPORTER.

include("${CMAKE_CURRENT_LIST_DIR}/OpsJitArtifacts.cmake")
set(OPSJIT_E2E_DIR "${CMAKE_CURRENT_LIST_DIR}/../test/e2e")

function(_ops_jit_add_test exe_target script mode tmp)
  if(NOT OPS_TEST OR NOT EXISTS "${script}")
    return()
  endif()
  # The script prints the command line for the executable that the build will produce.
  execute_process(COMMAND /bin/bash ${script} "${CMAKE_CURRENT_BINARY_DIR}/${exe_target}" ${mode}
                  OUTPUT_VARIABLE command RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT rc EQUAL 0 OR "${command}" STREQUAL "")
    return()
  endif()
  # The same run in three configurations of the JIT: default, loops not fused, every group checked against OPS.
  set(labels "jit-app")
  if(mode MATCHES "cuda")
    list(APPEND labels gpu)
  else()
    list(APPEND labels cpu)
  endif()
  foreach(variant "" "_nofusion" "_verify")
    set(environment "")
    if(variant STREQUAL "_nofusion")
      set(environment "OPS_JIT_FUSION=0")
    elseif(variant STREQUAL "_verify")
      set(environment "OPS_JIT_VERIFY=1")
    endif()
    add_test(NAME ${exe_target}${variant}
             COMMAND ${CMAKE_COMMAND} "-DCMD=${command}" "-DARG= " -DOPS_INSTALL_PATH=${OPS_INSTALL}
                     -P ${OPS_SRC}/apps/c/runtests.cmake
             WORKING_DIRECTORY "${tmp}")
    set(timeout 1800)
    if(variant STREQUAL "_verify")
      set(timeout 7200)
    endif()
    set_tests_properties(${exe_target}${variant} PROPERTIES LABELS "${labels}" RESOURCE_LOCK "${tmp}"
                         ENVIRONMENT "${environment}" TIMEOUT ${timeout})
  endforeach()
  # Two runs with a fresh disk cache: the second compiles nothing.
  add_test(NAME ${exe_target}_cache
           COMMAND ${CMAKE_COMMAND} "-DCMD=${command}" -DDIR=${CMAKE_CURRENT_BINARY_DIR}/${exe_target}_cache -DWORKDIR=${tmp}
                   -P ${OPSJIT_E2E_DIR}/CacheTest.cmake)
  set_tests_properties(${exe_target}_cache PROPERTIES LABELS "${labels}" RESOURCE_LOCK "${tmp}")
endfunction()

# For applications that print no PASSED line (TGV): run in verify mode; every group must match OPS and at least
# CLAIM_MIN percent of the dynamic loops must have been claimed by the JIT.
function(_ops_jit_add_verify_test exe_target mode tmp)
  if(NOT OPS_TEST)
    return()
  endif()
  set(command "${CMAKE_CURRENT_BINARY_DIR}/${exe_target}")
  if(mode STREQUAL "mpi")
    set(command "mpirun -np 4 --oversubscribe ${command}")
  elseif(mode STREQUAL "mpi_cuda")
    set(command "mpirun -np 1 --oversubscribe ${command}")
  endif()
  set(labels "jit-app")
  if(mode MATCHES "cuda")
    list(APPEND labels gpu)
  else()
    list(APPEND labels cpu)
  endif()
  add_test(NAME ${exe_target}_verify
           COMMAND ${CMAKE_COMMAND} "-DCMD=${command}" -DWORKDIR=${tmp} -DCLAIM_MIN=95 -P ${OPSJIT_E2E_DIR}/VerifyTest.cmake)
  set_tests_properties(${exe_target}_verify PROPERTIES LABELS "${labels}" RESOURCE_LOCK "${tmp}")
  if(exe_target MATCHES "HPSP")
    # Half storage with single precision arithmetic: clang rounds a whole expression once (and the stored quantities are
    # differences of nearly equal values), the JIT every operation: verify only catches gross errors (NaN, inf, garbage).
    set_tests_properties(${exe_target}_verify PROPERTIES ENVIRONMENT "OPS_JIT_VERIFY_HALF_TOL=1.0;OPS_JIT_VERIFY_FLOAT_TOL=1e-2")
  endif()
endfunction()

# Copies the sources of an application to <tmp>, applies PATCH, runs the translator (with --jit or without) and sets
# <prefix>_ops_sources, <prefix>_others in the caller's scope.
function(_ops_jit_prepare prefix tmp jit)
  file(REMOVE_RECURSE "${tmp}")
  file(MAKE_DIRECTORY "${tmp}")
  file(GLOB cpp "${ARG_SOURCE_DIR}/*.cpp")
  list(FILTER cpp EXCLUDE REGEX "ops.cpp")
  file(COPY ${cpp} DESTINATION "${tmp}")
  file(GLOB heads "${ARG_SOURCE_DIR}/*.h")
  file(COPY ${heads} DESTINATION "${tmp}")
  file(GLOB inputs "${ARG_SOURCE_DIR}/*.in")
  if(inputs)
    file(COPY ${inputs} DESTINATION "${tmp}")
  endif()
  if(ARG_INPUT_COPY)
    list(GET ARG_INPUT_COPY 0 copy_from)
    list(GET ARG_INPUT_COPY 1 copy_to)
    file(COPY_FILE "${tmp}/${copy_from}" "${tmp}/${copy_to}")
  endif()
  # PATCH <regex> <replacement> ... applied to PATCH_FILE of the copy (reduced problem sizes for tests).
  if(ARG_PATCH)
    file(READ "${tmp}/${ARG_PATCH_FILE}" patched)
    list(LENGTH ARG_PATCH npatch)
    math(EXPR last "${npatch} - 2")
    foreach(i RANGE 0 ${last} 2)
      math(EXPR j "${i} + 1")
      list(GET ARG_PATCH ${i} regex)
      list(GET ARG_PATCH ${j} replacement)
      string(REGEX REPLACE "${regex}" "${replacement}" patched "${patched}")
    endforeach()
    file(WRITE "${tmp}/${ARG_PATCH_FILE}" "${patched}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${cpp} ${heads})

  file(GLOB dev "${tmp}/*.cpp")
  if(ARG_ODD)
    list(FILTER dev EXCLUDE REGEX "${ARG_ODD}")
  endif()
  foreach(pattern ${ARG_OTHERS} ${ARG_EXTRA})
    list(FILTER dev EXCLUDE REGEX "${pattern}")
  endforeach()
  set(names "")
  foreach(path ${dev})
    get_filename_component(file "${path}" NAME)
    list(APPEND names "${file}")
  endforeach()

  set(jit_flag "")
  if(jit)
    set(jit_flag --jit)
  endif()
  execute_process(
    COMMAND ${OPS_PYTHON} ${OPS_TRANSLATOR} -DOPS_ACC_IGNORE -I ${OPS_INSTALL}/include ${ARG_TRANSLATOR_FLAGS}
            ${jit_flag} --file_paths ${names}
    WORKING_DIRECTORY "${tmp}" RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "ops-translator ${jit_flag} failed for ${name} (${rc}):\n${out}\n${err}")
  endif()

  file(GLOB ops_sources "${tmp}/*ops*.cpp")
  if(ARG_ODD)
    list(FILTER ops_sources EXCLUDE REGEX "${ARG_ODD}")
  endif()
  file(GLOB others "${tmp}/*.cpp")
  list(REMOVE_ITEM others ${ops_sources} ${dev})
  if(ARG_ODD)
    list(FILTER others EXCLUDE REGEX "${ARG_ODD}")
  endif()
  foreach(pattern ${ARG_EXTRA})
    list(FILTER others EXCLUDE REGEX "${pattern}")
  endforeach()
  set(${prefix}_ops_sources "${ops_sources}" PARENT_SCOPE)
  set(${prefix}_others "${others}" PARENT_SCOPE)
endfunction()

function(ops_jit_add_app name)
  cmake_parse_arguments(ARG "HDF5;NO_SEQ_TEST;VERIFY_ONLY;AOT_BASELINE" "SOURCE_DIR;TEST_SCRIPT;PATCH_FILE" "ODD;OTHERS;EXTRA;TRANSLATOR_FLAGS;IMPORTER_FLAGS;INPUT_COPY;PATCH" ${ARGN})
  set(tmp "${CMAKE_CURRENT_BINARY_DIR}/${name}_tmp")
  _ops_jit_prepare(jit "${tmp}" TRUE)
  set(ops_sources "${jit_ops_sources}")
  set(others "${jit_others}")

  # The artifacts, imported once per application: they depend on the application's headers, not on the backend.
  add_library(${name}_jit_artifacts OBJECT)
  target_include_directories(${name}_jit_artifacts PRIVATE "${tmp}" "${OPS_INSTALL}/include")
  set_target_properties(${name}_jit_artifacts PROPERTIES POSITION_INDEPENDENT_CODE ON)
  ops_add_jit_artifacts(${name}_jit_artifacts KERNELS_JSON "${tmp}/jit_kernels.json"
                        IMPORTER ${OPSJIT_IMPORTER} CLANG_ARGS -DOPS_ACC_IGNORE ${ARG_IMPORTER_FLAGS})

  set(common_libs opsjit OpenMP::OpenMP_CXX)
  if(ARG_HDF5)
    set(hdf5_seq OPS::ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl MPI::MPI_CXX)
    set(hdf5_mpi OPS::ops_hdf5_mpi hdf5::hdf5 hdf5::hdf5_hl MPI::MPI_CXX)
  endif()

  # <suffix> <kernels> <definitions> <libraries> <test mode>
  function(_variant suffix kernels defs libs mode)
    set(exe ${name}_jit${suffix})
    add_executable(${exe} ${ops_sources} ${others} "${tmp}/${kernels}" $<TARGET_OBJECTS:${name}_jit_artifacts>)
    target_include_directories(${exe} PRIVATE "${tmp}")
    target_compile_definitions(${exe} PRIVATE OPS_LAZY ${defs})
    target_link_libraries(${exe} PRIVATE ${libs} ${common_libs})
    if(NOT (ARG_NO_SEQ_TEST AND mode STREQUAL "seq"))
      if(ARG_VERIFY_ONLY)
      _ops_jit_add_verify_test(${exe} "${mode}" "${tmp}")
    else()
      _ops_jit_add_test(${exe} "${ARG_TEST_SCRIPT}" "${mode}" "${tmp}")
    endif()
    endif()
  endfunction()

  _variant("" mpi_openmp/mpi_openmp_kernels.cpp "" "${hdf5_seq};OPS::ops_seq" seq)
  if(TARGET OPS::ops_mpi)
    _variant("_mpi" mpi_openmp/mpi_openmp_kernels.cpp "OPS_MPI" "${hdf5_mpi};OPS::ops_mpi;MPI::MPI_CXX" mpi)
  endif()
  if(OPSJIT_APPS_CUDA AND TARGET OPS::ops_cuda)
    _variant("_cuda" cuda/cuda_kernels.cu "" "${hdf5_seq};OPS::ops_cuda;CUDA::cudart_static;-lcurand;-lnvidia-ml" cuda)
    target_compile_options(${name}_jit_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    if(TARGET OPS::ops_mpi_cuda)
      _variant("_mpi_cuda" cuda/cuda_kernels.cu "OPS_MPI;MPICH_IGNORE_CXX_SEEK"
               "${hdf5_mpi};OPS::ops_mpi_cuda;MPI::MPI_CXX;CUDA::cudart_static;-lcurand;-lnvidia-ml" "mpi_cuda 1")
      target_compile_options(${name}_jit_mpi_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    endif()
  endif()

  # The ahead-of-time code of the same application (no --jit, eager execution, no libopsjit), for comparisons: the standard
  # OPS builds `<name>_aot` (sequential/OpenMP), `_aot_mpi`, `_aot_cuda`, `_aot_mpi_cuda`.
  if(ARG_AOT_BASELINE)
    set(aot_tmp "${CMAKE_CURRENT_BINARY_DIR}/${name}_aot_tmp")
    _ops_jit_prepare(aot "${aot_tmp}" FALSE)
    set(aot_common_libs OpenMP::OpenMP_CXX)
    function(_aot suffix kernels defs libs)
      set(exe ${name}_aot${suffix})
      add_executable(${exe} ${aot_ops_sources} ${aot_others} "${aot_tmp}/${kernels}")
      target_include_directories(${exe} PRIVATE "${aot_tmp}")
      target_compile_definitions(${exe} PRIVATE ${defs})
      target_link_libraries(${exe} PRIVATE ${libs} ${aot_common_libs})
    endfunction()
    _aot("" mpi_openmp/mpi_openmp_kernels.cpp "" "${hdf5_seq};OPS::ops_seq")
    if(TARGET OPS::ops_mpi)
      _aot("_mpi" mpi_openmp/mpi_openmp_kernels.cpp "OPS_MPI" "${hdf5_mpi};OPS::ops_mpi;MPI::MPI_CXX")
    endif()
    if(OPSJIT_APPS_CUDA AND TARGET OPS::ops_cuda)
      _aot("_cuda" cuda/cuda_kernels.cu "" "${hdf5_seq};OPS::ops_cuda;CUDA::cudart_static;-lcurand;-lnvidia-ml")
      target_compile_options(${name}_aot_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
      if(TARGET OPS::ops_mpi_cuda)
        _aot("_mpi_cuda" cuda/cuda_kernels.cu "OPS_MPI;MPICH_IGNORE_CXX_SEEK"
             "${hdf5_mpi};OPS::ops_mpi_cuda;MPI::MPI_CXX;CUDA::cudart_static;-lcurand;-lnvidia-ml")
        target_compile_options(${name}_aot_mpi_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
      endif()
    endif()
  endif()
endfunction()
