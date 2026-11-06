# Builds an OPS Fortran application with the JIT:
#
#   ops_jit_add_fortran_app(<name> SOURCE_DIR <dir>
#                           MAIN <file.F90>...       sources with ops_par_loop calls (translated)
#                           [MODULES <file.F90>...]  modules the kernels use (OPS_CONSTANTS), in dependency order
#                           [OTHERS <file.F90>...]   other sources, compiled as they are
#                           [TRANSLATOR_FLAGS <flags>...])
#
# The application is translated with `ops-translator --jit -t f2c_mpi_openmp` at configure time, its kernels are
# imported with provider F (ops_add_jit_artifacts), and <name>_jit is linked with OPS's f2c library and libopsjit.
#
# Expects, besides what OpsApps.cmake expects: OPS_FORTRAN_F2C_LIB (libops_f2c_seq.a) and OPS_FORTRAN_MOD_DIR (the
# .mod files of OPS's Fortran API, built with -DOPS_F2C_INTEROP by CMAKE_Fortran_COMPILER).

include("${CMAKE_CURRENT_LIST_DIR}/OpsJitArtifacts.cmake")

function(ops_jit_add_fortran_app name)
  cmake_parse_arguments(ARG "" "SOURCE_DIR" "MAIN;MODULES;OTHERS;TRANSLATOR_FLAGS" ${ARGN})
  if(NOT ARG_MAIN)
    message(FATAL_ERROR "ops_jit_add_fortran_app(${name}): MAIN is required")
  endif()
  set(tmp "${CMAKE_CURRENT_BINARY_DIR}/${name}_tmp")
  file(REMOVE_RECURSE "${tmp}")
  file(MAKE_DIRECTORY "${tmp}")
  file(GLOB sources "${ARG_SOURCE_DIR}/*.F90" "${ARG_SOURCE_DIR}/*.inc")
  file(COPY ${sources} DESTINATION "${tmp}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${sources})

  execute_process(
    COMMAND ${OPS_PYTHON} ${OPS_TRANSLATOR} -I ${OPS_INSTALL}/include ${ARG_TRANSLATOR_FLAGS} --jit -t f2c_mpi_openmp
            --file_paths ${ARG_MAIN}
    WORKING_DIRECTORY "${tmp}" RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "ops-translator --jit failed for ${name} (${rc}):\n${out}\n${err}")
  endif()

  set(modules "")
  foreach(m ${ARG_MODULES})
    list(APPEND modules "${tmp}/${m}")
  endforeach()
  set(others "")
  foreach(o ${ARG_OTHERS})
    list(APPEND others "${tmp}/${o}")
  endforeach()
  set(translated "")
  foreach(m ${ARG_MAIN})
    get_filename_component(stem "${m}" NAME_WE)
    list(APPEND translated "${tmp}/${stem}_ops.F90")
  endforeach()
  file(GLOB hosts "${tmp}/f2c_mpi_openmp/*.F90")

  add_library(${name}_jit_artifacts OBJECT)
  target_include_directories(${name}_jit_artifacts PRIVATE "${tmp}" "${OPS_INSTALL}/include")
  set_target_properties(${name}_jit_artifacts PROPERTIES POSITION_INDEPENDENT_CODE ON)
  ops_add_jit_artifacts(${name}_jit_artifacts KERNELS_JSON "${tmp}/jit_kernels.json" IMPORTER ${OPSJIT_IMPORTER}
                        FORTRAN_MODULES ${modules})

  set(exe ${name}_jit)
  add_executable(${exe} ${modules} ${others} ${hosts} ${translated} "${tmp}/f2c_mpi_openmp/f2c_mpi_openmp_kernels.cpp"
                 $<TARGET_OBJECTS:${name}_jit_artifacts>)
  set_target_properties(${exe} PROPERTIES LINKER_LANGUAGE Fortran
                        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${exe}_mod")
  target_include_directories(${exe} PRIVATE "${tmp}" "${OPS_INSTALL}/include" "${OPS_FORTRAN_MOD_DIR}")
  target_compile_definitions(${exe} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:OPS_F2C_INTEROP>
                                            $<$<COMPILE_LANGUAGE:CXX>:OPS_LAZY OPS_FTN>)
  target_link_libraries(${exe} PRIVATE "${OPS_FORTRAN_F2C_LIB}" opsjit OpenMP::OpenMP_CXX stdc++)
  # Run with no arguments in the application's own directory (OPS's Fortran apps print PASSED when they are correct).
  _ops_jit_add_test_command(${exe} "${CMAKE_CURRENT_BINARY_DIR}/${exe}" seq "${tmp}")
endfunction()
