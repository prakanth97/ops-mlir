# OPS applications built through the JIT toolchain (the `*_jit*` executables).
# A separate project, configured by scripts/build_apps.sh against
#   * an installed OPS (OPS_INSTALL: headers, libraries, translator with its Python venv), and
#   * a build of ops-mlir (OPSJIT_BUILD: bin/ops-kernel-import, lib/libopsjit.so),
# with the same compilers as OPS (see scripts/build_ops.sh).
cmake_minimum_required(VERSION 3.20)
project(OPS_JIT_APPS C CXX)

set(OPS_INSTALL "$ENV{HOME}/ops-mlir-work/install/ops" CACHE PATH "installed OPS")
set(OPS_SRC "$ENV{HOME}/ops-mlir-work/OPS" CACHE PATH "OPS source tree (apps/c)")
set(OPSJIT_BUILD "$ENV{HOME}/ops-mlir-work/build/ops-mlir" CACHE PATH "ops-mlir build tree")
set(TGV_ROOT "$ENV{HOME}/ops-mlir-work/reguly-tests/apps/TGV_mixed" CACHE PATH "TGV_mixed (github.com/reguly/tests)")
option(OPS_TEST "Register the apps' tests with ctest" ON)
option(OPSJIT_APPS_CUDA "Build the CUDA variants" OFF)
set(OPSJIT_APPS "" CACHE STRING "build only these applications (semicolon separated names as in this file); empty: all")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_C_FLAGS_RELEASE "-O3 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

find_package(OpenMP REQUIRED)
find_package(MPI)
find_package(HDF5 COMPONENTS C HL)
if(OPSJIT_APPS_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
# After the packages above: OPSConfig only imports the backends whose dependencies were found.
find_package(OPS REQUIRED HINTS "${OPS_INSTALL}/lib/cmake")

set(OPS_TRANSLATOR "${OPS_INSTALL}/translator/ops_translator/ops-translator")
set(OPS_PYTHON "${OPS_INSTALL}/translator/ops_translator/ops_venv/bin/python3")
set(OPSJIT_IMPORTER "${OPSJIT_BUILD}/bin/ops-kernel-import")
foreach(f OPS_TRANSLATOR OPS_PYTHON OPSJIT_IMPORTER)
  if(NOT EXISTS "${${f}}")
    message(FATAL_ERROR "${f}: ${${f}} does not exist")
  endif()
endforeach()

add_library(opsjit SHARED IMPORTED)
set_target_properties(opsjit PROPERTIES IMPORTED_LOCATION "${OPSJIT_BUILD}/lib/libopsjit.so"
                      INTERFACE_LINK_OPTIONS "-Wl,-rpath,${OPSJIT_BUILD}/lib")

enable_testing()
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OpsApps.cmake")

set(c_apps "${OPS_SRC}/apps/c")

macro(ops_app name dir)
  if(OPSJIT_APPS AND NOT "${name}" IN_LIST OPSJIT_APPS)
    # not selected
  elseif(EXISTS "${dir}")
    ops_jit_add_app(${name} SOURCE_DIR "${dir}" TEST_SCRIPT "${dir}/cmake_test.sh" ${ARGN})
  else()
    message(WARNING "${dir} does not exist: ${name} skipped")
  endif()
endmacro()

ops_app(laplace2d "${c_apps}/laplace2d_tutorial/step7")
ops_app(poisson "${c_apps}/poisson")
ops_app(lazy_triggers "${c_apps}/lazy_triggers")
ops_app(clover_leaf "${c_apps}/CloverLeaf"
        INPUT_COPY clover_bm_short.in clover.in NO_SEQ_TEST AOT_BASELINE ODD profile OTHERS start.cpp time_step.cpp build_field.cpp read_input.cpp initialise.cpp advection.cpp)
ops_app(clover_leaf_3D "${c_apps}/CloverLeaf_3D"
        NO_SEQ_TEST AOT_BASELINE ODD profile OTHERS start.cpp time_step.cpp build_field.cpp read_input.cpp initialise.cpp advection.cpp)
ops_app(tea_leaf "${c_apps}/TeaLeaf"
        OTHERS build_field.cpp diffuse.cpp initialise.cpp read_input.cpp tea_solve.cpp timestep.cpp)
# mblock and multiDim use the legacy header (ops_seq.h with per-loop globals xdim0...): OPS itself does not build them
# (apps/c/CMakeLists.txt), so they are not built here either. lowdim needs HDF5, like in OPS.
if(HDF5_FOUND AND TARGET OPS::ops_hdf5_seq)
  ops_app(lowdim "${c_apps}/lowdim_test" HDF5)
endif()

# TGV (OpenSBLI), double, single and mixed precision. tgv_<p>: the benchmark as it is (for the performance runs);
# tgvs_<p>: 33^3 points and 3 steps, run in verify mode by ctest (the application prints no QA value).
if(HDF5_FOUND AND TARGET OPS::ops_hdf5_seq)
  foreach(variant DP SP SPDP HP HPSP)
    if(EXISTS "${TGV_ROOT}/TGsym_${variant}")
      if(NOT OPSJIT_APPS OR tgv_${variant} IN_LIST OPSJIT_APPS)
      ops_jit_add_app(tgv_${variant} SOURCE_DIR "${TGV_ROOT}/TGsym_${variant}" HDF5 AOT_BASELINE
                      IMPORTER_FLAGS -I${HDF5_INCLUDE_DIRS})
      endif()
      if(NOT OPSJIT_APPS OR tgvs_${variant} IN_LIST OPSJIT_APPS)
      ops_jit_add_app(tgvs_${variant} SOURCE_DIR "${TGV_ROOT}/TGsym_${variant}" HDF5 VERIFY_ONLY
                      IMPORTER_FLAGS -I${HDF5_INCLUDE_DIRS}
                      PATCH_FILE opensbli.cpp
                      PATCH "block0np0 = mult\\*512\\+1" "block0np0 = mult*32+1"
                            "block0np1 = 513" "block0np1 = 33"
                            "block0np2 = 513" "block0np2 = 33"
                            "niter = 20" "niter = 3")
      endif()
    endif()
  endforeach()
else()
  message(STATUS "HDF5 or OPS::ops_hdf5_seq not found: TGV_mixed skipped")
endif()
