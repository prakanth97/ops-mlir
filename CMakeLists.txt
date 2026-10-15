cmake_minimum_required(VERSION 3.20)
project(ops-mlir CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===============================================================================
# MLIR/LLVM Setup
#===============================================================================

# Paths are read from environment variables. See README.md for setup
# instructions (source env_setup or export these yourself before running cmake).
if(NOT DEFINED ENV{MLIR_BUILD_DIR})
  message(FATAL_ERROR "Environment variable MLIR_BUILD_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{LLVM_BUILD_DIR})
  message(FATAL_ERROR "Environment variable LLVM_BUILD_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{LLVM_SOURCE_DIR})
  message(FATAL_ERROR "Environment variable LLVM_SOURCE_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{MLIR_SOURCE_DIR})
  message(FATAL_ERROR "Environment variable MLIR_SOURCE_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{OPS_ROOT})
  message(FATAL_ERROR "Environment variable OPS_ROOT is not set. See README.md.")
endif()

set(MLIR_BUILD_DIR $ENV{MLIR_BUILD_DIR})
set(LLVM_BUILD_DIR $ENV{LLVM_BUILD_DIR})
set(LLVM_SOURCE_DIR $ENV{LLVM_SOURCE_DIR})
set(MLIR_SOURCE_DIR $ENV{MLIR_SOURCE_DIR})
set(OPS_ROOT $ENV{OPS_ROOT})
set(OPS_INCLUDE_DIR ${OPS_ROOT}/include)

# Find MLIR and LLVM packages
find_package(MLIR REQUIRED CONFIG
  HINTS "${MLIR_BUILD_DIR}/lib/cmake/mlir"
)
find_package(LLVM REQUIRED CONFIG
  HINTS "${LLVM_BUILD_DIR}/lib/cmake/llvm"
)
find_package(Clang REQUIRED CONFIG
  HINTS "${LLVM_BUILD_DIR}/lib/cmake/clang"
)

# Include MLIR cmake utilities
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_BUILD_DIR}/cmake/modules")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

#===============================================================================
# Fortran Frontend Configuration
#===============================================================================
# The Fortran frontend needs an LLVM build with the flang project enabled:
# the JIT links flang's frontend to lower kernel modules to FIR at run time,
# as it links clang for C++ kernel headers. Everything comes from one LLVM
# build, so FIR and MLIR stay ABI-compatible.
option(OPS_ENABLE_FORTRAN "Build the Fortran frontend (requires flang)" OFF)

if(OPS_ENABLE_FORTRAN)
  set(OPS_FLANG_ROOT "${LLVM_BUILD_DIR}" CACHE PATH
    "LLVM build/install prefix providing bin/flang and the flang libraries")

  if(NOT EXISTS "${OPS_FLANG_ROOT}/bin/flang")
    message(FATAL_ERROR
      "OPS_ENABLE_FORTRAN=ON but ${OPS_FLANG_ROOT}/bin/flang is missing.\n"
      "Rebuild LLVM with flang enabled, e.g.\n"
      "  cmake -S llvm -B build -DLLVM_ENABLE_PROJECTS=\"clang;mlir;flang\" \\\n"
      "        -DLLVM_ENABLE_RUNTIMES=\"flang-rt\"\n"
      "  ninja flang\n"
      "or point -DOPS_FLANG_ROOT at an LLVM build that has it.")
  endif()

  # Imports flang's library targets (flangFrontend, FIRDialect, ...) with
  # their link interfaces, the way Clang's package provides clangTooling.
  find_package(Flang REQUIRED CONFIG
    HINTS "${OPS_FLANG_ROOT}/lib/cmake/flang"
  )

  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
  message(STATUS "Fortran frontend enabled (flang root: ${OPS_FLANG_ROOT})")

  # Linking a Fortran application needs flang-rt, which the kernel
  # translator itself does not: it only runs flang's frontend. Keeping the
  # Fortran applications optional lets the translator and its tests build
  # against a flang that has no runtime yet.
  option(OPS_BUILD_FORTRAN_APPS "Build the Fortran example applications" ON)
endif()

#===============================================================================
# CUDA GPU Backend Configuration
#===============================================================================
option(OPS_ENABLE_CUDA "Build the NVIDIA CUDA/NVVM backend pipeline" OFF)

if(OPS_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

#===============================================================================
# OpenMP Setup
#===============================================================================
# omp.h and libomp.so live in the OpenMP runtime build alongside LLVM.
# Derive the path from MLIR_BUILD_DIR (same LLVM tree) — no extra env var needed.
cmake_path(GET MLIR_BUILD_DIR PARENT_PATH LLVM_PROJECT_DIR)
set(OMP_RUNTIME_DIR "${LLVM_PROJECT_DIR}/openmp/build/runtime/src")
if(NOT EXISTS "${OMP_RUNTIME_DIR}/omp.h")
  message(FATAL_ERROR "omp.h not found at ${OMP_RUNTIME_DIR}. "
    "Ensure the LLVM OpenMP runtime was built in-tree alongside MLIR_BUILD_DIR.")
endif()

add_compile_options(-fopenmp)
add_link_options(-fopenmp)

#===============================================================================
# Project Structure
#===============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OPS_INCLUDE_DIR})
include_directories(${OMP_RUNTIME_DIR})
link_directories(${OMP_RUNTIME_DIR})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${MLIR_BUILD_DIR}/include)
include_directories(${LLVM_BUILD_DIR}/include)
include_directories(${MLIR_SOURCE_DIR}/include)
include_directories(${LLVM_SOURCE_DIR}/include)

add_subdirectory(lib)
add_subdirectory(apps)

enable_testing()
add_subdirectory(tools)
add_subdirectory(unittests)

#===============================================================================
# Compilation Database for Language Server
#===============================================================================
# Automatically generated at ${CMAKE_BINARY_DIR}/compile_commands.json
