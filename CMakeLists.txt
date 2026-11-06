cmake_minimum_required(VERSION 3.20)
project(OpsMlir VERSION 0.1.0 LANGUAGES C CXX)

# MLIR 23 is C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LLVMVersion)

option(OPSJIT_BUILD_RUNTIME "Build libopsjit, the OPS-facing runtime (needs an OPS install in OPS_INSTALL)" ON)
set(OPS_INSTALL "$ENV{OPS_INSTALL}" CACHE PATH "Install prefix of OPS (include/ and lib/); see scripts/build_ops.sh")

# Hosts without zstd development files (the WSL development image: no sudo to install them) can point
# OPSJIT_ZSTD_LIBRARY at the zstd runtime library. LLVM's exported targets require zstd::libzstd_shared,
# and nothing in this project includes a zstd header.
set(OPSJIT_ZSTD_LIBRARY "" CACHE FILEPATH "zstd runtime library to use when its development package is missing")
if(OPSJIT_ZSTD_LIBRARY AND NOT TARGET zstd::libzstd_shared)
  add_library(zstd::libzstd_shared SHARED IMPORTED)
  set_target_properties(zstd::libzstd_shared PROPERTIES IMPORTED_LOCATION "${OPSJIT_ZSTD_LIBRARY}")
endif()

# The pinned LLVM/MLIR/clang install: see scripts/env.sh (MLIR_DIR) and cmake/LLVMVersion.cmake.
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
opsjit_check_llvm_version()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Clang: the build-time kernel importer (ops/lib/Frontend) parses OPS kernels with it.
find_package(Clang REQUIRED CONFIG HINTS "${LLVM_DIR}/../clang")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
# clang finds its builtin headers (stddef.h...) relative to its own executable; ours is not clang.
set(OPSJIT_CLANG_RESOURCE_DIR "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}" CACHE PATH "clang resource directory")

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include(OpsJitTesting)

# Layering (see docs/design.md): core/ (namespace opdsl) knows nothing about OPS; ops/ is the OPS dialect,
# analyses, transforms and the runtime that plugs into OPS; tools/ are command-line drivers.
add_subdirectory(core)
add_subdirectory(ops)
add_subdirectory(tools)
add_subdirectory(test)
