# ops/: the OPS-specific part (namespace ops_mlir): the `ops` dialect, analyses, transforms, conversions,
# pipelines, the build-time kernel importer and the runtime that plugs into OPS's lazy execution.

# `#include "ops-mlir/..."` finds both the sources and the generated .inc files.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include/ops-mlir/Dialect/OPS/IR)
add_subdirectory(lib/Dialect/OPS/IR)
add_subdirectory(include/ops-mlir/Transforms)
add_subdirectory(lib/Analysis)
add_subdirectory(lib/Transforms)
add_subdirectory(include/ops-mlir/Conversion)
add_subdirectory(lib/Conversion)
add_subdirectory(lib/Pipelines)
add_subdirectory(lib/Frontend)
add_subdirectory(lib/Runtime)
add_subdirectory(unittests)
