find_package(Python3 REQUIRED COMPONENTS Interpreter)

# lit is not vendored. The pinned LLVM install ships FileCheck, not and count, but no lit: use the one on PATH
# (`pip install -r requirements-test.txt`), one next to the LLVM tools (a source build of LLVM has llvm-lit), or
# -DOPSJIT_LIT=<path> (the environment variable of the same name is the default; scripts/env.sh sets it when it finds one).
if(NOT OPSJIT_LIT AND NOT "$ENV{OPSJIT_LIT}" STREQUAL "")
  set(OPSJIT_LIT "$ENV{OPSJIT_LIT}" CACHE FILEPATH "lit executable")
endif()
find_program(OPSJIT_LIT NAMES lit llvm-lit HINTS ${LLVM_TOOLS_BINARY_DIR} DOC "lit executable")
set(OPSJIT_LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})
set(OPSJIT_LLVM_LIBRARY_DIR ${LLVM_LIBRARY_DIR})
# The importer tests parse real OPS kernels: they need the headers of an OPS source tree (OPS_SRC; default: the submodule).
set(_opsjit_ops_default "$ENV{OPS_SRC}")
if("${_opsjit_ops_default}" STREQUAL "" AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/OPS/ops/c/include")
  set(_opsjit_ops_default "${CMAKE_SOURCE_DIR}/third_party/OPS")
endif()
set(OPSJIT_OPS_SRC "${_opsjit_ops_default}" CACHE PATH "OPS source tree (the importer tests use ops/c/include)")
set(OPSJIT_OPS_INCLUDE_DIR "${OPSJIT_OPS_SRC}/ops/c/include")

configure_file(lit/lit.site.cfg.py.in ${CMAKE_CURRENT_BINARY_DIR}/lit/lit.site.cfg.py @ONLY)

if(OPSJIT_LIT)
  add_custom_target(check-ops-mlir-lit
    COMMAND ${OPSJIT_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR}/lit
    DEPENDS opsjit-tools
    USES_TERMINAL)
else()
  message(WARNING "lit not found: check-ops-mlir-lit will fail (pip install -r requirements-test.txt, or -DOPSJIT_LIT=<path>)")
  add_custom_target(check-ops-mlir-lit
    COMMAND ${CMAKE_COMMAND} -E echo "lit not found: pip install -r requirements-test.txt, or configure with -DOPSJIT_LIT=<path>"
    COMMAND ${CMAKE_COMMAND} -E false
    USES_TERMINAL)
endif()
add_dependencies(check-ops-mlir check-ops-mlir-lit)
