# Test infrastructure.
#
#   check-ops-mlir        everything: lit tests + unit tests (what scripts/ci.sh runs)
#   check-ops-mlir-lit    lit/FileCheck tests under test/lit (needs the tools: target opsjit-tools)
#   check-ops-mlir-unit   unit tests (ctest label "unit")
#
# Unit tests are plain executables with the small harness in core/include/opdsl/Testing/TestHarness.h
# (the pinned LLVM install ships no gtest). Add one with add_opsjit_unittest().

enable_testing()

# All command-line tools register here so that the lit tests can depend on them.
add_custom_target(opsjit-tools)
# All unit-test executables register here.
add_custom_target(opsjit-unittests)

add_custom_target(check-ops-mlir-unit
  COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} -L unit --output-on-failure
  DEPENDS opsjit-unittests
  USES_TERMINAL)

add_custom_target(check-ops-mlir DEPENDS check-ops-mlir-unit)

# add_opsjit_unittest(<name> <sources...> [LINK <libs...>] [LABELS <extra labels...>])
function(add_opsjit_unittest name)
  cmake_parse_arguments(ARG "" "" "LINK;LABELS" ${ARGN})
  add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
  target_link_libraries(${name} PRIVATE OPDSLTestMain ${ARG_LINK})
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "unit;${ARG_LABELS}")
  add_dependencies(opsjit-unittests ${name})
endfunction()

# add_opsjit_tool(<name> <sources...> [LINK <libs...>])
function(add_opsjit_tool name)
  cmake_parse_arguments(ARG "" "" "LINK" ${ARGN})
  add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
  target_link_libraries(${name} PRIVATE ${ARG_LINK})
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  add_dependencies(opsjit-tools ${name})
endfunction()
