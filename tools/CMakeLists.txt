# Standalone developer utilities (not part of any app or test suite).

add_executable(kernel-ir-dump kernel-ir-dump.cpp)
target_link_libraries(kernel-ir-dump PRIVATE OPSRuntime LLVMSupport)
# This LLVM/MLIR build has LLVM_ENABLE_RTTI=OFF, so its static libraries
# contain no typeinfo for LLVM classes. kernel-ir-dump.cpp is the first
# target here to use llvm::cl::opt/cl::list, which need cl::Option's
# typeinfo -- compiling this TU with RTTI on (the GCC default) then leaves
# "undefined reference to typeinfo for llvm::cl::Option" at link time.
# Match the library's RTTI setting instead.
target_compile_options(kernel-ir-dump PRIVATE -fno-rtti)

# --- CTest coverage --------------------------------------------------------
# KernelIRBuilderTest.cpp already covers KernelIRBuilder itself against
# synthetic snippets; these exercise the *tool* end-to-end against real
# kernel sources shipped in this repo, so a regression in either the CLI
# plumbing or the real kernels' translatability shows up in `ctest`.

add_test(NAME KernelIRDump_SingleReturn
  COMMAND kernel-ir-dump
    ${CMAKE_SOURCE_DIR}/apps/c/laplace_2d/laplace_kernels.h
    apply_stencil 2)
set_tests_properties(KernelIRDump_SingleReturn PROPERTIES
  PASS_REGULAR_EXPRESSION "func.func private @apply_stencil.*-> f64")

add_test(NAME KernelIRDump_OutPointer
  COMMAND kernel-ir-dump
    ${CMAKE_SOURCE_DIR}/apps/c/taylor_green_vortex/opensbliblock00_kernels.h
    opensbliblock00Kernel039 3
    -double Delta0block0=0.1 -double Delta1block0=0.1 -double Delta2block0=0.1
    -double Minf=0.1 -double gama=1.4)
set_tests_properties(KernelIRDump_OutPointer PROPERTIES
  PASS_REGULAR_EXPRESSION "memref<3xi32>.*memref<5xf64>")

add_test(NAME KernelIRDump_UnknownKernelFails
  COMMAND kernel-ir-dump
    ${CMAKE_SOURCE_DIR}/apps/c/laplace_2d/laplace_kernels.h
    does_not_exist 2)
set_tests_properties(KernelIRDump_UnknownKernelFails PROPERTIES WILL_FAIL TRUE)

#===============================================================================
# Fortran frontend
#===============================================================================
if(OPS_ENABLE_FORTRAN)
  add_executable(fortran-kernel-ir-dump fortran-kernel-ir-dump.cpp)
  target_link_libraries(fortran-kernel-ir-dump PRIVATE OPSRuntime LLVMSupport)
  target_compile_options(fortran-kernel-ir-dump PRIVATE -fno-rtti)

  # These tests translate the Laplace app's kernel sources directly, so the
  # translator stays testable in a build that has no Fortran compiler
  # runtime and no Fortran applications enabled.
  set(_laplace_app "${CMAKE_SOURCE_DIR}/apps/fortran/laplace_2d")
  set(_laplace_kernels "${_laplace_app}/laplace_kernels.F90"
      -dep "${_laplace_app}/laplace_constants.F90")

  # A pure scalar kernel: five stencil neighbours in, one value out. Nothing
  # of Fortran's by-reference ABI should survive the translation.
  add_test(NAME FortranKernelIRDump_ScalarKernel
    COMMAND fortran-kernel-ir-dump ${_laplace_kernels} apply_stencil 2 -scalars 5)
  set_tests_properties(FortranKernelIRDump_ScalarKernel PROPERTIES
    PASS_REGULAR_EXPRESSION
      "func.func private @apply_stencil\\(%[^)]*f64.*-> f64")

  # An ops_arg_idx kernel, which also exercises baking a module variable
  # (jmax) and folding a Fortran `parameter` (pi).
  add_test(NAME FortranKernelIRDump_IndexKernel
    COMMAND fortran-kernel-ir-dump ${_laplace_kernels} left_bndcon 2
            -idx -int jmax=4094)
  set_tests_properties(FortranKernelIRDump_IndexKernel PROPERTIES
    PASS_REGULAR_EXPRESSION "memref<2xi32>.*math.sin")

  add_test(NAME FortranKernelIRDump_UnknownKernelFails
    COMMAND fortran-kernel-ir-dump ${_laplace_kernels} does_not_exist 2)
  set_tests_properties(FortranKernelIRDump_UnknownKernelFails PROPERTIES
    WILL_FAIL TRUE)

  # An unregistered module variable must be reported, not silently mistaken
  # for zero.
  add_test(NAME FortranKernelIRDump_UnregisteredConstantFails
    COMMAND fortran-kernel-ir-dump ${_laplace_kernels} left_bndcon 2 -idx)
  set_tests_properties(FortranKernelIRDump_UnregisteredConstantFails PROPERTIES
    WILL_FAIL TRUE)

  # Without its OPS_CONSTANTS dependency the kernel module does not compile;
  # that has to surface as a failure, not as an empty translation.
  add_test(NAME FortranKernelIRDump_MissingDependencyFails
    COMMAND fortran-kernel-ir-dump "${_laplace_app}/laplace_kernels.F90"
            apply_stencil 2 -scalars 5)
  set_tests_properties(FortranKernelIRDump_MissingDependencyFails PROPERTIES
    WILL_FAIL TRUE)
endif()
