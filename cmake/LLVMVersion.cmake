# LLVM/MLIR/clang version pin for ops-mlir v2.
#
# Local development (WSL) and the `cl` cluster use the SAME LLVM install:
#   cluster : ~/MLIR/lib_install/llvm-project-23.1.0-CUDA-Release
#   local   : ~/ops-mlir-work/toolchain/llvm-23.1.0   (an rsync copy of the cluster install)
# See scripts/env.sh. Using one install on both machines removes version skew between what is
# developed and what is measured.
#
# The pin is the release series (major.minor). A different patch release only warns; a different
# series is a configure error, because MLIR's C++ API is not stable across series.
# A ClangIR experiment (scripts/cir) uses its own LLVM build and is exempt: pass
# -DOPSJIT_ALLOW_LLVM_MISMATCH=ON there.

set(OPSJIT_LLVM_PIN_MAJOR 23)
set(OPSJIT_LLVM_PIN_MINOR 1)
set(OPSJIT_LLVM_PIN_PATCH 0)

option(OPSJIT_ALLOW_LLVM_MISMATCH "Do not fail the configure step when LLVM is not the pinned release series" OFF)

# Call after find_package(LLVM) / find_package(MLIR).
function(opsjit_check_llvm_version)
  if(NOT DEFINED LLVM_VERSION_MAJOR)
    message(FATAL_ERROR "opsjit_check_llvm_version: call find_package(LLVM) first")
  endif()
  set(_found "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
  set(_pin "${OPSJIT_LLVM_PIN_MAJOR}.${OPSJIT_LLVM_PIN_MINOR}.${OPSJIT_LLVM_PIN_PATCH}")
  if(NOT LLVM_VERSION_MAJOR EQUAL OPSJIT_LLVM_PIN_MAJOR OR
     NOT LLVM_VERSION_MINOR EQUAL OPSJIT_LLVM_PIN_MINOR)
    if(OPSJIT_ALLOW_LLVM_MISMATCH)
      message(WARNING "LLVM ${_found} found, ops-mlir is pinned to ${_pin} "
                      "(continuing because OPSJIT_ALLOW_LLVM_MISMATCH=ON)")
    else()
      message(FATAL_ERROR "LLVM ${_found} found in ${LLVM_DIR}, but ops-mlir is pinned to the "
                          "${OPSJIT_LLVM_PIN_MAJOR}.${OPSJIT_LLVM_PIN_MINOR} release series "
                          "(${_pin}). Source scripts/env.sh, or pass "
                          "-DOPSJIT_ALLOW_LLVM_MISMATCH=ON to override.")
    endif()
  elseif(NOT LLVM_VERSION_PATCH EQUAL OPSJIT_LLVM_PIN_PATCH)
    message(WARNING "LLVM ${_found} found; ops-mlir is pinned to ${_pin} (same series, different patch)")
  else()
    message(STATUS "LLVM ${_found} matches the pin")
  endif()
  if(NOT LLVM_ENABLE_ASSERTIONS)
    message(STATUS "LLVM was built without assertions: development builds get weaker diagnostics")
  endif()
endfunction()
