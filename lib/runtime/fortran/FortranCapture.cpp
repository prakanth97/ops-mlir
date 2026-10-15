//===- FortranCapture.cpp - C ABI used by the Fortran capture module ------===//
//
// Part of OPS-MLIR Project
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include "runtime/fortran/FortranCapture.h"

#include "runtime/common/JITEngine.h"

#include <cstddef>

extern "C" void ops_mlir_fortran_enqueue(const char *kernelName, void *block,
                                         int dims, const int *range,
                                         const void *args, int nargs) {
  if (nargs < 0)
    return;

  // Fortran kernel bodies are always translated from source, so there is no
  // host function pointer to fall back on and the token stays zero -- the
  // kernel name string carries the identity from here on. (JITEngine keys
  // its module cache on the loop descriptors' kernel names, not tokens.)
  ops_mlir::JITEngine::instance().enqueueParLoop(
      /*kernelToken=*/0, kernelName, static_cast<ops_block>(block), dims, range,
      static_cast<const ops_arg *>(args), static_cast<std::size_t>(nargs));
}

extern "C" void ops_mlir_fortran_compile_and_execute(void) {
  ops_mlir::JITEngine::instance().compile_and_execute();
}

extern "C" void ops_mlir_fortran_set_kernel_source_file(const char *path) {
  ops_mlir::JITEngine::instance().setFortranKernelSourceFile(path);
}

extern "C" void ops_mlir_fortran_add_kernel_dependency(const char *path) {
  ops_mlir::JITEngine::instance().addFortranKernelDependency(path);
}

extern "C" void ops_mlir_fortran_register_constant(const char *name,
                                                   const void *data,
                                                   int bytes) {
  if (bytes <= 0 || !data)
    return;
  ops_mlir::JITEngine::instance().registerKernelConstantValue(
      name, data, static_cast<std::size_t>(bytes));
}

extern "C" void ops_mlir_fortran_sync_all_host_buffers(void) {
  ops_mlir::JITEngine::instance().syncAllHostBuffers();
}

extern "C" void ops_mlir_fortran_exit(void) { ops_mlir::exitIntercepted(); }
