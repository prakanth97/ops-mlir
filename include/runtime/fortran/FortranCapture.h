//===- FortranCapture.h - C ABI used by the Fortran capture module -*- C -*-===//
//
// Part of OPS-MLIR Project
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
// The C entry points that OPS_MLIR_Fortran_Reference (see
// lib/runtime/fortran/ops_mlir_fortran_reference.F90) binds to. They are the
// Fortran counterpart of the inline wrappers in include/ops/OPSWrapper.h.
//
// `block` is an OPS `ops_block` and `args` points to a contiguous array of
// OPS `ops_arg` values. Both are spelled as void pointers so this header
// stays usable from an ISO_C_BINDING interface without exposing C++ types.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_FORTRAN_CAPTURE_H
#define OPS_MLIR_RUNTIME_FORTRAN_CAPTURE_H

#ifdef __cplusplus
extern "C" {
#endif

/// Captures one `ops_par_loop`. `range` is already zero-based and half-open
/// -- the Fortran side converts from OPS Fortran's one-based inclusive form
/// before calling, matching what the translator-generated host stubs do.
void ops_mlir_fortran_enqueue(const char *kernel_name, void *block, int dims,
                              const int *range, const void *args, int nargs);

/// Compiles and runs everything captured since the last call.
void ops_mlir_fortran_compile_and_execute(void);

/// Registers the Fortran module holding the application's kernel
/// procedures. Fortran's analogue of set_kernel_source_file().
void ops_mlir_fortran_set_kernel_source_file(const char *path);

/// Registers a module the kernel module `use`s, compiled before it.
void ops_mlir_fortran_add_kernel_dependency(const char *path);

/// Snapshots a module constant's value so the kernel translator can bake it into
/// translated kernel bodies. The bytes are copied: a Fortran actual argument
/// need not have a stable address, so retaining the pointer would be unsafe.
void ops_mlir_fortran_register_constant(const char *name, const void *data,
                                        int bytes);

void ops_mlir_fortran_sync_all_host_buffers(void);

/// Tears down the JIT's cached ExecutionEngines before the real ops_exit
/// runs -- see JITEngine::shutdown for why the order matters.
void ops_mlir_fortran_exit(void);

#ifdef __cplusplus
}
#endif

#endif // OPS_MLIR_RUNTIME_FORTRAN_CAPTURE_H
