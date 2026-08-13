//===- OPSWrapper.h - OPS JIT capture wrapper ----------------------*- C++
//-*-===//
//
// Intercepts OPS API calls to capture program structure for JIT compilation.
// Forwards to the real OPS library for correctness during development.
//
// Users only need: #include "ops/OPSWrapper.h"
//
//===----------------------------------------------------------------------===//

#ifndef OPS_WRAPPER_H
#define OPS_WRAPPER_H

#include "runtime/JITEngine.h"
#include "ops_lib_core.h"

#include <array>
#include <type_traits>

//===----------------------------------------------------------------------===//
// ops_par_loop interception
//===----------------------------------------------------------------------===//
// This template overrides the ops_par_loop to capture loop metadata.
// The captured loops are queued for JIT compilation.
//===----------------------------------------------------------------------===//

template <typename KernelFn, typename... Args>
void ops_par_loop(KernelFn kernel, const char *name, ops_block block, int dims,
                  int *range, Args... opsArgs) {
  static_assert((std::is_same_v<std::decay_t<Args>, ops_arg> && ...),
                "All ops_par_loop variadic arguments must be ops_arg values");

  std::array<ops_arg, sizeof...(Args)> packedArgs{opsArgs...};

  auto token =
      static_cast<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(kernel));

  ops_mlir::JITEngine::instance().enqueueParLoop(
      token, name, block, dims, range, packedArgs.data(), packedArgs.size());
}

void compile_and_execute() {
  ops_mlir::JITEngine::instance().compile_and_execute();
}

// Equvalent to ops_decl_const
template <typename T>
void ops_register_kernel_constant(const char *name, T *data) {
  ops_mlir::JITEngine::instance().registerKernelConstant(name, data);
}

void set_kernel_source_file(const std::string &filePath) {
  ops_mlir::JITEngine::instance().setKernelSourceFile(filePath);
}

// TODO: Use dat.data_d instead of deviceBuffers_. or improve the logic re copy only affected dats.
// Invalidate cached device buffers, forcing a host->device re-copy on next use.
#define ops_halo_transfer(group) ops_mlir::haloTransferIntercepted(group)

#define ops_NaNcheck(dat)                                                    \
  (ops_mlir::JITEngine::instance().syncHostBuffer(dat), ::ops_NaNcheck(dat))
#define ops_fetch_dat_hdf5_file(dat, file)                                   \
  (ops_mlir::JITEngine::instance().syncHostBuffer(dat),                      \
   ::ops_fetch_dat_hdf5_file(dat, file))

// Tear down JITEngine's cached ExecutionEngines deterministically before
// the real ops_exit runs -- see JITEngine::shutdown's comment for why.
#define ops_exit() ops_mlir::exitIntercepted()

#endif // OPS_WRAPPER_H
