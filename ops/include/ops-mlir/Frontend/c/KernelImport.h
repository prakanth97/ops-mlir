//===- KernelImport.h - OPS kernel (C++) -> ops.kernel + manifest ------*- C++ -*-===//
//
// Provider A of the plan: a build-time frontend that reads one OPS user kernel with clang (in the translation unit
// of the application, with its include paths and definitions) and emits
//
//   * an MLIR module holding `ops.kernel @<name>`, and
//   * a manifest (JSON, plan section 4.2): what the kernel's parameters are, which constants it reads, which
//     offsets it accesses, whether it reduces, ...
//
// Anything outside the supported subset makes the import fail with a reason (status "failed" in the manifest): the
// loops that call the kernel then run through the stock OPS code.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_FRONTEND_C_KERNELIMPORT_H
#define OPS_MLIR_FRONTEND_C_KERNELIMPORT_H

#include "ops-mlir/Frontend/common/Import.h"

namespace ops_mlir::frontend {

/// "A/1.0": the provider and its version, recorded in manifests and part of their digests.
const char *importerId();

/// Imports the kernel. `ctx` needs the ops, arith, math and scf dialects.
ImportResult importKernel(mlir::MLIRContext &ctx, const ImportOptions &options);

/// A manifest for a kernel that could not be imported.
std::string failedManifest(const ImportOptions &options, const std::string &reason);

} // namespace ops_mlir::frontend

#endif // OPS_MLIR_FRONTEND_C_KERNELIMPORT_H
