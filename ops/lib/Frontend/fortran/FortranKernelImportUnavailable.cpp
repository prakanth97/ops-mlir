//===- FortranKernelImportUnavailable.cpp - provider F without flang -----------===//

#include "ops-mlir/Frontend/fortran/FortranKernelImport.h"

#include "ops-mlir/Frontend/common/Manifest.h"

namespace ops_mlir::frontend {

const char *fortranImporterId() { return "F/1.0"; }

bool fortranImportAvailable() { return false; }

ImportResult importFortranKernel(mlir::MLIRContext &, const ImportOptions &options) {
  ImportResult result;
  result.reason = "this build of ops-mlir has no Fortran importer (configure with -DOPSJIT_ENABLE_FORTRAN=ON against "
                  "an LLVM build that has flang)";
  result.manifest = writeFailedManifest(options.kernelName, options.ndim, result.reason, fortranImporterId());
  return result;
}

} // namespace ops_mlir::frontend
