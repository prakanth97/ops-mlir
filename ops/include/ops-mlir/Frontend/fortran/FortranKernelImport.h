//===- FortranKernelImport.h - OPS kernel (Fortran) -> ops.kernel + manifest ---*- C++ -*-===//

#ifndef OPS_MLIR_FRONTEND_FORTRAN_FORTRANKERNELIMPORT_H
#define OPS_MLIR_FRONTEND_FORTRAN_FORTRANKERNELIMPORT_H

#include "ops-mlir/Frontend/common/Import.h"

namespace ops_mlir::frontend {

const char *fortranImporterId();

bool fortranImportAvailable();

ImportResult importFortranKernel(mlir::MLIRContext &ctx, const ImportOptions &options);

} // namespace ops_mlir::frontend

#endif // OPS_MLIR_FRONTEND_FORTRAN_FORTRANKERNELIMPORT_H
