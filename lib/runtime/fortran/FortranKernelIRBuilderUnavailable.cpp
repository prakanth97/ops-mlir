//===- FortranKernelIRBuilderUnavailable.cpp - no-flang stand-in ----------===//
//
// Part of OPS-MLIR Project
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
// Compiled in place of FortranKernelIRBuilder.cpp when ops-mlir is built
// without a flang installation. Keeping the choice here rather than behind
// preprocessor branches inside JITEngine means the Fortran frontend fails
// with one clear diagnostic instead of a link error, and the C++ frontend is
// unaffected.
//
//===----------------------------------------------------------------------===//

#include "runtime/fortran/FortranKernelIRBuilder.h"

namespace ops_mlir {

bool fortranFIRImportAvailable() { return false; }

mlir::func::FuncOp FortranKernelIRBuilder::generate(
    const FortranKernelSource &source, const std::string &kernelName,
    int indexRank, mlir::FunctionType opsABI,
    const std::map<std::string, const void *> &constants,
    llvm::raw_ostream &errs) {
  (void)source;
  (void)indexRank;
  (void)opsABI;
  (void)constants;
  errs << "FortranKernelIRBuilder: cannot translate '" << kernelName
       << "' -- this build of ops-mlir has no Fortran support. Reconfigure with "
          "-DOPS_ENABLE_FORTRAN=ON against an LLVM build that has the flang "
          "project enabled.\n";
  return nullptr;
}

} // namespace ops_mlir
