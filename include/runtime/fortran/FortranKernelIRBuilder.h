//===- FortranKernelIRBuilder.h - Fortran kernel body -> MLIR ---*- C++ -*-===//
//
// Part of OPS-MLIR Project
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
// Translates a single OPS kernel procedure from Fortran source into a real
// MLIR function body, so backends (notably CUDA) get device code rather than
// a symbol resolved against a compiled host binary.
//
// This is the Fortran counterpart of KernelIRBuilder: same contract, same
// resulting `func.func` shape, different source language. Where
// KernelIRBuilder runs clang in-process and walks its AST, this runs flang's
// frontend in-process to lower the kernel module to FIR, then walks that.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_FORTRAN_KERNEL_IR_BUILDER_H
#define OPS_MLIR_RUNTIME_FORTRAN_KERNEL_IR_BUILDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <vector>

namespace ops_mlir {

/// Where a Fortran application's kernels live. Unlike a C++ kernel header,
/// a Fortran kernel module usually `use`s other modules (the application's
/// constants, for instance), and flang needs their `.mod` files to check
/// it. Those modules are compiled from source first, in the order given --
/// a module before the sources that `use` it -- the same way a compiler
/// would see them, so the application never hands over build-tree paths.
struct FortranKernelSource {
  std::string file;
  std::vector<std::string> dependencies;
};

class FortranKernelIRBuilder {
public:
  explicit FortranKernelIRBuilder(mlir::MLIRContext &context)
      : context_(context) {}

  /// Translates `kernelName` from the Fortran `source`, producing a private
  /// `func.func` whose signature matches `opsABI` -- the declaration xDSL's
  /// stencil lowering left behind, which fixes how many scalars the kernel
  /// receives and whether it returns a value.
  ///
  /// `indexRank` is the block dimensionality, used to size the `memref` an
  /// `ops_arg_idx` argument arrives as. `constants` maps Fortran source
  /// names to the live addresses registered through
  /// JITEngine::registerKernelConstant; module globals the kernel reads are
  /// baked in from those.
  ///
  /// Returns null (leaving diagnostics on `errs`) if the source does not
  /// compile, the procedure is absent, or its body uses FIR outside the
  /// supported subset. Callers can then fall back to symbol binding.
  mlir::func::FuncOp generate(const FortranKernelSource &source,
                              const std::string &kernelName, int indexRank,
                              mlir::FunctionType opsABI,
                              const std::map<std::string, const void *> &constants,
                              llvm::raw_ostream &errs);

private:
  mlir::MLIRContext &context_;
};

/// True when this build can translate Fortran at all. Compiled out (returning
/// false) when ops-mlir was configured without a flang installation, so the
/// Fortran frontend degrades to a clear diagnostic rather than a link error.
bool fortranFIRImportAvailable();

} // namespace ops_mlir

#endif // OPS_MLIR_RUNTIME_FORTRAN_KERNEL_IR_BUILDER_H
