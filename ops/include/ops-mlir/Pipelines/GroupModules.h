//===- GroupModules.h - split a planned module into compilable groups -*- C++ -*-===//
//
// After ops-jit-plan a module holds the caller chain (a list of `ops.call`s) and one `ops.chain @group_<n>` per
// group. The executor compiles each group on its own, so each is cloned into a module of its own, in call order,
// where the lowering pipeline can rename and change it freely.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_PIPELINES_GROUPMODULES_H
#define OPS_MLIR_PIPELINES_GROUPMODULES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"

#include <string>
#include <vector>

namespace ops_mlir::ops {

struct GroupModule {
  std::string name;                       ///< symbol of the group chain
  mlir::OwningOpRef<mlir::ModuleOp> module; ///< a module holding only that chain
};

/// The groups called by the chains of `planned`, in call order. Fails (empty result and a diagnostic) if a call does
/// not name a chain of the module.
std::vector<GroupModule> extractGroupModules(mlir::ModuleOp planned);

/// The module as MLIR bytecode: the canonical form that group cache keys are computed from.
std::string moduleBytecode(mlir::ModuleOp module);

/// Registers the dialects the JIT pipelines produce and consume on `registry` (the ops dialect, arith, func, math,
/// memref, scf, cf, llvm, openmp).
void registerJITDialects(mlir::DialectRegistry &registry);

} // namespace ops_mlir::ops

#endif // OPS_MLIR_PIPELINES_GROUPMODULES_H
