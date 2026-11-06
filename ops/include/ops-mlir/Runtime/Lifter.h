//===- Lifter.h - queued OPS loops -> an ops.chain of unmaterialised par_loops -*- C++ -*-===//
//
// Plan section 4.5 (buildPlan, step 1). A *run* is a sequence of claimed descriptors. The lifter builds one module
// with the `ops.kernel`s the run uses and an `ops.chain @chain` holding one unmaterialised `ops.par_loop` per
// descriptor:
//
//   * every distinct dat (by dat->index) is one chain argument (a `!ops.dat`), in order of first appearance;
//   * after the dats, per loop, its range bounds (2 * rank index arguments) and the components of its read-only
//     globals (scalars);
//   * the frontend pipeline adds one scalar argument per constant the kernels read (`ops.binding_consts`);
//   * the chain returns the reduction values, loop by loop.
//
// `Lifted::args` records for every chain argument where its value comes from when the compiled code is called: the
// binding recipe is built from it (Binding.h). Nothing in it refers to pointers of the run, so a plan can be reused
// for another run with the same structure.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_LIFTER_H
#define OPS_MLIR_RUNTIME_LIFTER_H

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Runtime/Artifact.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "ops_lib_core.h"
#include "ops_internal2.h"

#include <string>
#include <vector>

namespace ops_mlir::runtime {

/// Where the value of a chain argument comes from.
struct ChainArg {
  enum Kind { Dat, Range, Gbl, Const } kind = Dat;
  int slot = 0;                  ///< Dat: the distinct dat (order of first appearance in the run)
  int loop = 0;                  ///< Range, Gbl: index of the loop in the run
  int dim = 0;                   ///< Range: dimension
  bool hi = false;               ///< Range: upper bound (else lower)
  int arg = 0;                   ///< Gbl: descriptor argument
  int comp = 0;                  ///< Gbl: component
  std::string name;              ///< Const: name registered with ops_decl_const
  int64_t offset = 0;            ///< Const: byte offset in the constant's value
  std::string elem;              ///< Gbl, Const: element type (f64, f32, i32)
};

/// One reduction result of the chain.
struct ReductionRef {
  int loop = 0;
  int arg = 0; ///< descriptor argument (the handle)
  ops::ReductionKind kind = ops::ReductionKind::Add;
  bool assign = false; ///< `*r = v` style: the value replaces what the handle holds
  std::string elem;
};

struct Lifted {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ops::ChainOp chain;
  std::vector<ChainArg> args;          ///< per chain argument
  std::vector<ReductionRef> reductions; ///< per chain result
  int numDatSlots = 0;
  std::vector<std::string> kernelNames; ///< per loop
};

/// The dat slots of a run: `dat->index` -> slot, in order of first appearance.
std::vector<ops_dat> datSlots(ops_kernel_descriptor *const *run, int n);

/// Builds the chain for `run[0..n)`. `ranges[k]` are the local ranges (2 * dims entries); `artifacts[k]` the decoded
/// artifact of loop k. Fails with a diagnostic text in `error`.
bool liftRun(mlir::MLIRContext &ctx, ops_kernel_descriptor *const *run, int n, const std::vector<std::vector<int>> &ranges,
             const std::vector<const KernelArtifact *> &artifacts, bool keepPrints, Lifted &out, std::string &error);

/// After ops-jit-frontend: add the constants it appended to the chain's arguments to `lifted.args`.
bool recordBindingConsts(Lifted &lifted, std::string &error);

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_LIFTER_H
