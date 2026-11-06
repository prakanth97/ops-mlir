//===- Binding.h - from queued loops to the arguments of a compiled group ----*- C++ -*-===//
//
// Plan sections 4.5 (binding recipe) and 4.6 (ABI of a compiled group). A compiled group is a function
//
//   (dat memref x D, shift x (D * rank), range lo/hi x (2 * rank), disp x rank, scalar x S) -> reduction values
//
// called through the packed wrapper of opdsl::JITSession. The recipe says, for each of those, where the value comes
// from in the current run of descriptors (dats by slot, ranges by loop, globals and constants by name); binding
// evaluates it.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_BINDING_H
#define OPS_MLIR_RUNTIME_BINDING_H

#include "ops-mlir/Runtime/Lifter.h"
#include "ops-mlir/Pipelines/GroupModules.h"

#include "ops-mlir/Runtime/PackedArgs.h"

#include "opdsl/JIT/JITSession.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ops_mlir::runtime {

/// How to call one group of a planned chain.
struct GroupRecipe {
  int rank = 0;
  std::vector<int> datSlots;          ///< memref arguments: dats by slot of the run
  std::vector<ChainArg> range;        ///< 2 * rank entries: lo0, hi0, lo1, hi1, ...
  std::vector<ChainArg> scalars;      ///< scalar arguments (globals and constants)
  std::vector<int> reductionOf;       ///< per result of the group: index into Lifted::reductions, or -1
  std::vector<std::string> resultElem; ///< element type name per result
  std::vector<int> members;           ///< loops of the run the group executes, in order
};

/// The recipes of the groups of a planned module, in call order (same order as extractGroupModules). `lifted` supplies
/// the meaning of the chain arguments.
bool buildRecipes(const Lifted &lifted, mlir::ModuleOp planned, std::vector<GroupRecipe> &recipes, std::string &error);

/// The arguments of one call of a compiled group, evaluated from a run of descriptors.
class Invocation {
public:
  /// Evaluates `recipe` for the loops `run` (with their local `ranges`). `soa`: OPS stores components apart.
  /// `reductionBuffers`: for a group lowered for a device, one device buffer of `reductionUnits` elements per
  /// reduction (the compiled function takes them after the scalars and returns nothing); null for the host.
  bool bind(const GroupRecipe &recipe, ops_kernel_descriptor *const *run, const std::vector<std::vector<int>> &ranges,
            const std::vector<ops_dat> &slots, bool soa, std::string &error, const std::vector<void *> *reductionBuffers = nullptr, int64_t reductionUnits = 0);

  /// Calls the compiled group. After it, result(i) reads the reduction values.
  void call(opdsl::JITSession::PackedFn fn) { args_.call(fn); }

  /// Result `i` (in the order of the group's results).
  double resultAsDouble(size_t i) const { return args_.resultAsDouble(i); }
  int64_t resultAsInt(size_t i) const { return args_.resultAsInt(i); }

  const PackedArgs &args() const { return args_; }

private:
  PackedArgs args_;
};

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_BINDING_H
