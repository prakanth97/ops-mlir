#include "ops-mlir/Runtime/Binding.h"

#include "ops_lazy_executor.h"

#include <cstring>
#include <map>

namespace ops_mlir::runtime {

using namespace mlir;
using namespace ops_mlir::ops;

namespace {

/// The chain argument (index) behind an operand of a call, or -1.
int chainArgIndex(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  return arg ? static_cast<int>(arg.getArgNumber()) : -1;
}

} // namespace

bool buildRecipes(const Lifted &lifted, ModuleOp planned, std::vector<GroupRecipe> &recipes, std::string &error) {
  recipes.clear();
  ChainOp caller;
  for (ChainOp c : planned.getOps<ChainOp>())
    if (c.getSymName() == "chain") caller = c;
  if (!caller) {
    error = "the planned module has no caller chain";
    return false;
  }
  SmallVector<Value> returned;
  caller.walk([&](ReturnOp ret) { returned.assign(ret.getOperands().begin(), ret.getOperands().end()); });

  for (Operation &op : caller.getBody().front()) {
    auto call = dyn_cast<CallOp>(&op);
    if (!call) continue;
    ChainOp group = planned.lookupSymbol<ChainOp>(call.getCallee());
    if (!group) {
      error = "call of an unknown group " + call.getCallee().str();
      return false;
    }
    ParLoopOp loop;
    for (Operation &inner : group.getBody().front())
      if (auto l = dyn_cast<ParLoopOp>(&inner)) loop = l;
    if (!loop) {
      error = "group " + call.getCallee().str() + " has no loop";
      return false;
    }

    // The source of a group operand: the chain argument the call passes for it.
    auto source = [&](Value groupValue, const char *what) -> const ChainArg * {
      auto arg = dyn_cast<BlockArgument>(groupValue);
      if (!arg) {
        error = std::string("a ") + what + " operand of a group is not an argument of the group";
        return nullptr;
      }
      int chainIndex = chainArgIndex(call.getOperand(arg.getArgNumber()));
      if (chainIndex < 0 || chainIndex >= (int)lifted.args.size()) {
        error = std::string("a ") + what + " operand of a call is not an argument of the chain";
        return nullptr;
      }
      return &lifted.args[chainIndex];
    };

    GroupRecipe r;
    r.rank = loop.getRank();
    // Dats, distinct, in the order of the group's operands (the ABI of convert-ops-to-scf).
    llvm::DenseSet<Value> seen;
    for (Value dat : loop.getDats()) {
      if (!seen.insert(dat).second) continue;
      const ChainArg *s = source(dat, "dat");
      if (!s) return false;
      if (s->kind != ChainArg::Dat) {
        error = "a dat operand does not come from a dat argument";
        return false;
      }
      r.datSlots.push_back(s->slot);
    }
    for (Value range : loop.getRange()) {
      const ChainArg *s = source(range, "range");
      if (!s) return false;
      if (s->kind != ChainArg::Range) {
        error = "a range operand does not come from a range argument";
        return false;
      }
      r.range.push_back(*s);
    }
    seen.clear();
    for (Value scalar : loop.getScalars()) {
      if (!seen.insert(scalar).second) continue;
      const ChainArg *s = source(scalar, "scalar");
      if (!s) return false;
      if (s->kind != ChainArg::Gbl && s->kind != ChainArg::Const) {
        error = "a scalar operand does not come from a global or constant argument";
        return false;
      }
      r.scalars.push_back(*s);
    }
    // Results: which reduction of the chain each one is.
    for (Value result : call.getResults()) {
      int index = -1;
      for (auto [i, v] : llvm::enumerate(returned))
        if (v == result) index = static_cast<int>(i);
      r.reductionOf.push_back(index);
      Type t = result.getType();
      r.resultElem.push_back(t.isF64() ? "f64" : t.isF32() ? "f32" : t.isInteger(32) ? "i32" : t.isInteger(64) ? "i64" : "");
    }
    // Members.
    if (auto fused = loop->getAttrOfType<ArrayAttr>("ops.fused_loops"))
      for (Attribute a : fused) r.members.push_back(static_cast<int>(cast<IntegerAttr>(a).getInt()));
    else
      r.members.push_back(static_cast<int>(loop.getLoopId()));
    recipes.push_back(std::move(r));
  }
  return true;
}

bool Invocation::bind(const GroupRecipe &recipe, ops_kernel_descriptor *const *run, const std::vector<std::vector<int>> &ranges,
                      const std::vector<ops_dat> &slots, bool soa, std::string &error,
                      const std::vector<void *> *reductionBuffers, int64_t reductionUnits) {
  const int rank = recipe.rank;
  args_ = PackedArgs();
  ops_kernel_descriptor *first = run[recipe.members.front()];
  OPS_instance *instance = first->block->instance;

  // Dats: whole allocations as memrefs, z/y/x (slowest first), then the component.
  for (int slot : recipe.datSlots) {
    ops_dat dat = slots[slot];
    char *ptr = first->isdevice ? dat->data_d : dat->data;
    const int ncomp = dat->dim;
    int64_t points = 1;
    for (int d = 0; d < rank; ++d) points *= dat->size[d];
    std::vector<int64_t> sizes(rank + (ncomp > 1 ? 1 : 0)), strides(sizes.size());
    int64_t stride = (ncomp > 1 && !soa) ? ncomp : 1; // x
    for (int d = 0; d < rank; ++d) {
      sizes[rank - 1 - d] = dat->size[d];
      strides[rank - 1 - d] = stride;
      stride *= dat->size[d];
    }
    if (ncomp > 1) {
      sizes[rank] = ncomp;
      strides[rank] = soa ? points : 1;
    }
    args_.addMemref(ptr, sizes, strides);
  }
  // Shifts, dat-major.
  for (int slot : recipe.datSlots) {
    int shift[OPS_MAX_DIM];
    ops_lazy_dat_shift(slots[slot], shift);
    for (int d = 0; d < rank; ++d) args_.addIndex(shift[d]);
  }
  // Ranges.
  for (const ChainArg &r : recipe.range) args_.addIndex(ranges[r.loop][2 * r.dim + (r.hi ? 1 : 0)]);
  // Displacement of this process's part of the block.
  int disp[OPS_MAX_DIM];
  ops_lazy_block_disp(first->block, disp);
  for (int d = 0; d < rank; ++d) args_.addIndex(disp[d]);

  // Scalars.
  for (const ChainArg &s : recipe.scalars) {
    const char *src = nullptr;
    if (s.kind == ChainArg::Gbl) {
      const ops_arg &a = run[s.loop]->args[s.arg];
      src = a.data + static_cast<size_t>(s.comp) * a.elem_size;
    } else {
      const ops_const_entry *c = ops_find_const(instance, s.name.c_str());
      if (!c) {
        error = "constant '" + s.name + "' is not registered";
        return false;
      }
      src = c->snapshot + s.offset;
    }
    args_.addScalar(s.elem, src);
  }
  if (reductionBuffers) {
    for (void *buffer : *reductionBuffers) args_.addMemref(buffer, {reductionUnits}, {1});
    args_.setResults({});
  } else {
    args_.setResults(recipe.resultElem);
  }
  args_.finalize();
  return true;
}

} // namespace ops_mlir::runtime
