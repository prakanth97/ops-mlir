#include "ops-mlir/Runtime/Lifter.h"

#include "ops-mlir/Runtime/Options.h"

#include "ops_lazy_executor.h"

#include "mlir/IR/Builders.h"

#include <map>
#include <set>

namespace ops_mlir::runtime {

using namespace mlir;
using namespace ops_mlir::ops;

namespace {

Type elementType(MLIRContext *ctx, const std::string &name) {
  Builder b(ctx);
  if (name == "f64") return b.getF64Type();
  if (name == "f32") return b.getF32Type();
  if (name == "f16") return b.getF16Type();
  if (name == "i32") return b.getI32Type();
  if (name == "i64") return b.getI64Type();
  return Type();
}

AccessMode accessMode(int acc) {
  switch (acc) {
  case OPS_READ: return AccessMode::Read;
  case OPS_WRITE: return AccessMode::Write;
  case OPS_RW: return AccessMode::RW;
  case OPS_INC: return AccessMode::Inc;
  case OPS_MIN: return AccessMode::Min;
  default: return AccessMode::Max;
  }
}

} // namespace

std::vector<ops_dat> datSlots(ops_kernel_descriptor *const *run, int n) {
  std::vector<ops_dat> slots;
  std::set<int> seen;
  for (int k = 0; k < n; ++k)
    for (int a = 0; a < run[k]->nargs; ++a)
      if (run[k]->args[a].argtype == OPS_ARG_DAT && seen.insert(run[k]->args[a].dat->index).second)
        slots.push_back(run[k]->args[a].dat);
  return slots;
}

bool liftRun(MLIRContext &ctx, ops_kernel_descriptor *const *run, int n, const std::vector<std::vector<int>> &ranges,
             const std::vector<const KernelArtifact *> &artifacts, bool keepPrints, Lifted &out, std::string &error) {
  Location loc = UnknownLoc::get(&ctx);
  OpBuilder b(&ctx);
  const int rank = run[0]->block->dims;

  std::vector<ops_dat> slots = datSlots(run, n);
  std::map<int, int> slotOf;
  for (size_t s = 0; s < slots.size(); ++s) slotOf[slots[s]->index] = static_cast<int>(s);
  out.numDatSlots = static_cast<int>(slots.size());

  // Chain signature: dats, then per loop its range bounds and the components of its globals.
  SmallVector<Type> inputs, results;
  out.args.clear();
  out.reductions.clear();
  out.kernelNames.clear();
  for (size_t s = 0; s < slots.size(); ++s) {
    std::string elem = elemName(slots[s]->type ? slots[s]->type : "");
    Type t = elementType(&ctx, elem);
    if (!t) {
      error = std::string("dat of unsupported type ") + (slots[s]->type ? slots[s]->type : "?");
      return false;
    }
    inputs.push_back(DatType::get(&ctx, t, rank, slots[s]->dim));
    ChainArg arg;
    arg.kind = ChainArg::Dat;
    arg.slot = static_cast<int>(s);
    out.args.push_back(arg);
  }
  for (int k = 0; k < n; ++k) {
    const Manifest &m = artifacts[k]->manifest;
    for (int d = 0; d < rank; ++d)
      for (int hi = 0; hi < 2; ++hi) {
        inputs.push_back(b.getIndexType());
        ChainArg arg;
        arg.kind = ChainArg::Range;
        arg.loop = k;
        arg.dim = d;
        arg.hi = hi;
        out.args.push_back(arg);
      }
    for (size_t i = 0; i < m.params.size(); ++i) {
      const ParamInfo &p = m.params[i];
      if (p.kind == ParamInfo::Gbl) {
        Type t = elementType(&ctx, p.elem);
        if (!t) {
          error = "global of unsupported type " + p.elem;
          return false;
        }
        for (int c = 0; c < p.dim; ++c) {
          inputs.push_back(t);
          ChainArg arg;
          arg.kind = ChainArg::Gbl;
          arg.loop = k;
          arg.arg = static_cast<int>(i);
          arg.comp = c;
          arg.elem = p.elem;
          out.args.push_back(arg);
        }
      } else if (p.kind == ParamInfo::Reduce) {
        Type t = elementType(&ctx, p.elem);
        if (!t) {
          error = "reduction of unsupported type " + p.elem;
          return false;
        }
        results.push_back(t);
        const ops_arg &a = run[k]->args[i];
        ReductionRef ref;
        ref.loop = k;
        ref.arg = static_cast<int>(i);
        ref.kind = a.acc == OPS_INC ? ReductionKind::Add : a.acc == OPS_MIN ? ReductionKind::Min : ReductionKind::Max;
        ref.assign = p.style == "assign";
        ref.elem = p.elem;
        out.reductions.push_back(ref);
      }
    }
  }

  out.module = ModuleOp::create(loc);
  b.setInsertionPointToEnd(out.module->getBody());

  // The kernels, once each.
  std::set<std::string> cloned;
  for (int k = 0; k < n; ++k) {
    KernelOp kernelOp = artifacts[k]->kernel;
    const std::string name = kernelOp.getSymName().str();
    out.kernelNames.push_back(name);
    if (cloned.insert(name).second) b.insert(artifacts[k]->kernel->clone());
  }

  if (!keepPrints) {
    // OPS_JIT_PRINTF=drop: the output of printf in kernels is not produced (the AOT code of OPS would produce it).
    SmallVector<Operation *> prints;
    out.module->walk([&](PrintOp op) { prints.push_back(op); });
    for (Operation *op : prints) op->erase();
  }
  out.chain = ChainOp::create(b, loc, "chain", b.getFunctionType(inputs, results));
  Block *entry = out.chain.addEntryBlock();
  b.setInsertionPointToEnd(entry);

  // Position of the next per-loop argument in the chain.
  unsigned argPos = slots.size();
  SmallVector<Value> returned;
  SmallVector<Attribute> rangeValues;
  for (int k = 0; k < n; ++k) {
    ops_kernel_descriptor *desc = run[k];
    const Manifest &m = artifacts[k]->manifest;

    SmallVector<Value> range, dats, scalars;
    for (int i = 0; i < 2 * rank; ++i) range.push_back(entry->getArgument(argPos++));

    SmallVector<Attribute> accesses, stencils, kinds;
    SmallVector<Type> accTypes, scalarTypes, redTypes;
    for (int a = 0; a < desc->nargs; ++a) {
      const ops_arg &arg = desc->args[a];
      const ParamInfo &p = m.params[a];
      if (arg.argtype == OPS_ARG_DAT) {
        int slot = slotOf.at(arg.dat->index);
        dats.push_back(entry->getArgument(slot));
        accesses.push_back(AccessModeAttr::get(&ctx, accessMode(arg.acc)));
        SmallVector<int64_t> points, strides;
        for (int pnt = 0; pnt < arg.stencil->points; ++pnt)
          for (int d = 0; d < rank; ++d) points.push_back(arg.stencil->stencil[pnt * arg.stencil->dims + d]);
        for (int d = 0; d < rank; ++d) strides.push_back(arg.stencil->stride[d]);
        stencils.push_back(StencilAttr::get(&ctx, rank, points, strides));
        accTypes.push_back(AccType::get(&ctx, cast<DatType>(entry->getArgument(slot).getType()).getElementType(), rank, arg.dat->dim));
      } else if (p.kind == ParamInfo::Gbl) {
        Type t = elementType(&ctx, p.elem);
        for (int c = 0; c < p.dim; ++c) {
          scalars.push_back(entry->getArgument(argPos++));
          scalarTypes.push_back(t);
        }
      } else if (p.kind == ParamInfo::Reduce) {
        Type t = elementType(&ctx, p.elem);
        redTypes.push_back(RedType::get(&ctx, t));
        ReductionKind kind = arg.acc == OPS_INC ? ReductionKind::Add : arg.acc == OPS_MIN ? ReductionKind::Min : ReductionKind::Max;
        kinds.push_back(ReductionKindAttr::get(&ctx, kind));
      }
    }

    SmallVector<Type> resultTypes;
    for (Type t : redTypes) resultTypes.push_back(cast<RedType>(t).getElementType());
    ArrayAttr kindsAttr = kinds.empty() ? ArrayAttr() : b.getArrayAttr(kinds);
    auto loop = ParLoopOp::create(b, loc, resultTypes, range, dats, scalars, FlatSymbolRefAttr::get(&ctx, out.kernelNames[k]),
                                  b.getI64IntegerAttr(k), b.getI64IntegerAttr(desc->block->index), b.getArrayAttr(accesses),
                                  b.getArrayAttr(stencils), kindsAttr);
    returned.append(loop.getResults().begin(), loop.getResults().end());

    SmallVector<Type> blockTypes(accTypes);
    blockTypes.append(scalarTypes);
    blockTypes.append(redTypes);
    SmallVector<Location> locs(blockTypes.size(), loc);
    {
      OpBuilder::InsertionGuard guard(b);
      b.createBlock(&loop.getBody(), {}, blockTypes, locs);
      YieldOp::create(b, loc);
    }

    SmallVector<int64_t> values(ranges[k].begin(), ranges[k].end());
    rangeValues.push_back(b.getDenseI64ArrayAttr(values));
  }
  ReturnOp::create(b, loc, returned);
  out.chain->setAttr("ops.range_values", b.getArrayAttr(rangeValues));
  if (options().specialize == "shapes") {
    // The sizes of the dats (OPS order: x first) are part of the code: static memref types, constant strides.
    SmallVector<Attribute> shapeValues;
    for (const ChainArg &arg : out.args) {
      SmallVector<int64_t> sizes;
      if (arg.kind == ChainArg::Dat)
        for (int d = 0; d < rank; ++d) sizes.push_back(slots[arg.slot]->size[d]);
      shapeValues.push_back(b.getDenseI64ArrayAttr(sizes));
    }
    out.chain->setAttr("ops.shape_values", b.getArrayAttr(shapeValues));
  }

  // Constants whose values are folded into the code (role bound): their values as of now, and the roles.
  SmallVector<NamedAttribute> roles, values;
  std::set<std::string> done;
  for (int k = 0; k < n; ++k) {
    for (const ConstInfo &c : artifacts[k]->manifest.consts) {
      if (!c.bound || !done.insert(c.name).second) continue;
      const ops_const_entry *entryConst = ops_find_const(run[k]->block->instance, c.name.c_str());
      if (!entryConst) {
        error = "constant '" + c.name + "' is not registered with ops_decl_const";
        return false;
      }
      size_t bytes = static_cast<size_t>(entryConst->dim) * entryConst->size;
      SmallVector<int8_t> snapshot(bytes);
      std::memcpy(snapshot.data(), entryConst->snapshot, bytes);
      roles.push_back(b.getNamedAttr(c.name, b.getStringAttr("bound")));
      values.push_back(b.getNamedAttr(c.name, b.getDenseI8ArrayAttr(snapshot)));
    }
  }
  if (!roles.empty()) {
    out.chain->setAttr("ops.const_roles", b.getDictionaryAttr(roles));
    out.chain->setAttr("ops.const_values", b.getDictionaryAttr(values));
  }
  return true;
}

bool recordBindingConsts(Lifted &lifted, std::string &error) {
  auto bindings = lifted.chain->getAttrOfType<ArrayAttr>("ops.binding_consts");
  if (!bindings) return true;
  for (Attribute a : bindings) {
    auto d = cast<DictionaryAttr>(a);
    ChainArg arg;
    arg.kind = ChainArg::Const;
    arg.name = cast<StringAttr>(d.get("name")).getValue().str();
    arg.offset = cast<IntegerAttr>(d.get("offset")).getInt();
    unsigned index = cast<IntegerAttr>(d.get("arg")).getInt();
    Type t = lifted.chain.getArgument(index).getType();
    arg.elem = t.isF64() ? "f64" : t.isF32() ? "f32" : t.isF16() ? "f16" : t.isInteger(32) ? "i32" : t.isInteger(64) ? "i64" : "";
    if (arg.elem.empty()) {
      error = "constant '" + arg.name + "' has a type the binding cannot read";
      return false;
    }
    if (index != lifted.args.size()) {
      error = "constant arguments are not appended in order";
      return false;
    }
    lifted.args.push_back(std::move(arg));
  }
  return true;
}

} // namespace ops_mlir::runtime
