//===- Passes.td - transformation passes of the ops dialect ----*- tablegen -*-===//

#ifndef OPS_MLIR_TRANSFORMS_PASSES
#define OPS_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MaterializeKernels : Pass<"ops-materialize-kernels", "::mlir::ModuleOp"> {
  let summary = "clone the body of each ops.kernel into the ops.par_loops that call it";
  let description = [{
    A `par_loop` with a `kernel` symbol and an empty body is the unmaterialised form the lifter produces. This
    pass fills in its body from the `ops.kernel` of that name:

    - the i-th accessor parameter of the kernel is the i-th dat operand's block argument, the i-th reduction
      parameter the i-th reduction's;
    - a global parameter `!ops.gbl<T, n>` takes the next n scalar operands: `ops.gbl_read %g[c]` becomes the
      scalar block argument of component c;
    - `ops.const_ref` without an index becomes a new scalar argument of the chain (one per constant, offset
      and type), passed to every loop that uses it; the arguments are recorded in the chain attribute
      `ops.binding_consts`, an array of `{name, offset, arg}` the lifter reads back to bind them. A
      `const_ref` with a dynamic index stays and is handled by `ops-specialize-consts`.

    The result is verified: a read or write at an offset outside the operand's stencil is an error.
  }];
  let dependentDialects = ["::ops_mlir::ops::OPSDialect"];
}

def SpecializeConsts : Pass<"ops-specialize-consts", "::mlir::ModuleOp"> {
  let summary = "fold constants whose values are known, and unroll the loops that index them";
  let description = [{
    The values of constants come from the chain attribute `ops.const_values`, a dictionary from the name of
    a constant to its bytes (`array<i8: ...>`, little endian, as `ops_decl_const` registered it). Which
    constants to fold: the option `names`, by default every constant whose entry in the chain attribute
    `ops.const_roles` is `"bound"`.

    - arguments recorded in `ops.binding_consts` for such a constant are replaced by their value in every loop and
      removed from the chain signature;
    - `scf.for` loops with a constant trip count of at most `max-trip-count` that index such a constant with a
      dynamic `ops.const_ref` are fully unrolled;
    - `ops.const_ref` of such a constant with a constant (or no) index is replaced by its value;
    - canonicalisation then folds what became constant.

    A `ops.const_ref` with a dynamic index that remains is an error: a chain with such a reference cannot be
    compiled and the executor declines it.
  }];
  let options = [
    ListOption<"names", "names", "std::string", "constants to fold (default: those with role bound)">,
    Option<"maxTripCount", "max-trip-count", "int64_t", /*default=*/"256",
           "largest constant trip count of a loop that is fully unrolled">
  ];
  let dependentDialects = ["::ops_mlir::ops::OPSDialect", "::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect"];
}


def FusePointLocal : Pass<"ops-fuse-point-local", "::mlir::ModuleOp"> {
  let summary = "merge the loops of a chain that can run point by point in one kernel";
  let description = [{
    Plans the chain with the dependence analysis (`ops.range_values` gives the concrete ranges) and rewrites it
    along the plan: a group of loops becomes one `ops.par_loop` whose dat operands are the union of the members'
    (deduplicated by value, access modes joined, stencil points united) and whose body is the members' bodies, in
    order. Independent loops move next to the loops they fuse with. Loops with a reduction are never fused and
    nothing moves across them. The kernels of all loops must be materialised (`ops-materialize-kernels`).
    `ops.range_values` is rewritten for the new loops; the fused loops carry `ops.fused_loops`, the `loop_id`s
    of their members.
  }];
  let options = [
    Option<"maxLoops", "max-loops", "int64_t", /*default=*/"8", "upper bound on the loops of a group">,
    Option<"reorder", "reorder", "bool", /*default=*/"true", "let independent loops move past each other">,
    Option<"placement", "placement", "std::string", /*default=*/"\"earliest\"",
           "which legal group a loop joins: earliest or latest">,
    Option<"guarded", "guarded", "bool", /*default=*/"false",
           "fuse loops of different ranges over their bounding box (not implemented yet)">,
    Option<"boxRatio", "box-ratio", "double", /*default=*/"1.0",
           "largest bounding box volume relative to the members' volumes (guarded fusion)">
  ];
  let dependentDialects = ["::ops_mlir::ops::OPSDialect"];
}

def ForwardStores : Pass<"ops-forward-stores", "::mlir::ModuleOp"> {
  let summary = "use the value stored to a dat point instead of reading it back; narrow access modes";
  let description = [{
    Inside a loop body, `ops.write %a[o] {comp = c}, %v` followed by `ops.read %a[o] {comp = c}` of the same
    accessor, with no write to that accessor in between, uses `%v`. The pass then narrows the access modes of
    the loop: an operand that is no longer read becomes `write` (it need not be loaded), one that is never written
    `read`.
  }];
  let dependentDialects = ["::ops_mlir::ops::OPSDialect"];
}

def OutlineGroups : Pass<"ops-outline-groups", "::mlir::ModuleOp"> {
  let summary = "make each loop of a chain its own ops.chain, called from the original";
  let description = [{
    Each `ops.par_loop` of a chain becomes `ops.chain @group_<n>` with only the operands of the loop, and
    the original chain becomes a list of `ops.call`s in execution order. Constants the loop body captured from
    the chain are copied into the group. The planner and the executor compile and launch the groups.
  }];
  let dependentDialects = ["::ops_mlir::ops::OPSDialect"];
}


def OpsSpecialize : Pass<"ops-specialize", "::mlir::ModuleOp"> {
  let summary = "replace what is known at JIT time by constants";
  let description = [{
    `mode=ranges`: the range operands of every loop become the constants of the chain attribute `ops.range_values`
    (compile-time trip counts, so the code generator can unroll and vectorise without remainder checks). The chain
    keeps its arguments, the ABI does not change. The compiled code is only valid for these ranges: the cache key
    must contain them. `mode=shapes` does that and also uses the extents of the dats (the chain attribute `ops.shape_values`,
    per argument the sizes in OPS order): `convert-ops-to-scf` gives the memrefs static shapes and strides, which makes the
    address arithmetic constant. `mode=none` does nothing.
  }];
  let options = [Option<"mode", "mode", "std::string", /*default=*/"\"none\"", "none, ranges or shapes">];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

#endif // OPS_MLIR_TRANSFORMS_PASSES
