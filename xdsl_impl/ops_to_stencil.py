"""Lower ops.par_loop -> func.func + stencil.* using xDSL, mirroring
lib/passes/OPSToStencil.cpp's C++ design.
"""

import ctypes
from dataclasses import dataclass
from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import arith, func, memref, stencil
from xdsl.dialects.builtin import IndexType, IntegerAttr, MemRefType, ModuleOp, i32, f64, FloatAttr
from xdsl.ir import Block, Region, SSAValue
from xdsl.passes import ModulePass

from ops_dialect import ArgType, Access, DatAttr, ParLoopOp, StencilAttr

def field_bounds(dat: DatAttr) -> list[tuple[int, int]]:
    """Normalized 0-based field bounds: lb=0, ub=full allocated size per dim."""
    return [(0, size) for size in dat.size_list]

def halo_offsets(dat_args) -> list[int]:
    """Per-dim d_m values (negative) from the first dat — shared across all dats on a block."""
    return list(dat_args[0].dat.d_m_list)

def stencil_offsets(stencil_attr: StencilAttr) -> list[tuple[int, ...]]:
    """Per-point access offsets for an arg's stencil.

    Extract the offsets from the StencilAttr's raw pointer to the underlying C array.
    """
    dims = stencil_attr.dims.data
    points = stencil_attr.points.data
    addr = stencil_attr.stencil.data
    if addr == 0 or points == 0:
        return [tuple(0 for _ in range(dims))]

    flat = (ctypes.c_int32 * (dims * points)).from_address(addr)
    return [tuple(flat[p * dims + d] for d in range(dims)) for p in range(points)]

def range_bounds(rng: list[int], ndim: int) -> list[tuple[int, int]]:
    return [(rng[2 * i], rng[2 * i + 1]) for i in range(ndim)]

def normalized_range_bounds(rng: list[int], d_m: list[int], ndim: int) -> list[tuple[int, int]]:
    """Shift OPS iteration range by -d_m so bounds are relative to normalized (0-based) field."""
    return [(lb - dm, ub - dm) for (lb, ub), dm in zip(range_bounds(rng, ndim), d_m)]

def declare_kernel(
    module: ModuleOp, name: str, num_f64_args: int, num_idx_args: int, ndim: int,
    num_results: int, num_reduce_args: int
) -> None:
    if any(
        isinstance(o, func.FuncOp) and o.sym_name.data == name
        for o in module.body.block.ops
    ):
        return

    # Fix ret hidden pointer issue when returning multiple results
    # Idx arguments are passed as MemRefType(i32, [ndim]) to match the OPS kernel ABI
    param_types = (f64,) * num_f64_args + (MemRefType(i32, [ndim]),) * num_idx_args

    if num_results > 1:
        # Out-pointer ABI: kernel writes results into a trailing MemRefType(f64, [num_results]) param
        param_types = param_types + (MemRefType(f64, [num_results]),)
        result_types = ()
    else:
        result_types = (f64,) * max(num_results, 1)

    param_types= param_types + (MemRefType(f64, []),) * num_reduce_args

    decl = func.FuncOp(
        name,
        (param_types, result_types),
        region=Region(),
        visibility="private",
    )
    module.body.block.add_op(decl)

def convert_par_loop(op: ParLoopOp, index: int, module: ModuleOp) -> func.FuncOp:
    ndim = op.dims.value.data
    args = op.arg_list()
    dat_args = [a for a in args if a.argtype.data == ArgType.DAT]
    num_idx_args = sum(1 for a in args if a.argtype.data == ArgType.IDX)

    # using GBL type here as i's what reductions currently use in OPS dialect
    reduce_args = [a for a in args if a.argtype.data == ArgType.GBL]

    d_m = halo_offsets(dat_args)
    apply_bounds = stencil.StencilBoundsAttr(
        normalized_range_bounds(list(op.range.get_values()), d_m, ndim)
    )

    field_types = [stencil.FieldType(field_bounds(arg.dat), f64) for arg in dat_args]
    reduce_handle_types = [MemRefType(f64, [])] * len(reduce_args)
    
    # Outer function: ops_par_loop_<kernel>_<index>(fields...) -> ()
    kernel_name = op.kernel_name.data
    fn_name = f"ops_par_loop_{kernel_name}_{index}"
    fn = func.FuncOp(fn_name, (tuple(field_types) + tuple(reduce_handle_types), ()), visibility="private")
    block = fn.body.block
    fn_builder = Builder(InsertPoint.at_end(block))

    reduce_handles = list(block.args[len(dat_args):])  # reduce handles come after dat args

    # Partition dat args (by access mode) into stencil.apply's reads/writes
    reads: list[SSAValue] = []
    read_types = []
    read_stencils: list[StencilAttr] = []
    writes: list[SSAValue] = []
    for arg, field in zip(dat_args, block.args):
        access = arg.acc.data
        if access in (Access.READ, Access.RW):
            reads.append(field)
            read_types.append(field.type)
            read_stencils.append(arg.stencil)
        if access in (Access.WRITE, Access.RW, Access.INC):
            writes.append(field)

    # Apply block body: access reads, compute indices, call kernel
    apply_block = Block(arg_types=read_types)
    block_builder = Builder(InsertPoint.at_end(apply_block))

    # One stencil.access per real stencil point (from stencil_offsets), not
    # just a placeholder (0, ..., 0) -- so multi-point stencils (e.g. a 5pt
    # Laplacian) actually read all their neighbors.
    access_results = []
    for block_arg, stencil_attr in zip(apply_block.args, read_stencils):
        for point in stencil_offsets(stencil_attr):
            access = block_builder.insert(stencil.AccessOp(block_arg, point))
            access_results.append(access.res)

    # alloca_scope lowers to explicit stacksave/stackrestore bracketing its region
    # (MemRefToLLVM.cpp's AllocaScopeOpLowering), so every iteration's
    # allocas are released before the next one runs.
    num_results = len(writes) or 1
    scope_block = Block()
    scope_builder = Builder(InsertPoint.at_end(scope_block))

    # Compute per-dim index buffers for the kernel call, which are passed as MemRefType(i32, [ndim])
    idx_buffers = []
    # Offset d_m[dim] un-normalizes the loop index back to the OPS global index:
    # ops_global_idx = normalized_loop_idx + d_m  (d_m is negative, e.g. -1)
    for _ in range(num_idx_args):
        idx_buffer = scope_builder.insert(
            memref.AllocaOp.get(i32, shape=[ndim])
        )
        for d in range(ndim):
            idx_op = scope_builder.insert(
                stencil.IndexOp.build(
                    attributes={
                        "dim": IntegerAttr(d, IndexType()),
                        "offset": stencil.IndexAttr.from_indices(*[
                            d_m[i] if i == d else 0 for i in range(ndim)
                        ]),
                    },
                    result_types=[IndexType()],
                )
            )
            idx_i32 = scope_builder.insert(
                arith.IndexCastOp(idx_op.idx, i32)
            )
            dim_const = scope_builder.insert(
                arith.ConstantOp(IntegerAttr(d, IndexType()))
            )
            scope_builder.insert(
                memref.StoreOp.get(idx_i32.result, idx_buffer.memref, [dim_const.result])
            )
        idx_buffers.append(idx_buffer.memref)

    reduce_scratches = []
    reduce_identities = []
    for arg in reduce_args:
        identity, kind = reduction_identity(arg.acc.data)
        reduce_identities.append((identity, kind))
        scratch = scope_builder.insert(memref.AllocaOp.get(f64, shape=[]))
        id_const = scope_builder.insert(arith.ConstantOp(FloatAttr(identity, f64)))
        scope_builder.insert(memref.StoreOp.get(id_const.result, scratch.memref, []))
        reduce_scratches.append(scratch.memref)

    call_args = access_results + idx_buffers + reduce_scratches
    declare_kernel(
        module, kernel_name, len(access_results), len(idx_buffers), ndim, num_results, len(reduce_args)
    )

    if num_results > 1:
        out_buf = scope_builder.insert(
            memref.AllocaOp.get(f64, shape=[num_results])
        )
        scope_builder.insert(
            func.CallOp(kernel_name, call_args + [out_buf.memref], [])
        )
        write_results = []
        for i in range(num_results):
            idx_const = scope_builder.insert(arith.ConstantOp(IntegerAttr(i, IndexType())))
            loaded = scope_builder.insert(memref.LoadOp.get(out_buf.memref, [idx_const.result]))
            write_results.append(loaded.res)
    else:
        kernel = scope_builder.insert(
            func.CallOp(kernel_name, call_args, [f64] * num_results)
        )
        write_results = list(kernel.results)

    # Load each reduce arg's per-point contribution
    reduce_contribs = []
    for scratch in reduce_scratches:
        loaded = scope_builder.insert(memref.LoadOp.get(scratch, []))
        reduce_contribs.append(loaded.res)

    scope_builder.insert(memref.AllocaScopeReturnOp.build(operands=[write_results + reduce_contribs]))

    alloca_scope = block_builder.insert(
        memref.AllocaScopeOp.build(
            regions=[Region([scope_block])],
            result_types=[[f64] * (num_results + len(reduce_args))],
        )
    )
    scope_results = list(alloca_scope.res)
    write_vals, reduce_vals = scope_results[:num_results], scope_results[num_results:]

    # Fold each reduce arg's contribution across all grid points
    for val, (identity, kind) in zip(reduce_vals, reduce_identities):
        assert val is not None, "contrib is None"
        insert_reduce_combiner(block_builder, val, identity, kind)
    block_builder.insert(stencil.ReturnOp.get(write_vals))

    # Create ApplyOp in buffer semantic form
    fn_builder.insert(
        stencil.ApplyOp.build(
            operands=[reads, writes, reduce_handles],
            regions=[Region([apply_block])],
            result_types=[[]],  # buffer semantic does not return results
            properties={"bounds": apply_bounds},
        )
    )

    fn_builder.insert(func.ReturnOp())
    return fn

@dataclass(frozen=True)
class OPSToStencilPass(ModulePass):
    """Lower ops.par_loop -> func.func + stencil.* (see module docstring)."""
    name = "ops-to-stencil"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        loops = [o for o in op.body.block.ops if isinstance(o, ParLoopOp)]
        for index, loop_op in enumerate(loops):
            self.lower_par_loop(loop_op, index, op)

    def lower_par_loop(self, loop_op: ParLoopOp, index: int, module: ModuleOp) -> None:
        fn = convert_par_loop(loop_op, index, module)
        module.body.block.add_op(fn)
        loop_op.detach()
        loop_op.erase()


def reduction_identity(access: int) -> tuple[float, str]:
    """(identity element, combiner kind) for a reduce arg's access mode."""
    if access == Access.MAX:
        return (float("-inf"), "max")
    if access == Access.MIN:
        return (float("inf"), "min")
    if access == Access.INC:
        return (0.0, "add")
    raise ValueError(f"unsupported reduction access mode: {access}")


def insert_reduce_combiner(builder: Builder, contrib: SSAValue, identity: float, kind: str, ) -> None:
    """Insert stencil.reduce %contrib init %identity { combiner } : f64."""
    id_const = builder.insert(arith.ConstantOp(FloatAttr(identity, f64)))

    combiner_block = Block(arg_types=[f64, f64])
    cb = Builder(InsertPoint.at_end(combiner_block))
    lhs, rhs = combiner_block.args

    # NOTE: The reduction ops use a compare and select pattern; this is because openMP lowering does not support operations 
    # like arith.maximumf and arith.minimumf, but compare and select is explicitly supported in the MLIR source code

    if kind == "max":
        cmp = cb.insert(arith.CmpfOp(lhs, rhs, "ogt"))
        result = cb.insert(arith.SelectOp(cmp.result, lhs, rhs))
    elif kind == "min":
        cmp = cb.insert(arith.CmpfOp(lhs, rhs, "olt"))
        result = cb.insert(arith.SelectOp(cmp.result, lhs, rhs))
    elif kind == "add":
        result = cb.insert(arith.AddfOp(lhs, rhs))
    else:
        raise ValueError(f"unsupported combiner kind: {kind}")

    cb.insert(stencil.YieldOp(result.results[0]))
    assert id_const.result is not None, "id_const.result is None"

    builder.insert(
        stencil.ReduceOp(contrib, id_const.result, Region([combiner_block]))
    )