//===- Passes.td - conversion passes of the ops dialect --------*- tablegen -*-===//

#ifndef OPS_MLIR_CONVERSION_PASSES
#define OPS_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertOpsToSCF : Pass<"convert-ops-to-scf", "::mlir::ModuleOp"> {
  let summary = "lower a group (an ops.chain of one ops.par_loop) to a func.func with scf.parallel";
  let description = [{
    The compiled group ABI (plan section 4.6). The function takes, in this order:

    1. one memref per distinct dat of the loop, in order of first appearance;
    2. the shifts: one `index` per dat and dimension (dat-major, dimension-minor, OPS axis order);
    3. the range: lo, hi per dimension (OPS axis order, x first);
    4. the displacement of the block, one `index` per dimension (`ops.index d` is `i_d + disp_d`);
    5. the scalars of the loop (distinct chain arguments, in order of first appearance).

    It returns the values of the reductions (neutral element combined over the range). With `target=gpu` the function
    is the host side of a kernel launch instead: the arguments after the scalars are one rank-0 memref per reduction
    (device memory the caller initialised with the neutral element, and reads back afterwards), each thread
    combines its partial value into it with an atomic, and nothing is returned. The threads are a block of
    `block` threads per dimension (x fastest); a thread outside the range does nothing. A dat of an OPS block of
    rank r with c components is a `memref<?x..xT>` of the r dimensions in memory order (z, y, x: OPS dimension d is
    memref dimension r-1-d) with a trailing dimension of size c if c > 1. Array of structures (default): the strides
    are [?, .., c, 1] (or [?, .., 1] for c = 1); structure of arrays (`soa`): [?, .., 1, ?]. An access at offset o of
    a dat with stencil stride s along d reads the memref index `i_d * s + o_d + shift_d`.

    The function is named like the chain, or as the chain attribute `ops.func_name` says; a private declaration of
    that name in the module (for a caller) is replaced by the definition.

    With `loops=affine` (CPU only) the loops are `affine.parallel` (the outer dimensions) and `affine.for` (x), and the
    accesses at constant stencil offsets are `affine.load` / `affine.store` of `shift + i + offset` (the shifts and the range are
    function arguments, so symbols), which lets affine-level transformations (tiling, interchange, unroll-and-jam, ...) work on the
    group before `lower-affine`. Data-dependent accesses (`ops.read_dyn`) stay `memref.load`. A loop with reductions keeps the
    `scf` form (the function then has `ops.loops = "scf"`); the attribute `ops.loops` is set whenever `loops=affine` is asked for.

    The function is marked `llvm.emit_c_interface`, its memref arguments `llvm.noalias`, and carries the attribute
    `ops.abi` describing the argument groups. Chains that contain `ops.call`s, and chains without a loop, are left
    alone. A group called from another chain in the same module cannot be converted (its signature changes): split
    the groups into modules of their own first.
  }];
  let options = [Option<"soa", "soa", "bool", /*default=*/"false", "structure-of-arrays layout of multi-component dats">,
                 Option<"target", "target", "std::string", /*default=*/"\"cpu\"",
                        "cpu: scf.parallel over the outer dimensions with a sequential x loop; gpu: a gpu.launch with one thread per point">,
                 Option<"block", "block", "std::string", /*default=*/"\"32,4,1\"",
                        "gpu: thread block size x,y,z (the dimensions the block does not have are folded into x)">,
                 Option<"loopStyle", "loops", "std::string", /*default=*/"\"scf\"",
                        "cpu: scf (scf.parallel / scf.for, memref accesses) or affine (affine.parallel / affine.for, affine accesses; loops without reductions)">];
  let dependentDialects = ["::ops_mlir::ops::OPSDialect", "::mlir::arith::ArithDialect", "::mlir::func::FuncDialect",
                           "::mlir::gpu::GPUDialect", "::mlir::LLVM::LLVMDialect", "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect",
                           "::mlir::affine::AffineDialect"];
}


def OpsHoistAccumulators : Pass<"ops-hoist-accumulators", "::mlir::ModuleOp"> {
  let summary = "move the reduction accumulators of a lowered group out of the loop body";
  let description = [{
    `convert-ops-to-scf` accumulates a loop's reductions in a scalar `memref.alloca` (marked `ops.accumulator`) that
    each iteration of the parallel loop initialises with the neutral element. An alloca in a loop body is a dynamic
    allocation after lowering: it makes LLVM bracket the body with stack save/restore and keeps it from promoting the
    accumulator to a register (so no vectorised reduction). This pass moves each marked alloca to the entry of the
    nearest enclosing `omp.parallel` region (one per thread) or, without OpenMP, of the function. Run it after
    `convert-scf-to-openmp`.
  }];
  let dependentDialects = ["::mlir::memref::MemRefDialect", "::mlir::omp::OpenMPDialect"];
}

def OpsMarkNoalias : Pass<"ops-mark-noalias", "::mlir::ModuleOp"> {
  let summary = "mark the data pointers of the dat arguments of a lowered group noalias";
  let description = [{
    Run after the conversion of the group function to the LLVM dialect: the memref arguments of `func.func` are
    expanded to a descriptor (allocated pointer, aligned pointer, offset, sizes, strides) and the attribute
    `llvm.noalias` of the argument does not reach the pointers the code loads and stores through. This pass sets it on
    the aligned pointer of each dat, using `ops.abi.noalias_args` that `convert-ops-to-scf` recorded. Distinct dats of
    a group never overlap (the lifter deduplicates by dat), which lets LLVM vectorise without run-time overlap checks.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}


def OpsFpContract : Pass<"ops-fp-contract", "::mlir::ModuleOp"> {
  let summary = "allow floating point operations to be contracted into fused multiply-adds";
  let description = [{
    Sets the `contract` fast-math flag on every floating point `arith` operation, so that the code generator may fuse a
    multiplication and an addition (what nvcc and clang do by default within an expression). The results then differ from
    unfused code in the last bits. Run after `convert-ops-to-scf`.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

def OpsGpuKernelAttrs : Pass<"ops-gpu-kernel-attrs", "::mlir::gpu::GPUModuleOp"> {
  let summary = "mark the dat pointers of the kernels of a gpu.module noalias";
  let description = [{
    Run after `convert-gpu-to-nvvm`: a memref argument of a kernel is expanded to a descriptor (allocated pointer, aligned
    pointer, offset, sizes, strides) and the pointers a kernel loads and stores through are the aligned ones. The dats of a
    group never overlap (the lifter deduplicates by dat), which is the `__restrict__` of the stock CUDA code: with it the
    code generator may move loads past stores and share loads of the same address.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect", "::mlir::gpu::GPUDialect"];
}

#endif // OPS_MLIR_CONVERSION_PASSES
