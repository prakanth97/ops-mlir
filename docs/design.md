# Design

An MLIR JIT for OPS that plugs into OPS's lazy execution. This document describes the architecture as built ("plan §n" refers to the implementation plan this was built from, which is not part of the repository).

## 1. Idea

OPS can queue the loops of a program (`ops_par_loop` builds a *kernel descriptor*) instead of running them, and flush the queue
when the host observes data. The queue is the unit of optimisation: from it we can see which loops follow each other, what
they read and write, and fuse them or compile them for a target the AOT code generator does not cover. The JIT is a *chain
executor* (OPS API `ops_lazy_executor.h`): at a flush OPS hands it the descriptors in program order, and it executes each exactly once. It
runs the loops it understands with generated code and gives the rest back to OPS (`ops_lazy_execute_default`). Everything about
halo exchanges, host/device coherence, dirty bits and timing stays OPS's business, through per-loop calls
(`ops_lazy_prepare/finish/account`).

```
build time   app.cpp + kernels.h --ops-translator --jit--> AOT loops (unchanged) + jit_kernels.json
             jit_kernels.json --ops-kernel-import (clang AST walker)--> per kernel: ops.kernel bytecode + manifest
             ops_add_jit_artifacts: embeds them as `ops_jit_artifact_<kernel>`; the loops pass the pointer to OPS
run time     flush(descs) -> claim -> run of claimed loops -> lift to ops.chain -> frontend passes -> planner (fusion)
                          -> per group: lower (CPU: OpenMP, GPU: CUDA) -> LLVM -> object -> JIT -> call with bound arguments
```

## 2. Layering (the rule: dependencies point down)

| Layer | Directory | Knows about | Contents |
|---|---|---|---|
| core (`opdsl`) | `core/` | LLVM/MLIR only; **no OPS** (a test enforces it) | `Hash`, `PipelineRegistry` (named pipelines with option strings), `JITSession` (ORC LLJIT: compile a module to an object, load an object, packed call), `PersistentCache` (atomic, checksummed, LRU), `GpuShim` (the `mgpu*` runtime symbols over the CUDA driver) |
| dialect and passes | `ops/include`, `ops/lib/{Dialect,Analysis,Transforms,Conversion,Pipelines}` | MLIR; **no OPS library, no runtime** | the `ops` dialect, footprint and dependence analyses, the fusion planner, passes, the named pipelines |
| frontend | `ops/lib/Frontend`, `tools/ops-kernel-import` | the dialect, clang | kernel C++ -> `ops.kernel` + manifest |
| runtime | `ops/lib/Runtime` | everything above, the OPS headers; **no OPS library at link time** | claim, lifter, binding, executor, `libopsjit.so` |
| tools, apps, tests | `tools/`, `apps/`, `test/` | | `ops-opt`, `ops-kernel-import`, `ops-jit-replay`; the example applications; lit, unit and end-to-end tests |

`libopsjit` is the same file for sequential, OpenMP, MPI and CUDA builds of an application: it calls OPS only through
`ops_lazy_executor.h` and the data structures of the descriptors, and the OPS library of the application provides those symbols
at link time. `OPSMLIRRuntimeCore` (argument packing, call recorder, group compiler) has no OPS dependency at all and is what
`ops-jit-replay` and `op2-mlir` can link.

## 3. The dialect

`ops` models what an OPS loop *is*, at three levels.

* **Kernel level** (`ops.kernel`): the body of a user kernel, with accessor parameters (`!ops.acc<T, rank, ncomp>`), read-only
  globals (`!ops.gbl<T, n>`), reductions (`!ops.red<T>`), `ops.read/read_dyn/write` at stencil offsets, `ops.gbl_read`,
  `ops.reduce`, `ops.index d` (global index), `ops.const_ref @name` (a registered constant), plus `arith`, `math`, `scf`.
* **Loop level** (`ops.par_loop`): a kernel applied over a range, with `dats(...)` each carrying an access mode (`read write rw inc min max`) and
  a stencil attribute (points, per-dimension strides, 0 = the dat has no such dimension), scalars (components of read-only globals and
  constants), and reduction kinds. The *unmaterialised* form has an empty body and names a kernel; the body is the kernel with
  parameters mapped to block arguments.
* **Chain level** (`ops.chain`): dats, range bounds and scalars as arguments; a list of loops; results are the reductions. After
  planning a chain is a list of `ops.call`s to group chains, each with a single loop.

The verifier checks that every access is inside the stencil of its operand (this is what makes dependence analysis sound), that
write accesses are on writable operands (non-zero write offsets are legal: the TGV symmetry boundaries), and operand/argument
consistency.

## 4. Passes and pipelines

| Pass | Purpose |
|---|---|
| `ops-materialize-kernels` | clone kernel bodies into the loops, mapping parameters by kind; constants and global components become scalar arguments; forwards stores; diagnoses reads of write-only data |
| `ops-specialize-consts` | fold constants marked *bound* (loop bounds, array indices) to their values (given in `ops.const_values`), unroll loops that become constant-trip |
| `ops-fuse-point-local` | the fusion planner: loops with equal ranges whose dependences are point-local (zero-offset) are merged, reordering independent loops when allowed; reductions are barriers |
| `ops-forward-stores` | store-to-load forwarding inside a fused body |
| `ops-outline-groups` | each loop of the chain becomes a group chain; the chain becomes calls |
| `ops-specialize` | `mode=ranges`: the loop ranges become constants; `mode=shapes`: also the dat sizes (static memref shapes and strides in `convert-ops-to-scf`); the keys change accordingly |
| `convert-ops-to-scf` | one loop -> `func.func` in the group ABI (section 6); `target=cpu`: `scf.parallel` over the outer dimensions with a sequential x loop; `target=gpu`: a `gpu.launch` with one thread per point, warp-shuffle reduction and atomics |
| `ops-hoist-accumulators` | reduction accumulators out of the loop body (one per OpenMP thread) |
| `ops-mark-noalias` | `noalias` on the data pointers after lowering to LLVM (host function) |
| `ops-fp-contract` | the `contract` fast-math flag on the floating point operations (`fma=true`, `OPS_JIT_FMA`): the code generator may fuse multiply-adds; off by default (the last bits of the results change) |
| `ops-gpu-kernel-attrs` | `noalias` on the dat pointers of the kernels of a `gpu.module` (what `__restrict__` does in the stock CUDA code) |

Pipelines (named, with option strings, in `opdsl::PipelineRegistry`; also usable as `ops-opt -ops-jit-cpu=...`):
`ops-jit-frontend` = materialise, specialise constants, canonicalise, CSE; `ops-jit-plan` = fuse, forward stores, canonicalise, CSE,
outline; `ops-jit-cpu{omp specialize soa}` = specialise, lower, OpenMP, down to the LLVM dialect; `ops-jit-cuda{chip features block ptxas
specialize soa fma restrict}` = specialise, lower for the GPU, our copy of the stages of MLIR's NVVM pipeline (with `ops-gpu-kernel-attrs` between the kernel conversion and the
serialisation; fatbin through `ptxas`). All passes and pipelines with their options: `docs/passes.md` (generated by `scripts/gen_passes_doc.py`).

**Analyses** (`Analysis/`): `Footprint` (the box a loop touches per dat: range extended by the stencil, unbounded for stride != 1),
`ChainDependence` (RAW/WAR/WAW edges and whether each is point-local), `FusionPlanner` (consecutive and DAG planners; the pass uses the
DAG planner). A randomised oracle test checks the planner's plans for semantic equivalence and was itself mutation-checked.

## 5. Frontend (provider A)

`ops-kernel-import` parses the *translation unit of the application* with its real flags (so globals, structs and macros are the
real ones), finds the kernel function and walks its AST (`CSubsetWalker` is a generic C-subset-to-MLIR walker; `OPSKernelWalker` adds
accessors `ACC<T>`/legacy `OPS_ACCn` macros, global and reduction parameters, registered constants, `idx`). Locals are SSA values
threaded through `scf.if`/`scf.for`; loops over components are unrolled. Helper functions defined in the file are inlined. Anything
outside the subset is a failure with a reason in the manifest (`status: failed`) and the loop runs AOT. The manifest (plan §4.2) records
parameter kinds and types, constants (with offsets in structs, `role: bound`), the constant offsets read and written per accessor,
reductions' style. Over the OPS example applications 588 of 626 kernels import (`scripts/import_coverage.py` regenerates the report). Provider B (ClangIR) was evaluated on a corpus of 62 translation units, of which 38 emit CIR, and is not adopted; provider C (LLVM IR) is a possible fallback.

## 6. Runtime

**Claim** (`Claim.cpp`): per descriptor, cheap: artifact present and known, arguments match the kernel's parameters, types are
double/float/half/int, regular stencils, accesses inside the declared stencils, stride-0 dims on written dats only in loops of one point in that dimension,
edge dats (size 1 in a dimension) only on one process (with MPI OPS exchanges them itself), registered constants, reduction style OK. The reason for every decline is a sentence (`OPS_JIT_EXPLAIN`). Optional (switched-off) dat arguments
are passed like any other.

**Executor** (`Executor.cpp`): a run of consecutive claimed loops of the same kind (host or device) is processed together:
local ranges (`ops_lazy_compute_local_range`; loops with nothing to do on this MPI rank are skipped as OPS does), then the *plan*
for the structure of the run. The plan key hashes artifact digests, block, argument kinds/accesses, dat slots/types/dims, stencils,
the values of *bound* constants (the bytes the kernel reads), the local ranges, the dat sizes under `shapes`, and the type of the handle of each reduction; plans live in memory, and the module
after the plan passes is cached on disk by plan key (a later process skips the frontend and plan pipelines). A miss lifts the run (`Lifter`) to an `ops.chain` of
unmaterialised loops, runs `ops-jit-frontend` and `ops-jit-plan`, derives a *binding recipe* for each group from the planned chain,
and gets the compiled code of each group from the `GroupCompiler`. The group code is keyed by the SHA-256 of the canonical bytecode of the
group (ranges, loop numbers and constant values removed unless specialised), pipeline and options, target and LLVM version; it is cached
in memory and on disk (`PersistentCache`), so a second run of the application compiles nothing; a group that fails to compile runs with OPS (the rest of the run goes on). Executing a plan: for each group,
`ops_lazy_prepare` of its loops, bind, call, combine reductions into the handle slots (always `handle op= value`, as OPS does: a handle that is never read back keeps accumulating),
`ops_lazy_finish`, `ops_lazy_account`.

**Group ABI** (plan §4.6): `func(dat memrefs..., shifts..., range lo/hi..., displacement..., scalars...[, reduction buffers])`. A dat is a
memref over its whole allocation (z, y, x, then components; AoS or SoA); the index of an access is `i_d * stride_d + offset_d + shift_d` with
`shift_d = -base - d_m (- d_im under MPI)` from `ops_lazy_dat_shift`. Reductions come back as the function's results (CPU) or through
device buffers (GPU). The packed wrapper `void(void **args)` hides the C interface conventions (struct results, result pointers).

**Verify** (`OPS_JIT_VERIFY`): per group, snapshot written dats and reduction slots, run the group, save results, restore, run the
members with OPS, compare. This is the regression test of the whole chain and runs in ctest for every example application.

**GPU**: loops with `isdevice` use `dat->data_d`; the group is compiled by `ops-jit-cuda` and launched through the shim on the legacy
default stream (so it is ordered with OPS's own CUDA work); every warp reduces its threads with shuffles and lane 0 stores a partial value in a per-group device buffer (after the scalars in the argument list); the executor copies
the buffer back (a synchronising copy) and combines the partial values into the handle on the host like on the CPU.

## 7. Where to change what

* **A new optimisation pass** over groups: add it to `ops/lib/Transforms`, declare it in `Passes.td`, put it into the pipeline in
  `lib/Pipelines/Pipelines.cpp` (it becomes part of the key through the pipeline name and options; bump `kCodeFormat` in
  `GroupCompiler.h` when the output of an existing pipeline changes), add a lit test with FileCheck and, if it can change numbers, run
  `OPS_JIT_VERIFY=1` on CloverLeaf/TeaLeaf/TGV.
* **A new target**: a lowering pass (as `convert-ops-to-scf target=gpu` is one variant) and a pipeline registered in `Pipelines.cpp`; a
  shim in `core/GPU` if it needs runtime symbols; the executor chooses the pipeline in `Executor::buildPlan`.
* **A new frontend**: produce the same `ops.kernel` + manifest (`importerId()` is part of the digest); `test/lit/Frontend` is the
  conformance suite.
* **A new loop feature** (e.g. multigrid): the claim, the lifter (what the chain looks like), the dialect verifier, the binding.
* **Reusing the core for another DSL** (`op2-mlir`): `core/` and `OPSMLIRRuntimeCore` have no OPS dependency.

## 8. Testing

`scripts/ci.sh`: lit tests (`test/lit`: dialect round trips, analyses, passes, conversions executed with mlir-runner, importer on real
kernels), unit tests (`core/unittests`, `ops/unittests`: planner oracle, runtime on a real OPS world, end-to-end CPU/GPU), then the
cluster: `ctest -L jit-app` runs the example applications built with the JIT in several configurations (default, no fusion, verify,
two runs with a fresh cache that must not compile the second time) with MPI and on GPUs.
