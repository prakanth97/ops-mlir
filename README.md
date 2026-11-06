# ops-mlir

An MLIR JIT for the [OPS](https://github.com/OP-DSL/OPS) structured-mesh library. It is an optimising tier that plugs into
OPS's own lazy execution: OPS queues the loops of a program as kernel descriptors and, at a flush, hands them to the JIT, which
fuses them, specialises them and compiles them (CPUs with OpenMP, NVIDIA GPUs with CUDA). Loops the JIT does not take run with
OPS's generated code.

## How it works

1. **Build time.** The OPS translator (`--jit`) runs `ops-kernel-import`, a clang AST walker, over the user kernels and embeds one
   MLIR artifact per kernel (bytecode and a manifest) next to the ordinary generated code.
2. **Run time.** `ops_par_loop` queues a descriptor instead of running the loop. When the program needs the data, OPS flushes the
   queue through the registered chain executor (`ops_register_chain_executor`).
3. **The executor** (`libopsjit`) claims the loops it supports and lifts them into an `ops.chain` of SSA values. It plans fusion
   groups (`ops-jit-plan`), lowers each group (`ops-jit-cpu`: scf and OpenMP, optionally through the affine dialect;
   `ops-jit-cuda`: NVVM), compiles it with ORC LLJIT and caches the code in memory and on disk, keyed by the bytecode of the group
   and the options.
4. **OPS keeps** the memory, MPI halos, dirty bits, reductions and timing; the JIT only runs loops. In verify mode
   (`OPS_JIT_VERIFY=1`) every group is compared with OPS's own code.

More: [docs/design.md](docs/design.md) (architecture), [docs/user-guide.md](docs/user-guide.md) (building an application, the
environment variables), [docs/passes.md](docs/passes.md) (passes and pipelines, generated),
[docs/performance.md](docs/performance.md) (measurements).

## Layout

| Directory | Contents |
|---|---|
| `core/` | `opdsl`: DSL-independent infrastructure (hashing, artifacts, JIT session, caches, GPU shim). Must not depend on OPS. |
| `ops/` | the `ops` dialect, analyses, transforms, conversions, pipelines, kernel importer, runtime |
| `tools/` | `ops-opt`, `ops-kernel-import`, `ops-jit-replay` |
| `test/` | lit tests (`test/lit`), end-to-end suites |
| `apps/`, `cmake/` | the OPS applications built with the JIT, and the CMake helpers for them |
| `scripts/` | environment (`env.sh`), OPS build, CI entry point, performance scripts |
| `cluster/v2/` | `sync.sh`: publishes the local branches to the `cl` cluster |
| `third_party/OPS/` | git submodule: the OPS sources (branch `feature/jit-executor`), used for the OPS build, the importer tests and the apps |

## Getting the sources

```bash
git clone --recurse-submodules https://github.com/OP-DSL/ops-mlir.git     # or: git submodule update --init third_party/OPS
```

`scripts/env.sh` takes `OPS_SRC` from the submodule when it is checked out; set `OPS_SRC` to use your own OPS clone
(for work on the OPS side). The submodule is pinned to a commit: to move it, `git -C third_party/OPS checkout <commit>`
and commit the new pin. Third-party code is never copied into this repository.

## Build and test

The toolchain is pinned (LLVM/MLIR/clang 23.1.0, `cmake/LLVMVersion.cmake`); `scripts/env.sh` selects it on the
cluster and in WSL.

```bash
source scripts/env.sh
scripts/ci.sh          # configure, build, check-ops-mlir
```

Targets: `check-ops-mlir` (everything), `check-ops-mlir-lit`, `check-ops-mlir-unit`.

The lit tests need `lit`, which the pinned LLVM install does not ship: `pip install -r requirements-test.txt`, or
point `OPSJIT_LIT` (environment or `-DOPSJIT_LIT=`) at the `llvm-lit` of an LLVM build. Without it
`check-ops-mlir-lit` fails with a message saying so; the rest builds and runs.

## Citation

If you use this work, please cite:

```
A. Dascombe, P. Thilakaraj, and G. R. Mudalige, “Code Generation with MLIR for the OPS Structured-Mesh DSL,”
in Proceedings of the 2026 IEEE/ACM 12th Workshop on the LLVM Compiler Infrastructure in HPC (LLVM-HPC 2026),
Chicago, IL, USA, Nov. 16, 2026, to be published.
```

The paper describes the first version of ops-mlir, which is in the history of `main`; this version keeps its idea and replaces
the embedded Python/xDSL lowering with the executor described above.
