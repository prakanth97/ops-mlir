# User guide

How to build an OPS application with the MLIR JIT and run it. For how it works inside, see `design.md`.

## 1. What you get

An OPS application translated with `ops-translator --jit` embeds, next to its ahead-of-time (AOT) loop code, an
intermediate representation of every user kernel. At run time `libopsjit` takes over the lazy execution queue of OPS:
it fuses and compiles the loops it understands (CPU: OpenMP; GPU: CUDA) and leaves the rest to the stock OPS code. A
loop the JIT cannot handle is never an error: it runs with the AOT code, and `OPS_JIT_EXPLAIN=1` says why.

## 2. Build

You need the pinned LLVM/MLIR/clang (23.1.0, see `scripts/env.sh`), OPS from its `feature/jit-executor` branch, and this
repository.

```bash
source scripts/env.sh                  # toolchain, MPI/CUDA/HDF5 paths (cluster: modules), OPS_SRC, OPS_INSTALL
scripts/build_ops.sh                   # OPS libraries, headers and translator -> $OPS_INSTALL   (OPS_CUDA=on for CUDA)
scripts/build_mlir.sh                  # ops-kernel-import, ops-opt, libopsjit -> $OPSJIT_BUILD_ROOT/ops-mlir
scripts/build_apps.sh                  # the example applications built with the JIT (needs the two steps above)
```

`scripts/ci.sh` builds and runs the unit and lit tests.

### Your own application

1. Translate with `--jit` (in a build directory that has the application's `.cpp` and kernel headers):

   ```bash
   $OPS_INSTALL/translator/ops_translator/ops_venv/bin/python3 \
       $OPS_INSTALL/translator/ops_translator/ops-translator -DOPS_ACC_IGNORE -I $OPS_INSTALL/include --jit --file_paths app.cpp
   ```

   This writes the usual generated loops plus `jit_kernels.json` (the kernels, their translation unit and parameter kinds).
   The generated `ops_init_backend()` calls `ops_jit_init()`: the executable must link `libopsjit`.

2. Import the kernels and embed them: in CMake, `ops_add_jit_artifacts(<target> KERNELS_JSON jit_kernels.json)` from
   `cmake/OpsJitArtifacts.cmake` runs `ops-kernel-import` per kernel (with the compiler flags of `<target>`) and adds a generated
   source that defines `ops_jit_artifact_<kernel>` for every kernel. `cmake/OpsApps.cmake: ops_jit_add_app` does translation,
   import and the executables for the OPS example apps (see `apps/CMakeLists.txt`).

3. Link `libopsjit.so` (`-Wl,-rpath` to its directory) together with the OPS library of your configuration (`ops_seq`, `ops_mpi`,
   `ops_cuda`, `ops_mpi_cuda`). The same `libopsjit` serves all of them. Compile with `-DOPS_LAZY` as for the AOT lazy builds.

An application translated without `--jit` contains no reference to the JIT at all.

## 3. Running

Nothing changes on the command line. Behaviour is controlled by environment variables (`ops_jit_set_option(name, value)` does the
same from code, before the first loop):

| Variable | Default | Meaning |
|---|---|---|
| `OPS_JIT` | 1 | 0: the JIT registers nothing; the application runs as if built without it |
| `OPS_JIT_CPU` | omp | `seq`: no OpenMP in generated code |
| `OPS_JIT_FUSION` | 1 | 0: one group per loop |
| `OPS_JIT_FUSION_MAX` | 8 | most loops in a group |
| `OPS_JIT_FUSION_REORDER`, `_PLACEMENT` | 1, earliest | planner options (see `design.md`) |
| `OPS_JIT_FUSION_GUARDED` | 0 | fusing loops of different ranges: not implemented (setting it prints a message and is ignored) |
| `OPS_JIT_SPECIALIZE` | none | `ranges`, `shapes`: fold loop ranges / dat shapes into the code (one compile per distinct value) |
| `OPS_JIT_CACHE`, `OPS_JIT_CACHE_DIR` | 1, `~/.cache/opdsl-jit` | persistent cache of compiled code (also `$XDG_CACHE_HOME/opdsl-jit`) |
| `OPS_JIT_RUN_MAX` | 512 | most loops planned together |
| `OPS_JIT_GPU_BLOCK` | 32,4,1 | thread block of generated CUDA kernels |
| `OPS_JIT_GPU_ARCH` | the device | e.g. `sm_80` |
| `OPS_JIT_PTXAS_OPTS` | | e.g. `-v` to see register use |
| `OPS_JIT_SYNC` | 0 | synchronise after every GPU launch (debugging) |
| `OPS_JIT_FMA` | 0 | let the CPU code generator fuse multiply-adds (no measurable gain on the machines tried; the last bits of the results change, so `OPS_JIT_VERIFY` reports differences) |
| `OPS_JIT_LOOPS` | scf | `affine`: the CPU lowering goes through the affine dialect (`affine.parallel`, `affine.for`, `affine.load/store`) until `lower-affine`; for transformations that work on affine loops. Loops with reductions and the GPU keep the scf form (see "The affine path") |
| `OPS_JIT_AFFINE_PASSES` | none | a pass pipeline run on the affine form (implies `OPS_JIT_LOOPS=affine`), e.g. `func.func(affine-loop-unroll{unroll-factor=4})`; write `+` for a comma and `;` for a blank (the option syntax of the pipelines uses those), e.g. `func.func(affine-loop-tile{tile-sizes=32+32})` |
| `OPS_JIT_VERIFY_HALF_TOL` | 1e-2 | relative tolerance of verify mode for half precision dats (the host code rounds a whole expression once, the JIT every operation) |
| `OPS_JIT_VERIFY_FLOAT_TOL` | 1e-5 | relative tolerance of verify mode for single precision dats |
| `CUDA_ROOT` | `$CUDA_INSTALL_PATH` | CUDA toolkit (ptxas, libdevice) used to compile kernels |

Diagnostics:

| Variable | What it does |
|---|---|
| `OPS_JIT_EXPLAIN=1` | one line per kernel the JIT declined, with the reason |
| `OPS_JIT_PLAN=1` | the groups of every plan |
| `OPS_JIT_STATS=1` | counters and phase times at exit; the line `ops_jit counters:` is machine readable |
| `OPS_JIT_TRACE=1` | every group before it runs (to find the one that crashes) |
| `OPS_JIT_VERIFY=1` | run every group with the JIT and again with OPS's code on the same inputs and compare (dats: relative to the largest value, 1e-12 double / 1e-5 float; reductions) |
| `OPS_JIT_DUMP_DIR=<dir>` | write the chain, the group IR and the lowered IR of every compile |
| `OPS_JIT_DUMP_DATA=1` | also record the first call of every group (inputs, checksums) for `ops-jit-replay` |
| `OPS_JIT_BACKTRACE=1` | a stack trace on a crash inside compiled code |

`ops-jit-replay <dump dir> <group>` recompiles a recorded group and runs it on the recorded inputs without OPS.

### The affine path

`OPS_JIT_LOOPS=affine` (or the option `loops=affine` of `ops-jit-cpu`) lowers a group through MLIR's affine dialect instead of straight to `scf`:

* the outer dimensions are an `affine.parallel`, x an `affine.for`; the bounds are the range arguments of the group (symbols) or constants (`OPS_JIT_SPECIALIZE`);
* an access at constant stencil offsets is an `affine.load` / `affine.store` of `shift + i + offset`, per dimension (the shifts are function arguments, so symbols);
* `ops.read_dyn` (data-dependent offsets) stays a `memref.load`, and data-dependent control flow stays `scf.if`: affine passes treat those parts of the loop nest conservatively;
* a loop with a reduction keeps the `scf` form (the function carries `ops.loops = "scf"` to say so); the GPU lowering is unchanged;
* `OPS_JIT_AFFINE_PASSES` is run on the affine form, then `lower-affine` and the usual pipeline (OpenMP, LLVM). The passes of MLIR's affine dialect are registered; a pass from elsewhere has to be linked into `libopsjit` and registered with MLIR's pass registry. Both options are part of the cache key of a group.

Fusion across loops stays where it is (`ops-jit-plan`): it knows which dats are temporaries, which affine loop fusion cannot. `affine-loop-fusion` on the result is possible but only useful for loops that the planner did not fuse. Many affine transformations want static shapes: combine with `OPS_JIT_SPECIALIZE=shapes`.

## 4. What the JIT runs, and what it declines

A loop is run by the JIT when its kernel was imported (`ops-kernel-import`, provider A: a clang AST walker; `scripts/import_coverage.py` reports what imports) and
the arguments are in the supported set: dats of type double, float or int; regular (non multigrid) stencils; read-only globals;
reductions (INC/MIN/MAX); the loop index; constants registered with `ops_decl_const`. Optional dat arguments that are switched off
are fine. Edge dats (size 1 in a dimension, read through stride-0 stencils) are plain dats on one process; with MPI they are
replicated and exchanged by OPS, and the loops that use them run with OPS. Declined (run by OPS, reason in `OPS_JIT_EXPLAIN`):
multigrid stencils, edge dats under MPI, stride-0 dimensions on dats that are written in loops of more than one point in that
dimension, kernels the importer cannot read (loops with data-dependent trip counts, pointer arithmetic, calls other than the math
library and helper functions defined in the same file, writes to globals...), and `isdevice` loops when no CUDA device is usable.
`printf` in a kernel is dropped with a warning.

## 5. Troubleshooting

* *The application crashes in compiled code*: `OPS_JIT_BACKTRACE=1 OPS_JIT_TRACE=1`, then `OPS_JIT_FUSION=0` and `OPS_JIT_CPU=seq` to
  narrow it down; `OPS_JIT_VERIFY=1` finds the first group that computes something different from OPS. Report with
  `OPS_JIT_DUMP_DIR=... OPS_JIT_DUMP_DATA=1` and `ops-jit-replay`.
* *Results differ in the last digits*: fusion removes memory round trips and LLVM contracts multiply-adds; bitwise equality with
  the AOT code is not promised. `OPS_JIT=0` gives the AOT result.
* *GPU: "CUDA path ... does not exist"*: set `CUDA_ROOT` to a toolkit that has `bin/ptxas` and `nvvm/libdevice`.
* *Cold start is slow*: the first run compiles every distinct group (about 0.05-0.3 s each on the CPU, more on the GPU because of
  `ptxas`). The disk cache makes later runs start immediately; keep `OPS_JIT_CACHE_DIR` on a fast, writable, per-user location.
