# Performance

Measurements of the JIT (branch `v2`) against OPS's own ahead-of-time (AOT) code, on the `cl` cluster. The AOT executables are the standard
OPS builds of the same applications (`OPS-apps-new`: OpenMP, CUDA); the JIT executables are the same sources translated with `--jit`
(`apps/`, `scripts/build_apps.sh`). Both are built by the same clang 23. Times are the wall time of the whole run (start-up, the JIT's
compilation from the disk cache, output included), the best of 3 unless said otherwise. The JIT runs once before the timed runs to fill its
disk cache (`OPS_JIT_CACHE_DIR`); a cold start costs more, see "Compilation cost". Scripts: `scripts/perf/{cpu,gpu,cpu-kernels,cpu-groups,
cpu-fusion-sweep,gpu-blocks,sample}.sh` (each in a SLURM job after `source scripts/env.sh`), raw results in `~/ops-jit-cl/results/`.

Machines: CPU nodes of the `cpu` partition (euclid: 48 cores; hajos, kemeny: 64 cores; one run per node, `--exclusive`, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, as many threads as cores), GPU: NVIDIA A100 (40 GB, `renyi`), CUDA 12.3.

## CPU

| Case | Node | AOT [s] | JIT [s] | JIT, no fusion [s] | JIT / AOT |
|---|---|---:|---:|---:|---:|
| CloverLeaf 2D, 3840^2, 87 steps | hajos/kemeny (64 thr) | 27.4 | 27.7 | 28.9 | 1.01 |
| CloverLeaf 3D, 256^3, 87 steps | hajos/kemeny (64 thr) | 56.4 | **54.0** | 59.4 | 0.96 |
| TGV double, 512^3 | euclid (48 thr) | 1257 | **253** | 231 | 0.20 |
| TGV single | euclid (48 thr) | 1184 | **163** | 122 | 0.14 (0.10 unfused) |
| TGV single/double | hajos (64 thr) | 888 | **214** | 224 | 0.24 |

* CloverLeaf is at parity: per kernel (OPS's own timing table, `scripts/perf/cpu-kernels.sh`, 20 steps of the 3D case, 64 threads) the 38 heaviest
  kernels are within 2% of the AOT code; the loops that were slow before the fixes below were the loops of the 1-D coordinate dats.
* TGV (OpenSBLI, many large point-wise and stencil kernels per time step) is 4-10 times faster than the stock OpenMP code: the JIT fuses
  point-local chains and specialises the ranges and the constants; the stock code streams every temporary through memory.
* Fusion is a small win or neutral on the CPU, and machine dependent: CloverLeaf 3D is 10% faster fused (as soon as pairs of loops are fused), TGV double
  7% faster at 64 threads; TGV single is neutral at 64 threads but 34% *slower* fused on euclid (163 s against 122 s). There is no register or cache model
  yet (plan item P7.4: register-aware fusion). The size sweep is in the section "Fusion size".


### Fusion size

`OPS_JIT_FUSION_MAX` (largest group; 1 = no fusion), 64 threads on hajos/kemeny, wall time [s] of the whole run, `scripts/perf/cpu-fusion-sweep.sh`:

| Case | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| CloverLeaf 3D | 59.4 | 54.3 | 53.7 | 54.0 | 53.8 | 54.1 |
| TGV single | 164.9 | 161.9 | 160.2 | 163.8 | 167.8 | 169.8 |
| TGV double | 322.7 | 317.5 | 311.4 | 308.4 | 303.3 | 300.0 |

CloverLeaf gets all of its gain from pairs of loops (the loops of a CloverLeaf step are mostly memory-bound sweeps whose temporaries do not survive a pair);
TGV double keeps gaining up to the largest groups (2.7% from 8 to 32), TGV single loses from 8 loops on. The default (`OPS_JIT_FUSION_MAX=8`) is a compromise; a per-case
choice needs the register-aware cost model of plan item P7.4.

### What the first cluster run found

The first run (before the fixes described below) had CloverLeaf 2D at 46.5 s against 18.1 s for AOT (euclid, 48
threads) and 3D at 124.8 s against 39.2 s. Causes, in the order of their weight:

1. OPS ran every loop that reads a 1-D (edge) dat after a write to it eagerly with the stock code and flushed the queue first; the rule is meant for
   the halo exchange of MPI runs, but its dirty bit is only cleared by the MPI code, so on one process it fired for ever. The executor never saw these
   loops (6% of them, the heavy advection and viscosity ones). OPS now applies the rule only with more than one process, and the JIT claims edge dats
   on one process.
2. A group that failed to lower ran with OPS: `calc_dt_kernel` of 3D (a conditional expression loading from two dats, whose branches the LLVM-dialect
   canonicaliser merged into a block with memref arguments): 2.7 s against 0.4 s per 20 steps. Pass order fixed.
3. The fusion planner's reachability test (a graph search per loop and candidate group) took seconds on the 500-loop chains of CloverLeaf 3D.
4. Constants and globals read inside an `if` of a kernel were not materialised (the group then ran with OPS).

## GPU (A100)

| Case | AOT CUDA [s] | JIT [s] | JIT, no fusion [s] | JIT / AOT |
|---|---:|---:|---:|---:|
| CloverLeaf 2D, 3840^2 | 5.86 | 6.49 | 6.60 | 1.11 |
| CloverLeaf 3D, 256^3 | 7.64 | 8.19 | 8.53 | 1.07 |
| TGV single, 512^3 | 13.17 | 13.95 | 19.63 | 1.06 (1.49 unfused) |

* TGV double and single/double do not fit in 40 GB with either code (AOT and JIT both run out of memory at the benchmark size).
* The thread block (`OPS_JIT_GPU_BLOCK`, default `32,4,1`, the one OPS uses) is the best or tied in a sweep of twelve shapes of 128 to 512 threads
  (`scripts/perf/gpu-blocks.sh`): 2D 6.45 s for every shape, 3D from 8.58 s (default) to 9.19 s, TGV from 14.47 s (default) to 16.17 s.
* **Per kernel** (CloverLeaf 3D, 20 steps, `scripts/perf/gpu-kernels.sh`: OPS's timing table of the AOT code against the JIT without fusion and with
  `OPS_JIT_SYNC=1`): the sum over the kernels is 0.900 s for AOT and 0.925 s for the JIT with `OPS_JIT_SPECIALIZE=shapes` (+2.8%), 0.933 s with `ranges`,
  0.940 s with none (+4.4%). The first version of the lowering was at 0.961 s (+6.8%): the kernels of ours were not told that the dats do not overlap
  (`ops-gpu-kernel-attrs`: `noalias` on the dat pointers, what `__restrict__` does in the stock code; `accelerate_kernel` 0.0486 -> 0.0357 s, AOT 0.0307 s).
  Allowing fused multiply-adds (`OPS_JIT_FMA=1`) changes nothing. What is left is the address arithmetic: the SASS of `accelerate_kernel` has 448-480 instructions
  against 256 for the stock code, the same floating point instructions (30 DMUL, 20 DFMA, 19 DADD, 22 LDG) and about 200 more 64-bit integer instructions (MLIR
  indices are 64 bit; nvcc computes with 32-bit `int`). A 32-bit index path in the GPU lowering is the next step.
* **Host side.** The fixed cost of the JIT is small: a one-step 16^2 run takes 2.38 s with the JIT and 2.25 s with the stock code on the A100 node (CUDA
  initialisation is most of both; `OPS_JIT=0` on the JIT executable: 2.26 s), i.e. about 0.13 s to load the library, the plans and the cached modules. The rest of the gap is
  per step and not yet attributed: kernel time explains +0.1 s of the 0.55 s of CloverLeaf 3D (87 steps); the executor's own work is small (`bind 0.09 s`, `claim 0.03 s`
  per run) so the candidates are the read-back of reductions (a synchronous copy per reduction group) and the launches (9090 fused groups of the 2D run). The persistent plan
  cache (the plan after the frontend and plan passes, by plan key) removed 0.4 s (3D: 8.59 -> 8.19 s).
* Correctness: the whole CUDA suite (`ctest -L jit-app -R cuda`: laplace2d, poisson, lazy_triggers, CloverLeaf 2D and 3D, TeaLeaf, lowdim, TGV DP/SP/SPDP/HP
  and HPSP, with and without MPI) passes on the A100 in verify mode (every group is compared with OPS's stock code), and CloverLeaf 2D/3D and TeaLeaf pass with two
  MPI ranks on the two A100s.

## Compilation cost

Per group (local 16-thread WSL machine, LLVM -O3, host CPU features): `lowering` (MLIR passes) 3 ms, `llvm` (optimisation and code generation) 0.2 to 0.25 s;
CloverLeaf 2D compiles 65 distinct groups on a cold cache (15 to 17 s in total), 3D 95 groups (5 s for a 64^3 case, the groups are shared between
loops of equal structure). With a disk cache hit a group costs about 3 ms (`cache io`: 40 ms for 65 groups). On the GPU the lowering includes ptxas:
CloverLeaf 3D, 94 groups, cold: lowering 4.6 s, LLVM 3.6 s; warm: cache io 0.19 s.

## Reproducing

```bash
cluster/v2/sync.sh                                   # local branches -> the cluster work trees
# then, in SLURM jobs after `source scripts/env.sh`:
scripts/build_ops.sh; scripts/build_mlir.sh; scripts/build_apps.sh   # OPS, ops-mlir, the applications
scripts/perf/cpu.sh                                  # CPU table (CASES=regex, SKIP_AOT=1, THREADS, NRUNS)
scripts/perf/gpu.sh                                  # on a GPU node (-p gpu --gres=gpu:a100:1)
scripts/perf/cpu-kernels.sh                          # per-kernel AOT against JIT (DIM=2|3)
scripts/perf/cpu-groups.sh                           # time per JIT group (OPS_JIT_TRACE) and executor statistics
```
