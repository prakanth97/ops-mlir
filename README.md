# ops-mlir
MLIR Compiler Implementation for OPS DSL

## Dependencies

- CMake >= 3.20
- A C++17 compiler
- [LLVM/MLIR](https://github.com/llvm/llvm-project) built from source, with the `mlir` project enabled
- [OPS](https://github.com/OP-DSL/OPS) (the OPS DSL C library)
- OpenMP

## Python Dependencies

- xDSL (for generating OPS DSL code from Python)

## Installation

### 1. Build LLVM/MLIR

Clone and build `llvm-project` with the MLIR project enabled, e.g.:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
ninja
```

### 2. Clone OPS

Clone the [OPS](https://github.com/OP-DSL/OPS) repository — `ops-mlir`
builds the OPS sequential backend (`ops_seq`) from its source tree and
links against it.

### 3. Set environment variables

`ops-mlir`'s CMake configuration reads the LLVM/MLIR build/source
locations and the OPS checkout location from environment variables — it
does not hardcode any paths. Set the following before configuring the
project:

```bash
export MLIR_BUILD_DIR=/path/to/llvm-project/build
export LLVM_BUILD_DIR=/path/to/llvm-project/build
export LLVM_SOURCE_DIR=/path/to/llvm-project/llvm
export MLIR_SOURCE_DIR=/path/to/llvm-project/mlir
export OPS_ROOT=/path/to/OPS/ops/c
```

A template is provided in [`env_setup_template`](env_setup_template) — copy/edit it for your
own paths and `source` it before building:

```bash
source env_setup
```

### 4. Build ops-mlir

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j$(nproc)
```

The project must be compiled with the clang++ bundled in the LLVM build
(`$MLIR_BUILD_DIR/bin/clang++`, added to `PATH` by `env_setup`). GCC 12 is
incompatible with the generated MLIR headers in LLVM 23. The OpenMP
headers and library are located automatically from `MLIR_BUILD_DIR` — no
extra env vars or `-D` flags are needed.

CMake configuration will fail with a clear error if any of the five
environment variables above are not set.

## 5. Use xDSL Fork
Needed for OPS reductions.

```bash
# Remove standard xDSL (if needed)
pip uninstall xdsl

git clone https://github.com/Archii0/xdsl.git
# you may need to change to the branch "stencil-reduce"
# git checkout stencil-reduce

pip install -e path/to/xdsl-fork

```

TLDR - only partial reduction support is upstreamed into xDSL, waiting on stencil dialect stakeholders to validate my full changes still...

## Usage

Run the 2D Laplace example:

```bash
./build/apps/c/laplace_2d/laplace_2d
```

## Fortran Frontend

`ops-mlir` can capture and JIT-compile OPS **Fortran** applications as well
as C++ ones. Just as the C++ frontend runs clang in-process on the kernel
header, the Fortran frontend runs flang in-process on the kernel module and
translates the FIR it produces, so a Fortran kernel ends up as the same MLIR
the C++ frontend produces from a C++ kernel — and reaches the same backends.

### Requirements

The Fortran frontend needs an LLVM build with the `flang` project enabled:
the runtime links flang's frontend libraries to lower kernel modules to FIR
at run time. They come from the same LLVM build as MLIR, so FIR and MLIR stay
ABI-compatible.

```bash
cd /path/to/llvm-project/build
cmake -S ../llvm -B . \
   -DLLVM_ENABLE_PROJECTS="clang;mlir;flang" \
   -DLLVM_ENABLE_RUNTIMES="flang-rt"
ninja flang
ninja runtimes/flang-rt runtimes/flang-rt-mod
```

Then configure `ops-mlir` with the frontend enabled:

```bash
cmake -S . -B build -DOPS_ENABLE_FORTRAN=ON \
   -DCMAKE_Fortran_COMPILER=$LLVM_BUILD_DIR/bin/flang
cmake --build build -j$(nproc)
```

`OPS_FLANG_ROOT` defaults to `LLVM_BUILD_DIR`; point it elsewhere if flang
lives in a different LLVM build. Set `-DOPS_BUILD_FORTRAN_APPS=OFF` to build
the translator and its tests without linking any Fortran application — the
translator only needs flang's frontend, not `flang-rt`. (Kernels that `use`
an intrinsic module such as `iso_c_binding` do need `flang-rt-mod`.)

### Running

```bash
OPS_BACKEND=seq ./build/apps/fortran/laplace_2d/laplace_2d_ftn
```

### Writing a Fortran application

There is **no translator step**. An OPS Fortran application changes one line:

```fortran
-   use OPS_Fortran_Reference
+   use OPS_MLIR_Fortran_Reference
```

`OPS_MLIR_Fortran_Reference` re-exports the ordinary OPS Fortran API and
replaces `ops_par_loop` with a capturing version — OPS Fortran's
`ops_par_loop` already carries the kernel *name*, so loops can be identified
without rewriting call sites. Iteration ranges stay in OPS Fortran's
one-based inclusive form; the capture layer converts them. Beyond the `use`
line an application only adds:

```fortran
call ops_mlir_set_kernel_source_file(dir // "/kernels.F90")    ! once, after ops_init
call ops_mlir_add_kernel_dependency(dir // "/constants.F90")   ! modules it uses
call ops_mlir_compile_and_execute()                            ! flush captured loops
call ops_mlir_exit()                                           ! instead of ops_exit
```

This mirrors `set_kernel_source_file()` in a C++ application, and `dir` is
found the same way, from `__FILE__` (see
[laplace2d.F90](apps/fortran/laplace_2d/laplace2d.F90)). A kernel module
usually `use`s other modules; register each as a dependency, in the order a
compiler would need them, and the JIT compiles them first.

Constants declared with the usual `ops_decl_const` are baked into kernel
bodies at compile time. A Fortran `parameter` needs nothing at all — flang
folds its value itself.

### Kernel convention

Kernels follow the same convention as ops-mlir's C++ kernels: a pure
function of its stencil neighbours, one scalar per stencil point, returning
the value written at the centre. This is what the stencil lowering calls, so
it maps one-to-one onto the emitted kernel declaration.

```fortran
real(kind=8) function apply_stencil(a, b, c, d, e)
    real(kind=8), intent(in) :: a, b, c, d, e
    apply_stencil = 0.25_8 * (b + c + d + e)
end function apply_stencil
```

An `ops_arg_idx` argument arrives as a rank-one `integer(kind=4)` array of
the block's dimensionality, indexed with ordinary one-based subscripts.

Supported bodies are straight-line expressions over the kernel's arguments,
locals, module constants, and intrinsic math functions — the same restriction
the C++ frontend imposes. Control flow, array-valued dummies other than the
index argument, and derived types are rejected with a diagnostic naming the
construct.

### Inspecting a translated kernel

`fortran-kernel-ir-dump` shows exactly what a kernel translates to, without
building or running the app — the counterpart of `kernel-ir-dump` for C++:

```bash
./build/tools/fortran-kernel-ir-dump \
    apps/fortran/laplace_2d/laplace_kernels.F90 \
    -dep apps/fortran/laplace_2d/laplace_constants.F90 \
    left_bndcon 2 -idx -int jmax=4094
```
