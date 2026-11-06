// Kernels with ACC accessors in 2D: the ideal gas equation of state of CloverLeaf (locals, arithmetic, sqrt) and the
// stencil of the Laplace tutorial (reads at offsets, a max reduction spelled with fmax).
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel ideal_gas_kernel --source %s --ndim 2 -o - --manifest %t.ig.json -- -I%ops_include | FileCheck %s --check-prefix=IDEAL
// RUN: FileCheck %s --check-prefix=IDEAL-MANIFEST --input-file=%t.ig.json
// RUN: ops-kernel-import --kernel apply_stencil --source %s --ndim 2 -o - --manifest %t.as.json -- -I%ops_include | FileCheck %s --check-prefix=STENCIL
// RUN: FileCheck %s --check-prefix=STENCIL-MANIFEST --input-file=%t.as.json

#include <ops_lib_core.h>

// The application's own globals (laplace2d.cpp).
extern double pi;
extern int jmax;

#include "Inputs/cloverleaf_ideal_gas_kernel.h"
#include "Inputs/laplace2d_kernels.h"

// IDEAL-LABEL: ops.kernel @ideal_gas_kernel(%arg0: !ops.acc<f64, 2, 1>, %arg1: !ops.acc<f64, 2, 1>, %arg2: !ops.acc<f64, 2, 1>, %arg3: !ops.acc<f64, 2, 1>)
// IDEAL: arith.divf
// IDEAL: ops.read %arg0 [0, 0]
// IDEAL: ops.write %arg2 [0, 0]
// IDEAL: math.sqrt
// IDEAL: ops.write %arg3 [0, 0]
// IDEAL: ops.return
// The accessors that are only read are const in C++; the manifest says so.
// IDEAL-MANIFEST: "importer": "A/1.0"
// IDEAL-MANIFEST: "kernel": "ideal_gas_kernel"
// IDEAL-MANIFEST: "ndim": 2
// IDEAL-MANIFEST: "params": [
// IDEAL-MANIFEST: "const": true,
// IDEAL-MANIFEST: "elem": "f64",
// IDEAL-MANIFEST: "kind": "acc",
// IDEAL-MANIFEST: "ncomp": 1
// IDEAL-MANIFEST: "status": "ok"

// STENCIL-LABEL: ops.kernel @apply_stencil(%arg0: !ops.acc<f64, 2, 1>, %arg1: !ops.acc<f64, 2, 1>, %arg2: !ops.red<f64>)
// The float literal 0.25f is a float, converted to double like in C.
// STENCIL: arith.constant 2.500000e-01 : f32
// STENCIL: arith.extf
// STENCIL: ops.read %arg0 [1, 0]
// STENCIL: ops.read %arg0 [-1, 0]
// STENCIL: ops.read %arg0 [0, -1]
// STENCIL: ops.read %arg0 [0, 1]
// STENCIL: ops.write %arg1 [0, 0]
// The output is read back after it was written: the kernel body keeps the read (materialisation forwards it).
// STENCIL: ops.read %arg1 [0, 0]
// STENCIL: math.absf
// STENCIL: ops.reduce %arg2
// STENCIL-MANIFEST: "kind": "reduce"
// STENCIL-MANIFEST: "kind_hint": "max"
// STENCIL-MANIFEST: "style": "accumulate"
