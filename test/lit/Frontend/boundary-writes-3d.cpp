// The symmetry boundary kernel of TGV: it writes the halo (offset -1, -2) from the interior (offset 1, 2) of RW
// accessors. Writes at non-zero offsets are legal and recorded in the manifest.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel opensbliblock00Kernel081 --source %s --ndim 3 -o - --manifest %t.json -- -I%ops_include | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json

#include <ops_lib_core.h>

extern double Delta0block0, Delta1block0, Delta2block0, Minf, gama;

#include "Inputs/tgv_dp_kernels.h"

// CHECK-LABEL: ops.kernel @opensbliblock00Kernel081(
// CHECK: %[[R:.*]] = ops.read %arg1 [1, 0, 0]
// CHECK-NEXT: ops.write %arg1 [-1, 0, 0], %[[R]]
// The momentum changes sign across the wall.
// CHECK: arith.negf
// CHECK: ops.write %arg2 [-1, 0, 0]
// CHECK: ops.read %arg1 [2, 0, 0]
// CHECK-NEXT: ops.write %arg1 [-2, 0, 0]
// CHECK: ops.return
// MANIFEST: "traits": {
// MANIFEST: "reads": {
// MANIFEST: "0": [
// MANIFEST: 1,
// MANIFEST: 0,
// MANIFEST: 0
// MANIFEST: "writes": {
// MANIFEST: "0": [
// MANIFEST: -2,
// MANIFEST: 0,
// MANIFEST: 0
// MANIFEST: -1,
