// A pointer-style kernel of TeaLeaf (OPS_ACC0 / OPS_ACC1), also with the parameter kinds given by the caller (what the
// translator does from the call sites).
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel set_field_kernel --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json
// RUN: ops-kernel-import --kernel set_field_kernel --kinds acc,acc --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s
// RUN: not ops-kernel-import --strict --kernel set_field_kernel --kinds gbl,acc --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=WRONG

#include <ops_lib_core.h>

#include "Inputs/tealeaf_set_field_kernel.h"

// CHECK-LABEL: ops.kernel @set_field_kernel(%arg0: !ops.acc<f64, 2, 1>, %arg1: !ops.acc<f64, 2, 1>)
// CHECK: %[[V:.*]] = ops.read %arg0 [0, 0]
// CHECK: ops.write %arg1 [0, 0], %[[V]]
// MANIFEST: "const": true,
// MANIFEST: "legacy_macro": true,
// MANIFEST: "const": false,
// MANIFEST: "legacy_macro": true,
// A caller that says the first parameter is a global is told it is not (the kernel indexes it with OPS_ACC0).
// WRONG: set_field_kernel: not imported: {{.*}}not a compile-time constant
