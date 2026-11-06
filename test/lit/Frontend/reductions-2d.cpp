// Reductions: accumulations (field_summary of CloverLeaf: five += into reduction arguments), a minimum written with the
// MIN macro (calc_dt), a maximum written with fmax (Laplace), and the spellings the importer recognises: a test followed
// by an assignment, and a plain assignment (recorded as the "assign" style, valid for loops of a single point only).
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel field_summary_kernel --source %s --ndim 2 -o - --manifest %t.fs.json -- -I%ops_include | FileCheck %s --check-prefix=SUMMARY
// RUN: FileCheck %s --check-prefix=SUMMARY-MANIFEST --input-file=%t.fs.json
// RUN: ops-kernel-import --kernel calc_dt_kernel_min --source %s --ndim 2 -o - --manifest %t.min.json -- -I%ops_include | FileCheck %s --check-prefix=MIN
// RUN: FileCheck %s --check-prefix=MIN-MANIFEST --input-file=%t.min.json
// RUN: ops-kernel-import --kernel test_then_assign --source %s --ndim 2 -o - --manifest %t.t.json -- -I%ops_include | FileCheck %s --check-prefix=TEST
// RUN: FileCheck %s --check-prefix=TEST-MANIFEST --input-file=%t.t.json
// RUN: ops-kernel-import --kernel assign_style --source %s --ndim 2 -o - --manifest %t.a.json -- -I%ops_include | FileCheck %s --check-prefix=ASSIGN
// RUN: FileCheck %s --check-prefix=ASSIGN-MANIFEST --input-file=%t.a.json
// RUN: ops-kernel-import --kernel mixed_kinds --source %s --ndim 2 -o - --manifest %t.m.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=MIXED

#include <ops_lib_core.h>

#include "Inputs/cloverleaf_field_summary_kernel.h"
#include "Inputs/cloverleaf_calc_dt_kernel.h"

void test_then_assign(const ACC<double> &a, double *big) {
  if (a(0, 0) > *big) *big = a(0, 0);
}

void assign_style(const ACC<double> &a, double *last) { *last = a(0, 0); }

void mixed_kinds(const ACC<double> &a, double *r) {
  *r += a(0, 0);
  *r = MIN(*r, a(1, 0));
}

// Five accumulations: `*vol = *vol + cell_vol` is an accumulation spelled out.
// SUMMARY-LABEL: ops.kernel @field_summary_kernel(
// SUMMARY-SAME: %arg6: !ops.red<f64>, %arg7: !ops.red<f64>, %arg8: !ops.red<f64>, %arg9: !ops.red<f64>, %arg10: !ops.red<f64>)
// SUMMARY: ops.reduce %arg6,
// SUMMARY: ops.reduce %arg7,
// SUMMARY: ops.reduce %arg8,
// SUMMARY: ops.reduce %arg9,
// SUMMARY: ops.reduce %arg10,
// SUMMARY-MANIFEST: "kind": "reduce",
// SUMMARY-MANIFEST: "kind_hint": "add",
// SUMMARY-MANIFEST: "style": "accumulate"

// *dt_min_val = MIN(*dt_min_val, dt_min(0,0)) contributes dt_min(0,0) to a minimum.
// MIN-LABEL: ops.kernel @calc_dt_kernel_min(%arg0: !ops.acc<f64, 2, 1>, %arg1: !ops.red<f64>)
// MIN: %[[V:.*]] = ops.read %arg0 [0, 0]
// MIN-NEXT: ops.reduce %arg1, %[[V]]
// MIN-MANIFEST: "kind": "reduce",
// MIN-MANIFEST: "kind_hint": "min",

// TEST-LABEL: ops.kernel @test_then_assign
// TEST: ops.reduce %arg1,
// TEST-MANIFEST: "kind_hint": "max",
// TEST-MANIFEST: "style": "accumulate"

// A plain assignment replaces the value; the manifest says so, the loop must cover a single point to use it.
// ASSIGN-LABEL: ops.kernel @assign_style
// ASSIGN: ops.reduce %arg1,
// ASSIGN-MANIFEST: "kind_hint": "none",
// ASSIGN-MANIFEST: "style": "assign"

// MIXED: mixed_kinds: not imported: {{.*}}a reduction that is updated as both add and min
