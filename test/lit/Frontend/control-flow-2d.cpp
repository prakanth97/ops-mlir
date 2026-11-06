// Control flow and locals: if/else threading locals through scf.if, loops as scf.for with iter_args, local arrays,
// the ternary operator, short-circuit operators, MIN/MAX/SIGN, and constant loops that select components being unrolled.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel if_else --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s --check-prefix=IFELSE
// RUN: ops-kernel-import --kernel loop_sum --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s --check-prefix=LOOP
// RUN: ops-kernel-import --kernel ternary_minmax --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s --check-prefix=TERNARY
// RUN: ops-kernel-import --kernel local_array --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s --check-prefix=ARRAY
// RUN: ops-kernel-import --kernel short_circuit --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s --check-prefix=SHORT
// RUN: ops-kernel-import --kernel unrolled_components --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s --check-prefix=UNROLL
// RUN: FileCheck %s --check-prefix=UNROLL-MANIFEST --input-file=%t.json

#include <ops_lib_core.h>

extern int nsteps;

void if_else(const ACC<double> &a, ACC<double> &b) {
  double v = a(0, 0);
  double w = 1.0;
  if (v > 0.5) {
    w = v * 2.0;
    v = v - 1.0;
  } else if (v < -0.5) {
    w = -v;
  }
  b(0, 0) = v + w;
}

// IFELSE-LABEL: ops.kernel @if_else
// IFELSE: %[[A:.*]] = ops.read %arg0 [0, 0]
// IFELSE: %[[GT:.*]] = arith.cmpf ogt, %[[A]]
// The locals v and w leave the branch as results: both are assigned (v, w in the then branch; w in the else branch).
// IFELSE: scf.if %[[GT]] -> (f64, f64) {
// IFELSE: scf.yield %{{.*}}, %{{.*}} : f64, f64
// IFELSE: } else {
// IFELSE: arith.cmpf olt
// IFELSE: scf.if %{{.*}} -> (f64) {
// IFELSE: ops.write %arg1 [0, 0]

void loop_sum(const ACC<double> &a, ACC<double> &b, const double *scale) {
  double s = 0.0;
  for (int i = 0; i < nsteps; i++) {
    s += scale[0] * i;
    if (s > 100.0) s = 100.0;
  }
  b(0, 0) = a(0, 0) + s;
}

// LOOP-LABEL: ops.kernel @loop_sum
// A loop with a run-time bound is an scf.for that carries s.
// LOOP: ops.const_ref @nsteps : i32
// LOOP: arith.index_cast
// LOOP: %[[S:.*]] = scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (f64) {
// LOOP: ops.gbl_read %arg2[0]
// LOOP: scf.yield %{{.*}} : f64
// LOOP: ops.write %arg1 [0, 0]

void ternary_minmax(const ACC<double> &a, const ACC<double> &b, ACC<double> &c) {
  double lo = MIN(a(0, 0), b(0, 0));
  double hi = MAX(a(0, 0), b(0, 0));
  c(0, 0) = (lo < 0.0 ? -lo : lo) + SIGN(hi, a(1, 0));
}

// TERNARY-LABEL: ops.kernel @ternary_minmax
// Pure branches are evaluated and selected.
// TERNARY: arith.cmpf olt
// TERNARY: arith.select
// TERNARY: arith.cmpf ogt
// TERNARY: arith.select
// TERNARY: arith.select
// TERNARY: ops.write %arg2 [0, 0]

void local_array(const ACC<double> &a, ACC<double> &b) {
  double w[3] = {0.25, 0.5, 0.25};
  double s = 0.0;
  for (int k = 0; k < 3; k++) s += w[k] * a(k - 1, 0);
  w[1] = s;
  b(0, 0) = w[1] + w[0];
}

// ARRAY-LABEL: ops.kernel @local_array
// The elements of a local array are scalars; the loop over them selects offsets, so it is unrolled into constant reads.
// ARRAY: ops.read %arg0 [-1, 0]
// ARRAY: ops.read %arg0 [0, 0]
// ARRAY: ops.read %arg0 [1, 0]
// ARRAY-NOT: scf.for
// ARRAY: ops.write %arg1 [0, 0]

void short_circuit(const ACC<double> &a, const ACC<int> &flag, ACC<double> &b) {
  if (flag(0, 0) != 0 && a(0, 0) > 0.0 || a(1, 0) < 0.0) b(0, 0) = 1.0;
  else b(0, 0) = 0.0;
}

// SHORT-LABEL: ops.kernel @short_circuit
// SHORT: arith.andi
// SHORT: arith.ori
// SHORT: scf.if
// SHORT: ops.write %arg2 [0, 0]

void unrolled_components(const ACC<double> &q, ACC<double> &r) {
  for (int m = 0; m < 3; m++) r(m, 0, 0) = q(m, 0, 0) * 2.0 + m;
}

// UNROLL-LABEL: ops.kernel @unrolled_components
// The component is the loop variable: the loop is unrolled and the accessors get three components.
// UNROLL-SAME: (%arg0: !ops.acc<f64, 2, 3>, %arg1: !ops.acc<f64, 2, 3>)
// UNROLL-NOT: scf.for
// UNROLL: ops.read %arg0 [0, 0] {comp = 1 : i64}
// UNROLL: ops.write %arg1 [0, 0], %{{.*}} {comp = 2 : i64}
// UNROLL-MANIFEST: "ncomp": 3
