// ops-specialize-consts: what cannot be compiled is rejected.
// RUN: ops-opt %s -split-input-file -ops-materialize-kernels -ops-specialize-consts -verify-diagnostics

// A table indexed dynamically whose constant is not folded.
ops.kernel @table(%p: !ops.acc<f64, 1, 1>) {
  %i = ops.index 0
  // expected-error @+1 {{ops.const_ref @table has a dynamic index that cannot be resolved}}
  %e = ops.const_ref @table[%i : i32] {array_stride = 8 : i64} : f64
  ops.write %p[0], %e : !ops.acc<f64, 1, 1>
  ops.return
}

ops.chain @c1(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  ops.par_loop @table range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
  }
  ops.return
}

// -----

// Folded, but the index is a grid coordinate: no loop to unroll.
ops.kernel @coord_table(%p: !ops.acc<f64, 1, 1>) {
  %i = ops.index 0
  // expected-error @+1 {{ops.const_ref @table has a dynamic index that cannot be resolved}}
  %e = ops.const_ref @table[%i : i32] {array_stride = 8 : i64} : f64
  ops.write %p[0], %e : !ops.acc<f64, 1, 1>
  ops.return
}

ops.chain @c2(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index)
    attributes {ops.const_roles = {table = "bound"}, ops.const_values = {table = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>}} {
  ops.par_loop @coord_table range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
  }
  ops.return
}

// -----

// The loop is too long to unroll.
ops.kernel @long_loop(%p: !ops.acc<f64, 1, 1>) {
  %lo = arith.constant 0 : index
  %hi = arith.constant 1000 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %sum = scf.for %i = %lo to %hi step %c1 iter_args(%acc = %zero) -> (f64) {
    %ii = arith.index_cast %i : index to i32
    // expected-error @+1 {{ops.const_ref @table has a dynamic index that cannot be resolved}}
    %e = ops.const_ref @table[%ii : i32] {array_stride = 8 : i64} : f64
    %next = arith.addf %acc, %e : f64
    scf.yield %next : f64
  }
  ops.write %p[0], %sum : !ops.acc<f64, 1, 1>
  ops.return
}

ops.chain @c3(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index)
    attributes {ops.const_roles = {table = "bound"}, ops.const_values = {table = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>}} {
  ops.par_loop @long_loop range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
  }
  ops.return
}

// -----

// A bound constant without a value.
ops.kernel @no_value(%p: !ops.acc<f64, 1, 1>) {
  %dt = ops.const_ref @dt : f64
  ops.write %p[0], %dt : !ops.acc<f64, 1, 1>
  ops.return
}

// expected-error @+1 {{no value for constant @dt in ops.const_values}}
ops.chain @c4(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) attributes {ops.const_roles = {dt = "bound"}} {
  ops.par_loop @no_value range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
  }
  ops.return
}

// -----

// The value is too short for the type.
ops.kernel @short_value(%p: !ops.acc<f64, 1, 1>) {
  %dt = ops.const_ref @dt : f64
  ops.write %p[0], %dt : !ops.acc<f64, 1, 1>
  ops.return
}

// expected-error @+1 {{constant @dt has 4 bytes: cannot read 'f64' at byte offset 0}}
ops.chain @c5(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index)
    attributes {ops.const_roles = {dt = "bound"}, ops.const_values = {dt = array<i8: 0, 0, 0, 0>}} {
  ops.par_loop @short_value range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
  }
  ops.return
}
