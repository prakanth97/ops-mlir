// Verifiers of ops.par_loop and of accesses inside its body.
// RUN: ops-opt %s -split-input-file -verify-diagnostics

ops.chain @odd_range(%a: !ops.dat<f64, 2, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{range must hold lower and upper bound of each dimension, got 3 values}}
  ops.par_loop range(%lo, %hi, %lo) dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

ops.chain @rank_mismatch(%a: !ops.dat<f64, 3, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{dat operand #0 has rank 3, the loop has rank 2}}
  ops.par_loop range(%lo, %hi, %lo, %hi) dats(%a : !ops.dat<f64, 3, 1> read #ops.stencil<[[0, 0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 3, 1>):
  }
  ops.return
}

// -----

ops.chain @stencil_rank(%a: !ops.dat<f64, 2, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{stencil of dat operand #0 has rank 1, the loop has rank 2}}
  ops.par_loop range(%lo, %hi, %lo, %hi) dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

ops.chain @reduction_kinds(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{expects one entry of reduction_kinds per result, got 0 for 1 results}}
  %m = ops.par_loop range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars() -> f64 {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %r: !ops.red<f64>):
  }
  ops.return %m : f64
}

// -----

ops.chain @body_arity(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{body must have 1 arguments (1 accessors, 0 scalars, 0 reductions), got 2}}
  ops.par_loop range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %y: !ops.acc<f64, 1, 1>):
  }
  ops.return
}

// -----

ops.chain @body_acc_type(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{body argument #0 must be '!ops.acc<f64, 1, 1>' to match dat operand #0}}
  ops.par_loop range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f32, 1, 1>):
  }
  ops.return
}

// -----

ops.chain @body_scalar_type(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index, %dt: f64) {
  // expected-error @+1 {{body argument #1 must have the type of scalar operand #0}}
  ops.par_loop range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars(%dt : f64) {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %s: f32):
  }
  ops.return
}

// -----

ops.chain @body_red_type(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{body argument #1 must be '!ops.red<f64>' to match result #0}}
  %m = ops.par_loop range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars()
      attributes {reduction_kinds = [#ops.red_kind<add>]} -> f64 {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %r: !ops.red<f32>):
  }
  ops.return %m : f64
}

// -----

ops.chain @write_read_only(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  ops.par_loop range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
    %c = arith.constant 1.0 : f64
    // expected-error @+1 {{writes dat operand #0 of the enclosing ops.par_loop, which is only read}}
    ops.write %x[0], %c : !ops.acc<f64, 1, 1>
  }
  ops.return
}

// -----

ops.chain @offset_outside_stencil(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  ops.par_loop range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0], [1]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
    // expected-error @+1 {{offset [-1] is not a point of the stencil of dat operand #0}}
    %v = ops.read %x[-1] : !ops.acc<f64, 1, 1> -> f64
  }
  ops.return
}

// -----

ops.chain @write_offset_outside_stencil(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  ops.par_loop range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> rw #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
    %c = arith.constant 1.0 : f64
    // expected-error @+1 {{offset [-1] is not a point of the stencil of dat operand #0}}
    ops.write %x[-1], %c : !ops.acc<f64, 1, 1>
  }
  ops.return
}
