// Verifiers of the operations of kernel bodies.
// RUN: ops-opt %s -split-input-file -verify-diagnostics

ops.kernel @read_arity(%a: !ops.acc<f64, 2, 1>) {
  // expected-error @+1 {{expects 2 offsets, got 1}}
  %v = ops.read %a[0] : !ops.acc<f64, 2, 1> -> f64
  ops.return
}

// -----

ops.kernel @write_arity(%a: !ops.acc<f64, 2, 1>) {
  %c = arith.constant 1.0 : f64
  // expected-error @+1 {{expects 2 offsets, got 3}}
  ops.write %a[0, 0, 0], %c : !ops.acc<f64, 2, 1>
  ops.return
}

// -----

ops.kernel @component_range(%a: !ops.acc<f64, 2, 2>) {
  // expected-error @+1 {{component 2 is out of range for an accessor with 2 components}}
  %v = ops.read %a[0, 0] {comp = 2 : i64} : !ops.acc<f64, 2, 2> -> f64
  ops.return
}

// -----

ops.kernel @dyn_arity(%a: !ops.acc<f64, 2, 1>) {
  %i = ops.index 0
  // expected-error @+1 {{expects 2 offsets, got 1}}
  %v = ops.read_dyn %a[%i : i32] : !ops.acc<f64, 2, 1> -> f64
  ops.return
}

// -----

ops.kernel @gbl_range(%g: !ops.gbl<f64, 2>) {
  // expected-error @+1 {{index 2 is out of range for a global of size 2}}
  %v = ops.gbl_read %g[2] : !ops.gbl<f64, 2> -> f64
  ops.return
}

// -----

ops.kernel @index_dim(%a: !ops.acc<f64, 2, 1>) {
  // expected-error @+1 {{dimension must be between 0 and 2}}
  %i = ops.index 3
  ops.return
}

// -----

ops.chain @index_beyond_rank(%a: !ops.dat<f64, 2, 1>, %lo: index, %hi: index) {
  ops.par_loop range(%lo, %hi, %lo, %hi) dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
    // expected-error @+1 {{dimension 2 does not exist in a loop of rank 2}}
    %i = ops.index 2
  }
  ops.return
}

// -----

ops.kernel @const_negative_offset() {
  // expected-error @+1 {{offset must not be negative}}
  %v = ops.const_ref @c {offset = -4 : i64} : f64
  ops.return
}

// -----

ops.kernel @const_stride() {
  %i = arith.constant 0 : i32
  // expected-error @+1 {{an indexed constant needs a positive array_stride}}
  %v = ops.const_ref @c[%i : i32] : f64
  ops.return
}
