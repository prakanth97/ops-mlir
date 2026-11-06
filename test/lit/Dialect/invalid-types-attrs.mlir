// Verifiers of types and attributes.
// RUN: ops-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{element type must be f16, f32, f64, i32 or i64, got 'i8'}}
ops.chain @bad_element(%a: !ops.dat<i8, 2, 1>) { ops.return }

// -----

// expected-error @+1 {{rank must be between 1 and 3, got 4}}
ops.chain @bad_rank(%a: !ops.dat<f64, 4, 1>) { ops.return }

// -----

// expected-error @+1 {{rank must be between 1 and 3, got 0}}
ops.chain @zero_rank(%a: !ops.dat<f64, 0, 1>) { ops.return }

// -----

// expected-error @+1 {{number of components must be at least 1, got 0}}
ops.chain @no_components(%a: !ops.dat<f64, 2, 0>) { ops.return }

// -----

// expected-error @+1 {{size must be at least 1, got 0}}
ops.kernel @empty_gbl(%g: !ops.gbl<f64, 0>) { ops.return }

// -----

// expected-error @+1 {{element type must be f16, f32, f64, i32 or i64, got 'index'}}
ops.kernel @bad_red(%r: !ops.red<index>) { ops.return }

// -----

ops.chain @bad_stencil_points(%a: !ops.dat<f64, 2, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{all stencil points must have the same number of offsets}}
  ops.par_loop range(%lo, %hi, %lo, %hi) dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

ops.chain @empty_stencil(%a: !ops.dat<f64, 2, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{a stencil needs at least one point}}
  ops.par_loop range(%lo, %hi, %lo, %hi) dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

ops.chain @bad_strides(%a: !ops.dat<f64, 2, 1>, %lo: index, %hi: index) {
  // expected-error @+2 {{strides must be 0 or 1, got 2}}
  ops.par_loop range(%lo, %hi, %lo, %hi)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]], strides = [1, 2]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

ops.chain @bad_stride_count(%a: !ops.dat<f64, 2, 1>, %lo: index, %hi: index) {
  // expected-error @+2 {{expected 2 strides, got 1}}
  ops.par_loop range(%lo, %hi, %lo, %hi)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]], strides = [1]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

ops.chain @unknown_access(%a: !ops.dat<f64, 2, 1>, %lo: index, %hi: index) {
  // expected-error @+1 {{unknown access mode 'append'}}
  ops.par_loop range(%lo, %hi, %lo, %hi) dats(%a : !ops.dat<f64, 2, 1> append #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}
