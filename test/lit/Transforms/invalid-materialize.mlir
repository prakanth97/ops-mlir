// ops-materialize-kernels reports what does not fit.
// RUN: ops-opt %s -split-input-file -ops-materialize-kernels -verify-diagnostics

ops.kernel @reads_too_far(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>) {
  // expected-error @+1 {{offset [1, 0] is not a point of the stencil}}
  %v = ops.read %x[1, 0] : !ops.acc<f64, 2, 1> -> f64
  ops.write %y[0, 0], %v : !ops.acc<f64, 2, 1>
  ops.return
}

ops.chain @c1(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index, %hi1: index) {
  ops.par_loop @reads_too_far range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

ops.kernel @one(%x: !ops.acc<f64, 2, 1>) { ops.return }

ops.chain @c2(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index, %hi1: index) {
  // expected-error @+1 {{kernel @one has 1 accessor parameters but the loop has 2 dats}}
  ops.par_loop @one range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

ops.chain @c3(%a: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index, %hi1: index) {
  // expected-error @+1 {{no ops.kernel named @missing}}
  ops.par_loop @missing range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

ops.kernel @needs_two(%x: !ops.acc<f64, 2, 1>, %g: !ops.gbl<f64, 2>) { ops.return }

ops.chain @c4(%a: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index, %hi1: index, %s: f64) {
  // expected-error @+1 {{kernel @needs_two takes 2 global components but the loop passes 1 scalars}}
  ops.par_loop @needs_two range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars(%s : f64) {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %t: f64):
  }
  ops.return
}

// -----

ops.kernel @wrong_type(%x: !ops.acc<f32, 2, 1>) { ops.return }

ops.chain @c5(%a: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index, %hi1: index) {
  // expected-error @+1 {{parameter 0 has type '!ops.acc<f32, 2, 1>' but the dat operand has accessor type '!ops.acc<f64, 2, 1>'}}
  ops.par_loop @wrong_type range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// -----

// A kernel that reads a point of a write-only operand that it has not written.
ops.kernel @reads_output(%x: !ops.acc<f64, 1, 1>) {
  // expected-error @+1 {{kernel @reads_output reads the write-only dat operand #0 before it has written that point}}
  %v = ops.read %x[0] : !ops.acc<f64, 1, 1> -> f64
  ops.write %x[0], %v : !ops.acc<f64, 1, 1>
  ops.return
}

ops.chain @c6(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  ops.par_loop @reads_output range(%lo, %hi) dats(%a : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
  }
  ops.return
}
