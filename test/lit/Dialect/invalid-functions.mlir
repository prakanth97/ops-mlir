// Verifiers of ops.kernel and ops.chain.
// RUN: ops-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{parameter #0 must be !ops.acc, !ops.gbl or !ops.red, got 'f64'}}
ops.kernel @scalar_param(%x: f64) { ops.return }

// -----

// expected-error @+1 {{must not have results}}
ops.kernel @has_result(%a: !ops.acc<f64, 1, 1>) -> f64 {
  %v = ops.read %a[0] : !ops.acc<f64, 1, 1> -> f64
  ops.return %v : f64
}

// -----

ops.kernel @return_value(%a: !ops.acc<f64, 1, 1>) {
  %v = ops.read %a[0] : !ops.acc<f64, 1, 1> -> f64
  // expected-error @+1 {{'ops.return' op of a kernel must not have operands}}
  ops.return %v : f64
}

// -----

// expected-error @+1 {{argument #0 must be a dat or a scalar, got '!ops.acc<f64, 1, 1>'}}
ops.chain @acc_arg(%a: !ops.acc<f64, 1, 1>) { ops.return }

// -----

// expected-error @+1 {{result #0 must be a reduction value (f16, f32, f64, i32, i64), got 'index'}}
ops.chain @index_result(%x: index) -> index { ops.return %x : index }

// -----

ops.chain @return_mismatch(%x: f64) -> f64 {
  %y = arith.constant 1 : i32
  // expected-error @+1 {{operand types do not match the results of the chain}}
  ops.return %y : i32
}

// -----

// ops.return must be inside a kernel or chain.
func.func @wrong_parent() {
  // expected-error @+1 {{expects parent op to be one of 'ops.kernel, ops.chain'}}
  ops.return
}
