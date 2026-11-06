// ops.print runs as printf on the host (sequential and with OpenMP: the order of the lines of different points is not
// defined, so one point only).
// RUN: ops-opt %s -ops-jit-cpu="omp=false" \
// RUN:   | mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:   | FileCheck %s

ops.chain @talk_group(%lo0: index, %hi0: index, %u: !ops.dat<f64, 1, 1>) attributes {ops.func_name = "talk", ops.range_values = [array<i64: 2, 3>]} {
  ops.par_loop range(%lo0, %hi0) dats(%u : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars() {
  ^bb0(%a: !ops.acc<f64, 1, 1>):
    %v = ops.read %a[0] : !ops.acc<f64, 1, 1> -> f64
    %i = ops.index 0
    ops.print "point %d has value %.2f\0A", %i, %v : i32, f64
  }
  ops.return
}
func.func private @talk(memref<?xf64, strided<[1], offset: ?>>, index, index, index, index)
func.func @main() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %buf = memref.alloc() : memref<8xf64>
  %v = arith.constant 4.5 : f64
  scf.for %i = %c0 to %c3 step %c2 {
    memref.store %v, %buf[%i] : memref<8xf64>
  }
  memref.store %v, %buf[%c2] : memref<8xf64>
  %s = memref.cast %buf : memref<8xf64> to memref<?xf64, strided<[1], offset: ?>>
  // arguments: dat, shift, range lo/hi, displacement
  func.call @talk(%s, %c0, %c2, %c3, %c0) : (memref<?xf64, strided<[1], offset: ?>>, index, index, index, index) -> ()
  return
}
// CHECK: point 2 has value 4.50
