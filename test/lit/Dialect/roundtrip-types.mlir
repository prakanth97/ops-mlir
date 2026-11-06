// Types of the ops dialect: every element type, ranks 1-3, components, globals and reductions.
// RUN: ops-opt %s --verify-roundtrip | FileCheck %s
// RUN: ops-opt %s | ops-opt | FileCheck %s

// CHECK-LABEL: ops.chain @dats
// CHECK-SAME: %{{.*}}: !ops.dat<f16, 1, 2>, %{{.*}}: !ops.dat<f32, 2, 1>, %{{.*}}: !ops.dat<f64, 3, 5>
// CHECK-SAME: %{{.*}}: !ops.dat<i32, 1, 1>, %{{.*}}: !ops.dat<i64, 2, 1>
ops.chain @dats(%a: !ops.dat<f16, 1, 2>, %b: !ops.dat<f32, 2, 1>, %c: !ops.dat<f64, 3, 5>,
                %d: !ops.dat<i32, 1, 1>, %e: !ops.dat<i64, 2, 1>) {
  ops.return
}

// CHECK-LABEL: ops.kernel @params
// CHECK-SAME: !ops.gbl<f64, 4>, %{{.*}}: !ops.red<i32>, %{{.*}}: !ops.acc<f32, 3, 1>
ops.kernel @params(%g: !ops.gbl<f64, 4>, %r: !ops.red<i32>, %a: !ops.acc<f32, 3, 1>) {
  ops.return
}

// CHECK-LABEL: ops.chain @scalars
// CHECK-SAME: index, %{{.*}}: f64, %{{.*}}: i32
ops.chain @scalars(%lo: index, %dt: f64, %n: i32) {
  ops.return
}

// Results of a chain are reduction values.
// CHECK-LABEL: ops.chain @results
// CHECK-SAME: -> (f64, i32)
ops.chain @results(%x: f64, %y: i32) -> (f64, i32) {
  ops.return %x, %y : f64, i32
}
