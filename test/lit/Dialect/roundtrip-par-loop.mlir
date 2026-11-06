// ops.par_loop: dats with access modes and stencils, scalars, reductions, the unmaterialised form.
// RUN: ops-opt %s --verify-roundtrip | FileCheck %s
// RUN: ops-opt %s | ops-opt | FileCheck %s

ops.kernel @k(%a: !ops.acc<f64, 2, 1>) { ops.return }

// CHECK-LABEL: ops.chain @plain
ops.chain @plain(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index, %hi1: index) {
  // CHECK: ops.par_loop @k range(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) dats(%{{.*}} : !ops.dat<f64, 2, 1> read #ops.stencil<{{\[}}[0, 0], [1, 0], [-1, 0]]>, %{{.*}} : !ops.dat<f64, 2, 1> write #ops.stencil<{{\[}}[0, 0]]>) scalars()
  ops.par_loop @k range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1, 0], [-1, 0]]>,
           %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// All access modes, a loop id and a block.
// CHECK-LABEL: ops.chain @modes
ops.chain @modes(%a: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  // CHECK: read #ops.stencil
  // CHECK-SAME: write #ops.stencil
  // CHECK-SAME: rw #ops.stencil
  // CHECK-SAME: inc #ops.stencil
  // CHECK-SAME: min #ops.stencil
  // CHECK-SAME: max #ops.stencil
  // CHECK: attributes {block = 2 : i64, loop_id = 7 : i64}
  ops.par_loop range(%lo, %hi)
      dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>, %a : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>,
           %a : !ops.dat<f64, 1, 1> rw #ops.stencil<[[0]]>, %a : !ops.dat<f64, 1, 1> inc #ops.stencil<[[0]]>,
           %a : !ops.dat<f64, 1, 1> min #ops.stencil<[[0]]>, %a : !ops.dat<f64, 1, 1> max #ops.stencil<[[0]]>)
      scalars() attributes {loop_id = 7 : i64, block = 2 : i64} {
  ^bb0(%a0: !ops.acc<f64, 1, 1>, %a1: !ops.acc<f64, 1, 1>, %a2: !ops.acc<f64, 1, 1>, %a3: !ops.acc<f64, 1, 1>,
       %a4: !ops.acc<f64, 1, 1>, %a5: !ops.acc<f64, 1, 1>):
  }
  ops.return
}

// Scalars and a reduction, 3D, strides, multi-component.
// CHECK-LABEL: ops.chain @scalars_and_reduction
ops.chain @scalars_and_reduction(%w: !ops.dat<f32, 3, 2>, %lo0: index, %hi0: index, %lo1: index, %hi1: index,
                                 %lo2: index, %hi2: index, %dt: f64, %n: i32) -> f64 {
  // CHECK: scalars(%{{.*}}, %{{.*}} : f64, i32)
  // CHECK: attributes {reduction_kinds = [#ops.red_kind<min>]} -> f64
  %m = ops.par_loop range(%lo0, %hi0, %lo1, %hi1, %lo2, %hi2)
      dats(%w : !ops.dat<f32, 3, 2> read #ops.stencil<[[0, 0, 0]]>)
      scalars(%dt, %n : f64, i32)
      attributes {reduction_kinds = [#ops.red_kind<min>]} -> f64 {
  ^bb0(%a: !ops.acc<f32, 3, 2>, %s: f64, %k: i32, %r: !ops.red<f64>):
    %v = ops.read %a[0, 0, 0] {comp = 1 : i64} : !ops.acc<f32, 3, 2> -> f32
    %vd = arith.extf %v : f32 to f64
    ops.reduce %r, %vd : !ops.red<f64>
  }
  ops.return %m : f64
}

// A low-dimensional dat: stride 0 along y.
// CHECK-LABEL: ops.chain @strided
ops.chain @strided(%xx: !ops.dat<i32, 2, 1>, %lo0: index, %hi0: index, %lo1: index, %hi1: index) {
  // CHECK: #ops.stencil<{{\[}}[0, 0], [1, 0]], strides = [1, 0]>
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%xx : !ops.dat<i32, 2, 1> read #ops.stencil<[[0, 0], [1, 0]], strides = [1, 0]>) scalars() {
  ^bb0(%a: !ops.acc<i32, 2, 1>):
  }
  ops.return
}

// A loop whose body reads and writes at stencil points (what ops-materialize-kernels produces).
// CHECK-LABEL: ops.chain @materialised
ops.chain @materialised(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index,
                        %lo1: index, %hi1: index) {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1, 0]]>,
           %b : !ops.dat<f64, 2, 1> rw #ops.stencil<[[0, 0], [-1, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %x[1, 0] : !ops.acc<f64, 2, 1> -> f64
    %w = ops.read %y[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %s = arith.addf %v, %w : f64
    ops.write %y[-1, 0], %s : !ops.acc<f64, 2, 1>
  }
  ops.return
}
