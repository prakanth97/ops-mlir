// ops-jit-rocm: a group with a reduction to an AMDGPU kernel (assembly) and its HIP launch.
// REQUIRES: rocm
// RUN: ops-opt %s -ops-jit-rocm="chip=gfx90a format=isa rocm=%rocm_path" | FileCheck %s
// RUN: ops-opt %s -ops-jit-rocm="chip=gfx1100 wave64=false format=isa rocm=%rocm_path" | FileCheck %s --check-prefix=WAVE32

// CHECK: llvm.func @sum_max(
// CHECK: gpu.launch_func @sum_max_kernel::@sum_max_kernel
// CHECK: gpu.binary @sum_max_kernel
// CHECK-SAME: #rocdl.target<O = 3, chip = "gfx90a">
// CHECK-SAME: ds_bpermute_b32
// CHECK-SAME: .amdhsa_kernel sum_max_kernel
// CHECK-SAME: .wavefront_size: 64
// WAVE32: .wavefront_size: 32
ops.chain @sum_max(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %w: !ops.dat<f64, 2, 1>, %o: !ops.dat<f64, 2, 1>) -> f64 {
  %s = ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%w : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %o : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars()
      attributes {reduction_kinds = [#ops.red_kind<max>]} -> f64 {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>, %acc: !ops.red<f64>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %a = math.absf %v : f64
    ops.write %y[0, 0], %a : !ops.acc<f64, 2, 1>
    ops.reduce %acc, %v : !ops.red<f64>
  }
  ops.return %s : f64
}
