// ops-fuse-point-local, ops-forward-stores: loops of a chain fused into one kernel.
// RUN: ops-opt %s -ops-fuse-point-local | FileCheck %s
// RUN: ops-opt %s -ops-fuse-point-local -ops-forward-stores | FileCheck %s --check-prefix=FWD
// RUN: ops-opt %s -ops-fuse-point-local=max-loops=2 | FileCheck %s --check-prefix=CAP
// RUN: ops-opt %s -ops-fuse-point-local=reorder=false | FileCheck %s --check-prefix=NOREORDER
// RUN: ops-opt %s -ops-fuse-point-local | ops-opt | FileCheck %s

// A point-wise pipeline: b = 2a; c = b + 1; d = c * c. One kernel.
// CHECK-LABEL: ops.chain @pipeline
// CHECK-SAME: ops.range_values = [array<i64: 0, 8, 0, 8>]
// CHECK: ops.par_loop
// CHECK-SAME: %arg0 : !ops.dat<f64, 2, 1> read
// CHECK-SAME: %arg1 : !ops.dat<f64, 2, 1> rw
// CHECK-SAME: %arg2 : !ops.dat<f64, 2, 1> rw
// CHECK-SAME: %arg3 : !ops.dat<f64, 2, 1> write
// CHECK-SAME: ops.fused_loops = [-1, -1, -1]
// CHECK-NOT: ops.par_loop
//
// With store forwarding the intermediates are only stored: 1 load (a), 3 stores (b, c, d).
// FWD-LABEL: ops.chain @pipeline
// FWD: ops.par_loop
// FWD-SAME: %arg0 : !ops.dat<f64, 2, 1> read
// FWD-SAME: %arg1 : !ops.dat<f64, 2, 1> write
// FWD-SAME: %arg2 : !ops.dat<f64, 2, 1> write
// FWD-SAME: %arg3 : !ops.dat<f64, 2, 1> write
// FWD: ^bb0([[A:%[a-z0-9]+]]: !ops.acc<f64, 2, 1>, [[B:%[a-z0-9]+]]: !ops.acc<f64, 2, 1>, [[C:%[a-z0-9]+]]: !ops.acc<f64, 2, 1>, [[D:%[a-z0-9]+]]: !ops.acc<f64, 2, 1>):
// FWD: ops.read [[A]] [0, 0]
// FWD-NOT: ops.read
// FWD: ops.write [[D]] [0, 0]
// FWD-NOT: ops.par_loop
// FWD-LABEL: ops.chain @jacobi
//
// With a cap of 2 loops per kernel: 2 + 1.
// CAP-LABEL: ops.chain @pipeline
// CAP-SAME: ops.range_values = [array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>]
// CAP: ops.fused_loops = [-1, -1]
// CAP: ops.par_loop
// CAP-NOT: ops.fused_loops
// CAP-LABEL: ops.chain @jacobi
ops.chain @pipeline(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %c: !ops.dat<f64, 2, 1>, %d: !ops.dat<f64, 2, 1>,
                    %lo0: index, %hi0: index, %lo1: index, %hi1: index,
                    %lo2: index, %hi2: index, %lo3: index, %hi3: index,
                    %lo4: index, %hi4: index, %lo5: index, %hi5: index)
    attributes {"ops.range_values" = [array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %two = arith.constant 2.0 : f64
    %w = arith.mulf %v, %two : f64
    ops.write %y[0, 0], %w : !ops.acc<f64, 2, 1>
  }
  ops.par_loop range(%lo2, %hi2, %lo3, %hi3)
      dats(%b : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %c : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %one = arith.constant 1.0 : f64
    %w = arith.addf %v, %one : f64
    ops.write %y[0, 0], %w : !ops.acc<f64, 2, 1>
  }
  ops.par_loop range(%lo4, %hi4, %lo5, %hi5)
      dats(%c : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %d : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %w = arith.mulf %v, %v : f64
    ops.write %y[0, 0], %w : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// Jacobi: lap(a) -> anew, anew -> a. The second loop overwrites what the first reads around each point: two kernels.
// CHECK-LABEL: ops.chain @jacobi
// CHECK: ops.par_loop
// CHECK-NOT: ops.fused_loops
// CHECK: ops.par_loop
// CHECK-NOT: ops.fused_loops
ops.chain @jacobi(%a: !ops.dat<f64, 2, 1>, %anew: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index,
                  %hi1: index, %lo2: index, %hi2: index, %lo3: index, %hi3: index)
    attributes {"ops.range_values" = [array<i64: 1, 7, 1, 7>, array<i64: 1, 7, 1, 7>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1, 0], [-1, 0]]>,
           %anew : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %l = ops.read %x[-1, 0] : !ops.acc<f64, 2, 1> -> f64
    %r = ops.read %x[1, 0] : !ops.acc<f64, 2, 1> -> f64
    %s = arith.addf %l, %r : f64
    ops.write %y[0, 0], %s : !ops.acc<f64, 2, 1>
  }
  ops.par_loop range(%lo2, %hi2, %lo3, %hi3)
      dats(%anew : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %a : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.write %y[0, 0], %v : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// Reordering: two independent stencil chains, interleaved. Side by side they need 2 kernels instead of 3.
// L0: b = a   L1: c = b(5pt)   L2: e = d   L3: f = e(5pt)  ->  {L0, L2} then {L1, L3}
// CHECK-LABEL: ops.chain @reorder
// CHECK-SAME: ops.range_values = [array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>]
// CHECK: ops.par_loop
// CHECK-SAME: ops.fused_loops = [10, 12]
// CHECK: ops.par_loop
// CHECK-SAME: ops.fused_loops = [11, 13]
// CHECK-NOT: ops.par_loop
// NOREORDER-LABEL: ops.chain @reorder
// NOREORDER-SAME: ops.range_values = [array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>]
// NOREORDER: ops.fused_loops = [11, 12]
ops.chain @reorder(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %c: !ops.dat<f64, 2, 1>,
                   %d: !ops.dat<f64, 2, 1>, %e: !ops.dat<f64, 2, 1>, %f: !ops.dat<f64, 2, 1>,
                   %lo0: index, %hi0: index, %lo1: index, %hi1: index, %lo2: index, %hi2: index, %lo3: index,
                   %hi3: index, %lo4: index, %hi4: index, %lo5: index, %hi5: index, %lo6: index, %hi6: index,
                   %lo7: index, %hi7: index)
    attributes {"ops.range_values" = [array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>,
                                      array<i64: 0, 8, 0, 8>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() attributes {loop_id = 10 : i64} {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.write %y[0, 0], %v : !ops.acc<f64, 2, 1>
  }
  ops.par_loop range(%lo2, %hi2, %lo3, %hi3)
      dats(%b : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1, 0], [-1, 0]]>,
           %c : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() attributes {loop_id = 11 : i64} {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %l = ops.read %x[-1, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.write %y[0, 0], %l : !ops.acc<f64, 2, 1>
  }
  ops.par_loop range(%lo4, %hi4, %lo5, %hi5)
      dats(%d : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %e : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() attributes {loop_id = 12 : i64} {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.write %y[0, 0], %v : !ops.acc<f64, 2, 1>
  }
  ops.par_loop range(%lo6, %hi6, %lo7, %hi7)
      dats(%e : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [0, 1], [0, -1]]>,
           %f : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() attributes {loop_id = 13 : i64} {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %l = ops.read %x[0, 1] : !ops.acc<f64, 2, 1> -> f64
    ops.write %y[0, 0], %l : !ops.acc<f64, 2, 1>
  }
  ops.return
}
