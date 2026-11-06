// Dependence analysis of chains: footprints, RAW/WAR/WAW edges and whether they allow fusion.
// RUN: ops-opt %s -ops-print-chain-dependence -o /dev/null 2>&1 | FileCheck %s
// RUN: ops-opt %s -ops-print-fusion-plan -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN

// Jacobi: anew = lap(a); a = anew. The second loop overwrites what the first reads around each point.
// CHECK-LABEL: chain @jacobi
// CHECK: loop 0: block 0 range [1, 9, 1, 9]
// CHECK-NEXT: d0: read nonlocal
// CHECK-NEXT: d1: write local
// CHECK: loop 1: block 0 range [1, 9, 1, 9]
// CHECK-NEXT: d0: write local
// CHECK-NEXT: d1: read local
// CHECK: edge 0 -> 1: raw war {d0, d1} blocking
// PLAN-LABEL: chain @jacobi
// PLAN: plan: 0|1
// PLAN: blocked-by dependence: 1
ops.chain @jacobi(%a: !ops.dat<f64, 2, 1>, %anew: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index,
                  %hi1: index, %lo2: index, %hi2: index, %lo3: index, %hi3: index)
    attributes {"ops.range_values" = [array<i64: 1, 9, 1, 9>, array<i64: 1, 9, 1, 9>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]>,
           %anew : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop range(%lo2, %hi2, %lo3, %hi3)
      dats(%anew : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>,
           %a : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// A point-wise pipeline: every edge stays within a point and fuses.
// CHECK-LABEL: chain @pointwise
// CHECK: edge 0 -> 1: raw {d1} fusable
// CHECK: edge 1 -> 2: raw {d2} fusable
// CHECK-NOT: blocking
// PLAN-LABEL: chain @pointwise
// PLAN: plan: 0,1,2
// PLAN: kernels: 1 of 3 loops
// PLAN: traffic: unfused 2880 fused 1920
ops.chain @pointwise(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %c: !ops.dat<f64, 2, 1>,
                     %d: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index, %hi1: index, %lo2: index,
                     %hi2: index, %lo3: index, %hi3: index, %lo4: index, %hi4: index, %lo5: index, %hi5: index)
    attributes {"ops.range_values" = [array<i64: 0, 10, 0, 6>, array<i64: 0, 10, 0, 6>, array<i64: 0, 10, 0, 6>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop range(%lo2, %hi2, %lo3, %hi3)
      dats(%b : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %c : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop range(%lo4, %hi4, %lo5, %hi5)
      dats(%c : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %d : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// Reordering: two independent chains, interleaved. Side by side, they need 2 kernels instead of 3.
// CHECK-LABEL: chain @reorder
// CHECK: edge 0 -> 1: raw {d1} blocking
// CHECK-NOT: edge 0 -> 2
// CHECK: edge 2 -> 3: raw {d4} blocking
// PLAN-LABEL: chain @reorder
// PLAN: plan: 0,2|1,3
// PLAN: reordered: 2
ops.chain @reorder(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %c: !ops.dat<f64, 2, 1>,
                   %d: !ops.dat<f64, 2, 1>, %e: !ops.dat<f64, 2, 1>, %f: !ops.dat<f64, 2, 1>,
                   %lo0: index, %hi0: index, %lo1: index, %hi1: index, %lo2: index, %hi2: index, %lo3: index,
                   %hi3: index, %lo4: index, %hi4: index, %lo5: index, %hi5: index, %lo6: index, %hi6: index,
                   %lo7: index, %hi7: index)
    attributes {"ops.range_values" = [array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>,
                                      array<i64: 0, 8, 0, 8>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop range(%lo2, %hi2, %lo3, %hi3)
      dats(%b : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1, 0], [-1, 0]]>,
           %c : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop range(%lo4, %hi4, %lo5, %hi5)
      dats(%d : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %e : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop range(%lo6, %hi6, %lo7, %hi7)
      dats(%e : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [0, 1], [0, -1]]>,
           %f : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.return
}

// A boundary kernel that writes the halo at non-zero offsets: its write footprint is the halo, so a loop over
// the interior is independent of it, while a loop that reads the halo is not and cannot fuse with it.
// CHECK-LABEL: chain @boundary
// CHECK: loop 0: block 0 range [0, 1, 0, 8]
// CHECK-NEXT: d0: read nonlocal write nonlocal
// CHECK-NOT: edge 0 -> 1
// CHECK: edge 0 -> 2: raw {d0} blocking
// PLAN-LABEL: chain @boundary
// PLAN: plan: 0|1|2
ops.chain @boundary(%u: !ops.dat<f64, 2, 1>, %v: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index,
                    %hi1: index, %lo2: index, %hi2: index, %lo3: index, %hi3: index, %lo4: index, %hi4: index,
                    %lo5: index, %hi5: index)
    attributes {"ops.range_values" = [array<i64: 0, 1, 0, 8>, array<i64: 1, 8, 0, 8>, array<i64: -2, 0, 0, 8>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%u : !ops.dat<f64, 2, 1> rw #ops.stencil<[[0, 0], [1, 0], [2, 0], [-1, 0], [-2, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
    %c = arith.constant 0.0 : f64
    ops.write %x[-1, 0], %c : !ops.acc<f64, 2, 1>
    ops.write %x[-2, 0], %c : !ops.acc<f64, 2, 1>
  }
  ops.par_loop range(%lo2, %hi2, %lo3, %hi3)
      dats(%u : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %v : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop range(%lo4, %hi4, %lo5, %hi5)
      dats(%u : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.return
}
