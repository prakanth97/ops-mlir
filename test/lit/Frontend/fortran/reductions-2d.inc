! Reductions: r = r + x, r = r - x, MIN, MAX (flang spells them as a compare and a select) and integer counts become
! ops.reduce; a plain assignment is the "assign" style; reading the reduction value otherwise is rejected.
! RUN: ops-kernel-import --kernel sums_kernel --source %s --ndim 2 --kinds acc,red,red,red -o - --manifest %t.s.json | FileCheck %s --check-prefix=SUMS
! RUN: FileCheck %s --check-prefix=SUMS-MANIFEST --input-file=%t.s.json
! RUN: ops-kernel-import --kernel minmax_kernel --source %s --ndim 2 --kinds acc,red,red -o - --manifest %t.m.json | FileCheck %s --check-prefix=MINMAX
! RUN: FileCheck %s --check-prefix=MINMAX-MANIFEST --input-file=%t.m.json
! RUN: ops-kernel-import --kernel assign_kernel --source %s --ndim 2 --kinds acc,red -o - --manifest %t.a.json | FileCheck %s --check-prefix=ASSIGN
! RUN: FileCheck %s --check-prefix=ASSIGN-MANIFEST --input-file=%t.a.json
! RUN: ops-kernel-import --kernel mixed_kernel --source %s --ndim 2 --kinds acc,red -o - 2>&1 | FileCheck %s --check-prefix=MIXED
! RUN: ops-kernel-import --kernel read_kernel --source %s --ndim 2 --kinds acc,acc,red -o - 2>&1 | FileCheck %s --check-prefix=READ

SUBROUTINE sums_kernel(a, total, diff, cnt)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8) :: total, diff
    INTEGER(KIND=4) :: cnt
    total = total + a(OPS_ACC1(0,0))
    diff = diff - a(OPS_ACC1(1,0))
    cnt = cnt + 1
END SUBROUTINE

SUBROUTINE minmax_kernel(a, lo, hi)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8) :: lo, hi
    lo = MIN(lo, a(OPS_ACC1(0,0)))
    hi = MAX(a(OPS_ACC1(0,0)), hi)
END SUBROUTINE

SUBROUTINE assign_kernel(a, last)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8) :: last
    last = a(OPS_ACC1(0,0))
END SUBROUTINE

SUBROUTINE mixed_kernel(a, r)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8) :: r
    r = r + a(OPS_ACC1(0,0))
    r = MIN(r, a(OPS_ACC1(1,0)))
END SUBROUTINE

SUBROUTINE read_kernel(a, b, r)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8), DIMENSION(1) :: b
    REAL(KIND=8) :: r
    b(OPS_ACC2(0,0)) = r * a(OPS_ACC1(0,0))
END SUBROUTINE

! SUMS-LABEL: ops.kernel @sums_kernel(%arg0: !ops.acc<f64, 2, 1>, %arg1: !ops.red<f64>, %arg2: !ops.red<f64>, %arg3: !ops.red<i32>)
! SUMS: %[[A:.*]] = ops.read %arg0 [0, 0]
! SUMS-NEXT: ops.reduce %arg1, %[[A]] : !ops.red<f64>
! SUMS: %[[B:.*]] = ops.read %arg0 [1, 0]
! SUMS-NEXT: %[[NB:.*]] = arith.negf %[[B]]
! SUMS-NEXT: ops.reduce %arg2, %[[NB]] : !ops.red<f64>
! SUMS: ops.reduce %arg3, %{{.*}} : !ops.red<i32>
! SUMS-MANIFEST-COUNT-3: "kind_hint": "add",

! MINMAX-LABEL: ops.kernel @minmax_kernel
! MINMAX: %[[A:.*]] = ops.read %arg0 [0, 0]
! MINMAX-NEXT: ops.reduce %arg1, %[[A]]
! MINMAX: ops.reduce %arg2,
! MINMAX-MANIFEST: "kind_hint": "min",
! MINMAX-MANIFEST: "kind_hint": "max",

! ASSIGN-LABEL: ops.kernel @assign_kernel
! ASSIGN: ops.reduce %arg1,
! ASSIGN-MANIFEST: "kind_hint": "none",
! ASSIGN-MANIFEST-NEXT: "style": "assign"

! MIXED: not imported: reductions-2d.inc:{{[0-9]+}}: a reduction that is updated as both add and min
! READ: not imported: reductions-2d.inc:{{[0-9]+}}: the value of a reduction argument is read
