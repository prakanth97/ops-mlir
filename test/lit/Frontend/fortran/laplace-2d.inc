! The kernels of OPS's Fortran Laplace tutorial: OPS_ACCn accessors, a MAX reduction, the one-based index, a module
! variable registered with ops_decl_const and a module parameter. Several kernels in one file, as in OPS apps.
! RUN: ops-kernel-import --kernel apply_stencil_kernel --source %s --module %S/Inputs/constants.F90 --ndim 2 --kinds acc,acc,red -o - --manifest %t.as.json | FileCheck %s --check-prefix=STENCIL
! RUN: FileCheck %s --check-prefix=STENCIL-MANIFEST --input-file=%t.as.json
! RUN: ops-kernel-import --kernel left_bndcon_kernel --source %s --module %S/Inputs/constants.F90 --ndim 2 -o - --manifest %t.lb.json | FileCheck %s --check-prefix=BNDCON
! RUN: FileCheck %s --check-prefix=BNDCON-MANIFEST --input-file=%t.lb.json

SUBROUTINE apply_stencil_kernel(A, Anew, error)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: A
    REAL(KIND=8), DIMENSION(1) :: Anew
    REAL(KIND=8) :: error

    Anew(OPS_ACC2(0,0)) = 0.25_8 * ( A(OPS_ACC1(1,0)) + A(OPS_ACC1(-1,0)) &
                                 & + A(OPS_ACC1(0,-1)) + A(OPS_ACC1(0,1)) )
    error = MAX( error, ABS( Anew(OPS_ACC2(0,0)) - A(OPS_ACC1(0,0)) ) )
END SUBROUTINE

SUBROUTINE left_bndcon_kernel(A, idx)
    REAL(KIND=8), DIMENSION(1) :: A
    INTEGER(KIND=4), DIMENSION(2), INTENT(IN) :: idx

    A(OPS_ACC1(0,0)) = SIN(pi * idx(2) / (jmax+2))
END SUBROUTINE

! STENCIL-LABEL: ops.kernel @apply_stencil_kernel(%arg0: !ops.acc<f64, 2, 1>, %arg1: !ops.acc<f64, 2, 1>, %arg2: !ops.red<f64>)
! STENCIL: ops.read %arg0 [1, 0]
! STENCIL: ops.read %arg0 [-1, 0]
! STENCIL: ops.read %arg0 [0, -1]
! STENCIL: ops.read %arg0 [0, 1]
! STENCIL: %[[NEW:.*]] = arith.mulf
! STENCIL-NEXT: ops.write %arg1 [0, 0], %[[NEW]]
! STENCIL: %[[DIFF:.*]] = math.absf
! STENCIL-NEXT: ops.reduce %arg2, %[[DIFF]] : !ops.red<f64>
! STENCIL-NOT: fastmath
! STENCIL-MANIFEST: "importer": "F/1.0",
! STENCIL-MANIFEST: "const": true,
! STENCIL-MANIFEST-NEXT: "elem": "f64",
! STENCIL-MANIFEST-NEXT: "kind": "acc",
! STENCIL-MANIFEST: "kind": "reduce",
! STENCIL-MANIFEST-NEXT: "kind_hint": "max",
! STENCIL-MANIFEST-NEXT: "style": "accumulate"

! Without --kinds: idx is the index argument. idx(2) is one-based: ops.index 1 plus one. pi is folded, jmax is read
! from the registered constant.
! BNDCON-LABEL: ops.kernel @left_bndcon_kernel(%arg0: !ops.acc<f64, 2, 1>)
! BNDCON: arith.constant 3.1415926535897931 : f64
! BNDCON: %[[I:.*]] = ops.index 1
! BNDCON-NEXT: %[[ONE:.*]] = arith.constant 1 : i32
! BNDCON-NEXT: arith.addi %[[I]], %[[ONE]] : i32
! BNDCON: ops.const_ref @jmax : i32
! BNDCON: math.sin
! BNDCON: ops.write %arg0 [0, 0]
! BNDCON-MANIFEST: "name": "jmax",
! BNDCON-MANIFEST: "kind": "idx"
! BNDCON-MANIFEST: "idx_axes_read": [
! BNDCON-MANIFEST-NEXT: 1
