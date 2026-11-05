! Control flow: IF / ELSE IF chains with logicals become nested scf.if yielding the locals they assign; a DO loop whose
! body subscripts with its variable and has a constant trip count is unrolled (the OPS_ACCn offsets become constants);
! other DO loops become scf.for over [lb, ub + 1), with the constants in their bounds marked "bound".
! RUN: ops-kernel-import --kernel branches_kernel --source %s --module %S/Inputs/constants.F90 --ndim 2 -o - | FileCheck %s --check-prefix=IF
! RUN: ops-kernel-import --kernel loops_kernel --source %s --module %S/Inputs/constants.F90 --ndim 2 -o - --manifest %t.json | FileCheck %s --check-prefix=LOOP
! RUN: FileCheck %s --check-prefix=LOOP-MANIFEST --input-file=%t.json

SUBROUTINE branches_kernel(a, b)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8), DIMENSION(1) :: b
    REAL(KIND=8) :: t, s
    LOGICAL :: big

    t = a(OPS_ACC1(0,0))
    big = t > 10.0_8
    IF (big .AND. t < 100.0_8) THEN
        s = SQRT(t)
    ELSE IF (t < 0.0_8) THEN
        s = -t
    ELSE
        s = t ** 2
    END IF
    b(OPS_ACC2(0,0)) = s
END SUBROUTINE

SUBROUTINE loops_kernel(a, b)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8), DIMENSION(1) :: b
    REAL(KIND=8) :: s, tmp(3)
    INTEGER :: i

    DO i = -1, 1
        tmp(i+2) = a(OPS_ACC1(i,0)) * w(i+2)
    END DO
    s = tmp(1) + tmp(2) + tmp(3)
    DO i = 1, nsteps
        s = s * 0.5_8 + coef(MOD(i,3)+1)
    END DO
    b(OPS_ACC2(0,0)) = s
END SUBROUTINE

! IF-LABEL: ops.kernel @branches_kernel
! IF: %[[T:.*]] = ops.read %arg0 [0, 0]
! IF: %[[C:.*]] = arith.andi
! IF: %[[S:.*]] = scf.if %[[C]] -> (f64) {
! IF:   math.sqrt %[[T]]
! IF: } else {
! IF:   scf.if %{{.*}} -> (f64) {
! IF:     arith.negf %[[T]]
! IF:   } else {
! IF:     arith.mulf %[[T]], %[[T]]
! IF: ops.write %arg1 [0, 0], %[[S]]

! LOOP-LABEL: ops.kernel @loops_kernel
! LOOP: ops.read %arg0 [-1, 0]
! LOOP: arith.constant 2.500000e-01
! LOOP: ops.read %arg0 [0, 0]
! LOOP: arith.constant 5.000000e-01
! LOOP: ops.read %arg0 [1, 0]
! LOOP: %[[N:.*]] = ops.const_ref @nsteps : i32
! LOOP: %[[UB:.*]] = arith.index_cast %[[N]] : i32 to index
! LOOP: %[[END:.*]] = arith.addi %[[UB]], %{{.*}} : index
! LOOP: %[[R:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %[[END]]
! LOOP:   ops.const_ref @coef[%{{.*}} : i32] {array_stride = 8 : i64} : f64
! LOOP: ops.write %arg1 [0, 0], %[[R]]#2
! LOOP-MANIFEST: "dynamic_index": true,
! LOOP-MANIFEST-NEXT: "elem": "f64",
! LOOP-MANIFEST-NEXT: "name": "coef",
! LOOP-MANIFEST-NEXT: "offset": 0,
! LOOP-MANIFEST-NEXT: "role": "bound"
! LOOP-MANIFEST: "name": "nsteps",
! LOOP-MANIFEST-NEXT: "offset": 0,
! LOOP-MANIFEST-NEXT: "role": "bound"
