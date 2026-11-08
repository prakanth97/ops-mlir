! Kernels outside the supported subset are not imported; the manifest says why (with the line) and the loops run
! the code OPS generated.
! RUN: ops-kernel-import --kernel helper_kernel --source %s --ndim 2 -o - --manifest %t.h.json 2>&1 | FileCheck %s --check-prefix=HELPER
! RUN: FileCheck %s --check-prefix=HELPER-MANIFEST --input-file=%t.h.json
! RUN: ops-kernel-import --kernel wrong_acc_kernel --source %s --ndim 2 -o - 2>&1 | FileCheck %s --check-prefix=WRONG
! RUN: ops-kernel-import --kernel dyn_write_kernel --source %s --ndim 2 -o - 2>&1 | FileCheck %s --check-prefix=DYNWRITE
! RUN: ops-kernel-import --kernel gbl_write_kernel --source %s --ndim 2 --kinds acc,gbl -o - 2>&1 | FileCheck %s --check-prefix=GBL
! RUN: ops-kernel-import --kernel missing_kernel --source %s --ndim 2 -o - 2>&1 | FileCheck %s --check-prefix=MISSING
! RUN: not ops-kernel-import --strict --kernel missing_kernel --source %s --ndim 2 -o /dev/null

REAL(KIND=8) FUNCTION twice(x)
    REAL(KIND=8), INTENT(IN) :: x
    twice = 2.0_8 * x
END FUNCTION

SUBROUTINE helper_kernel(a, b)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8), DIMENSION(1) :: b
    b(OPS_ACC2(0,0)) = twice(a(OPS_ACC1(0,0)))
END SUBROUTINE

SUBROUTINE wrong_acc_kernel(a, b)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8), DIMENSION(1) :: b
    b(OPS_ACC2(0,0)) = a(OPS_ACC2(0,0))
END SUBROUTINE

SUBROUTINE dyn_write_kernel(a, b)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    REAL(KIND=8), DIMENSION(1) :: b
    INTEGER :: k
    k = INT(a(OPS_ACC1(0,0)))
    b(OPS_ACC2(k,0)) = 1.0_8
END SUBROUTINE

SUBROUTINE gbl_write_kernel(a, g)
    REAL(KIND=8), DIMENSION(1) :: a
    REAL(KIND=8) :: g
    g = a(OPS_ACC1(0,0))
END SUBROUTINE

! HELPER: not imported: unsupported-2d.inc:{{[0-9]+}}: call to 'twice' (only intrinsic math functions are supported; helper procedures are not)
! HELPER-MANIFEST: "importer": "F/1.0",
! HELPER-MANIFEST: "reason": "unsupported-2d.inc:{{[0-9]+}}: call to 'twice'
! HELPER-MANIFEST: "status": "failed"
! WRONG: not imported: unsupported-2d.inc:{{[0-9]+}}: OPS_ACC2 subscripts argument 1 ('a'): OPS_ACCn must subscript the n-th argument
! DYNWRITE: not imported: unsupported-2d.inc:{{[0-9]+}}: write to a dat at an offset that is not a compile-time constant
! GBL: not imported: unsupported-2d.inc:{{[0-9]+}}: assignment to the global argument 'g'
! MISSING: not imported: the subroutine 'missing_kernel' was not found in
