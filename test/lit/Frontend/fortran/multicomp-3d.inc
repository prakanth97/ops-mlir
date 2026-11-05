! 3D kernels: OPS_ACC_MDn (one-based components) on multi-component dats, read-only globals (one-based elements),
! the index in 3D, and a data-dependent read offset (ops.read_dyn).
! RUN: ops-kernel-import --kernel mdim_kernel --source %s --ndim 3 --kinds acc,acc,gbl,idx -o - --manifest %t.md.json | FileCheck %s --check-prefix=MD
! RUN: FileCheck %s --check-prefix=MD-MANIFEST --input-file=%t.md.json
! RUN: ops-kernel-import --kernel gather_kernel --source %s --ndim 3 -o - --manifest %t.g.json | FileCheck %s --check-prefix=DYN
! RUN: FileCheck %s --check-prefix=DYN-MANIFEST --input-file=%t.g.json

SUBROUTINE mdim_kernel(vel, speed, g, idx)
    REAL(KIND=8), DIMENSION(3), INTENT(IN) :: vel
    REAL(KIND=8), DIMENSION(1) :: speed
    REAL(KIND=8), DIMENSION(2), INTENT(IN) :: g
    INTEGER(KIND=4), DIMENSION(3), INTENT(IN) :: idx
    speed(OPS_ACC2(0,0,0)) = g(2) * SQRT(vel(OPS_ACC_MD1(1,0,0,0))**2 + vel(OPS_ACC_MD1(2,0,0,0))**2 &
                           & + vel(OPS_ACC_MD1(3,0,0,-1))**2) + g(1) * idx(3)
END SUBROUTINE

SUBROUTINE gather_kernel(a, shift, b)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: a
    INTEGER(KIND=4), DIMENSION(1), INTENT(IN) :: shift
    REAL(KIND=8), DIMENSION(1) :: b
    b(OPS_ACC3(0,0,0)) = a(OPS_ACC1(shift(OPS_ACC2(0,0,0)),0,0))
END SUBROUTINE

! MD-LABEL: ops.kernel @mdim_kernel(%arg0: !ops.acc<f64, 3, 3>, %arg1: !ops.acc<f64, 3, 1>, %arg2: !ops.gbl<f64, 2>)
! MD: ops.gbl_read %arg2[1]
! MD: ops.read %arg0 [0, 0, 0] :
! MD: ops.read %arg0 [0, 0, 0] {comp = 1 : i64}
! MD: ops.read %arg0 [0, 0, -1] {comp = 2 : i64}
! MD: ops.gbl_read %arg2[0]
! MD: ops.index 2
! MD: ops.write %arg1 [0, 0, 0]
! MD-MANIFEST: "ncomp": 3
! MD-MANIFEST: "dim": 2,
! MD-MANIFEST-NEXT: "elem": "f64",
! MD-MANIFEST-NEXT: "kind": "gbl"

! DYN-LABEL: ops.kernel @gather_kernel(%arg0: !ops.acc<f64, 3, 1>, %arg1: !ops.acc<i32, 3, 1>, %arg2: !ops.acc<f64, 3, 1>)
! DYN: %[[S:.*]] = ops.read %arg1 [0, 0, 0]
! DYN: ops.read_dyn %arg0[%[[S]], %{{.*}}, %{{.*}}]
! DYN-MANIFEST: "dynamic_reads": [
! DYN-MANIFEST-NEXT: "0"
