// ops-gpu-kernel-attrs: the aligned pointer of each memref descriptor of a kernel is noalias (NVVM and ROCDL kernels).
// RUN: ops-opt %s -pass-pipeline="builtin.module(gpu.module(ops-gpu-kernel-attrs))" | FileCheck %s

// CHECK-LABEL: llvm.func @nvvm_kernel(
// CHECK-SAME: %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr {llvm.noalias}, %{{.*}}: i64, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr {llvm.noalias}, %{{.*}}: i64)
// CHECK-LABEL: llvm.func @rocdl_kernel(
// CHECK-SAME: %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr {llvm.noalias}, %{{.*}}: i64)
// CHECK-LABEL: llvm.func @helper(
// CHECK-NOT: llvm.noalias
gpu.module @kernels {
  llvm.func @nvvm_kernel(%a0: !llvm.ptr, %a1: !llvm.ptr, %o: i64, %b0: !llvm.ptr, %b1: !llvm.ptr, %p: i64) attributes {nvvm.kernel} {
    llvm.return
  }
  llvm.func @rocdl_kernel(%a0: !llvm.ptr, %a1: !llvm.ptr, %o: i64) attributes {rocdl.kernel} {
    llvm.return
  }
  llvm.func @helper(%a0: !llvm.ptr, %a1: !llvm.ptr) {
    llvm.return
  }
}
