// The lit setup works and the pinned toolchain is the one the tests run against.
// RUN: mlir-opt --version | FileCheck %s --check-prefix=VERSION
// RUN: mlir-opt %s --canonicalize | FileCheck %s
// RUN: not mlir-opt %S/Inputs/not-mlir.txt 2>&1 | FileCheck %s --check-prefix=ERR

// VERSION: LLVM version 23.1

// CHECK-LABEL: func.func @identity
// CHECK-SAME: (%[[A:.*]]: i32) -> i32
// CHECK-NEXT: return %[[A]] : i32
func.func @identity(%a: i32) -> i32 {
  %0 = arith.addi %a, %a : i32
  %1 = arith.subi %0, %a : i32
  return %1 : i32
}

// ERR: error
