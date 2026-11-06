// What the importer declines, with the reason: the loops that call these kernels run through the stock OPS code.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel uses_while --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=WHILE
// RUN: FileCheck %s --check-prefix=WHILE-MANIFEST --input-file=%t.json
// RUN: ops-kernel-import --kernel writes_dynamic --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=DYNWRITE
// RUN: ops-kernel-import --kernel unknown_call --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=CALL
// RUN: ops-kernel-import --kernel bf16_input --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=BF16
// RUN: ops-kernel-import --kernel writes_global --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=GBLWRITE
// RUN: ops-kernel-import --kernel gbl_by_variable --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=GBLVAR
// RUN: ops-kernel-import --kernel with_printf --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=PRINTF
// RUN: ops-kernel-import --kernel missing --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=MISSING

#include <ops_lib_core.h>
#include <stdio.h>

void uses_while(const ACC<double> &a, ACC<double> &b) {
  double v = a(0, 0);
  while (v > 1.0) v = v / 2.0;
  b(0, 0) = v;
}
// WHILE: ops-kernel-import: uses_while: not imported: {{.*}}unsupported statement (WhileStmt)
// WHILE-MANIFEST: "reason": "{{.*}}WhileStmt
// WHILE-MANIFEST: "status": "failed"

void writes_dynamic(const ACC<int> &which, ACC<double> &b) {
  b(which(0, 0), 0) = 1.0;
}
// DYNWRITE: writes_dynamic: not imported: {{.*}}write to an accessor at a data-dependent offset

double erfcinv_not_in_the_library(double);
void unknown_call(const ACC<double> &a, ACC<double> &b) { b(0, 0) = erfcinv_not_in_the_library(a(0, 0)); }
// CALL: unknown_call: not imported: {{.*}}call to unsupported function 'erfcinv_not_in_the_library'

void bf16_input(const ACC<__bf16> &a, ACC<double> &b) { b(0, 0) = a(0, 0); }
// BF16: bf16_input: not imported: bfloat16 is not supported

void writes_global(const ACC<double> &a, double *out) { out[1] = a(0, 0); }
// GBLWRITE: writes_global: not imported: {{.*}}write to the global or reduction argument 'out'

void gbl_by_variable(const ACC<double> &a, ACC<double> &b, const double *table, const int *idx) {
  b(0, 0) = a(0, 0) * table[idx[0]];
}
// GBLVAR: gbl_by_variable: not imported: {{.*}}not a compile-time constant

void with_printf(const ACC<double> &a, ACC<double> &b) {
  printf("value %f\n", a(0, 0));
  b(0, 0) = a(0, 0);
}
// PRINTF: ops.kernel @with_printf
// PRINTF: ops.print "value %f\0A", %{{.*}} : f64
// PRINTF: ops.write %arg1 [0, 0]

// MISSING: missing: not imported: the definition of kernel 'missing' was not found
