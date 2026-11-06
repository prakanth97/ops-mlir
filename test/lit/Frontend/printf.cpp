// printf with a literal format string becomes ops.print (arguments promoted as for C varargs); other output calls are dropped.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel noisy --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>%t.err | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json
// RUN: FileCheck %s --check-prefix=WARN --input-file=%t.err

#include <ops_lib_core.h>
#include <stdio.h>

void noisy(ACC<double> &a, const int *idx) {
  a(0, 0) = 2.0;
  float f = 1.5f;
  printf("at %d,%d value %g\n", idx[0], idx[1], f);
  puts("dropped");
}

// CHECK-LABEL: ops.kernel @noisy
// CHECK: ops.print "at %d,%d value %g\0A", %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, f64
// MANIFEST: "dropped_printf": 1
// MANIFEST: "prints": 1
// WARN: call to puts is dropped
