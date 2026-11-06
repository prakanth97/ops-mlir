// Constants that shape the code are recorded with role "bound": a loop bound, the index of an array constant, the
// array constant that is indexed. Plain values (here `scale`) are not: the runtime passes them as arguments.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel weighted --source %s --ndim 2 -o %t.mlir --manifest %t.json -- -I%ops_include
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json

#include <ops_lib_core.h>

extern int nstates;
extern double weights[4];
extern double scale;

void weighted(ACC<double> &a, const ACC<double> &b) {
  double s = 0.0;
  for (int i = 0; i < nstates; i++) s += weights[i] * b(0, 0);
  a(0, 0) = s * scale;
}

// MANIFEST: "consts": [
// MANIFEST: "name": "nstates",
// MANIFEST: "offset": 0,
// MANIFEST: "role": "bound"
// MANIFEST: "name": "scale",
// MANIFEST: "offset": 0
// MANIFEST-NOT: "role"
// MANIFEST: "array_stride": 8,
// MANIFEST: "dynamic_index": true,
// MANIFEST: "name": "weights",
// MANIFEST: "offset": 0,
// MANIFEST: "role": "bound"
