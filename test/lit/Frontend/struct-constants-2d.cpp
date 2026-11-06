// Registered constants that are arrays of structs, indexed by a value known only at run time (the states of
// CloverLeaf generate_chunk): ops.const_ref with a dynamic index and the stride of the element.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel generate_chunk_kernel --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json

#include <ops_lib_core.h>

// The types of CloverLeaf user_types.h, reduced to what the kernel uses.
struct state_type {
  int defined;
  double density, energy, xvel, yvel;
  int geometry;
  double xmin, xmax, ymin, ymax, radius;
};
extern state_type *states;
extern int number_of_states;
const int g_rect = 1, g_circ = 2, g_point = 3;

#include "Inputs/cloverleaf_generate_chunk_kernel.h"

// CHECK-LABEL: ops.kernel @generate_chunk_kernel(
// The background state: element 0 of the array, constant offsets.
// CHECK: ops.const_ref @states {offset = 16 : i64} : f64
// CHECK: ops.const_ref @number_of_states : i32
// The loop over the states has a run-time trip count: an scf.for whose elements are read with a dynamic index.
// CHECK: scf.for
// CHECK: ops.const_ref @states[%{{.*}} : i32] {array_stride = 88 : i64, offset = 48 : i64} : f64
// g_rect, g_circ and g_point are compile-time constants of the translation unit: no const_ref.
// CHECK-NOT: ops.const_ref @g_rect
// MANIFEST: "array_stride": 88,
// MANIFEST: "dynamic_index": true,
// MANIFEST: "name": "states",
