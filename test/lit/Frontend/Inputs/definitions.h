// Stands in for CloverLeaf's definitions.h (which the real advec_*_kernel.h headers include): only what those kernels
// use. The layout of field_type is the one of CloverLeaf's user_types.h.
#ifndef IMPORTER_TEST_DEFINITIONS_H
#define IMPORTER_TEST_DEFINITIONS_H
struct field_type {
  int x_min, x_max, y_min, y_max;
};
extern field_type field;
#endif
