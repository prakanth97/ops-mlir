#include "math.h"
//===----------------------------------------------------------------------===//
// Laplace 2D Stencil Kernels
//===----------------------------------------------------------------------===//

extern int jmax;
extern double pi;

double set_zero() {
  return 0.0;
}

double copy(double a) {
  return a;
}

double left_bndcon(const int *idx) {
  return sin(pi * (idx[1] + 1) / (jmax + 1));
}

double right_bndcon(const int *idx) {
  return sin(pi * (idx[1] + 1) / (jmax + 1)) * exp(-pi);
}

double apply_stencil(double a, double b, double c, double d, double e, double *error) {
  double anew = 0.25f * (b + c + d + e);
  *error = fmax(*error, fabs(anew - a));
  return anew;
}
