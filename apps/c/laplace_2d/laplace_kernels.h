//===----------------------------------------------------------------------===//
// Laplace 2D Stencil Kernels
//===----------------------------------------------------------------------===//

double set_zero() {
  return 0.0;
}

void copy(ACC<double> &A, const ACC<double> &Anew) { A(0, 0) = Anew(0, 0); }

double left_bndcon(const int *idx) {
  return sin(pi * (idx[1] + 1) / (jmax + 1));
}

double right_bndcon(const int *idx) {
  return sin(pi * (idx[1] + 1) / (jmax + 1)) * exp(-pi);
}

void apply_stencil(const ACC<double> &A, ACC<double> &Anew, double *error) {
  Anew(0, 0) = 0.25f * (A(1, 0) + A(-1, 0) + A(0, -1) + A(0, 1));
  *error = fmax(*error, fabs(Anew(0, 0) - A(0, 0)));
}
