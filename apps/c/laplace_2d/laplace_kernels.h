//===----------------------------------------------------------------------===//
// Laplace 2D Stencil Kernels
//===----------------------------------------------------------------------===//

double set_zero() {
  return 0.0;
}

double left_bndcon(const int *idx) {
  return sin(pi * (idx[1] + 1) / (jmax + 1));
}

double right_bndcon(const int *idx) {
  return sin(pi * (idx[1] + 1) / (jmax + 1)) * exp(-pi);
}
