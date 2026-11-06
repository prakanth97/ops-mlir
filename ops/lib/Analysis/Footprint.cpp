#include "ops-mlir/Analysis/Footprint.h"

#include <algorithm>

namespace ops_mlir::ops {

bool isEmpty(llvm::ArrayRef<int64_t> range) {
  for (size_t d = 0; d + 1 < range.size(); d += 2)
    if (range[d] >= range[d + 1]) return true;
  return false;
}

bool overlaps(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (isEmpty(a) || isEmpty(b)) return false;
  size_t n = std::min(a.size(), b.size());
  for (size_t d = 0; d + 1 < n; d += 2)
    if (!(a[d] < b[d + 1] && b[d] < a[d + 1])) return false;
  return true;
}

double volume(llvm::ArrayRef<int64_t> range) {
  double v = 1.0;
  for (size_t d = 0; d + 1 < range.size(); d += 2)
    v *= static_cast<double>(std::max<int64_t>(0, range[d + 1] - range[d]));
  return v;
}

Range boundingBox(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (a.empty()) return Range(b.begin(), b.end());
  Range box(a.begin(), a.end());
  for (size_t d = 0; d + 1 < box.size(); d += 2) {
    box[d] = std::min(a[d], b[d]);
    box[d + 1] = std::max(a[d + 1], b[d + 1]);
  }
  return box;
}

void Reach::addOffset(llvm::ArrayRef<int64_t> offset) {
  for (size_t d = 0; d < lo.size() && d < offset.size(); ++d) {
    lo[d] = any ? std::min(lo[d], offset[d]) : offset[d];
    hi[d] = any ? std::max(hi[d], offset[d]) : offset[d];
  }
  // lo/hi of a dimension are only meaningful once the first offset set them: do it for all dims at once.
  any = true;
  for (int64_t o : offset)
    if (o != 0) pointLocal = false;
}

void Reach::setStrides(llvm::ArrayRef<int64_t> strides) {
  for (size_t d = 0; d < unbounded.size() && d < strides.size(); ++d)
    if (strides[d] != 1) {
      unbounded[d] = true;
      pointLocal = false;
    }
}

Reach Reach::of(StencilAttr stencil) {
  Reach r(stencil.getRank());
  for (int64_t i = 0, n = stencil.getNumPoints(); i < n; ++i) r.addOffset(stencil.getPoint(i));
  r.setStrides(stencil.getStrides());
  return r;
}

Range footprint(llvm::ArrayRef<int64_t> range, const Reach &reach) {
  Range r(range.begin(), range.end());
  for (size_t d = 0; d < reach.lo.size() && 2 * d + 1 < r.size(); ++d) {
    if (reach.unbounded[d]) {
      r[2 * d] = -kUnbounded;
      r[2 * d + 1] = kUnbounded;
      continue;
    }
    if (!reach.any) continue; // no offsets: the zero offset
    r[2 * d] += reach.lo[d];
    r[2 * d + 1] += reach.hi[d];
  }
  return r;
}

Range footprint(llvm::ArrayRef<int64_t> range, StencilAttr stencil) {
  return footprint(range, Reach::of(stencil));
}

} // namespace ops_mlir::ops
