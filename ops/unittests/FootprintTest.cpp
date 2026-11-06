#include "ops-mlir/Analysis/Footprint.h"

#include "ChainBuilder.h"
#include "opdsl/Testing/TestHarness.h"

using namespace ops_mlir::ops;
using ops_mlir::ops::testing::context;

namespace {
StencilAttr stencil(std::vector<int64_t> flat, int64_t rank = 2, std::vector<int64_t> strides = {}) {
  if (strides.empty()) strides.assign(rank, 1);
  return StencilAttr::get(&context(), rank, flat, strides);
}
Range r(std::initializer_list<int64_t> v) { return Range(v.begin(), v.end()); }
} // namespace

OPDSL_TEST(Footprint, Geometry) {
  OPDSL_EXPECT(isEmpty(r({0, 0, 0, 5})));
  OPDSL_EXPECT(!isEmpty(r({0, 1, 0, 5})));
  OPDSL_EXPECT(overlaps(r({0, 5}), r({4, 9})));
  OPDSL_EXPECT(!overlaps(r({0, 5}), r({5, 9}))); // half-open
  OPDSL_EXPECT(!overlaps(r({0, 5, 0, 5}), r({0, 5, 5, 9})));
  OPDSL_EXPECT(!overlaps(r({3, 3}), r({0, 9}))); // an empty box overlaps nothing
  OPDSL_EXPECT_EQ(volume(r({0, 4, 2, 5})), 12.0);
  OPDSL_EXPECT(boundingBox(r({0, 4, 2, 5}), r({3, 9, -1, 3})) == r({0, 9, -1, 5}));
  OPDSL_EXPECT(boundingBox({}, r({1, 2})) == r({1, 2}));
}

OPDSL_TEST(Footprint, StencilExtent) {
  // five-point
  StencilAttr five = stencil({0, 0, 1, 0, -1, 0, 0, 1, 0, -1});
  OPDSL_EXPECT(footprint(r({0, 10, 0, 10}), five) == r({-1, 11, -1, 11}));
  // only offsets on one side: the zero offset is not implied
  StencilAttr ahead = stencil({1, 0, 2, 0});
  OPDSL_EXPECT(footprint(r({0, 10, 0, 10}), ahead) == r({1, 12, 0, 10}));
  // a point stencil does not change the range
  OPDSL_EXPECT(footprint(r({3, 5, 4, 6}), stencil({0, 0})) == r({3, 5, 4, 6}));
}

OPDSL_TEST(Footprint, StridesMakeADimensionUnbounded) {
  StencilAttr broadcast = stencil({0, 0}, 2, {1, 0});
  Range f = footprint(r({0, 10, 0, 10}), broadcast);
  OPDSL_EXPECT(f[0] == 0 && f[1] == 10);
  OPDSL_EXPECT(f[2] == -kUnbounded && f[3] == kUnbounded);
  OPDSL_EXPECT(overlaps(f, r({2, 3, 500, 600})));
}

OPDSL_TEST(Footprint, PointLocality) {
  OPDSL_EXPECT(Reach::of(stencil({0, 0})).pointLocal);
  OPDSL_EXPECT(!Reach::of(stencil({0, 0, 1, 0})).pointLocal);
  OPDSL_EXPECT(!Reach::of(stencil({0, 0}, 2, {1, 0})).pointLocal);
  Reach none(2);
  OPDSL_EXPECT(none.pointLocal && !none.any);
  none.addOffset(r({0, -1}));
  OPDSL_EXPECT(!none.pointLocal && none.any);
}
