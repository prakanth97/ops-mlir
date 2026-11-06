#include "ChainBuilder.h"
#include "ops-mlir/Analysis/ChainDependence.h"

#include "opdsl/Testing/TestHarness.h"

#include "llvm/Support/raw_ostream.h"

using namespace ops_mlir::ops;
using namespace ops_mlir::ops::testing;

namespace {
using R = std::vector<int64_t>;
const R kInner = {0, 10, 0, 10};

ArgSpec arg(int id, AccessMode acc, Points points = {}) {
  ArgSpec a;
  a.dat = id;
  a.access = acc;
  a.points = std::move(points);
  return a;
}
LoopSpec loop(std::vector<ArgSpec> args, R range = kInner) {
  LoopSpec l;
  l.args = std::move(args);
  l.range = std::move(range);
  return l;
}
ChainDependence analyse(const std::vector<LoopSpec> &q) {
  BuiltChain b = buildChain(q, 2);
  auto dep = ChainDependence::build(b.chain);
  OPDSL_EXPECT(succeeded(dep));
  return succeeded(dep) ? std::move(*dep) : ChainDependence{};
}
const Points kFive = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
} // namespace

OPDSL_TEST(ChainDependence, ExtractionReadsRangesAndAccesses) {
  ChainDependence dep = analyse({loop({arg(1, AccessMode::Read, kFive), arg(2, AccessMode::Write)})});
  OPDSL_EXPECT_EQ(dep.loops.size(), size_t(1));
  const LoopInfo &l = dep.loops[0];
  OPDSL_EXPECT(l.range == Range(kInner.begin(), kInner.end()));
  OPDSL_EXPECT_EQ(l.args.size(), size_t(2));
  OPDSL_EXPECT(l.args[0].reads && !l.args[0].writes && l.args[0].readNonLocal);
  OPDSL_EXPECT(l.args[1].writes && !l.args[1].reads && !l.args[1].writeNonLocal);
  OPDSL_EXPECT_EQ(l.args[0].elemBytes, int64_t(8));
  OPDSL_EXPECT(l.args[0].readBox == Range({-1, 11, -1, 11}));
  OPDSL_EXPECT(!l.hasReduction);
  OPDSL_EXPECT_EQ(l.axes, 3u);
}

OPDSL_TEST(ChainDependence, MissingRangeIsADiagnostic) {
  BuiltChain b = buildChain({loop({arg(1, AccessMode::Write)})}, 2);
  b.chain->removeAttr("ops.range_values");
  OPDSL_EXPECT(failed(extractLoops(b.chain)));
}

OPDSL_TEST(ChainDependence, EdgesAndFusability) {
  // L0: d2 = f(d1)    L1: d3 = g(d2) pointwise    L2: d1 = h(d3 5pt)    L3: d2 = k(d2) (rw)
  ChainDependence dep = analyse({loop({arg(1, AccessMode::Read), arg(2, AccessMode::Write)}),
                                 loop({arg(2, AccessMode::Read), arg(3, AccessMode::Write)}),
                                 loop({arg(3, AccessMode::Read, kFive), arg(1, AccessMode::Write)}),
                                 loop({arg(2, AccessMode::RW)})});
  auto find = [&](size_t from, size_t to) -> const DepEdge * {
    for (const DepEdge &e : dep.edges)
      if (e.from == from && e.to == to) return &e;
    return nullptr;
  };
  const DepEdge *e01 = find(0, 1);
  OPDSL_EXPECT(e01 && e01->kinds == RAW && e01->fusable);
  const DepEdge *e12 = find(1, 2);
  OPDSL_EXPECT(e12 && e12->kinds == RAW && !e12->fusable); // 5-point read of what L1 wrote
  const DepEdge *e02 = find(0, 2);
  OPDSL_EXPECT(e02 && e02->kinds == WAR && e02->fusable); // L0 read d1 at zero, L2 writes it
  const DepEdge *e03 = find(0, 3);
  OPDSL_EXPECT(e03 && e03->kinds == (RAW | WAW) && e03->fusable);
  const DepEdge *e13 = find(1, 3);
  OPDSL_EXPECT(e13 && e13->kinds == WAR && e13->fusable);
  OPDSL_EXPECT(find(2, 3) == nullptr);
}

OPDSL_TEST(ChainDependence, PrintFormat) {
  ChainDependence dep = analyse({loop({arg(1, AccessMode::Read), arg(2, AccessMode::Write)}),
                                 loop({arg(2, AccessMode::Read, kFive), arg(3, AccessMode::Write)})});
  std::string text;
  llvm::raw_string_ostream os(text);
  dep.print(os);
  OPDSL_EXPECT(text.find("edge 0 -> 1: raw {d2} blocking") != std::string::npos);
  OPDSL_EXPECT(text.find("loop 1: block 1 range [0, 10, 0, 10]") != std::string::npos);
}
