// Tests of the fusion planner on ops.chain: every case of the legacy planner's suite, then the cases the
// legacy planner could not express (writes at non-zero offsets), then a randomised oracle: a plan must
// compute exactly what program order does.

#include "ChainBuilder.h"
#include "ops-mlir/Analysis/FusionPlanner.h"

#include "opdsl/Testing/TestHarness.h"

#include <map>
#include <random>
#include <string>
#include <vector>

using namespace ops_mlir::ops;
using namespace ops_mlir::ops::testing;

namespace {

using R = std::vector<int64_t>;

const R kFull = {-1, 11, -1, 11};
const R kInner = {0, 10, 0, 10};
const R kLeft = {-1, 0, -1, 11};

const Points kFive = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
const std::vector<int64_t> kBroadcast = {1, 0};
const std::vector<int64_t> kAlongY = {0, 1};

ArgSpec dat(int id, AccessMode acc, bool five = false) {
  ArgSpec a;
  a.dat = id;
  a.access = acc;
  if (five) a.points = kFive;
  return a;
}
ArgSpec rd(int id, bool five = false) { return dat(id, AccessMode::Read, five); }
ArgSpec wr(int id) { return dat(id, AccessMode::Write); }
ArgSpec rw(int id) { return dat(id, AccessMode::RW); }
ArgSpec red() {
  ArgSpec a;
  a.dat = kReduction;
  return a;
}
ArgSpec global() {
  ArgSpec a;
  a.dat = kGlobal;
  return a;
}
ArgSpec along(int id, AccessMode acc, std::vector<int64_t> strides) {
  ArgSpec a = dat(id, acc);
  a.strides = std::move(strides);
  return a;
}

LoopSpec loop(std::vector<ArgSpec> args, R range = kInner, int64_t block = 1) {
  LoopSpec l;
  l.args = std::move(args);
  l.range = std::move(range);
  l.block = block;
  return l;
}

FusionOptions unguarded() {
  FusionOptions o;
  o.allowGuarded = false;
  return o;
}
FusionOptions guarded(double ratio = 1.0) {
  FusionOptions o;
  o.allowGuarded = true;
  o.boxRatio = ratio;
  return o;
}
FusionOptions consecutive() {
  FusionOptions o;
  o.reorder = false;
  return o;
}
FusionOptions latest() {
  FusionOptions o;
  o.placement = FusionOptions::Placement::Latest;
  return o;
}

FusionPlan planOf(const std::vector<LoopSpec> &q, const FusionOptions &o = {}, unsigned rank = 2) {
  BuiltChain built = buildChain(q, rank);
  auto loops = extractLoops(built.chain);
  OPDSL_EXPECT(succeeded(loops));
  if (failed(loops)) return {};
  return planFusion(*loops, o);
}

std::string plan(const std::vector<LoopSpec> &q, const FusionOptions &o = {}) { return planOf(q, o).digest(); }

} // namespace

//===----------------------------------------------------------------------===//
// Ported from the legacy suite
//===----------------------------------------------------------------------===//

OPDSL_TEST(FusionPlanner, PointwiseChainFuses) {
  std::vector<LoopSpec> q = {loop({rd(1), wr(2)}), loop({rd(2), wr(3)}), loop({rd(3), wr(4)})};
  OPDSL_EXPECT_EQ(plan(q), "0,1,2");
}

OPDSL_TEST(FusionPlanner, StencilRawStaysSplit) {
  std::vector<LoopSpec> q = {loop({rd(1), wr(2)}), loop({rd(2, true), wr(3)})};
  OPDSL_EXPECT_EQ(plan(q), "0|1");
}

OPDSL_TEST(FusionPlanner, StencilWarJacobiStaysSplit) {
  std::vector<LoopSpec> q = {loop({rd(1, true), wr(2)}), loop({rd(2), wr(1)})};
  OPDSL_EXPECT_EQ(plan(q), "0|1");
}

OPDSL_TEST(FusionPlanner, TwoStencilReadersOfOneDatFuse) {
  std::vector<LoopSpec> q = {loop({rd(1, true), wr(2)}), loop({rd(1, true), wr(3)})};
  OPDSL_EXPECT_EQ(plan(q), "0,1");
}

OPDSL_TEST(FusionPlanner, WawAndZeroOffsetRawWarFuse) {
  std::vector<LoopSpec> q = {loop({wr(1)}), loop({wr(1)}), loop({rw(1)}), loop({rd(1), wr(2)}),
                             loop({rd(2), wr(1)})};
  OPDSL_EXPECT_EQ(plan(q), "0,1,2,3,4");
}

OPDSL_TEST(FusionPlanner, ReductionLoopIsABarrier) {
  std::vector<LoopSpec> q = {loop({wr(1)}), loop({rd(1), red()}), loop({wr(2)})};
  OPDSL_EXPECT_EQ(plan(q), "0|1|2");
}

OPDSL_TEST(FusionPlanner, ReadOnlyGlobalsFuse) {
  std::vector<LoopSpec> q = {loop({wr(1), global()}), loop({wr(2), global()})};
  OPDSL_EXPECT_EQ(plan(q), "0,1");
}

OPDSL_TEST(FusionPlanner, DifferentBlocksStaySplit) {
  std::vector<LoopSpec> q = {loop({wr(1)}, kInner, 1), loop({wr(2)}, kInner, 2)};
  OPDSL_EXPECT_EQ(plan(q), "0|1");
}

OPDSL_TEST(FusionPlanner, StridedReadAfterWriteStaysSplit) {
  std::vector<LoopSpec> q = {loop({wr(2)}), loop({along(2, AccessMode::Read, kBroadcast), wr(3)})};
  OPDSL_EXPECT_EQ(plan(q), "0|1");
}

OPDSL_TEST(FusionPlanner, LoopsOverDifferentAxesNeverShareAKernel) {
  std::vector<LoopSpec> mixed = {loop({along(1, AccessMode::Write, kBroadcast)}),
                                 loop({along(2, AccessMode::Write, kAlongY)})};
  OPDSL_EXPECT_EQ(plan(mixed), "0|1");
  std::vector<LoopSpec> same = {loop({along(1, AccessMode::Write, kBroadcast)}),
                                loop({along(2, AccessMode::Write, kBroadcast)})};
  OPDSL_EXPECT_EQ(plan(same), "0,1");
  std::vector<LoopSpec> full = {loop({along(1, AccessMode::Write, kBroadcast)}), loop({wr(2)})};
  OPDSL_EXPECT_EQ(plan(full), "0|1");
}

OPDSL_TEST(FusionPlanner, DifferentRangesSplitUnlessGuarded) {
  std::vector<LoopSpec> q = {loop({wr(1)}, kFull), loop({wr(1)}, kLeft)};
  OPDSL_EXPECT_EQ(plan(q, unguarded()), "0|1");
  OPDSL_EXPECT_EQ(plan(q), "0,1g");
  OPDSL_EXPECT_EQ(plan(q, guarded()), "0,1g");
  FusionPlan p = planOf(q, guarded());
  OPDSL_EXPECT(p.groups.size() == 1 && p.groups[0].guarded);
  OPDSL_EXPECT(p.groups[0].range == Range(kFull.begin(), kFull.end()));
}

OPDSL_TEST(FusionPlanner, SetupPlusBoundaryStripsFuse) {
  const R bottom = {-1, 11, -1, 0}, top = {-1, 11, 10, 11}, left = {-1, 0, -1, 11}, right = {10, 11, -1, 11};
  std::vector<LoopSpec> q = {loop({wr(1)}, kFull), loop({wr(1)}, bottom), loop({wr(1)}, top),
                             loop({wr(1)}, left), loop({wr(1)}, right)};
  OPDSL_EXPECT_EQ(plan(q, guarded()), "0,1,2,3,4g");
}

OPDSL_TEST(FusionPlanner, BoundingBoxRatioSplitsThinLoopsFarApart) {
  const R rowA = {0, 100, 0, 1}, rowB = {0, 100, 99, 100};
  std::vector<LoopSpec> q = {loop({wr(1)}, rowA), loop({wr(2)}, rowB)};
  OPDSL_EXPECT_EQ(plan(q, guarded(1.0)), "0|1");
  OPDSL_EXPECT_EQ(plan(q, guarded(100.0)), "0,1g");
}

OPDSL_TEST(FusionPlanner, GuardedStillRespectsRaw) {
  std::vector<LoopSpec> q = {loop({wr(1)}, kFull), loop({rd(1, true), wr(2)}, kInner)};
  OPDSL_EXPECT_EQ(plan(q, guarded()), "0|1");
}

OPDSL_TEST(FusionPlanner, StencilMemberCannotShrinkInsideABox) {
  // 5-point consumer over the interior + a full-grid writer: the writer grows the box past the consumer.
  std::vector<LoopSpec> q = {loop({rd(1, true), wr(2)}, kInner), loop({wr(3)}, kFull)};
  OPDSL_EXPECT_EQ(plan(q, guarded(4.0)), "0|1");
  // ...but a stencil member that IS the box is fine.
  std::vector<LoopSpec> q2 = {loop({wr(3)}, kInner), loop({rd(1, true), wr(2)}, kFull)};
  OPDSL_EXPECT_EQ(plan(q2, guarded(4.0)), "0,1g");
  // A later, bigger loop must not retroactively strand an earlier stencil member.
  std::vector<LoopSpec> q3 = {loop({rd(1, true), wr(2)}, kInner), loop({wr(3)}, kInner), loop({wr(4)}, kFull)};
  OPDSL_EXPECT_EQ(plan(q3, guarded(4.0)), "0,1|2");
}

OPDSL_TEST(FusionPlanner, GroupSizeCap) {
  std::vector<LoopSpec> q;
  for (int i = 0; i < 5; ++i) q.push_back(loop({wr(i + 1)}));
  FusionOptions o;
  o.maxGroupSize = 2;
  OPDSL_EXPECT_EQ(plan(q, o), "0,1|2,3|4");
}

OPDSL_TEST(FusionPlanner, DisabledGivesSingletons) {
  std::vector<LoopSpec> q = {loop({wr(1)}), loop({wr(2)})};
  FusionOptions o;
  o.enabled = false;
  OPDSL_EXPECT_EQ(plan(q, o), "0|1");
}

OPDSL_TEST(FusionPlanner, EmptyChain) { OPDSL_EXPECT_EQ(plan({}), ""); }

OPDSL_TEST(FusionPlanner, EachLoopAppearsOnceInOrder) {
  std::vector<LoopSpec> q = {loop({wr(1)}), loop({rd(1, true), wr(2)}), loop({rd(2), wr(3)}),
                             loop({rd(3), red()}), loop({wr(4)})};
  FusionPlan p = planOf(q, guarded());
  std::vector<size_t> flat;
  for (auto &g : p.groups) flat.insert(flat.end(), g.loops.begin(), g.loops.end());
  bool ok = flat.size() == q.size();
  for (size_t i = 0; ok && i < flat.size(); ++i) ok = flat[i] == i;
  OPDSL_EXPECT(ok);
}

//===----------------------------------------------------------------------===//
// Reordering (DAG planner)
//===----------------------------------------------------------------------===//

OPDSL_TEST(FusionPlanner, InterleavedChainsRunSideBySide) {
  // a1 a2 b1 b2 a3 b3: a_k reads a_{k-1} with 5 points, same for b.
  std::vector<LoopSpec> q = {
      loop({rd(1), wr(2)}),        // a1
      loop({rd(2, true), wr(3)}),  // a2
      loop({rd(11), wr(12)}),      // b1
      loop({rd(12, true), wr(13)}),// b2
      loop({rd(3, true), wr(4)}),  // a3
      loop({rd(13, true), wr(14)}) // b3
  };
  OPDSL_EXPECT_EQ(plan(q, consecutive()), "0|1,2|3,4|5");
  OPDSL_EXPECT_EQ(plan(q), "0,2|1,3|4,5");
  OPDSL_EXPECT_EQ(plan(q, latest()), "0|1,2|3,4|5");
  OPDSL_EXPECT_EQ(planOf(q, consecutive()).numReordered(), size_t(0));
  OPDSL_EXPECT(planOf(q, FusionOptions()).numReordered() > 0);
}

OPDSL_TEST(FusionPlanner, FootprintsNotWholeDatsDefineDependences) {
  const R strip = {-1, 0, -1, 11}, inner = {3, 10, 0, 10};
  std::vector<LoopSpec> q = {loop({wr(1)}, strip),                 // L0 strip
                             loop({rd(1, true), wr(2)}, inner),    // L1 reads dat1 far from the strip
                             loop({wr(3)}, strip)};                // L2 strip
  OPDSL_EXPECT_EQ(plan(q, consecutive()), "0|1|2");
  OPDSL_EXPECT_EQ(plan(q), "0,2|1");
  // The same read overlapping the strip IS a dependence.
  const R near = {0, 10, 0, 10};
  std::vector<LoopSpec> q2 = {loop({wr(1)}, strip), loop({rd(1, true), wr(2)}, near), loop({wr(3)}, strip)};
  OPDSL_EXPECT_EQ(plan(q2), "0,2|1");
  FusionPlan p2 = planOf(q2);
  OPDSL_EXPECT(p2.groups.size() == 2 && p2.groups[0].loops.front() == 0 && p2.groups[1].loops.front() == 1);
}

OPDSL_TEST(FusionPlanner, ReductionIsABarrierForReordering) {
  std::vector<LoopSpec> q = {loop({wr(1)}), loop({rd(1), red()}), loop({wr(2)})};
  OPDSL_EXPECT_EQ(plan(q), "0|1|2");
}

OPDSL_TEST(FusionPlanner, DependentLoopStaysBehindItsProducer) {
  std::vector<LoopSpec> q = {loop({wr(1)}),                  // 0: a
                             loop({rd(1, true), wr(2)}),     // 1: b = f(a)
                             loop({rd(2), wr(3)})};          // 2: c = g(b)
  OPDSL_EXPECT_EQ(plan(q), "0|1,2");
}

OPDSL_TEST(FusionPlanner, BlockedByReasons) {
  std::vector<LoopSpec> q = {loop({wr(1)}), loop({rd(1, true), wr(2)})};
  FusionPlan p = planOf(q);
  OPDSL_EXPECT_EQ(p.blockedBy["first"], size_t(1));
  OPDSL_EXPECT_EQ(p.blockedBy["dependence"], size_t(1));
  std::vector<LoopSpec> blocks = {loop({wr(1)}, kInner, 1), loop({wr(2)}, kInner, 2)};
  OPDSL_EXPECT_EQ(planOf(blocks).blockedBy["block"], size_t(1));
  FusionOptions o;
  o.maxGroupSize = 1;
  std::vector<LoopSpec> two = {loop({wr(1)}), loop({wr(2)})};
  OPDSL_EXPECT_EQ(planOf(two, o).blockedBy["size"], size_t(1));
  std::vector<LoopSpec> ranges = {loop({wr(1)}, kFull), loop({wr(1)}, kLeft)};
  OPDSL_EXPECT_EQ(planOf(ranges, unguarded()).blockedBy["range"], size_t(1));
}

OPDSL_TEST(FusionPlanner, TrafficEstimate) {
  // a = f(b); c = g(a): fused, `a` is neither loaded nor stored twice.
  std::vector<LoopSpec> q = {loop({rd(1), wr(2)}), loop({rd(2), wr(3)})};
  BuiltChain built = buildChain(q, 2);
  auto loops = extractLoops(built.chain);
  OPDSL_EXPECT(succeeded(loops));
  FusionPlan fused = planFusion(*loops, FusionOptions());
  FusionOptions off;
  off.enabled = false;
  FusionPlan split = planFusion(*loops, off);
  TrafficEstimate tf = estimateTraffic(*loops, fused), ts = estimateTraffic(*loops, split);
  const double plane = 100.0 * 8; // 10 x 10 points of f64
  OPDSL_EXPECT_EQ(ts.unfusedBytes, 4 * plane);
  OPDSL_EXPECT_EQ(ts.fusedBytes, 4 * plane);
  // fused: d1 loaded, d2 only stored (written before it is read), d3 stored
  OPDSL_EXPECT_EQ(tf.unfusedBytes, 4 * plane);
  OPDSL_EXPECT_EQ(tf.fusedBytes, 3 * plane);
}

//===----------------------------------------------------------------------===//
// Writes at non-zero offsets (TGV boundary kernels). The legacy planner assumed OPS writes are at offset 0.
//===----------------------------------------------------------------------===//

namespace {

// A boundary kernel as TGV has them: reads the interior side (+1, +2), writes the halo (-1, -2).
ArgSpec boundaryRW(int id) {
  ArgSpec a = rw(id);
  a.points = {{0, 0}, {1, 0}, {2, 0}, {-1, 0}, {-2, 0}};
  a.writes = Points{{-1, 0}, {-2, 0}};
  return a;
}

LoopSpec materialised(LoopSpec l) {
  l.materialize = true;
  return l;
}

const R kColumn = {0, 1, 0, 10}; // the first interior column

} // namespace

OPDSL_TEST(FusionPlanner, NonZeroWriteBlocksFusionWithAPointwiseReader) {
  // Both loops over the same range: fusing them is only blocked by the dependence.
  ArgSpec w = wr(1);
  w.points = {{-1, 0}};
  for (bool body : {false, true}) {
    LoopSpec a = loop({w}, kInner), b = loop({rd(1), wr(2)}, kInner);
    a.materialize = b.materialize = body;
    // writes d1 at -1; the next loop reads d1 at 0 over the same range: RAW overlap.
    OPDSL_EXPECT_EQ(plan({a, b}), "0|1");
    // and the other way round: WAR.
    OPDSL_EXPECT_EQ(plan({b, a}), "0|1");
    // WAW with a point-wise writer.
    LoopSpec c = loop({wr(1)}, kInner);
    c.materialize = body;
    OPDSL_EXPECT_EQ(plan({a, c}), "0|1");
    OPDSL_EXPECT_EQ(plan({c, a}), "0|1");
  }
}

OPDSL_TEST(FusionPlanner, NonZeroWriteBlocksFusionAlsoWhenGuardedAndBoxed) {
  ArgSpec w = wr(1);
  w.points = {{1, 0}};
  std::vector<LoopSpec> q = {loop({w}, kInner), loop({rd(1), wr(2)}, kInner)};
  OPDSL_EXPECT_EQ(plan(q, guarded(100.0)), "0|1");
  OPDSL_EXPECT_EQ(plan(q, consecutive()), "0|1");
}

OPDSL_TEST(FusionPlanner, DisjointFootprintsHaveNoDependenceDespiteNonZeroWrite) {
  // A single column: writing at -1 lands in the halo column, which the neighbour never touches.
  ArgSpec w = wr(1);
  w.points = {{-1, 0}};
  std::vector<LoopSpec> q = {loop({w}, kColumn), loop({rd(1), wr(2)}, kColumn)};
  OPDSL_EXPECT_EQ(plan(q), "0,1");
  // Moved one column inwards the footprints touch again.
  const R col1 = {1, 2, 0, 10};
  std::vector<LoopSpec> q2 = {loop({w}, col1), loop({rd(1), wr(2)}, R{0, 1, 0, 10})};
  OPDSL_EXPECT_EQ(plan(q2), "0|1");
}

OPDSL_TEST(FusionPlanner, BoundaryKernelBlocksALaterInteriorWriter) {
  // The boundary loop reads +1..+2 of d1; a later loop overwrites d1 on the interior: WAR across points.
  for (bool body : {false, true}) {
    LoopSpec a = loop({boundaryRW(1)}, kColumn);
    a.materialize = body;
    LoopSpec b = loop({wr(1)}, kInner);
    b.materialize = body;
    OPDSL_EXPECT_EQ(plan({a, b}), "0|1");
    OPDSL_EXPECT_EQ(plan({b, a}), "0|1"); // and the writer first: WAW + RAW with the boundary reads
  }
}

OPDSL_TEST(FusionPlanner, BoundaryKernelNextToAnIndependentInteriorReaderHasNoEdge) {
  // The boundary loop writes the halo only: an interior point-wise reader (other range) is independent.
  LoopSpec a = materialised(loop({boundaryRW(1)}, kColumn));
  LoopSpec b = loop({rd(1), wr(2)}, R{1, 10, 0, 10});
  BuiltChain built = buildChain({a, b}, 2);
  auto dep = ChainDependence::build(built.chain);
  OPDSL_EXPECT(succeeded(dep));
  // a reads [-2, 3) of d1: that overlaps what b reads, but reads do not conflict; b does not write d1.
  OPDSL_EXPECT_EQ(dep->edges.size(), size_t(0));
}

OPDSL_TEST(FusionPlanner, ReorderingPastANonZeroWriterNeedsDisjointFootprints) {
  ArgSpec w = wr(1);
  w.points = {{-1, 0}};
  // L0 writes d1 at -1 over the column; L1 reads d1 right at the halo; L2 is unrelated.
  const R halo = {-1, 0, 0, 10};
  std::vector<LoopSpec> dependent = {loop({w}, kColumn), loop({rd(1), wr(2)}, halo), loop({wr(3)}, kColumn)};
  // L1 depends on L0 (blocking): it cannot join; L2 hoists next to L0.
  OPDSL_EXPECT_EQ(plan(dependent, unguarded()), "0,2|1");
  // L1 reading far from the halo is independent of L0, whatever its range.
  const R far = {4, 8, 0, 10};
  std::vector<LoopSpec> independent = {loop({w}, kColumn), loop({rd(1), wr(2)}, far), loop({wr(3)}, kColumn)};
  OPDSL_EXPECT_EQ(plan(independent, unguarded()), "0,2|1");
  BuiltChain built = buildChain(independent, 2);
  auto dep = ChainDependence::build(built.chain);
  OPDSL_EXPECT(succeeded(dep));
  OPDSL_EXPECT_EQ(dep->edges.size(), size_t(0));
}

OPDSL_TEST(FusionPlanner, MaterialisedBodyNarrowsTheWriteOffsets) {
  // RW with a wide stencil but a body that writes at zero only: the write is point-local, so the pair fuses.
  ArgSpec a = rw(1);
  a.points = {{0, 0}, {1, 0}, {-1, 0}};
  a.writes = Points{{0, 0}};
  LoopSpec first = loop({a}, kInner), second = loop({rd(1), wr(2)}, kInner);
  first.materialize = true;
  OPDSL_EXPECT_EQ(plan({first, second}), "0,1");
  // The unmaterialised loop is judged by its whole stencil.
  first.materialize = false;
  OPDSL_EXPECT_EQ(plan({first, second}), "0|1");
}

//===----------------------------------------------------------------------===//
// Randomised oracle
//===----------------------------------------------------------------------===//
//
// Random 1-D programs over a few dats. Reference: run the loops in order. Candidate: run the plan's groups
// in order; inside a group visit each box point and, at each, every member whose range contains it, in
// member order, writing in place. Any planner bug (a wrongly fused WAR/RAW, a move across a dependence, a
// bad bounding box) shows up as different data. The loops of a program may read and write at non-zero
// offsets (when `extended`), and are tried both with their kernel bodies materialised (write offsets known
// exactly) and without (write offsets assumed to be the whole stencil).

namespace {

const std::vector<std::vector<int64_t>> kReadSets = {{0}, {-1, 0, 1}, {1, 0}, {-1, 0}};
const std::vector<std::vector<int64_t>> kWriteSets = {{0}, {1}, {-1}, {0, 1}};

struct OArg {
  int dat;
  AccessMode acc;
  int rs, ws; // index into kReadSets / kWriteSets
};
struct Prog {
  std::vector<std::vector<OArg>> loops;
  std::vector<std::pair<int, int>> ranges;
};

constexpr int kGrid = 14, kBase = 2; // dat cell index = x + kBase
constexpr int64_t kMod = 1000003;
using Grid = std::vector<std::vector<int64_t>>;

void execAt(const Prog &pr, size_t li, int x, Grid &g) {
  int64_t v = 7 * (li + 1);
  for (const OArg &a : pr.loops[li]) {
    if (a.acc == AccessMode::Write) continue;
    for (int64_t o : kReadSets[a.rs]) v = (v * 31 + g[a.dat][x + o + kBase]) % kMod;
  }
  for (const OArg &a : pr.loops[li]) {
    if (a.acc == AccessMode::Read) continue;
    for (int64_t o : kWriteSets[a.ws]) g[a.dat][x + o + kBase] = (v + 17 * a.dat + 5 * o + 5) % kMod;
  }
}

Grid freshGrid() {
  Grid g(4, std::vector<int64_t>(kGrid + 2 * kBase));
  for (int d = 0; d < 4; ++d)
    for (int x = 0; x < kGrid + 2 * kBase; ++x) g[d][x] = (d * 101 + x * 13 + 5) % kMod;
  return g;
}

std::vector<LoopSpec> toSpecs(const Prog &pr, bool materialize) {
  std::vector<LoopSpec> out;
  for (size_t i = 0; i < pr.loops.size(); ++i) {
    LoopSpec l;
    l.range = {pr.ranges[i].first, pr.ranges[i].second};
    l.materialize = materialize;
    for (const OArg &a : pr.loops[i]) {
      ArgSpec s;
      s.dat = a.dat;
      s.access = a.acc;
      // The stencil of the operand: the offsets it reads and the offsets it writes.
      std::vector<int64_t> offs;
      auto add = [&](const std::vector<int64_t> &set) {
        for (int64_t o : set)
          if (std::find(offs.begin(), offs.end(), o) == offs.end()) offs.push_back(o);
      };
      if (a.acc != AccessMode::Write) add(kReadSets[a.rs]);
      if (a.acc != AccessMode::Read) add(kWriteSets[a.ws]);
      for (int64_t o : offs) s.points.push_back({o});
      if (a.acc != AccessMode::Read) {
        Points w;
        for (int64_t o : kWriteSets[a.ws]) w.push_back({o});
        s.writes = w;
      }
      l.args.push_back(std::move(s));
    }
    out.push_back(std::move(l));
  }
  return out;
}

struct OracleStats {
  int bad = 0, reordered = 0, fusedLoops = 0, total = 0;
  std::string firstBad;
};

OracleStats runOracle(bool extended, int trials, unsigned seed) {
  std::mt19937 rng(seed);
  auto rnd = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
  OracleStats st;
  for (int trial = 0; trial < trials; ++trial) {
    Prog pr;
    int nloops = rnd(2, 12);
    for (int i = 0; i < nloops; ++i) {
      std::vector<OArg> args;
      int nargs = rnd(1, 3);
      std::map<int, bool> used;
      bool writes = false;
      for (int a = 0; a < nargs; ++a) {
        int dat = rnd(0, 3);
        if (used[dat]) continue;
        used[dat] = true;
        int mode = rnd(0, 2);
        AccessMode acc = mode == 0 ? AccessMode::Read : mode == 1 ? AccessMode::Write : AccessMode::RW;
        OArg arg{dat, acc, 0, 0};
        if (acc == AccessMode::Read) arg.rs = rnd(0, 3);
        if (extended) {
          if (acc != AccessMode::Read) arg.ws = rnd(0, 3);
          if (acc == AccessMode::RW) arg.rs = rnd(0, 3);
        }
        writes |= acc != AccessMode::Read;
        args.push_back(arg);
      }
      if (!writes) {
        int dat = rnd(0, 3);
        if (!used[dat]) args.push_back({dat, AccessMode::Write, 0, extended ? rnd(0, 3) : 0});
        else for (OArg &a : args) if (a.dat == dat && a.acc == AccessMode::Read) { a.acc = AccessMode::RW; }
      }
      if (!extended) {
        // OPS requires a dat written in a loop to be accessed point-locally.
        std::map<int, bool> wr;
        for (const OArg &a : args) if (a.acc != AccessMode::Read) wr[a.dat] = true;
        for (OArg &a : args) if (a.acc == AccessMode::Read && wr[a.dat]) a.rs = 0;
      }
      pr.loops.push_back(args);
      int lo = rnd(0, kGrid - 2), hi = rnd(lo + 1, kGrid);
      pr.ranges.push_back({lo, hi});
    }

    Grid ref = freshGrid();
    for (size_t li = 0; li < pr.loops.size(); ++li)
      for (int x = pr.ranges[li].first; x < pr.ranges[li].second; ++x) execAt(pr, li, x, ref);

    for (bool materialize : {false, true}) {
      BuiltChain built = buildChain(toSpecs(pr, materialize), 1);
      auto loops = extractLoops(built.chain);
      if (failed(loops)) { st.bad++; if (st.firstBad.empty()) st.firstBad = "extraction failed"; continue; }
      for (int variant = 0; variant < 6; ++variant) {
        FusionOptions o;
        o.reorder = variant % 2 == 0;
        o.allowGuarded = variant % 3 != 0;
        o.boxRatio = variant < 3 ? 1.0 : 50.0;
        o.maxGroupSize = variant == 5 ? 3 : 8;
        o.placement = variant == 4 ? FusionOptions::Placement::Latest : FusionOptions::Placement::Earliest;
        FusionPlan plan = planFusion(*loops, o);

        // Every loop exactly once, ascending within a group.
        std::vector<int> seen(loops->size(), 0);
        bool wellFormed = true;
        for (const FusedGroup &g : plan.groups)
          for (size_t k = 0; k < g.loops.size(); ++k) {
            wellFormed &= ++seen[g.loops[k]] == 1;
            wellFormed &= k == 0 || g.loops[k - 1] < g.loops[k];
          }
        for (int c : seen) wellFormed &= c == 1;

        Grid got = freshGrid();
        for (const FusedGroup &g : plan.groups) {
          int lo = g.range[0], hi = g.range[1];
          for (int x = lo; x < hi; ++x)
            for (size_t li : g.loops)
              if (x >= pr.ranges[li].first && x < pr.ranges[li].second) execAt(pr, li, x, got);
          for (size_t li : g.loops) wellFormed &= pr.ranges[li].first >= lo && pr.ranges[li].second <= hi;
          if (g.loops.size() > 1) st.fusedLoops += static_cast<int>(g.loops.size());
        }
        ++st.total;
        if (plan.numReordered() > 0) ++st.reordered;
        if (!wellFormed || got != ref) {
          if (!st.bad++)
            st.firstBad = "trial " + std::to_string(trial) + " variant " + std::to_string(variant) +
                          (materialize ? " materialised" : " unmaterialised") + " plan " + plan.digest() +
                          (wellFormed ? "" : " (malformed)");
        }
      }
    }
  }
  return st;
}

} // namespace

OPDSL_TEST(FusionPlanner, RandomisedOracleZeroOffsetWrites) {
  OracleStats st = runOracle(/*extended=*/false, 1500, 12345);
  OPDSL_EXPECT_EQ(st.bad, 0);
  if (st.bad) std::fprintf(stderr, "first mismatch: %s\n", st.firstBad.c_str());
  OPDSL_EXPECT(st.reordered > 100);
  OPDSL_EXPECT(st.fusedLoops > 1000);
}

OPDSL_TEST(FusionPlanner, RandomisedOracleNonZeroOffsetWrites) {
  OracleStats st = runOracle(/*extended=*/true, 3000, 777);
  OPDSL_EXPECT_EQ(st.bad, 0);
  if (st.bad) std::fprintf(stderr, "first mismatch: %s\n", st.firstBad.c_str());
  OPDSL_EXPECT(st.reordered > 100);
  OPDSL_EXPECT(st.fusedLoops > 500);
}
