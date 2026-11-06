#include "ops-mlir/Analysis/FusionPlanner.h"

#include <algorithm>
#include <set>

namespace ops_mlir::ops {

std::string FusionPlan::digest() const {
  std::string out;
  for (size_t g = 0; g < groups.size(); ++g) {
    if (g) out += '|';
    for (size_t k = 0; k < groups[g].loops.size(); ++k) {
      if (k) out += ',';
      out += std::to_string(groups[g].loops[k]);
    }
    if (groups[g].guarded) out += 'g';
  }
  return out;
}

size_t FusionPlan::numReordered() const {
  size_t position = 0, moved = 0;
  for (const FusedGroup &g : groups)
    for (size_t l : g.loops) moved += (l != position++);
  return moved;
}

namespace {

//===----------------------------------------------------------------------===//
// Shared rules
//===----------------------------------------------------------------------===//

// Loops can only share a kernel if they iterate over the same space.
bool sameSpace(const LoopInfo &a, const LoopInfo &b) {
  return a.block == b.block && a.rank == b.rank && a.axes == b.axes;
}

// A member of a group, for the guarded-range rule. A guarded member is skipped outside its own range; it is
// kept honest by never letting a member that reaches across points (read or write) be smaller than the box.
struct Member {
  Range range;
  bool nonLocal;
};

struct GroupState {
  Range range;
  double memberVolume = 0.0;
  bool guarded = false;
  std::vector<Member> members;
};

// Decides whether a loop with range `loopRange` can join a group whose state is `g`, and what the group's
// box/guarded flag become. Different ranges need guarded fusion enabled, a bounding box not much larger than
// the members' volumes, and every member smaller than the box to be point-local.
bool rangeJoin(const GroupState &g, const Range &loopRange, bool loopNonLocal, const FusionOptions &options,
               Range &boxOut, bool &guardedOut) {
  boxOut = g.range;
  guardedOut = g.guarded;
  if (loopRange == g.range) return true;
  if (!options.allowGuarded) return false;
  boxOut = boundingBox(g.range, loopRange);
  guardedOut = true;
  if (volume(boxOut) > options.boxRatio * (g.memberVolume + volume(loopRange))) return false;
  if (loopNonLocal && loopRange != boxOut) return false;
  for (const Member &m : g.members)
    if (m.nonLocal && m.range != boxOut) return false;
  return true;
}

void addMember(GroupState &g, const Range &loopRange, bool nonLocal, Range box, bool guarded) {
  g.range = std::move(box);
  g.guarded = guarded;
  g.memberVolume += volume(loopRange);
  g.members.push_back({loopRange, nonLocal});
}

//===----------------------------------------------------------------------===//
// Consecutive planner: conflicts are decided per dat, not per footprint
//===----------------------------------------------------------------------===//

// Legality of appending a loop with accesses `next` to a group with accesses `group`.
bool compatible(const DatAccesses &group, const DatAccesses &next) {
  for (const auto &[dat, n] : next) {
    auto it = group.find(dat);
    if (it == group.end()) continue;
    const DatAccess &g = it->second;
    if (g.writes && n.reads && n.readNonLocal) return false; // RAW across points
    if (g.readNonLocal && n.writes) return false;            // WAR across points
    if (g.writeNonLocal || n.writeNonLocal) return false;    // writes at non-zero offsets stay ordered
  }
  return true;
}

void mergeAccesses(DatAccesses &group, const DatAccesses &next) {
  for (const auto &[dat, n] : next) {
    DatAccess &g = group[dat];
    g.reads |= n.reads;
    g.writes |= n.writes;
    g.readNonLocal |= n.readNonLocal;
    g.writeNonLocal |= n.writeNonLocal;
  }
}

FusionPlan planConsecutive(llvm::ArrayRef<LoopInfo> loops, const FusionOptions &options) {
  FusionPlan plan;
  DatAccesses groupAcc;
  GroupState state;
  bool open = false;

  auto startGroup = [&](size_t i) {
    FusedGroup g;
    g.loops.push_back(i);
    g.range = loops[i].range;
    plan.groups.push_back(std::move(g));
    groupAcc = loops[i].accesses();
    state = GroupState();
    state.range = loops[i].range;
    state.memberVolume = volume(loops[i].range);
    state.members = {{loops[i].range, loops[i].hasNonLocalAccess()}};
    open = true;
  };

  for (size_t i = 0; i < loops.size(); ++i) {
    const LoopInfo &loop = loops[i];

    // A loop with a reduction is never fused with its neighbours.
    if (!options.enabled || loop.hasReduction) {
      open = false;
      FusedGroup g;
      g.loops.push_back(i);
      g.range = loop.range;
      plan.groups.push_back(std::move(g));
      continue;
    }
    if (!open) {
      startGroup(i);
      continue;
    }

    FusedGroup &group = plan.groups.back();
    bool ok = sameSpace(loop, loops[group.loops.front()]) && group.loops.size() < options.maxGroupSize;
    DatAccesses nextAcc;
    Range box;
    bool guarded = false;
    if (ok) {
      nextAcc = loop.accesses();
      ok = compatible(groupAcc, nextAcc) &&
           rangeJoin(state, loop.range, loop.hasNonLocalAccess(), options, box, guarded);
    }
    if (!ok) {
      startGroup(i);
      continue;
    }
    group.loops.push_back(i);
    group.range = box;
    group.guarded = guarded;
    addMember(state, loop.range, loop.hasNonLocalAccess(), box, guarded);
    mergeAccesses(groupAcc, nextAcc);
  }
  return plan;
}

//===----------------------------------------------------------------------===//
// DAG planner
//===----------------------------------------------------------------------===//

// One group being built by the DAG planner.
struct DagGroup {
  std::vector<size_t> loops;
  GroupState state;
  std::vector<int> succ; // groups that must run after this one
  bool barrier = false;
};

FusionPlan planDag(llvm::ArrayRef<LoopInfo> loops, const FusionOptions &options) {
  const size_t n = loops.size();
  ChainDependence dep = ChainDependence::build(std::vector<LoopInfo>(loops.begin(), loops.end()));

  std::map<std::string, size_t> blockedBy;
  std::vector<DagGroup> groups;
  std::vector<int> groupOf(n, -1);
  int lastBarrier = -1;
  size_t firstJoinable = 0; // groups before a barrier are sealed

  // The transitive closure of the group graph, one bit row per group (a group is never more than the loop count), kept up
  // to date when an edge is added: the placement test asks `reaches` for every loop and every candidate group, and a graph
  // search per question made long chains (hundreds of loops between two reductions) cost seconds.
  const size_t words = (n + 63) / 64;
  std::vector<std::vector<uint64_t>> below; // below[g]: the groups strictly after g
  auto test = [&](const std::vector<uint64_t> &row, int bit) { return (row[bit / 64] >> (bit % 64)) & 1; };
  auto addEdge = [&](int from, int to) {
    if (from == to) return;
    auto &s = groups[from].succ;
    if (std::find(s.begin(), s.end(), to) == s.end()) s.push_back(to);
    if (test(below[from], to)) return; // already implied
    for (size_t x = 0; x < groups.size(); ++x) {
      if (static_cast<int>(x) != from && !test(below[x], from)) continue;
      below[x][to / 64] |= uint64_t(1) << (to % 64);
      for (size_t w = 0; w < words; ++w) below[x][w] |= below[to][w];
    }
  };
  auto reaches = [&](int from, int to) { return from == to || test(below[from], to); };
  auto newGroup = [&](size_t j) {
    below.emplace_back(words, uint64_t(0));
    DagGroup g;
    g.loops.push_back(j);
    g.state.range = loops[j].range;
    g.state.memberVolume = volume(loops[j].range);
    g.state.members = {{loops[j].range, loops[j].hasNonLocalAccess()}};
    groups.push_back(std::move(g));
    return static_cast<int>(groups.size()) - 1;
  };

  for (size_t j = 0; j < n; ++j) {
    const LoopInfo &loop = loops[j];
    const bool nonLocal = loop.hasNonLocalAccess();

    if (loop.hasReduction) {
      // Barrier: nothing moves across it.
      int b = newGroup(j);
      groups[b].barrier = true;
      for (int h = 0; h < b; ++h) addEdge(h, b);
      groupOf[j] = b;
      lastBarrier = b;
      firstJoinable = groups.size();
      continue;
    }

    int chosen = -1;
    Range chosenBox;
    bool chosenGuarded = false;
    std::set<std::string> whyNot; // reasons for this loop, for diagnostics
    for (size_t g = firstJoinable; g < groups.size(); ++g) {
      const DagGroup &grp = groups[g];
      if (grp.barrier) continue;
      if (grp.loops.size() >= options.maxGroupSize) {
        whyNot.insert("size");
        continue;
      }
      if (!sameSpace(loop, loops[grp.loops.front()])) {
        whyNot.insert("block");
        continue;
      }
      bool ok = true;
      for (size_t ei : dep.predecessors[j]) {
        const DepEdge &e = dep.edges[ei];
        int h = groupOf[e.from];
        if (h == static_cast<int>(g)) {
          if (!e.fusable) {
            ok = false;
            whyNot.insert("dependence");
            break;
          }
        } else if (reaches(static_cast<int>(g), h)) {
          ok = false; // would put j before something that must run before it
          whyNot.insert("order");
          break;
        }
      }
      if (!ok) continue;
      Range box;
      bool guarded = false;
      if (!rangeJoin(grp.state, loop.range, nonLocal, options, box, guarded)) {
        whyNot.insert("range");
        continue;
      }
      chosen = static_cast<int>(g);
      chosenBox = std::move(box);
      chosenGuarded = guarded;
      if (options.placement == FusionOptions::Placement::Earliest) break; // otherwise keep scanning for the latest
    }

    if (chosen >= 0) {
      DagGroup &g = groups[chosen];
      g.loops.push_back(j);
      addMember(g.state, loop.range, nonLocal, chosenBox, chosenGuarded);
    } else {
      chosen = newGroup(j);
      if (whyNot.empty()) ++blockedBy["first"];
      for (const std::string &r : whyNot) ++blockedBy[r];
    }
    groupOf[j] = chosen;
    for (size_t ei : dep.predecessors[j]) addEdge(groupOf[dep.edges[ei].from], chosen);
    if (lastBarrier >= 0) addEdge(lastBarrier, chosen);
  }

  // Emission: topological order, ties broken by program order.
  std::vector<int> indegree(groups.size(), 0);
  for (const DagGroup &g : groups)
    for (int s : g.succ) ++indegree[s];
  std::set<std::pair<size_t, int>> ready; // (first loop, group)
  for (size_t g = 0; g < groups.size(); ++g)
    if (indegree[g] == 0) ready.insert({groups[g].loops.front(), static_cast<int>(g)});

  FusionPlan plan;
  plan.blockedBy = blockedBy;
  while (!ready.empty()) {
    int g = ready.begin()->second;
    ready.erase(ready.begin());
    FusedGroup out;
    out.loops = groups[g].loops;
    out.guarded = groups[g].state.guarded;
    out.range = groups[g].state.range;
    plan.groups.push_back(std::move(out));
    for (int s : groups[g].succ)
      if (--indegree[s] == 0) ready.insert({groups[s].loops.front(), s});
  }
  // A cycle in the group graph would strand groups (and their loops). The placement checks rule that out; if
  // it ever happens, fall back to the always-correct adjacent grouping rather than drop work.
  if (plan.groups.size() != groups.size()) return planConsecutive(loops, options);
  return plan;
}

} // namespace

TrafficEstimate estimateTraffic(llvm::ArrayRef<LoopInfo> loops, const FusionPlan &plan) {
  TrafficEstimate t;
  for (const FusedGroup &g : plan.groups) {
    // Per dat: loaded if its first access in the kernel is a read; stored if any member writes it.
    struct Use {
      bool firstIsRead;
      bool written;
      double bytes;
    };
    std::map<int, Use> uses;
    for (size_t l : g.loops) {
      const LoopInfo &loop = loops[l];
      for (const ArgInfo &arg : loop.args) {
        double loopBytes = volume(loop.range) * static_cast<double>(arg.elemBytes);
        // OPS convention: read or write-only = 1 pass, read-write = 2.
        t.unfusedBytes += loopBytes * ((arg.access == AccessMode::Read || arg.access == AccessMode::Write) ? 1 : 2);
        auto it = uses.find(arg.dat);
        if (it == uses.end()) it = uses.emplace(arg.dat, Use{arg.reads, false, 0}).first;
        it->second.written |= arg.writes;
        it->second.bytes = volume(g.range) * static_cast<double>(arg.elemBytes);
      }
    }
    for (const auto &[dat, u] : uses) t.fusedBytes += u.bytes * ((u.firstIsRead ? 1 : 0) + (u.written ? 1 : 0));
  }
  return t;
}

FusionPlan planFusion(llvm::ArrayRef<LoopInfo> loops, const FusionOptions &options) {
  if (!options.enabled || !options.reorder) return planConsecutive(loops, options);
  return planDag(loops, options);
}

} // namespace ops_mlir::ops
