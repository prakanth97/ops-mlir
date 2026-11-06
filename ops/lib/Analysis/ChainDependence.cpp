#include "ops-mlir/Analysis/ChainDependence.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;

namespace ops_mlir::ops {

//===----------------------------------------------------------------------===//
// LoopInfo
//===----------------------------------------------------------------------===//

DatAccesses LoopInfo::accesses() const {
  DatAccesses out;
  for (const ArgInfo &a : args) {
    DatAccess &d = out[a.dat];
    if (a.reads) {
      d.reads = true;
      d.readNonLocal |= a.readNonLocal;
      d.readBox = boundingBox(d.readBox, a.readBox);
    }
    if (a.writes) {
      d.writes = true;
      d.writeNonLocal |= a.writeNonLocal;
      d.writeBox = boundingBox(d.writeBox, a.writeBox);
    }
  }
  return out;
}

bool LoopInfo::hasNonLocalRead() const {
  return llvm::any_of(args, [](const ArgInfo &a) { return a.readNonLocal; });
}

bool LoopInfo::hasNonLocalAccess() const {
  return llvm::any_of(args, [](const ArgInfo &a) { return a.readNonLocal || a.writeNonLocal; });
}

//===----------------------------------------------------------------------===//
// Extraction
//===----------------------------------------------------------------------===//

namespace {

/// Reach of the offsets `loop` writes accessor `i` at. The offsets of the `ops.write`s of the body when it is
/// materialised, otherwise (and when the body never writes the accessor) every point of the stencil.
Reach writeReach(ParLoopOp loop, unsigned i) {
  StencilAttr stencil = loop.getStencil(i);
  if (!loop.isBodyEmpty()) {
    Reach reach(stencil.getRank());
    Value acc = loop.getAccessorArg(i);
    loop.getBody().walk([&](WriteOp w) {
      if (w.getAcc() == acc) reach.addOffset(w.getOffsets());
    });
    if (reach.any) {
      reach.setStrides(stencil.getStrides());
      return reach;
    }
  }
  return Reach::of(stencil);
}

} // namespace

llvm::FailureOr<std::vector<LoopInfo>> extractLoops(ChainOp chain) {
  auto ranges = chain->getAttrOfType<ArrayAttr>("ops.range_values");
  llvm::DenseMap<Value, int> datIds;
  int nextExtraId = chain.getBody().front().getNumArguments();
  std::vector<LoopInfo> out;

  for (Operation &op : chain.getBody().front()) {
    auto loop = dyn_cast<ParLoopOp>(&op);
    if (!loop) continue;
    LoopInfo info;
    info.op = loop;
    info.block = static_cast<int64_t>(loop.getBlock());
    info.rank = loop.getRank();
    info.hasReduction = loop->getNumResults() > 0;

    // Concrete range.
    size_t idx = out.size();
    DenseI64ArrayAttr range;
    if (ranges && idx < ranges.size()) range = dyn_cast<DenseI64ArrayAttr>(ranges[idx]);
    if (!range) {
      loop.emitError("no concrete range: the chain needs an `ops.range_values` entry for loop ") << idx;
      return failure();
    }
    if (range.size() != 2 * info.rank) {
      loop.emitError("`ops.range_values` entry ") << idx << " has " << range.size() << " bounds, expected "
                                                  << 2 * info.rank;
      return failure();
    }
    info.range.assign(range.asArrayRef().begin(), range.asArrayRef().end());

    for (unsigned i = 0, n = loop.getNumDats(); i < n; ++i) {
      Value dat = loop.getDats()[i];
      ArgInfo arg;
      auto [it, inserted] = datIds.try_emplace(dat, 0);
      if (inserted) it->second = isa<BlockArgument>(dat) ? cast<BlockArgument>(dat).getArgNumber() : nextExtraId++;
      arg.dat = it->second;
      arg.access = loop.getAccess(i);
      auto datType = cast<DatType>(dat.getType());
      arg.elemBytes = (datType.getElementType().getIntOrFloatBitWidth() / 8) * datType.getNcomp();

      StencilAttr stencil = loop.getStencil(i);
      for (int64_t d = 0; d < stencil.getRank(); ++d)
        if (stencil.getStrides()[d] != 0) info.axes |= 1u << d;

      arg.reads = arg.access == AccessMode::Read || arg.access == AccessMode::RW;
      arg.writes = arg.access != AccessMode::Read;
      if (arg.reads) {
        Reach reach = Reach::of(stencil);
        arg.readNonLocal = !reach.pointLocal;
        arg.readBox = footprint(info.range, reach);
      }
      if (arg.writes) {
        Reach reach = writeReach(loop, i);
        arg.writeNonLocal = !reach.pointLocal;
        arg.writeBox = footprint(info.range, reach);
      }
      info.args.push_back(std::move(arg));
    }
    out.push_back(std::move(info));
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Edges
//===----------------------------------------------------------------------===//

ChainDependence ChainDependence::build(std::vector<LoopInfo> loops) {
  ChainDependence dep;
  dep.loops = std::move(loops);
  const size_t n = dep.loops.size();
  dep.predecessors.assign(n, {});

  std::vector<DatAccesses> acc(n);
  std::map<int, std::vector<size_t>> history; // dat -> loops touching it, ascending
  for (size_t j = 0; j < n; ++j) {
    acc[j] = dep.loops[j].accesses();
    std::map<size_t, DepEdge> edge; // from -> edge
    for (const auto &[dat, aj] : acc[j]) {
      for (size_t i : history[dat]) {
        const DatAccess &ai = acc[i].at(dat);
        unsigned kinds = 0;
        if (ai.writes && aj.reads && overlaps(ai.writeBox, aj.readBox)) kinds |= RAW;
        if (ai.reads && aj.writes && overlaps(ai.readBox, aj.writeBox)) kinds |= WAR;
        if (ai.writes && aj.writes && overlaps(ai.writeBox, aj.writeBox)) kinds |= WAW;
        if (!kinds) continue;
        bool fusable = true;
        if ((kinds & RAW) && aj.readNonLocal) fusable = false; // j reads across points what i wrote
        if ((kinds & WAR) && ai.readNonLocal) fusable = false; // i read across points what j overwrites
        if (ai.writeNonLocal || aj.writeNonLocal) fusable = false;
        auto [it, inserted] = edge.try_emplace(i);
        DepEdge &e = it->second;
        if (inserted) {
          e.from = i;
          e.to = j;
        }
        e.kinds |= kinds;
        e.fusable = e.fusable && fusable;
        e.dats.push_back(dat); // dats ascend: acc[j] is ordered by dat
      }
      history[dat].push_back(j);
    }
    for (auto &[from, e] : edge) {
      dep.predecessors[j].push_back(dep.edges.size());
      dep.edges.push_back(std::move(e));
    }
  }
  return dep;
}

llvm::FailureOr<ChainDependence> ChainDependence::build(ChainOp chain) {
  auto loops = extractLoops(chain);
  if (failed(loops)) return failure();
  return build(std::move(*loops));
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

static void printRange(llvm::raw_ostream &os, llvm::ArrayRef<int64_t> r) {
  os << '[';
  llvm::interleaveComma(r, os);
  os << ']';
}

void ChainDependence::print(llvm::raw_ostream &os) const {
  for (size_t j = 0; j < loops.size(); ++j) {
    const LoopInfo &l = loops[j];
    os << "loop " << j;
    if (l.op)
      if (auto k = l.op->getAttrOfType<FlatSymbolRefAttr>("kernel")) os << " @" << k.getValue();
    os << ": block " << l.block << " range ";
    printRange(os, l.range);
    if (l.hasReduction) os << " reduction";
    os << '\n';
    for (const auto &[dat, a] : l.accesses()) {
      os << "  d" << dat << ':';
      if (a.reads) os << " read " << (a.readNonLocal ? "nonlocal" : "local");
      if (a.writes) os << " write " << (a.writeNonLocal ? "nonlocal" : "local");
      os << '\n';
    }
  }
  for (const DepEdge &e : edges) {
    os << "edge " << e.from << " -> " << e.to << ':';
    if (e.kinds & RAW) os << " raw";
    if (e.kinds & WAR) os << " war";
    if (e.kinds & WAW) os << " waw";
    os << " {";
    for (size_t k = 0; k < e.dats.size(); ++k) os << (k ? ", d" : "d") << e.dats[k];
    os << "} " << (e.fusable ? "fusable" : "blocking") << '\n';
  }
}

} // namespace ops_mlir::ops
