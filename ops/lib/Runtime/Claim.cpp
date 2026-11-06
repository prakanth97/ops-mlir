#include "ops-mlir/Runtime/Claim.h"

#include "ops-mlir/Runtime/Options.h"

#include "ops_lazy_executor.h"

#include <algorithm>
#include <sstream>

namespace ops_mlir::runtime {

namespace {

const char *accessName(int acc) {
  switch (acc) {
  case OPS_READ: return "OPS_READ";
  case OPS_WRITE: return "OPS_WRITE";
  case OPS_RW: return "OPS_RW";
  case OPS_INC: return "OPS_INC";
  case OPS_MIN: return "OPS_MIN";
  case OPS_MAX: return "OPS_MAX";
  }
  return "?";
}

bool stencilContains(const ops_stencil_core &stencil, const std::vector<int64_t> &offset, int dims) {
  if ((int)offset.size() != dims) return false;
  for (int p = 0; p < stencil.points; ++p) {
    bool same = true;
    for (int d = 0; d < dims && same; ++d) same = stencil.stencil[p * stencil.dims + d] == offset[d];
    if (same) return true;
  }
  return false;
}

std::string offsetText(const std::vector<int64_t> &offset) {
  std::ostringstream os;
  os << "(";
  for (size_t d = 0; d < offset.size(); ++d) os << (d ? "," : "") << offset[d];
  os << ")";
  return os.str();
}

} // namespace

std::string claimReason(ops_kernel_descriptor *desc, const KernelArtifact &artifact, bool deviceSupported) {
  const Manifest &m = artifact.manifest;
  const int dims = desc->block->dims;
  if (dims > 3) return "the block has more than 3 dimensions";
  if (m.ndim != desc->dim) return "the artifact is for " + std::to_string(m.ndim) + "D blocks, the loop runs on " + std::to_string(desc->dim) + "D";
  if (desc->isdevice && !deviceSupported) return "loops that run on a device are not supported by this build of the JIT";
  if ((int)m.params.size() != desc->nargs)
    return "the kernel has " + std::to_string(m.params.size()) + " parameters but the loop passes " + std::to_string(desc->nargs) + " arguments";

  for (int i = 0; i < desc->nargs; ++i) {
    const ParamInfo &p = m.params[i];
    const ops_arg &a = desc->args[i];
    const std::string where = "argument " + std::to_string(i) + ": ";
    switch (p.kind) {
    case ParamInfo::Acc: {
      if (a.argtype != OPS_ARG_DAT) return where + "the kernel takes a dat accessor, the loop passes a global or index";
      // An optional argument that is switched off (a.opt == 0) is passed like any other: the kernel only touches it
      // under a run-time condition (CloverLeaf's `fields` mask), which the compiled code evaluates as well.
      ops_dat dat = a.dat;
      if (!dat) return where + "no dat";
      std::string elem = elemName(dat->type ? dat->type : "");
      if (elem.empty()) return where + "dat of type '" + std::string(dat->type ? dat->type : "?") + "' (the JIT handles double, float and int)";
      if (elem != p.elem) return where + "dat of type " + elem + " but the kernel's accessor is " + p.elem;
      if (!a.stencil) return where + "no stencil";
      if (a.stencil->type != 0) return where + "multigrid (restrict/prolong) stencil";
      // Edge dats (size 1 in a dimension, read with stride-0 stencils) are replicated and exchanged between processes by
      // OPS's MPI code: only on one process are they plain dats.
      if (dat->e_dat && ops_num_procs() > 1) return where + "edge dat (more than one process)";
      for (int d = 0; d < dims; ++d)
        if (dat->stride[d] > 1) return where + "coarse dat of a multigrid hierarchy";
      if (a.stencil->dims != dims) return where + "stencil of another dimension than the block";
      if (p.ncomp > dat->dim) return where + "the kernel uses " + std::to_string(p.ncomp) + " components of a dat that has " + std::to_string(dat->dim);
      for (int d = 0; d < dims; ++d) {
        if (a.stencil->stride[d] != 0 || a.acc == OPS_READ) continue;
        // Writing through a stride-0 dimension is one point per iteration only if the loop has one point in it (the
        // coordinates of CloverLeaf are initialised in loops over a single row).
        if (int64_t(desc->orig_range[2 * d + 1]) - desc->orig_range[2 * d] > 1)
          return where + "stride-0 dimension on a dat that is written, in a loop of more than one point in that dimension";
      }
      auto writes = m.writes.find(i);
      if (a.acc == OPS_READ && writes != m.writes.end() && !writes->second.empty()) return where + "the kernel writes a dat passed as OPS_READ";
      auto reads = m.reads.find(i);
      if (reads != m.reads.end())
        for (const auto &offset : reads->second)
          if (!stencilContains(*a.stencil, offset, dims)) return where + "the kernel reads at " + offsetText(offset) + " outside the stencil";
      if (writes != m.writes.end())
        for (const auto &offset : writes->second)
          if (!stencilContains(*a.stencil, offset, dims)) return where + "the kernel writes at " + offsetText(offset) + " outside the stencil";
      if (m.dynamicReads.count(i) && a.stencil->points == 0) return where + "data-dependent reads of a dat with an empty stencil";
      break;
    }
    case ParamInfo::Gbl: {
      if (a.argtype != OPS_ARG_GBL || a.acc != OPS_READ) return where + "the kernel takes a read-only global, the loop passes something else";
      if (a.elem_size != (int)elemBytes(p.elem)) return where + "global with " + std::to_string(a.elem_size) + "-byte elements, the kernel reads " + p.elem;
      if (a.dim < p.dim) return where + "global of " + std::to_string(a.dim) + " elements but the kernel reads " + std::to_string(p.dim);
      break;
    }
    case ParamInfo::Reduce: {
      if (a.argtype != OPS_ARG_GBL || (a.acc != OPS_INC && a.acc != OPS_MIN && a.acc != OPS_MAX))
        return where + "the kernel reduces, the loop passes an argument with access " + accessName(a.acc);
      if (a.dim != 1) return where + "reduction of " + std::to_string(a.dim) + " elements";
      // ops_arg_reduce does not set elem_size: the type is the handle's.
      ops_reduction handle = (ops_reduction)a.data;
      std::string handleElem = elemName(handle && handle->type ? handle->type : "");
      if (handleElem != p.elem) return where + "reduction of type " + std::string(handle && handle->type ? handle->type : "?") + ", the kernel reduces " + p.elem;
      if (p.style == "assign") {
        int64_t volume = 1;
        for (int d = 0; d < dims; ++d) volume *= std::max<int64_t>(0, int64_t(desc->orig_range[2 * d + 1]) - desc->orig_range[2 * d]);
        if (volume != 1) return where + "reduction written as an assignment in a loop of more than one point";
      }
      break;
    }
    case ParamInfo::Idx:
      if (a.argtype != OPS_ARG_IDX) return where + "the kernel takes the index, the loop passes something else";
      break;
    }
  }

  if (m.prints > 0 && options().printfPolicy == "decline") return "the kernel prints (OPS_JIT_PRINTF=decline)";
  for (const ConstInfo &c : m.consts) {
    if (c.dynamicIndex && !c.bound) return "constant '" + c.name + "' is indexed by a computed value and is not a loop bound";
    if (!ops_find_const(desc->block->instance, c.name.c_str()))
      return "constant '" + c.name + "' is not registered with ops_decl_const";
  }
  return "";
}

} // namespace ops_mlir::runtime
