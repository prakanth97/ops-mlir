#include "ops-mlir/Frontend/common/Manifest.h"

#include "opdsl/Support/Hash.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace ops_mlir::frontend {

using namespace mlir;
using namespace ops_mlir::ops;

std::string typeName(Type t) {
  if (t.isF64()) return "f64";
  if (t.isF32()) return "f32";
  if (t.isF16()) return "f16";
  if (t.isInteger(32)) return "i32";
  if (t.isInteger(64)) return "i64";
  std::string s;
  llvm::raw_string_ostream(s) << t;
  return s;
}

std::set<std::string> boundConstants(KernelOp kernel) {
  std::set<std::string> boundConsts;
  kernel.walk([&](ConstRefOp ref) {
    if (ref.getIndex()) boundConsts.insert(ref.getName().str());
    SmallVector<Value> work{ref.getResult()};
    DenseSet<Value> seen;
    while (!work.empty()) {
      Value v = work.pop_back_val();
      if (!seen.insert(v).second) continue;
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (auto loop = dyn_cast<scf::ForOp>(user)) {
          if (use.getOperandNumber() < 3) boundConsts.insert(ref.getName().str());
        } else if (auto other = dyn_cast<ConstRefOp>(user)) {
          if (other.getIndex() && use.get() == other.getIndex()) boundConsts.insert(ref.getName().str());
        } else if (isa<ReadDynOp>(user)) {
          boundConsts.insert(ref.getName().str());
        } else if (user->getDialect() && user->getDialect()->getNamespace() == "arith" && user->getNumResults() == 1) {
          work.push_back(user->getResult(0));
        }
      }
    }
  });
  return boundConsts;
}

std::string writeFailedManifest(const std::string &kernel, int ndim, const std::string &reason, const char *importer) {
  llvm::json::Object o;
  o["manifest_version"] = 1;
  o["kernel"] = kernel;
  o["status"] = "failed";
  o["reason"] = reason;
  o["importer"] = importer;
  o["ndim"] = ndim;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(o))) << "\n";
  return text;
}

std::string writeManifest(const KernelSummary &summary, KernelOp kernel, ModuleOp module, const char *importer) {
  std::set<std::string> boundConsts = boundConstants(kernel);

  auto offsetsJson = [](const std::set<std::vector<int64_t>> &set) {
    llvm::json::Array a;
    for (const auto &off : set) {
      llvm::json::Array o;
      for (int64_t v : off) o.push_back(v);
      a.push_back(std::move(o));
    }
    return a;
  };
  llvm::json::Array paramsJson;
  llvm::json::Object reads, writes;
  llvm::json::Array dynamicReads;
  for (auto [i, p] : llvm::enumerate(summary.params)) {
    llvm::json::Object o;
    switch (p.kind) {
    case ParamSummary::Acc:
      o["kind"] = "acc";
      o["elem"] = typeName(p.elem);
      o["ncomp"] = p.ncomp;
      o["const"] = p.isConst;
      o["legacy_macro"] = p.legacyMacro;
      reads[std::to_string(i)] = offsetsJson(p.reads);
      writes[std::to_string(i)] = offsetsJson(p.writes);
      if (p.dynamicRead) dynamicReads.push_back(std::to_string(i));
      break;
    case ParamSummary::Gbl:
      o["kind"] = "gbl";
      o["elem"] = typeName(p.elem);
      o["dim"] = p.gblDim;
      break;
    case ParamSummary::Red:
      o["kind"] = "reduce";
      o["elem"] = typeName(p.elem);
      o["dim"] = 1;
      o["style"] = p.style;
      o["kind_hint"] = p.kindHint.empty() ? "none" : p.kindHint;
      break;
    case ParamSummary::Idx: o["kind"] = "idx"; break;
    }
    paramsJson.push_back(std::move(o));
  }
  llvm::json::Array constsJson;
  for (const ConstUse &c : summary.consts) {
    llvm::json::Object o;
    o["name"] = c.name;
    o["elem"] = typeName(c.elem);
    o["offset"] = c.offset;
    if (c.dynamicIndex) {
      o["array_stride"] = c.stride;
      o["dynamic_index"] = true;
    }
    if (boundConsts.count(c.name)) o["role"] = "bound";
    constsJson.push_back(std::move(o));
  }
  llvm::json::Array idxAxes;
  for (int d = 0; d < 3; ++d)
    if (summary.idxAxesRead & (1u << d)) idxAxes.push_back(d);
  llvm::json::Object traits;
  traits["idx_axes_read"] = std::move(idxAxes);
  traits["dynamic_reads"] = std::move(dynamicReads);
  traits["dropped_printf"] = static_cast<int64_t>(summary.droppedPrintf);
  traits["prints"] = static_cast<int64_t>(summary.prints);
  traits["reads"] = std::move(reads);
  traits["writes"] = std::move(writes);

  llvm::json::Object manifest;
  manifest["manifest_version"] = 1;
  manifest["kernel"] = summary.kernel;
  manifest["status"] = "ok";
  manifest["importer"] = importer;
  manifest["ndim"] = summary.ndim;
  manifest["params"] = std::move(paramsJson);
  manifest["consts"] = std::move(constsJson);
  manifest["traits"] = std::move(traits);

  std::string moduleText;
  {
    llvm::raw_string_ostream ts(moduleText);
    module->print(ts);
  }
  std::string body;
  llvm::raw_string_ostream bs(body);
  bs << llvm::formatv("{0}", llvm::json::Value(llvm::json::Object(manifest)));
  manifest["digest"] = opdsl::Hasher().add(moduleText).add(body).add(importer).hex();

  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest))) << "\n";
  return text;
}

} // namespace ops_mlir::frontend
