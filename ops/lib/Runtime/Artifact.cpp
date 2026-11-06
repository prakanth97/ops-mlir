#include "ops-mlir/Runtime/Artifact.h"

#include "ops_jit_artifact.h"

#include "mlir/Parser/Parser.h"

#include "llvm/Support/JSON.h"

namespace ops_mlir::runtime {

std::string elemName(const std::string &opsType) {
  if (opsType == "double" || opsType == "real(8)") return "f64";
  if (opsType == "float" || opsType == "real(4)") return "f32";
  if (opsType == "int" || opsType == "integer(4)") return "i32";
  if (opsType == "half" || opsType == "_Float16" || opsType == "real(2)") return "f16";
  return "";
}

unsigned elemBytes(const std::string &name) {
  if (name == "f64" || name == "i64") return 8;
  if (name == "f32" || name == "i32") return 4;
  if (name == "f16" || name == "bf16" || name == "i16") return 2;
  return 0;
}

namespace {

Offsets decodeOffsets(const llvm::json::Array *a) {
  Offsets out;
  if (!a) return out;
  for (const llvm::json::Value &o : *a) {
    std::vector<int64_t> point;
    if (const llvm::json::Array *p = o.getAsArray())
      for (const llvm::json::Value &v : *p) point.push_back(v.getAsInteger().value_or(0));
    out.push_back(std::move(point));
  }
  return out;
}

} // namespace

bool parseManifest(const std::string &text, Manifest &m, std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
  if (!parsed) {
    error = "the manifest is not valid JSON: " + llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    error = "the manifest is not a JSON object";
    return false;
  }
  if (root->getInteger("manifest_version").value_or(0) != 1) {
    error = "unknown manifest version";
    return false;
  }
  if (root->getString("status").value_or("") != "ok") {
    error = "kernel not imported: " + root->getString("reason").value_or("no reason recorded").str();
    return false;
  }
  m.kernel = root->getString("kernel").value_or("").str();
  m.digest = root->getString("digest").value_or("").str();
  m.importer = root->getString("importer").value_or("").str();
  m.ndim = static_cast<int>(root->getInteger("ndim").value_or(0));

  if (const llvm::json::Array *params = root->getArray("params")) {
    for (const llvm::json::Value &pv : *params) {
      const llvm::json::Object *p = pv.getAsObject();
      if (!p) continue;
      ParamInfo info;
      llvm::StringRef kind = p->getString("kind").value_or("");
      info.kind = kind == "acc" ? ParamInfo::Acc : kind == "gbl" ? ParamInfo::Gbl : kind == "reduce" ? ParamInfo::Reduce : ParamInfo::Idx;
      info.elem = p->getString("elem").value_or("").str();
      info.ncomp = p->getInteger("ncomp").value_or(1);
      info.dim = p->getInteger("dim").value_or(1);
      info.isConst = p->getBoolean("const").value_or(false);
      info.style = p->getString("style").value_or("").str();
      m.params.push_back(std::move(info));
    }
  }
  if (const llvm::json::Array *consts = root->getArray("consts")) {
    for (const llvm::json::Value &cv : *consts) {
      const llvm::json::Object *c = cv.getAsObject();
      if (!c) continue;
      ConstInfo info;
      info.name = c->getString("name").value_or("").str();
      info.elem = c->getString("elem").value_or("").str();
      info.offset = c->getInteger("offset").value_or(0);
      info.dynamicIndex = c->getBoolean("dynamic_index").value_or(false);
      info.arrayStride = c->getInteger("array_stride").value_or(0);
      info.bound = c->getString("role").value_or("") == "bound";
      m.consts.push_back(std::move(info));
    }
  }
  if (const llvm::json::Object *traits = root->getObject("traits")) {
    if (const llvm::json::Object *reads = traits->getObject("reads"))
      for (const auto &entry : *reads) {
        int param;
        if (!llvm::StringRef(entry.first).getAsInteger(10, param)) m.reads[param] = decodeOffsets(entry.second.getAsArray());
      }
    if (const llvm::json::Object *writes = traits->getObject("writes"))
      for (const auto &entry : *writes) {
        int param;
        if (!llvm::StringRef(entry.first).getAsInteger(10, param)) m.writes[param] = decodeOffsets(entry.second.getAsArray());
      }
    if (const llvm::json::Array *dyn = traits->getArray("dynamic_reads"))
      for (const llvm::json::Value &v : *dyn) {
        int param;
        if (auto s = v.getAsString(); s && !s->getAsInteger(10, param)) m.dynamicReads.insert(param);
      }
    if (const llvm::json::Array *axes = traits->getArray("idx_axes_read"))
      for (const llvm::json::Value &v : *axes) m.idxAxesRead.push_back(static_cast<int>(v.getAsInteger().value_or(0)));
    m.droppedPrintf = static_cast<int>(traits->getInteger("dropped_printf").value_or(0));
    m.prints = static_cast<int>(traits->getInteger("prints").value_or(0));
  }
  return true;
}

const KernelArtifact *ArtifactCache::get(const ops_jit_artifact *artifact, std::string &reason) {
  if (!artifact) {
    reason = "no JIT artifact (the code generator did not embed one)";
    return nullptr;
  }
  auto cached = byPointer.find(artifact);
  if (cached != byPointer.end()) {
    reason = cached->second->error;
    return cached->second->error.empty() ? cached->second.get() : nullptr;
  }
  auto entry = std::make_unique<KernelArtifact>();
  KernelArtifact *raw = entry.get();
  byPointer[artifact] = std::move(entry);

  if (artifact->version != OPS_JIT_ARTIFACT_VERSION) {
    raw->error = "the artifact has version " + std::to_string(artifact->version) + ", this library reads version " +
                 std::to_string(OPS_JIT_ARTIFACT_VERSION);
  } else if (!artifact->manifest_json) {
    raw->error = "the artifact has no manifest";
  } else if (!parseManifest(artifact->manifest_json, raw->manifest, raw->error)) {
    // raw->error set by parseManifest
  } else if (artifact->status != 0 || !artifact->bytecode) {
    raw->error = "the importer did not produce an IR for " + std::string(artifact->kernel_name ? artifact->kernel_name : "?");
  } else {
    // Kernels of equal digest share the parsed module.
    auto known = byDigest.find(raw->manifest.digest);
    if (known != byDigest.end()) {
      raw->module = known->second->module ? mlir::OwningOpRef<mlir::ModuleOp>(llvm::cast<mlir::ModuleOp>(known->second->module->clone()))
                                          : mlir::OwningOpRef<mlir::ModuleOp>();
    } else {
      llvm::StringRef bytes(reinterpret_cast<const char *>(artifact->bytecode), artifact->bytecode_size);
      raw->module = mlir::parseSourceString<mlir::ModuleOp>(bytes, &ctx);
    }
    if (!raw->module) {
      raw->error = "the kernel IR of " + raw->manifest.kernel + " does not parse";
    } else {
      raw->kernel = ops::KernelOp();
      for (ops::KernelOp k : raw->module->getOps<ops::KernelOp>()) raw->kernel = k;
      if (!raw->kernel) raw->error = "the kernel IR of " + raw->manifest.kernel + " has no ops.kernel";
      else byDigest.emplace(raw->manifest.digest, raw);
    }
  }
  reason = raw->error;
  return raw->error.empty() ? raw : nullptr;
}

} // namespace ops_mlir::runtime
