#include "ops-mlir/Runtime/CallDump.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

namespace ops_mlir::runtime {

std::string checksumHex(const void *data, size_t n) {
  uint64_t h = 1469598103934665603ull;
  const unsigned char *p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h);
  return buf;
}

namespace {

/// Bytes of the allocation a memref covers: the largest extent over its dimensions.
size_t allocationBytes(const PackedArgs::MemrefInfo &m, const std::string &elem) {
  int64_t elems = 0;
  for (size_t i = 0; i < m.sizes.size(); ++i) elems = std::max<int64_t>(elems, m.sizes[i] * m.strides[i]);
  return static_cast<size_t>(elems) * sizeOfElem(elem);
}

std::string hexOf(const void *bytes, size_t n) {
  static const char *digits = "0123456789abcdef";
  std::string out;
  const unsigned char *p = static_cast<const unsigned char *>(bytes);
  for (size_t i = 0; i < n; ++i) {
    out += digits[p[i] >> 4];
    out += digits[p[i] & 15];
  }
  return out;
}

bool bytesOfHex(llvm::StringRef hex, void *out, size_t n) {
  if (hex.size() != 2 * n) return false;
  unsigned char *p = static_cast<unsigned char *>(out);
  for (size_t i = 0; i < n; ++i) {
    unsigned v;
    if (hex.substr(2 * i, 2).getAsInteger(16, v)) return false;
    p[i] = static_cast<unsigned char>(v);
  }
  return true;
}

bool writeFile(const std::string &path, const void *data, size_t n, std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    error = "cannot write " + path + ": " + ec.message();
    return false;
  }
  os.write(static_cast<const char *>(data), n);
  return true;
}

} // namespace

CallRecorder::CallRecorder(std::string dir, std::string entry, std::string pipeline, std::string pipelineOptions, const PackedArgs &args,
                           std::vector<std::string> datElems)
    : dir(std::move(dir)), entry(std::move(entry)), pipeline(std::move(pipeline)), pipelineOptions(std::move(pipelineOptions)),
      args(args), datElems(std::move(datElems)) {
  llvm::sys::fs::create_directories(this->dir);
  std::string error;
  inputsWritten = true;
  for (size_t i = 0; i < args.memrefInfos().size(); ++i) {
    const PackedArgs::MemrefInfo &m = args.memrefInfos()[i];
    if (!writeFile(this->dir + "/" + this->entry + ".dat" + std::to_string(i) + ".bin", m.data, allocationBytes(m, this->datElems[i]), error)) {
      std::fprintf(stderr, "ops_jit: %s\n", error.c_str());
      inputsWritten = false;
    }
  }
}

bool CallRecorder::finish(std::string &error) {
  if (!inputsWritten) {
    error = "the inputs could not be written";
    return false;
  }
  llvm::json::Object root;
  root["version"] = 1;
  root["entry"] = entry;
  root["pipeline"] = pipeline;
  root["pipeline_options"] = pipelineOptions;
  llvm::json::Array dats;
  for (size_t i = 0; i < args.memrefInfos().size(); ++i) {
    const PackedArgs::MemrefInfo &m = args.memrefInfos()[i];
    llvm::json::Object d;
    d["elem"] = datElems[i];
    llvm::json::Array sizes, strides;
    for (int64_t v : m.sizes) sizes.push_back(v);
    for (int64_t v : m.strides) strides.push_back(v);
    d["sizes"] = std::move(sizes);
    d["strides"] = std::move(strides);
    size_t bytes = allocationBytes(m, datElems[i]);
    d["bytes"] = static_cast<int64_t>(bytes);
    d["input"] = entry + ".dat" + std::to_string(i) + ".bin";
    d["checksum"] = checksumHex(m.data, bytes);
    dats.push_back(std::move(d));
  }
  root["dats"] = std::move(dats);
  llvm::json::Array indices;
  for (int64_t v : args.indexArgs()) indices.push_back(v);
  root["indices"] = std::move(indices);
  llvm::json::Array scalars;
  for (const PackedArgs::ScalarInfo &s : args.scalarInfos()) {
    llvm::json::Object o;
    o["elem"] = s.elem;
    o["hex"] = hexOf(s.bytes.data(), sizeOfElem(s.elem));
    scalars.push_back(std::move(o));
  }
  root["scalars"] = std::move(scalars);
  llvm::json::Array results;
  for (size_t i = 0; i < args.resultElems().size(); ++i) {
    llvm::json::Object o;
    o["elem"] = args.resultElems()[i];
    o["value"] = args.resultAsDouble(i);
    results.push_back(std::move(o));
  }
  root["results"] = std::move(results);

  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
  return writeFile(dir + "/" + entry + ".call.json", text.data(), text.size(), error);
}

bool loadCall(const std::string &dir, const std::string &entry, LoadedCall &out, std::string &error) {
  auto text = llvm::MemoryBuffer::getFile(dir + "/" + entry + ".call.json");
  if (!text) {
    error = "cannot read " + dir + "/" + entry + ".call.json";
    return false;
  }
  auto parsed = llvm::json::parse((*text)->getBuffer());
  if (!parsed) {
    error = "the call record is not valid JSON: " + llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root || root->getInteger("version").value_or(0) != 1) {
    error = "unknown call record version";
    return false;
  }
  out.entry = root->getString("entry").value_or("").str();
  out.pipeline = root->getString("pipeline").value_or("").str();
  out.pipelineOptions = root->getString("pipeline_options").value_or("").str();

  const llvm::json::Array *dats = root->getArray("dats");
  if (!dats) {
    error = "no dats in the call record";
    return false;
  }
  out.buffers.clear();
  out.buffers.resize(dats->size());
  size_t index = 0;
  for (const llvm::json::Value &dv : *dats) {
    const llvm::json::Object *d = dv.getAsObject();
    std::string elem = d->getString("elem").value_or("").str();
    std::vector<int64_t> sizes, strides;
    for (const llvm::json::Value &v : *d->getArray("sizes")) sizes.push_back(v.getAsInteger().value_or(0));
    for (const llvm::json::Value &v : *d->getArray("strides")) strides.push_back(v.getAsInteger().value_or(0));
    auto file = llvm::MemoryBuffer::getFile(dir + "/" + d->getString("input").value_or("").str());
    if (!file) {
      error = "cannot read the input of dat " + std::to_string(index);
      return false;
    }
    out.buffers[index].assign((*file)->getBufferStart(), (*file)->getBufferEnd());
    out.args.addMemref(out.buffers[index].data(), sizes, strides);
    out.datElems.push_back(elem);
    out.expectedChecksums.push_back(d->getString("checksum").value_or("").str());
    ++index;
  }
  for (const llvm::json::Value &v : *root->getArray("indices")) out.args.addIndex(v.getAsInteger().value_or(0));
  for (const llvm::json::Value &sv : *root->getArray("scalars")) {
    const llvm::json::Object *s = sv.getAsObject();
    std::string elem = s->getString("elem").value_or("").str();
    char bytes[8] = {0};
    if (!bytesOfHex(s->getString("hex").value_or(""), bytes, sizeOfElem(elem))) {
      error = "bad scalar in the call record";
      return false;
    }
    out.args.addScalar(elem, bytes);
  }
  std::vector<std::string> resultElems;
  for (const llvm::json::Value &rv : *root->getArray("results")) {
    const llvm::json::Object *r = rv.getAsObject();
    resultElems.push_back(r->getString("elem").value_or("").str());
    out.expectedResults.push_back(r->getNumber("value").value_or(0));
  }
  out.args.setResults(resultElems);
  out.args.finalize();
  return true;
}

} // namespace ops_mlir::runtime
