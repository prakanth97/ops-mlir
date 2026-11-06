#include "opdsl/Pipeline/PipelineRegistry.h"

#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <optional>

namespace opdsl {

namespace {

llvm::Error fail(const llvm::Twine &message) { return llvm::createStringError(llvm::errc::invalid_argument, message); }

std::optional<bool> parseBool(llvm::StringRef v) {
  std::string s = v.lower();
  if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
  if (s == "false" || s == "0" || s == "no" || s == "off") return false;
  return std::nullopt;
}

bool isBoolWord(llvm::StringRef v) {
  std::string s = v.lower();
  return s == "true" || s == "false" || s == "yes" || s == "no" || s == "on" || s == "off";
}

} // namespace

llvm::Expected<OptionMap> OptionMap::parse(llvm::StringRef text) {
  OptionMap map;
  llvm::SmallVector<llvm::StringRef> items;
  size_t start = 0;
  for (size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || std::isspace(static_cast<unsigned char>(text[i])) || text[i] == ',') {
      if (i > start) items.push_back(text.slice(start, i));
      start = i + 1;
    }
  }
  for (llvm::StringRef item : items) {
    auto [key, value] = item.split('=');
    if (key.empty()) return fail("empty option name in '" + text + "'");
    bool hasValue = item.contains('=');
    if (!map.values.emplace(key.str(), hasValue ? value.str() : "true").second)
      return fail("option '" + key + "' given twice");
  }
  return map;
}

std::string OptionMap::getString(llvm::StringRef key, llvm::StringRef fallback) const {
  auto it = values.find(key.str());
  return it == values.end() ? fallback.str() : it->second;
}

bool OptionMap::getBool(llvm::StringRef key, bool fallback) const {
  auto it = values.find(key.str());
  if (it == values.end()) return fallback;
  return parseBool(it->second).value_or(fallback);
}

int64_t OptionMap::getInt(llvm::StringRef key, int64_t fallback) const {
  auto it = values.find(key.str());
  if (it == values.end()) return fallback;
  int64_t v;
  return llvm::StringRef(it->second).getAsInteger(10, v) ? fallback : v;
}

llvm::Error OptionMap::validate(llvm::ArrayRef<std::pair<std::string, std::string>> known) const {
  for (const auto &[key, value] : values) {
    auto it = std::find_if(known.begin(), known.end(), [&](const auto &k) { return k.first == key; });
    if (it == known.end()) {
      std::string names;
      for (const auto &k : known) names += (names.empty() ? "" : ", ") + k.first;
      return fail("unknown option '" + key + "'" + (names.empty() ? " (there are none)" : " (known: " + names + ")"));
    }
    if (it->second == "bool" && !parseBool(value))
      return fail("option '" + key + "' expects true or false, got '" + value + "'");
    int64_t ignored;
    if (it->second == "int" && llvm::StringRef(value).getAsInteger(10, ignored))
      return fail("option '" + key + "' expects an integer, got '" + value + "'");
  }
  return llvm::Error::success();
}

std::string OptionMap::canonical() const {
  std::string out;
  for (const auto &[key, value] : values) {
    if (!out.empty()) out += ' ';
    out += key + "=" + (isBoolWord(value) ? (*parseBool(value) ? "true" : "false") : value);
  }
  return out;
}

void PipelineRegistry::add(Entry entry) {
  std::string name = entry.name;
  bool inserted = entriesByName.emplace(name, std::move(entry)).second;
  assert(inserted && "pipeline registered twice");
  (void)inserted;
}

const PipelineRegistry::Entry *PipelineRegistry::find(llvm::StringRef name) const {
  auto it = entriesByName.find(name.str());
  return it == entriesByName.end() ? nullptr : &it->second;
}

std::vector<const PipelineRegistry::Entry *> PipelineRegistry::entries() const {
  std::vector<const Entry *> out;
  for (const auto &[name, entry] : entriesByName) out.push_back(&entry);
  return out;
}

llvm::Error PipelineRegistry::build(llvm::StringRef name, llvm::StringRef optionText, mlir::OpPassManager &pm) const {
  const Entry *entry = find(name);
  if (!entry) {
    std::string names;
    for (const auto &[n, e] : entriesByName) names += (names.empty() ? "" : ", ") + n;
    return fail("unknown pipeline '" + name + "' (known: " + names + ")");
  }
  llvm::Expected<OptionMap> options = OptionMap::parse(optionText);
  if (!options) return options.takeError();
  std::vector<std::pair<std::string, std::string>> known;
  for (const Option &o : entry->options) known.push_back({o.name, o.kind});
  if (llvm::Error e = options->validate(known)) return llvm::joinErrors(fail("pipeline '" + name + "'"), std::move(e));
  return entry->build(pm, *options);
}

PipelineRegistry &PipelineRegistry::global() {
  static PipelineRegistry registry;
  return registry;
}

} // namespace opdsl
