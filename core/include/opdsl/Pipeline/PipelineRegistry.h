//===- PipelineRegistry.h - named pass pipelines with option strings -*- C++ -*-===//
//
// A DSL's JIT compiles a group of loops by running a pipeline chosen by name ("cpu", "cuda", ...) with options
// given as a string, from an environment variable or a tool's command line:
//
//     key=value key=value        (separators: blanks and commas)
//
// The registry maps a name to a function that fills an mlir::OpPassManager. Options are parsed into an OptionMap
// whose canonical form (sorted, normalised) goes into compiled-code cache keys. Nothing here knows about a DSL.
//
//===----------------------------------------------------------------------===//

#ifndef OPDSL_PIPELINE_PIPELINEREGISTRY_H
#define OPDSL_PIPELINE_PIPELINEREGISTRY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
class OpPassManager;
}

namespace opdsl {

/// Parsed `key=value` options.
class OptionMap {
public:
  /// Parses `text`. A key without `=value` means `key=true`. A repeated key is an error.
  static llvm::Expected<OptionMap> parse(llvm::StringRef text);

  bool has(llvm::StringRef key) const { return values.count(key.str()) != 0; }
  std::string getString(llvm::StringRef key, llvm::StringRef fallback = "") const;
  /// true/false/1/0/yes/no/on/off; the fallback for a missing or malformed value (call `validate` first).
  bool getBool(llvm::StringRef key, bool fallback) const;
  int64_t getInt(llvm::StringRef key, int64_t fallback) const;

  /// Fails, naming the first key that is not in `known` (pairs of name and kind: "bool", "int" or "string"), or a
  /// value that does not parse for the kind its key is declared with.
  llvm::Error validate(llvm::ArrayRef<std::pair<std::string, std::string>> known) const;

  /// Sorted `k=v k=v` with the spellings of booleans normalised to true/false: equal option sets give equal strings.
  std::string canonical() const;

private:
  std::map<std::string, std::string> values;
};

class PipelineRegistry {
public:
  using Builder = std::function<llvm::Error(mlir::OpPassManager &, const OptionMap &)>;
  struct Option {
    std::string name, kind /* bool|int|string */, description;
  };
  struct Entry {
    std::string name, description;
    std::vector<Option> options;
    Builder build;
  };

  /// Registers a pipeline; the name must be new.
  void add(Entry entry);
  const Entry *find(llvm::StringRef name) const;
  std::vector<const Entry *> entries() const;

  /// Parses and validates `optionText` against the entry's declared options and appends the pipeline to `pm`.
  llvm::Error build(llvm::StringRef name, llvm::StringRef optionText, mlir::OpPassManager &pm) const;

  /// The registry of the process.
  static PipelineRegistry &global();

private:
  std::map<std::string, Entry> entriesByName;
};

} // namespace opdsl

#endif // OPDSL_PIPELINE_PIPELINEREGISTRY_H
