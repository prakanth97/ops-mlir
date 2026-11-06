#include "opdsl/Pipeline/PipelineRegistry.h"
#include "opdsl/Testing/TestHarness.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Error.h"

using opdsl::OptionMap;
using opdsl::PipelineRegistry;

namespace {
std::string message(llvm::Error e) { return llvm::toString(std::move(e)); }
} // namespace

OPDSL_TEST(OptionMap, ParsesKeysAndValues) {
  auto opts = OptionMap::parse("omp=true  specialize=ranges,block=32 verbose");
  OPDSL_EXPECT(static_cast<bool>(opts));
  OPDSL_EXPECT(opts->getBool("omp", false));
  OPDSL_EXPECT_EQ(opts->getString("specialize"), "ranges");
  OPDSL_EXPECT_EQ(opts->getInt("block", 0), int64_t(32));
  OPDSL_EXPECT(opts->getBool("verbose", false)); // a bare key is true
  OPDSL_EXPECT(!opts->has("missing"));
  OPDSL_EXPECT_EQ(opts->getInt("missing", 7), int64_t(7));
}

OPDSL_TEST(OptionMap, EmptyAndErrors) {
  auto none = OptionMap::parse("");
  OPDSL_EXPECT(static_cast<bool>(none));
  OPDSL_EXPECT_EQ(none->canonical(), "");
  auto twice = OptionMap::parse("a=1 a=2");
  OPDSL_EXPECT(!twice);
  if (!twice) OPDSL_EXPECT_EQ(message(twice.takeError()), "option 'a' given twice");
  auto noName = OptionMap::parse("=3");
  OPDSL_EXPECT(!noName);
  if (!noName) llvm::consumeError(noName.takeError());
}

OPDSL_TEST(OptionMap, CanonicalFormIsOrderAndSpellingIndependent) {
  auto a = OptionMap::parse("b=yes a=2");
  auto b = OptionMap::parse("a=2,b=true");
  OPDSL_EXPECT(a && b);
  OPDSL_EXPECT_EQ(a->canonical(), "a=2 b=true");
  OPDSL_EXPECT_EQ(a->canonical(), b->canonical());
}

OPDSL_TEST(OptionMap, Validation) {
  std::vector<std::pair<std::string, std::string>> known = {{"omp", "bool"}, {"block", "int"}, {"mode", "string"}};
  auto ok = OptionMap::parse("omp=off block=8 mode=x");
  OPDSL_EXPECT(ok && !ok->validate(known));
  auto unknown = OptionMap::parse("ompp=1");
  OPDSL_EXPECT(static_cast<bool>(unknown));
  OPDSL_EXPECT_EQ(message(unknown->validate(known)), "unknown option 'ompp' (known: omp, block, mode)");
  auto badBool = OptionMap::parse("omp=maybe");
  OPDSL_EXPECT(static_cast<bool>(badBool));
  OPDSL_EXPECT_EQ(message(badBool->validate(known)), "option 'omp' expects true or false, got 'maybe'");
  auto badInt = OptionMap::parse("block=x");
  OPDSL_EXPECT(static_cast<bool>(badInt));
  OPDSL_EXPECT_EQ(message(badInt->validate(known)), "option 'block' expects an integer, got 'x'");
}

OPDSL_TEST(PipelineRegistry, BuildsByNameWithOptions) {
  PipelineRegistry registry;
  int built = 0;
  bool sawOmp = false;
  registry.add({"cpu", "a test pipeline", {{"omp", "bool", "use openmp"}},
                [&](mlir::OpPassManager &, const OptionMap &o) {
                  ++built;
                  sawOmp = o.getBool("omp", false);
                  return llvm::Error::success();
                }});
  OPDSL_EXPECT(registry.find("cpu") != nullptr);
  OPDSL_EXPECT(registry.find("gpu") == nullptr);

  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);
  OPDSL_EXPECT(!registry.build("cpu", "omp=true", pm));
  OPDSL_EXPECT_EQ(built, 1);
  OPDSL_EXPECT(sawOmp);
  // A bad option or an unknown pipeline: the builder does not run.
  std::string e1 = message(registry.build("cpu", "nope=1", pm));
  OPDSL_EXPECT(e1.find("unknown option 'nope'") != std::string::npos);
  OPDSL_EXPECT(e1.find("pipeline 'cpu'") != std::string::npos);
  std::string e2 = message(registry.build("gpu", "", pm));
  OPDSL_EXPECT_EQ(e2, "unknown pipeline 'gpu' (known: cpu)");
  OPDSL_EXPECT_EQ(built, 1);
}
