#include "opdsl/Support/Hash.h"
#include "opdsl/Testing/TestHarness.h"

using opdsl::Hasher;
using opdsl::sha256Hex;

OPDSL_TEST(Hash, KnownVectors) {
  OPDSL_EXPECT_EQ(sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  OPDSL_EXPECT_EQ(sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

OPDSL_TEST(Hash, HasherFieldsAreUnambiguous) {
  OPDSL_EXPECT(Hasher().add("ab").add("c").hex() != Hasher().add("a").add("bc").hex());
  OPDSL_EXPECT(Hasher().addInt(1).addInt(23).hex() != Hasher().addInt(12).addInt(3).hex());
  OPDSL_EXPECT(Hasher().add("1").hex() != Hasher().addInt(1).hex());
}

OPDSL_TEST(Hash, HasherIsDeterministicAndReusable) {
  Hasher h;
  h.add("kernel").addInt(42).addBool(true);
  std::string first = h.hex();
  OPDSL_EXPECT_EQ(first, h.hex());
  OPDSL_EXPECT_EQ(first, Hasher().add("kernel").addInt(42).addBool(true).hex());
  OPDSL_EXPECT(h.add("more").hex() != first);
}
