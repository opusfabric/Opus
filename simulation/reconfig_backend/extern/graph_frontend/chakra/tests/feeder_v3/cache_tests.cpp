#include <gtest/gtest.h>
#include "cache.h"

using namespace Chakra::FeederV3;

class CacheTest : public ::testing::Test {
 protected:
  CacheTest() : cache(3) {} // Initialize cache with capacity 3

  Cache<int, int> cache;
};

TEST_F(CacheTest, PutAndGetTest) {
  int key = 1;
  int value = 100;
  cache.put(key, value);
  auto cached_value = cache.get_locked(key);
  ASSERT_EQ(*cached_value, value);
}

TEST_F(CacheTest, HasTest) {
  int key = 1;
  int value = 100;
  cache.put(key, value);
  ASSERT_TRUE(cache.has(key));
  ASSERT_FALSE(cache.has(2));
}

TEST_F(CacheTest, EvictionTest) {
  cache.put(1, 100);
  cache.put(2, 200);
  cache.put(3, 300);
  cache.put(4, 400); // This should evict key 1

  ASSERT_FALSE(cache.has(1));
  ASSERT_TRUE(cache.has(2));
  ASSERT_TRUE(cache.has(3));
  ASSERT_TRUE(cache.has(4));
}

TEST_F(CacheTest, RemoveTest) {
  int key = 1;
  int value = 100;
  cache.put(key, value);
  cache.remove(key);
  ASSERT_FALSE(cache.has(key));
}

TEST_F(CacheTest, GetOrNullTest) {
  int key = 1;
  int value = 100;
  cache.put(key, value);
  auto cached_value = cache.get_or_null_locked(key);
  ASSERT_TRUE(cached_value);
  ASSERT_EQ(*cached_value, value);

  auto null_value = cache.get_or_null_locked(2);
  ASSERT_FALSE(null_value);
}

TEST_F(CacheTest, UpdateValueTest) {
  int key = 1;
  int value1 = 100;
  int value2 = 200;
  cache.put(key, value1);
  cache.put(key, value2);
  auto cached_value = cache.get_locked(key);
  ASSERT_EQ(*cached_value, value2);
}

// int main(int argc, char** argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }