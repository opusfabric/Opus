#include <gtest/gtest.h>
#include <string>
#include "et_feeder.h"

class ETFeederTest : public ::testing::Test {
 protected:
  ETFeederTest() {}
  virtual ~ETFeederTest() {}

  void SetUp(const std::string& filename) {
    trace = new Chakra::FeederV3::ETFeeder(filename);
  }

  virtual void TearDown() {
    delete trace;
  }

  Chakra::FeederV3::ETFeeder* trace;
};

const static std::string kTestFile = "../../../tests/data/chakra.0.et";

TEST_F(ETFeederTest, ConstructorNodeIDTest) {
  SetUp(kTestFile);
  std::shared_ptr<Chakra::FeederV3::ETFeederNode> node =
      trace->getNextIssuableNode();

  std::unordered_set<uint64_t> free_node_ids{216, 432, 1634, 619, 541};
  uint64_t firstNodeID = node->id();
  ASSERT_TRUE(free_node_ids.find(firstNodeID) != free_node_ids.end());
  free_node_ids.erase(firstNodeID);

  node = trace->getNextIssuableNode();
  uint64_t secondNodeID = node->id();
  ASSERT_TRUE(free_node_ids.find(secondNodeID) != free_node_ids.end());
  free_node_ids.erase(secondNodeID);

  node = trace->getNextIssuableNode();
  secondNodeID = node->id();
  ASSERT_TRUE(free_node_ids.find(secondNodeID) != free_node_ids.end());
  free_node_ids.erase(secondNodeID);

  node = trace->getNextIssuableNode();
  secondNodeID = node->id();
  ASSERT_TRUE(free_node_ids.find(secondNodeID) != free_node_ids.end());
  free_node_ids.erase(secondNodeID);

  node = trace->getNextIssuableNode();
  secondNodeID = node->id();
  ASSERT_TRUE(free_node_ids.find(secondNodeID) != free_node_ids.end());
  free_node_ids.erase(secondNodeID);
}

TEST_F(ETFeederTest, ConstructorNodeValuesTest) {
  SetUp(kTestFile);
  std::shared_ptr<Chakra::FeederV3::ETFeederNode> node = trace->lookupNode(216);
  ChakraProtoMsg::NodeType firstNodeType = node->type(); 
  ASSERT_EQ(firstNodeType, ChakraProtoMsg::COMP_NODE);
  ASSERT_TRUE(node->is_cpu_op());

  std::string attr = "rf_id";
  ChakraProtoMsg::AttributeProto rf_id = node->get_attr_msg(attr);
  ASSERT_EQ(rf_id.int64_val(), 2);
  ASSERT_EQ(node->get_attr<int64_t>(attr), 2);

  node = trace->lookupNode(432);
  uint64_t secondNodeType = node->type();
  ASSERT_EQ(secondNodeType, ChakraProtoMsg::COMM_COLL_NODE);
  ASSERT_TRUE(node->is_cpu_op());

  rf_id = node->get_attr_msg(attr);
  ASSERT_EQ(rf_id.int64_val(), 110);
  ASSERT_EQ(node->get_attr<int64_t>(attr), 110);
}

TEST_F(ETFeederTest, ConstructorETFeederTest) {
  SetUp(kTestFile);
  std::shared_ptr<Chakra::FeederV3::ETFeederNode> node = this->trace->lookupNode(216);
  const auto& dependancy_resolver = this->trace->getDependancyResolver().get_enabled_dependancy();
  const auto& childs = dependancy_resolver.get_children(node->id());
  // ASSERT_EQ(childs.size(), 3);
  ASSERT_TRUE(childs.find(217) != childs.end());
  ASSERT_TRUE(childs.find(430) != childs.end());
  ASSERT_TRUE(childs.find(435) != childs.end());
}

TEST_F(ETFeederTest, FreeChildrenTest) {
  SetUp(kTestFile);
  std::shared_ptr<Chakra::FeederV3::ETFeederNode> node = trace->lookupNode(216);
  ASSERT_EQ(node->id(), 216);
  auto& dep_resolver = trace->getDependancyResolver();
  assert(dep_resolver.get_dependancy_free_nodes().find(216) != dep_resolver.get_dependancy_free_nodes().end());
  dep_resolver.take_node(216);
  dep_resolver.finish_node(216);
  ASSERT_TRUE(dep_resolver.get_dependancy_free_nodes().find(217) != dep_resolver.get_dependancy_free_nodes().end());
}

TEST_F(ETFeederTest, HasNodesToIssueTest) {
  SetUp(kTestFile);
  auto& dep_resolver = trace->getDependancyResolver();
  ASSERT_TRUE(dep_resolver.get_dependancy_free_nodes().find(216) != dep_resolver.get_dependancy_free_nodes().end());
  dep_resolver.take_node(216);
  ASSERT_TRUE(trace->hasNodesToIssue());
}

TEST_F(ETFeederTest, PushBackIssuableNodeTest) {
  SetUp(kTestFile);
  auto& dep_resolver = trace->getDependancyResolver();
  ASSERT_TRUE(dep_resolver.get_dependancy_free_nodes().find(216) != dep_resolver.get_dependancy_free_nodes().end());
  dep_resolver.take_node(216);
  ASSERT_TRUE(dep_resolver.get_dependancy_free_nodes().find(216) == dep_resolver.get_dependancy_free_nodes().end());
  trace->pushBackIssuableNode(216);
  ASSERT_TRUE(dep_resolver.get_dependancy_free_nodes().find(216) != dep_resolver.get_dependancy_free_nodes().end());
}

TEST_F(ETFeederTest, NodeGetChildrenTest) {
  SetUp(kTestFile);
  auto& dep_resolver = trace->getDependancyResolver();
  ASSERT_TRUE(dep_resolver.get_dependancy_free_nodes().find(216) != dep_resolver.get_dependancy_free_nodes().end());
  auto& children = dep_resolver.get_enabled_dependancy().get_children(216);
  ASSERT_TRUE(children.find(217) != children.end());
  ASSERT_TRUE(children.find(435) != children.end());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}