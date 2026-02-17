# Makefile for building and running tests

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -I/usr/include/gtest -I/usr/include -I../../src/feeder_v3 -I../../schema/protobuf -O0 -g

# Directories
SRC_DIR = ../../src/feeder_v3
PROTO_DIR = ../../schema/protobuf
TEST_DIR = ./
BUILD_DIR = ./build

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
PROTO_FILES = $(wildcard $(PROTO_DIR)/*.cc)
TEST_FILES = $(wildcard $(TEST_DIR)/*.cpp)

# Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))
PROTO_OBJ_FILES = $(patsubst $(PROTO_DIR)/%.cc,$(BUILD_DIR)/%.o,$(PROTO_FILES))
TEST_OBJ_FILES = $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(TEST_FILES))

# Target executable
TARGET = $(BUILD_DIR)/run_tests

# Google Test library
GTEST_LIBS = -lgtest -lgtest_main -pthread
PROTOBUF_LIBS = -lprotobuf

# Default target
all: $(CLEAN) $(TARGET)

# Build target
$(TARGET): $(OBJ_FILES) $(TEST_OBJ_FILES) $(PROTO_OBJ_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(GTEST_LIBS) $(PROTOBUF_LIBS)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile protobuf files
$(BUILD_DIR)/%.o: $(PROTO_DIR)/%.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

# Run tests
test: $(TARGET)
	./$(TARGET)

.PHONY: all clean test
