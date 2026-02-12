# Makefile for controller

# Compiler settings for controller
CXX ?= g++
CXXFLAGS ?= -g -O0 -std=c++17 -Wall -Wextra

# CUDA settings for gpu target
# CUDA_HOME ?= /usr/local/cuda-12.6
# NVCC      ?= $(CUDA_HOME)/bin/nvcc
# CUDA_INC  ?= $(CUDA_HOME)/include
# CUDA_LIB  ?= $(CUDA_HOME)/lib64

# NCCL_PATH ?= /home/$(USER)/nccl/build

TARGETS := controller

all: $(TARGETS)

controller: controller.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< \
	-lpthread

clean:
	rm -f $(TARGETS)


