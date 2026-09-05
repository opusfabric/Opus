cmake_minimum_required(VERSION 3.22)
project(OpusAstraSimBuild LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILDTARGET "all" CACHE STRING "AstraSim targets to build")
set(NETWORK_BACKEND_BUILD_AS_LIBRARY ON CACHE BOOL "Build network backends as libraries" FORCE)

set(BACKEND_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
add_subdirectory("${BACKEND_ROOT}" astra_sim_core)
add_subdirectory("${BACKEND_ROOT}/extern/network_backend/analytical" analytical_network_backend)
add_subdirectory("${BACKEND_ROOT}/astra-sim/network_frontend/analytical" analytical_network_frontend)

# Generate the Chakra execution-trace bindings required by feeder_v3.
set(OPUS_PROTO_DIR "${BACKEND_ROOT}/extern/graph_frontend/chakra/schema/protobuf")
set(OPUS_PROTO_FILE "${OPUS_PROTO_DIR}/et_def.proto")
set(OPUS_PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/chakra_proto")
set(OPUS_PROTO_CC "${OPUS_PROTO_GEN_DIR}/et_def.pb.cc")
set(OPUS_PROTO_H "${OPUS_PROTO_GEN_DIR}/et_def.pb.h")
file(MAKE_DIRECTORY "${OPUS_PROTO_GEN_DIR}")

if(TARGET protobuf::protoc)
    set(OPUS_PROTOC "$<TARGET_FILE:protobuf::protoc>")
elseif(DEFINED Protobuf_PROTOC_EXECUTABLE)
    set(OPUS_PROTOC "${Protobuf_PROTOC_EXECUTABLE}")
elseif(DEFINED protobuf_PROTOC_EXECUTABLE)
    set(OPUS_PROTOC "${protobuf_PROTOC_EXECUTABLE}")
else()
    message(FATAL_ERROR "Could not locate protoc. Install protobuf-compiler or load a Protobuf module.")
endif()

add_custom_command(
    OUTPUT "${OPUS_PROTO_CC}" "${OPUS_PROTO_H}"
    COMMAND "${OPUS_PROTOC}"
            "--cpp_out=${OPUS_PROTO_GEN_DIR}"
            "-I${OPUS_PROTO_DIR}"
            "${OPUS_PROTO_FILE}"
    DEPENDS "${OPUS_PROTO_FILE}"
    VERBATIM
)

add_custom_target(opus_generate_chakra_proto DEPENDS "${OPUS_PROTO_CC}" "${OPUS_PROTO_H}")

foreach(target
        AstraSim
        AstraSim_CongestionAware
        AstraSim_Reconfigurable)
    target_sources(${target} PRIVATE "${OPUS_PROTO_CC}")
    add_dependencies(${target} opus_generate_chakra_proto)
    target_include_directories(${target} PUBLIC "${OPUS_PROTO_GEN_DIR}")
endforeach()

foreach(target
        AstraSim_Analytical_Congestion_Unaware
        AstraSim_Analytical_Congestion_Aware
        AstraSim_Analytical_Reconfigurable)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endforeach()
