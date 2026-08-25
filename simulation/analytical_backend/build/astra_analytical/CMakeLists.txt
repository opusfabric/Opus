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

foreach(target
        AstraSim_Analytical_Congestion_Unaware
        AstraSim_Analytical_Congestion_Aware
        AstraSim_Analytical_Reconfigurable)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endforeach()
