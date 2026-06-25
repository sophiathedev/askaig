cmake_minimum_required(VERSION 3.28.2)
project(askaig)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized Release build unless the caller picks a build type explicitly.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(askaig
    src/types.cpp
    src/tables.cpp
    src/position.cpp
    src/eval.cpp
    src/tt.cpp
    src/search.cpp
    src/tune.cpp
    src/uci.cpp
    src/wdl.cpp
    src/kpk.cpp
    src/main.cpp)

target_include_directories(askaig PRIVATE src)

find_package(Threads REQUIRED)
target_link_libraries(askaig PRIVATE Threads::Threads)

# Build-date version. CI passes -DASKAIG_VERSION=<YYYYMMDD> (shared across all targets, == the release
# tag); a plain local build auto-stamps the configure date. uci.cpp falls back to "dev" if undefined.
if(NOT DEFINED ASKAIG_VERSION)
    string(TIMESTAMP ASKAIG_VERSION "%Y%m%d" UTC)
endif()
target_compile_definitions(askaig PRIVATE ASKAIG_VERSION="${ASKAIG_VERSION}")
message(STATUS "Askaig version: ${ASKAIG_VERSION}")

# SIMD / hardware bit-intrinsics. ARCH auto-detects from the host (override with -DARCH=...).
option(SIMD "Enable SIMD/hardware bit intrinsics" ON)
set(ARCH "" CACHE STRING "Target SIMD arch: AVX2 or ARM_NEON (auto-detected when empty)")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

if(SIMD)
    if(ARCH STREQUAL "")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
            set(ARCH "ARM_NEON")
        else()
            set(ARCH "AVX2")
        endif()
    endif()

    target_compile_definitions(askaig PRIVATE SIMD)
    if(ARCH STREQUAL "AVX2")
        target_compile_definitions(askaig PRIVATE ARCH_AVX2)
        target_compile_options(askaig PRIVATE -mavx2 -mbmi -mbmi2 -mpopcnt) # -mbmi2 -> PEXT for sliding attacks
    elseif(ARCH STREQUAL "ARM_NEON")
        target_compile_definitions(askaig PRIVATE ARCH_ARM_NEON)  # NEON is baseline on arm64
    else()
        message(FATAL_ERROR "Unknown ARCH '${ARCH}' (expected AVX2 or ARM_NEON)")
    endif()
    message(STATUS "SIMD enabled, ARCH=${ARCH}")
endif()

target_compile_options(askaig PRIVATE -flto)
target_link_options(askaig PRIVATE -flto)

# Profile-guided optimization (two-stage; drive it with the pgo-generate / pgo-use presets, or
# tools/build_pgo.sh which orchestrates the train+merge between them):
#   -DPGO=generate                              -> instrumented build (writes .profraw when run)
#   -DPGO=use  [-DPGO_PROFILE=<merged.profdata>] -> rebuild optimised with that profile
# Empty (default) = a normal build. PGO is behaviour-neutral — it only improves code layout / speed,
# never what the engine searches. Clang-only here (Apple/LLVM clang; merged with llvm-profdata).
set(PGO "" CACHE STRING "Profile-guided optimization: 'generate', 'use', or empty (off)")
set(PGO_PROFILE "${CMAKE_BINARY_DIR}/pgo.profdata" CACHE FILEPATH "merged profile data for -DPGO=use")
if(NOT PGO STREQUAL "")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "PGO here is wired for Clang (-fprofile-instr-*); compiler is ${CMAKE_CXX_COMPILER_ID}")
    endif()
    if(PGO STREQUAL "generate")
        target_compile_options(askaig PRIVATE -fprofile-instr-generate)
        target_link_options(askaig PRIVATE -fprofile-instr-generate)
    elseif(PGO STREQUAL "use")
        target_compile_options(askaig PRIVATE -fprofile-instr-use=${PGO_PROFILE})
        target_link_options(askaig PRIVATE -fprofile-instr-use=${PGO_PROFILE})
    else()
        message(FATAL_ERROR "PGO must be 'generate', 'use', or empty (got '${PGO}')")
    endif()
    message(STATUS "PGO: ${PGO}")
endif()

