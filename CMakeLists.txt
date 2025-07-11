cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)

project(index)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apply whisper.cpp patches if available
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/whisper.cpp.patch)
  execute_process(COMMAND
    git apply ${CMAKE_CURRENT_SOURCE_DIR}/scripts/whisper.cpp.patch
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ERROR_QUIET
  )
endif()

# NAPI version setup
if(NOT DEFINED napi_build_version)
  set(napi_build_version 6)
endif()
add_definitions(-DNAPI_VERSION=${napi_build_version})
message(STATUS "NAPI_VERSION: ${napi_build_version}")

# Platform and architecture normalization (following llama.node pattern)
set(CMAKE_SYSTEM_PROCESSOR ${NODE_ARCH})
string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)
string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)

# Normalize platform to nodejs
string(REPLACE "windows" "win32" PLATFORM ${PLATFORM})

# Normalize arch to nodejs
string(REPLACE "amd64" "x64" ARCH ${ARCH})
string(REPLACE "x86_64" "x64" ARCH ${ARCH})
string(REPLACE "i686" "ia32" ARCH ${ARCH})
string(REPLACE "i386" "ia32" ARCH ${ARCH})
string(REPLACE "armv7l" "arm" ARCH ${ARCH})
string(REPLACE "arm64x" "arm64" ARCH ${ARCH})
string(REPLACE "aarch64" "arm64" ARCH ${ARCH})

# Build options
option(TO_PACKAGE "Build as package" OFF)
option(CLANG_USE_GOMP "Use GNU OpenMP in Clang" OFF)

# Variant support (default, vulkan, cuda)
if(DEFINED VARIANT)
  set(VARIANT -${VARIANT})
else()
  set(VARIANT "")
endif()

# Package output directory
if (TO_PACKAGE)
  set(PLATFORM_BINARY_DIR ${CMAKE_SOURCE_DIR}/packages/node-whisper-${PLATFORM}-${ARCH}${VARIANT})
else()
  set(PLATFORM_BINARY_DIR ${CMAKE_SOURCE_DIR}/build/Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${PLATFORM}")
message(STATUS "Architecture: ${ARCH}")
message(STATUS "Variant: ${VARIANT}")
message(STATUS "PLATFORM_BINARY_DIR: ${PLATFORM_BINARY_DIR}")

# Include cmake-js
include_directories(${CMAKE_JS_INC})

# Add necessary definitions for node-addon-api
add_definitions(-DNAPI_VERSION=6)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform-specific settings
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    set(CMAKE_MACOSX_RPATH ON)
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if((UNIX OR MINGW) AND NOT CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")
  endif()

  if (MSVC)
    if (NOT GGML_VULKAN)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy /GL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /Ob2 /Oi /Ot /Oy /GL")
      set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} /LTCG")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O1 /Ob1 /bigobj")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O1 /Ob1 /bigobj")
    endif()
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -flto")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -funroll-loops -flto")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -flto")
  endif()
endif()

# Position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MinGW settings
if (MINGW)
  add_definitions(-D_WIN32_WINNT=0x0601)
endif()

# Clang static linking
if (CLANG)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

# Vulkan SDK support
if (VULKAN_SDK)
  set(ENV{VULKAN_SDK} ${VULKAN_SDK})
  find_package(Vulkan REQUIRED)
endif()

# OpenMP for Clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT DEFINED GGML_OPENMP OR GGML_OPENMP AND CLANG_USE_GOMP)
  find_package(OpenMP)
  if (OpenMP_FOUND)
    set(OpenMP_C_FLAGS "-fopenmp=libgomp")
    set(OpenMP_CXX_FLAGS "-fopenmp=libgomp")
    set(OpenMP_C_LIB_NAMES "libgomp")
    set(OpenMP_CXX_LIB_NAMES "libgomp")
    set(OpenMP_libgomp_LIBRARY "gomp")
  endif()
endif()

# whisper.cpp configuration
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")

# Platform-specific optimizations
if(APPLE)
    set(WHISPER_METAL ON CACHE BOOL "" FORCE)
    set(WHISPER_ACCELERATE ON CACHE BOOL "" FORCE)
elseif(WIN32)
    # Windows-specific settings
    set(WHISPER_WIN32_LEAN_AND_MEAN ON)
else()
    # Linux-specific settings
    if(NOT DEFINED GGML_OPENMP)
      set(GGML_OPENMP ON CACHE BOOL "" FORCE)
    endif()
endif()

# Add whisper.cpp subdirectory
add_subdirectory(whisper.cpp)

# Add node-addon-api
execute_process(COMMAND node -p "require('node-addon-api').include"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NODE_ADDON_API_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
# Remove quotes from the path
string(REPLACE "\"" "" NODE_ADDON_API_DIR "${NODE_ADDON_API_DIR}")

# Source files
file(GLOB_RECURSE SOURCES "src/*.cc" "src/*.cpp" "src/*.c")
list(APPEND SOURCES "whisper.cpp/examples/common-whisper.cpp")

# Handle Windows cross-compilation without MSVC (similar to llama.node)
if (NOT MSVC AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
  file(GLOB WIN_DYNAMIC_LOAD_SRC "src/win_dynamic_load.c")

  add_library(win_dynamic_load ${WIN_DYNAMIC_LOAD_SRC})
  set_target_properties(win_dynamic_load PROPERTIES COMPILE_FLAGS "-Wno-implicit-function-declaration")
  
  unset(CMAKE_JS_SRC)
  unset(CMAKE_JS_LIB)
  unset(CMAKE_JS_NODELIB_DEF)
  unset(CMAKE_JS_NODELIB_TARGET)
  unset(CMAKE_JS_NODELIB_TARGET_NAME)
  string(REGEX REPLACE "/DELAYLOAD:NODE.EXE" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

  set(CMAKE_JS_LIB win_dynamic_load)
endif()

# Create the addon
add_library(${PROJECT_NAME} SHARED ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Link libraries
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} whisper)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_JS_INC}
    ${NODE_ADDON_API_DIR}
    ${CMAKE_SOURCE_DIR}/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/whisper.cpp/examples
    ${CMAKE_SOURCE_DIR}/src
)

# Compiler-specific flags
if(MSVC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
    target_compile_options(${PROJECT_NAME} PRIVATE /W3)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${PROJECT_NAME} PRIVATE -g -O0)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -O3)
    endif()
endif()

# Platform-specific linking
if(APPLE)
    target_link_libraries(${PROJECT_NAME} "-framework Accelerate")
    if(WHISPER_METAL)
        target_link_libraries(${PROJECT_NAME} "-framework Metal" "-framework Foundation")
    endif()
elseif(WIN32)
    # Windows-specific libraries
    target_link_libraries(${PROJECT_NAME} winmm)
else()
    # Linux-specific libraries
    target_link_libraries(${PROJECT_NAME} pthread)
    if(GGML_OPENMP)
        find_package(OpenMP)
        if(OpenMP_FOUND)
            target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()

# Package output handling
add_custom_target(copy_assets ALL DEPENDS ${PROJECT_NAME})

if (TO_PACKAGE)
  add_custom_command(TARGET copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${PLATFORM_BINARY_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
    COMMENT "Deploy as package"
  )
endif()

# Metal library handling for macOS
if (TO_PACKAGE)
  set(METAL_LIB_TARGET_PATH ${PLATFORM_BINARY_DIR})
else()
  set(METAL_LIB_TARGET_PATH ${CMAKE_BINARY_DIR}/bin/default.metallib)
endif()

# Generate node.lib if needed (for Windows)
if(CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

if (GGML_METAL AND NOT GGML_METAL_EMBED_LIBRARY)
  add_custom_command(
    TARGET copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/default.metallib ${METAL_LIB_TARGET_PATH}
    COMMENT "Copying default.metallib to bin folder"
  )
  add_dependencies(copy_assets ggml-metal)
endif()

# Install target
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
