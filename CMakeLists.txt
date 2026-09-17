cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)

project (index)

set(CMAKE_CXX_STANDARD 17)

option(WHISPER_NODE_WASM "Build the browser WASM package" OFF)
option(WHISPER_NODE_WASM_SINGLE_FILE "Embed WASM into the generated browser JS" OFF)
option(WHISPER_NODE_WASM_THREADS "Build the browser WASM package with pthread support" OFF)

# whisper.cpp comes from the whisper.rn submodule, which vendors it in upstream
# layout (src/whisper.rn/vendor/whisper.cpp) with whisper.rn's own patches
# already applied. scripts/whisper.cpp.patch carries only the whisper.node
# specific changes on top of that tree (MSVC ARM flags, Vulkan shader-gen
# toolchain when host and target arch differ).
set(WHISPER_RN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/whisper.rn)
set(WHISPER_CPP_DIR ${WHISPER_RN_DIR}/vendor/whisper.cpp)
set(WHISPER_CPP_PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/scripts/whisper.cpp.patch)

if (NOT EXISTS ${WHISPER_CPP_DIR}/CMakeLists.txt)
  message(FATAL_ERROR "src/whisper.rn submodule is missing; run: git submodule update --init src/whisper.rn")
endif()

execute_process(
  COMMAND git apply --check ${WHISPER_CPP_PATCH_FILE}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE WHISPER_CPP_PATCH_CHECK_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)

if (WHISPER_CPP_PATCH_CHECK_RESULT EQUAL 0)
  execute_process(
    COMMAND git apply ${WHISPER_CPP_PATCH_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE WHISPER_CPP_PATCH_APPLY_RESULT
  )

  if (NOT WHISPER_CPP_PATCH_APPLY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to apply scripts/whisper.cpp.patch")
  endif()

  message(STATUS "Applied scripts/whisper.cpp.patch")
else()
  execute_process(
    COMMAND git apply --reverse --check ${WHISPER_CPP_PATCH_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE WHISPER_CPP_PATCH_REVERSE_CHECK_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
  )

  if (WHISPER_CPP_PATCH_REVERSE_CHECK_RESULT EQUAL 0)
    message(STATUS "scripts/whisper.cpp.patch already applied")
  else()
    message(FATAL_ERROR "scripts/whisper.cpp.patch does not match the current src/whisper.rn/vendor/whisper.cpp checkout")
  endif()
endif()

if (WHISPER_NODE_WASM)
  if (NOT EMSCRIPTEN)
    message(FATAL_ERROR "WHISPER_NODE_WASM=ON requires configuring with Emscripten, for example: emcmake cmake -S . -B build/wasm -DWHISPER_NODE_WASM=ON")
  endif()

  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
  set(CMAKE_SYSTEM_PROCESSOR wasm32 CACHE STRING "Use ggml WASM CPU kernels" FORCE)
  set(GGML_NATIVE OFF CACHE BOOL "Disable native CPU flags for portable WASM" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP for WASM" FORCE)

  if (WHISPER_NODE_WASM_THREADS)
    add_compile_options("-pthread")
  endif()

  add_subdirectory(${WHISPER_CPP_DIR} ${CMAKE_BINARY_DIR}/whisper.cpp)

  set(WASM_TARGET whisper-node-wasm)
  if (WHISPER_NODE_WASM_THREADS)
    set(WASM_OUTPUT_NAME "whisper-node.threads")
  else()
    set(WASM_OUTPUT_NAME "whisper-node")
  endif()

  add_executable(${WASM_TARGET}
    "src/wasm/emscripten.cpp"
  )

  target_include_directories(${WASM_TARGET} PRIVATE
    ${WHISPER_CPP_DIR}
    ${WHISPER_CPP_DIR}/include
    ${WHISPER_CPP_DIR}/src
    ${WHISPER_CPP_DIR}/ggml/include
  )

  target_link_libraries(${WASM_TARGET} PRIVATE whisper parakeet)

  set_target_properties(${WASM_TARGET} PROPERTIES
    OUTPUT_NAME ${WASM_OUTPUT_NAME}
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  target_link_options(${WASM_TARGET} PRIVATE
    "--bind"
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sEXPORT_NAME=createWhisperNodeModule"
    "-sENVIRONMENT=web,worker"
    "-sINITIAL_MEMORY=512MB"
    "-sMAXIMUM_MEMORY=2000MB"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sFORCE_FILESYSTEM=1"
    "-sEXPORTED_RUNTIME_METHODS=['FS','HEAPU8']"
    "-sEXIT_RUNTIME=0"
  )

  if (WHISPER_NODE_WASM_THREADS)
    target_link_options(${WASM_TARGET} PRIVATE
      "-pthread"
      "-sUSE_PTHREADS=1"
      "-sPTHREAD_POOL_SIZE=8"
      "-sPTHREAD_POOL_SIZE_STRICT=0"
    )
  endif()

  if (WHISPER_NODE_WASM_SINGLE_FILE)
    target_link_options(${WASM_TARGET} PRIVATE "-sSINGLE_FILE=1")
  endif()

  return()
endif()

if(NOT DEFINED napi_build_version)
  set(napi_build_version 6)
endif()
add_definitions(-DNAPI_VERSION=${napi_build_version})
message(STATUS "NAPI_VERSION: ${napi_build_version}")

set(CMAKE_SYSTEM_PROCESSOR ${NODE_ARCH})
string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)
string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)

# normalize platform to nodejs
string(REPLACE "windows" "win32" PLATFORM ${PLATFORM})

# normalize arch to nodejs: 'arm', 'arm64', 'ia32', 'loong64', 'mips', 'mipsel', 'ppc', 'ppc64', 'riscv64', 's390', 's390x', and 'x64'.
string(REPLACE "amd64" "x64" ARCH ${ARCH})
string(REPLACE "x86_64" "x64" ARCH ${ARCH})
string(REPLACE "i686" "ia32" ARCH ${ARCH})
string(REPLACE "i386" "ia32" ARCH ${ARCH})
string(REPLACE "armv7l" "arm" ARCH ${ARCH})
string(REPLACE "arm" "arm" ARCH ${ARCH})
string(REPLACE "arm64x" "arm64" ARCH ${ARCH})
string(REPLACE "aarch64" "arm64" ARCH ${ARCH})

option(TO_PACKAGE "Build as package" OFF)
option(CLANG_USE_GOMP "Use GNU OpenMP in Clang" OFF)

if(DEFINED VARIANT)
  set(VARIANT -${VARIANT})
else()
  set(VARIANT "")
endif()

if (TO_PACKAGE)
  set(PACKAGE_NAME "node-whisper-${PLATFORM}-${ARCH}${VARIANT}")
  set(PLATFORM_BINARY_DIR ${CMAKE_SOURCE_DIR}/packages/${PACKAGE_NAME})
else()
  set(PLATFORM_BINARY_DIR ${CMAKE_SOURCE_DIR}/build/Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${PLATFORM}")
message(STATUS "Architecture: ${ARCH}")
message(STATUS "PLATFORM_BINARY_DIR: ${PLATFORM_BINARY_DIR}")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if((UNIX OR MINGW) AND NOT CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")
  endif()
endif()

# Improve speed
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if (MSVC)
    # Enable parallel compilation for all MSVC builds
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")

    if (NOT GGML_VULKAN AND NOT GGML_CUDA)
      # Full optimization with LTCG for default builds
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy /GL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /Ob2 /Oi /Ot /Oy /GL")
      set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} /LTCG")
    elseif(GGML_VULKAN)
      # Reduced optimization for Vulkan builds
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O1 /Ob1 /bigobj")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O1 /Ob1 /bigobj")
    else()
      # Faster linking for CUDA builds (no LTCG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /Ob2 /Oi")
    endif()
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -flto")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -funroll-loops -flto")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -flto")
  endif()
endif()

if (CLANG)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

# flags: -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MINGW)
  add_definitions(-D_WIN32_WINNT=0x0601)
endif()

# VULKAN_SDK
if (VULKAN_SDK)
  set(ENV{VULKAN_SDK} ${VULKAN_SDK})
  find_package(Vulkan REQUIRED)
endif()

# Avoid libomp is not installed commonly
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

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")

# Ensure ggml/Metal kernel compilation uses the same deployment target as the rest of the build.
# whisper.cpp's ggml-metal backend supports -mmacosx-version-min via GGML_METAL_MACOSX_VERSION_MIN.
if(APPLE AND NOT DEFINED GGML_METAL_MACOSX_VERSION_MIN)
  if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    set(GGML_METAL_MACOSX_VERSION_MIN "${CMAKE_OSX_DEPLOYMENT_TARGET}")
  elseif(DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
    set(GGML_METAL_MACOSX_VERSION_MIN "$ENV{MACOSX_DEPLOYMENT_TARGET}")
  endif()
endif()
add_subdirectory(${WHISPER_CPP_DIR} ${CMAKE_BINARY_DIR}/whisper.cpp)

if (NOT GGML_VULKAN AND NOT GGML_CUDA AND NOT GGML_METAL AND NOT GGML_CLBLAST AND NOT GGML_HEXAGON)
  add_definitions(-DNO_GPU_SUPPORT)
endif()

include_directories(
  ${CMAKE_JS_INC}
  ${WHISPER_CPP_DIR}
  ${WHISPER_CPP_DIR}/examples
  ${WHISPER_CPP_DIR}/src
)

file(
  GLOB SOURCE_FILES
    "src/addons.cc"
    "src/common.hpp"
    "src/common.cpp"
    "src/WhisperContext.cpp"
    "src/WhisperContext.h"
    "src/ParakeetContext.cpp"
    "src/ParakeetContext.h"
)
list(APPEND SOURCE_FILES ${WHISPER_CPP_DIR}/examples/common-whisper.cpp)

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

if (TO_PACKAGE AND GGML_HEXAGON)
  # The HTP skels (libggml-htp-*.so) ship next to index.node; ggml-hexagon
  # dlopens them through the FastRPC stub, so make the package dir an rpath
  # for plain Node and for Electron (asar-unpacked / resources) layouts.
  set(NODE_RPATH "node_modules/@fugood/${PACKAGE_NAME}")
  set(ELECTRON_ASAR_RPATH "resources/app.asar.unpacked/node_modules/@fugood/${PACKAGE_NAME}")
  set(ELECTRON_RES_RPATH "resources/node_modules/@fugood/${PACKAGE_NAME}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${NODE_RPATH} -Wl,-rpath,${ELECTRON_ASAR_RPATH} -Wl,-rpath,${ELECTRON_RES_RPATH}")
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} whisper parakeet ggml ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(copy_assets ALL DEPENDS ${PROJECT_NAME})

if (TO_PACKAGE)
  add_custom_command(TARGET copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${PLATFORM_BINARY_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
    COMMENT "Deploy as package"
  )
endif()

if (TO_PACKAGE)
  set(METAL_LIB_TARGET_PATH ${PLATFORM_BINARY_DIR})
else()
  set(METAL_LIB_TARGET_PATH ${CMAKE_BINARY_DIR}/bin/default.metallib)
endif()

if(CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

if (GGML_METAL AND NOT GGML_METAL_EMBED_LIBRARY)
  # copy ${CMAKE_BINARY_DIR}/bin/default.metallib  
  add_custom_command(
    TARGET copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/default.metallib ${METAL_LIB_TARGET_PATH}
    COMMENT "Copying default.metallib to bin folder"
  )
  add_dependencies(copy_assets ggml-metal)
endif()

if (GGML_HEXAGON)
  get_target_property(HTP_LIBS_DIR ggml-hexagon BINARY_DIR)
  add_custom_command(
    TARGET copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy ${HTP_LIBS_DIR}/libggml-htp-v73.so ${PLATFORM_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${HTP_LIBS_DIR}/libggml-htp-v75.so ${PLATFORM_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${HTP_LIBS_DIR}/libggml-htp-v79.so ${PLATFORM_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${HTP_LIBS_DIR}/libggml-htp-v81.so ${PLATFORM_BINARY_DIR}
    COMMENT "Copying HTP libraries to bin folder"
  )
endif()

if (GGML_CLBLAST AND TO_PACKAGE)
  find_package(CLBlast)
  if (CLBlast_FOUND)
    message(STATUS "CLBlast found: ${CLBlast_DIR}")
    file(
      GLOB CLBlast_SO_FILES
      ${CLBlast_DIR}/../../../bin/clblast.dll
      ${CLBlast_DIR}/../../../lib/libclblast.so
    )
    add_custom_command(
      TARGET copy_assets
      COMMAND ${CMAKE_COMMAND} -E copy ${CLBlast_SO_FILES} ${PLATFORM_BINARY_DIR}
      COMMENT "Copying CLBlast SO files to bin folder"
    )
  endif()
endif()
