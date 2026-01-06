# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Linux)
  message(FATAL_ERROR "This file is for Linux only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
  message(FATAL_ERROR "This file is for x86_64 only. Given: ${CMAKE_SYSTEM_PROCESSOR}")
endif()


set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1-patched.zip")
set(onnxruntime_URL2 "https://hub.nuaa.cf/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1-patched.zip")
set(onnxruntime_HASH "SHA256=1261de176e8d9d4d2019f8fa8c732c6d11494f3c6e73168ab6d2cc0903f22551")

# If you don't have access to the Internet,
# please download onnxruntime to local file and use -DONNXRUNTIME_FILE to specify it.
if(ONNXRUNTIME_FILE)
  if(NOT IS_ABSOLUTE ${ONNXRUNTIME_FILE})
    get_filename_component(ONNXRUNTIME_FILE "${ONNXRUNTIME_FILE}" ABSOLUTE)
  endif()
  if(EXISTS ${ONNXRUNTIME_FILE})
    set(onnxruntime_URL  "${ONNXRUNTIME_FILE}")
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    set(onnxruntime_URL2)
  else()
    message(FATAL_ERROR "ONNXRUNTIME_FILE ${ONNXRUNTIME_FILE} does not exist!")
  endif()
endif()

FetchContent_Declare(onnxruntime
  URL
    ${onnxruntime_URL}
    ${onnxruntime_URL2}
  URL_HASH          ${onnxruntime_HASH}
)

FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
  message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
  FetchContent_Populate(onnxruntime)
endif()

# FetchContent sets onnxruntime_SOURCE_DIR, map it to ONNXRUNTIME_SRC_DIR
set(ONNXRUNTIME_SRC_DIR ${onnxruntime_SOURCE_DIR})
message(STATUS "onnxruntime is downloaded to ${ONNXRUNTIME_SRC_DIR}")

find_library(location_onnxruntime onnxruntime
  PATHS
  "${ONNXRUNTIME_SRC_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)

message(STATUS "location_onnxruntime: ${location_onnxruntime}")

add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime}
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_SRC_DIR}/include"
)

find_library(location_onnxruntime_cuda_lib onnxruntime_providers_cuda
  PATHS
  "${ONNXRUNTIME_SRC_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)

add_library(onnxruntime_providers_cuda SHARED IMPORTED)
set_target_properties(onnxruntime_providers_cuda PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime_cuda_lib}
)
message(STATUS "location_onnxruntime_cuda_lib: ${location_onnxruntime_cuda_lib}")

# for libonnxruntime_providers_shared.so
find_library(location_onnxruntime_providers_shared_lib onnxruntime_providers_shared
  PATHS
  "${ONNXRUNTIME_SRC_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
add_library(onnxruntime_providers_shared SHARED IMPORTED)
set_target_properties(onnxruntime_providers_shared PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime_providers_shared_lib}
)
message(STATUS "location_onnxruntime_providers_shared_lib: ${location_onnxruntime_providers_shared_lib}")

file(GLOB onnxruntime_lib_files "${ONNXRUNTIME_SRC_DIR}/lib/libonnxruntime*")
message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
