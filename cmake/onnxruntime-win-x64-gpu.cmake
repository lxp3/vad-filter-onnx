

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  message(FATAL_ERROR "This file is for Windows only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
  message(FATAL_ERROR "This file is for Windows x64 only. Given architecture size: ${CMAKE_SIZEOF_VOID_P}")
endif()

set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-win-x64-gpu-1.17.1.zip")
set(onnxruntime_URL2 "https://hub.nuaa.cf/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-win-x64-gpu-1.17.1.zip")
set(onnxruntime_HASH "SHA256=b7a66f50ad146c2ccb43471d2d3b5ad78084c2d4ddbd3ea82d65f86c867408b2")

# If you don't have access to the Internet,
# please download onnxruntime to local file and use -DONNXRUNTIME_FILE to specify it.
if(ONNXRUNTIME_FILE)
  if(NOT IS_ABSOLUTE ${ONNXRUNTIME_FILE})
    get_filename_component(ONNXRUNTIME_FILE "${ONNXRUNTIME_FILE}" ABSOLUTE)
  endif()
  
  if(NOT EXISTS ${ONNXRUNTIME_FILE})
    message(STATUS "ONNXRUNTIME_FILE ${ONNXRUNTIME_FILE} does not exist. Downloading from ${onnxruntime_URL}...")
    get_filename_component(ONNXRUNTIME_FILE_DIR "${ONNXRUNTIME_FILE}" DIRECTORY)
    file(MAKE_DIRECTORY ${ONNXRUNTIME_FILE_DIR})
    file(DOWNLOAD 
      ${onnxruntime_URL} 
      ${ONNXRUNTIME_FILE}
      SHOW_PROGRESS
      EXPECTED_HASH ${onnxruntime_HASH}
    )
  endif()

  if(EXISTS ${ONNXRUNTIME_FILE})
    set(onnxruntime_URL  "${ONNXRUNTIME_FILE}")
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    set(onnxruntime_URL2)
  else()
    message(FATAL_ERROR "ONNXRUNTIME_FILE ${ONNXRUNTIME_FILE} does not exist and download failed!")
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
  if(IS_ABSOLUTE ${onnxruntime_URL} AND EXISTS ${onnxruntime_URL})
    message(STATUS "Extracting local onnxruntime from ${onnxruntime_URL}")
  else()
    message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
  endif()
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

set_property(TARGET onnxruntime
  PROPERTY
    IMPORTED_IMPLIB "${ONNXRUNTIME_SRC_DIR}/lib/onnxruntime.lib"
)

file(COPY ${ONNXRUNTIME_SRC_DIR}/lib/onnxruntime.dll
  DESTINATION
    ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
)

# for onnxruntime_providers_cuda.dll

find_library(location_onnxruntime_providers_cuda_lib onnxruntime_providers_cuda
  PATHS
  "${ONNXRUNTIME_SRC_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
message(STATUS "location_onnxruntime_providers_cuda_lib: ${location_onnxruntime_providers_cuda_lib}")

add_library(onnxruntime_providers_cuda SHARED IMPORTED)
set_target_properties(onnxruntime_providers_cuda PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime_providers_cuda_lib}
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_SRC_DIR}/include"
)

set_property(TARGET onnxruntime_providers_cuda
  PROPERTY
    IMPORTED_IMPLIB "${ONNXRUNTIME_SRC_DIR}/lib/onnxruntime_providers_cuda.lib"
)

# for onnxruntime_providers_shared.dll

find_library(location_onnxruntime_providers_shared_lib onnxruntime_providers_shared
  PATHS
  "${ONNXRUNTIME_SRC_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
message(STATUS "location_onnxruntime_providers_shared_lib: ${location_onnxruntime_providers_shared_lib}")
add_library(onnxruntime_providers_shared SHARED IMPORTED)
set_target_properties(onnxruntime_providers_shared PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime_providers_shared_lib}
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_SRC_DIR}/include"
)
set_property(TARGET onnxruntime_providers_shared
  PROPERTY
    IMPORTED_IMPLIB "${ONNXRUNTIME_SRC_DIR}/lib/onnxruntime_providers_shared.lib"
)

file(
  COPY
    ${ONNXRUNTIME_SRC_DIR}/lib/onnxruntime_providers_cuda.dll
    ${ONNXRUNTIME_SRC_DIR}/lib/onnxruntime_providers_shared.dll
  DESTINATION
    ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
)

# 设置全局变量供其他模块使用
file(GLOB onnxruntime_lib_files "${ONNXRUNTIME_SRC_DIR}/lib/*.dll")
file(GLOB onnxruntime_lib_libs "${ONNXRUNTIME_SRC_DIR}/lib/*.lib")

message(STATUS "onnxruntime DLL files: ${onnxruntime_lib_files}")
message(STATUS "onnxruntime LIB files: ${onnxruntime_lib_libs}")

# 安装DLL文件
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
install(FILES ${onnxruntime_lib_files} DESTINATION bin)

# 为Python模块安装DLL
if(ENABLE_PYTHON)
  install(FILES ${onnxruntime_lib_files} DESTINATION python/csrc)
endif()
