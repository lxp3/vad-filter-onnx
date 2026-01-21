include(FetchContent)

# Avoid warnings about download timestamps
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

option(ENABLE_GPU "Enable GPU support for ONNX Runtime" OFF)

set(ONNXRUNTIME_VERSION "1.23.2")

if(WIN32)
    if(ENABLE_GPU)
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip")
        set(ONNXRUNTIME_SHA256 "e77afdbbc2b8cb6da4e5a50d89841b48c44f3e47dce4fb87b15a2743786d0bb9")
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip")
        set(ONNXRUNTIME_SHA256 "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd")
    endif()
elseif(APPLE)
    # macOS provided is only for CPU
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-x86_64-1.23.2.tgz")
    set(ONNXRUNTIME_SHA256 "d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6")
elseif(UNIX)
    if(ENABLE_GPU)
        set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2-patched.zip")
        set(ONNXRUNTIME_SHA256 "e2f622513212304447e34512b99ae4eabb4fd8870dd1baac895f222179dede19")
    else()
        set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/onnxruntime-linux-x64-glibc2_17-Release-1.23.2.zip")
        set(ONNXRUNTIME_SHA256 "77ea3532dfdd8d5c66918429f7eacd80c1fea834941a14746adf3109f8e7b830")
    endif()
endif()

# Get filename from URL
get_filename_component(ONNXRUNTIME_FILENAME ${ONNXRUNTIME_URL} NAME)
set(DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/public/downloads")
set(LOCAL_ZIP_PATH "${DOWNLOAD_DIR}/${ONNXRUNTIME_FILENAME}")

# Create download directory if not exists
if(NOT EXISTS "${DOWNLOAD_DIR}")
    file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")
endif()

# Download if not exists or hash mismatch
if(NOT EXISTS "${LOCAL_ZIP_PATH}")
    message(STATUS "Downloading ONNX Runtime from ${ONNXRUNTIME_URL} to ${LOCAL_ZIP_PATH}...")
    file(DOWNLOAD ${ONNXRUNTIME_URL} "${LOCAL_ZIP_PATH}"
        EXPECTED_HASH SHA256=${ONNXRUNTIME_SHA256}
        SHOW_PROGRESS
    )
else()
    # Verify hash if file exists to ensure integrity
    file(SHA256 "${LOCAL_ZIP_PATH}" ACTUAL_HASH)
    if(NOT ACTUAL_HASH STREQUAL ONNXRUNTIME_SHA256)
        message(WARNING "Hash mismatch for ${LOCAL_ZIP_PATH}. Redownloading...")
        file(DOWNLOAD ${ONNXRUNTIME_URL} "${LOCAL_ZIP_PATH}"
            EXPECTED_HASH SHA256=${ONNXRUNTIME_SHA256}
            SHOW_PROGRESS
        )
    endif()
endif()

# Use FetchContent to extract
FetchContent_Declare(
    onnxruntime
    URL "${LOCAL_ZIP_PATH}"
)

FetchContent_MakeAvailable(onnxruntime)

# Define variables for ease of use
set(ONNXRUNTIME_ROOT_DIR ${onnxruntime_SOURCE_DIR})
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_DIR}/include)

if(WIN32)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.lib)
    set(ONNXRUNTIME_DLL ${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.dll)
elseif(APPLE)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib)
else()
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
endif()

# Create imported target
if(NOT TARGET onnxruntime)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
    )
    if(WIN32)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${ONNXRUNTIME_LIB}"
            IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
        )
    else()
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
        )
    endif()
endif()

message(STATUS "ONNX Runtime version: ${ONNXRUNTIME_VERSION}")
message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT_DIR}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime libraries: ${ONNXRUNTIME_LIB}")
