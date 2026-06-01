include(FetchContent)

# Avoid warnings about download timestamps
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(ONNXRUNTIME_VERSION "1.23.2")

if(WIN32)
    if(BUILD_SHARED_LIBS STREQUAL "ON")
        if(ENABLE_GPU STREQUAL "ON")
            set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip")
            set(ONNXRUNTIME_SHA256 "e77afdbbc2b8cb6da4e5a50d89841b48c44f3e47dce4fb87b15a2743786d0bb9")
        else()
            set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip")
            set(ONNXRUNTIME_SHA256 "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd")
        endif()
    else()
        # Static library for Windows
        set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/onnxruntime-win-x64-static_lib-1.23.2.tar.bz2")
        set(ONNXRUNTIME_SHA256 "86f2a87c029554bb685e528ff143090b4d4eb1a0b9ff5d08ba9b676d6c79b76c")
    endif()
elseif(APPLE)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-universal2-1.23.2.tgz")
    set(ONNXRUNTIME_SHA256 "49ae8e3a66ccb18d98ad3fe7f5906b6d7887df8a5edd40f49eb2b14e20885809")
elseif(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        if(BUILD_SHARED_LIBS STREQUAL "ON")
            set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-linux-aarch64-glibc2_17-Release-1.17.1-patched.zip")
            set(ONNXRUNTIME_SHA256 "6e0e68985f8dd1f643e5a4dbe7cd54c9e176a0cc62249c6bee0699b87fc6d4fb")
        else()
            # ARM64 currently uses the available 1.17.1 static package from onnxruntime-libs.
            set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-linux-aarch64-static_lib-1.17.1.zip")
            set(ONNXRUNTIME_SHA256 "831b9a3869501040b4399de85f34c4f170e2bcbd41881edaeb553f8dc4080985")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        if(BUILD_SHARED_LIBS STREQUAL "ON")
            set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-linux-arm-1.17.1-patched.zip")
            set(ONNXRUNTIME_SHA256 "4ec00f7adc7341c068babea3d0f607349655e598222d4212115ae4f52619efdb")
        else()
            set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-linux-arm-static_lib-1.17.1.zip")
            set(ONNXRUNTIME_SHA256 "3f2ba38156d2facfb732c0fe53bc1eaaf2791d9a91dd240380e3d53716798b09")
        endif()
    else()
        if(BUILD_SHARED_LIBS STREQUAL "ON")
            if(ENABLE_GPU STREQUAL "ON")
                set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2-patched.zip")
                set(ONNXRUNTIME_SHA256 "e2f622513212304447e34512b99ae4eabb4fd8870dd1baac895f222179dede19")
            else()
                set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/onnxruntime-linux-x64-glibc2_17-Release-1.23.2.zip")
                set(ONNXRUNTIME_SHA256 "77ea3532dfdd8d5c66918429f7eacd80c1fea834941a14746adf3109f8e7b830")
            endif()
        else()
            # Static library for Linux x64
            set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/onnxruntime-linux-x64-static_lib-1.23.2-glibc2_17.zip")
            set(ONNXRUNTIME_SHA256 "93a52b9d93a0932259a03090291be861ba21ad4b1b58057d3a0f57a4c4108671")
        endif()
    endif()
endif()

# Get filename from URL
get_filename_component(ONNXRUNTIME_FILENAME ${ONNXRUNTIME_URL} NAME)
set(VAD_FILTER_ONNX_ORT_DOWNLOAD_DIR
    "${CMAKE_BINARY_DIR}/_deps/onnxruntime-downloads"
    CACHE PATH
    "Directory used to cache downloaded ONNX Runtime archives"
)
set(DOWNLOAD_DIR "${VAD_FILTER_ONNX_ORT_DOWNLOAD_DIR}")
set(LOCAL_ZIP_PATH "${DOWNLOAD_DIR}/${ONNXRUNTIME_FILENAME}")
message(STATUS "ONNX Runtime URL: ${ONNXRUNTIME_URL}")
message(STATUS "ONNX Runtime filename: ${ONNXRUNTIME_FILENAME}")
message(STATUS "ONNX Runtime download directory: ${DOWNLOAD_DIR}")
message(STATUS "ONNX Runtime local zip path: ${LOCAL_ZIP_PATH}")

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
    if(NOT ACTUAL_HASH STREQUAL "${ONNXRUNTIME_SHA256}")
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
    if(BUILD_SHARED_LIBS STREQUAL "ON")
        set(ONNXRUNTIME_DLL ${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.dll)
    endif()
elseif(APPLE)
    if(BUILD_SHARED_LIBS STREQUAL "ON")
        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib)
    else()
        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.a)
    endif()
else()
    if(BUILD_SHARED_LIBS STREQUAL "ON")
        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
    else()
        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.a)
    endif()
endif()

# Create imported target
if(NOT TARGET onnxruntime)
    if(BUILD_SHARED_LIBS STREQUAL "ON")
        add_library(onnxruntime SHARED IMPORTED)
    else()
        add_library(onnxruntime STATIC IMPORTED)
    endif()

    set_target_properties(onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
    )
    if(WIN32)
        if(BUILD_SHARED_LIBS STREQUAL "ON")
            set_target_properties(onnxruntime PROPERTIES
                IMPORTED_IMPLIB "${ONNXRUNTIME_LIB}"
                IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
            )
        else()
            set_target_properties(onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
            )
        endif()
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
