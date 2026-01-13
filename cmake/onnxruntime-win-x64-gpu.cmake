

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  message(FATAL_ERROR "This file is for Windows only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
  message(FATAL_ERROR "This file is for Windows x64 only. Given architecture size: ${CMAKE_SIZEOF_VOID_P}")
endif()

# set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-win-x64-static_lib-1.17.1.tar.bz2")
# set(onnxruntime_HASH "SHA256=42a0c02fda945d1d72433b2a7cdb2187d51cb4d7f3af462c6ae07b25314d5fb3")
set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip")
set(onnxruntime_HASH "SHA256=e77afdbbc2b8cb6da4e5a50d89841b48c44f3e47dce4fb87b15a2743786d0bb9")

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
  else()
    message(FATAL_ERROR "ONNXRUNTIME_FILE ${ONNXRUNTIME_FILE} does not exist and download failed!")
  endif()
endif()

FetchContent_Declare(onnxruntime
  URL               ${onnxruntime_URL}
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

# Find the main onnxruntime library for linking
find_library(location_onnxruntime onnxruntime
  PATHS "${ONNXRUNTIME_SRC_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
message(STATUS "location_onnxruntime: ${location_onnxruntime}")

if(NOT TARGET onnxruntime)
  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${location_onnxruntime}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_SRC_DIR}/include"
    IMPORTED_IMPLIB "${ONNXRUNTIME_SRC_DIR}/lib/onnxruntime.lib"
  )
endif()

# Collect all DLLs and LIBs using wildcards
file(GLOB onnxruntime_lib_files "${ONNXRUNTIME_SRC_DIR}/lib/*.dll")
file(GLOB onnxruntime_lib_libs "${ONNXRUNTIME_SRC_DIR}/lib/*.lib")

message(STATUS "onnxruntime DLL files: ${onnxruntime_lib_files}")
message(STATUS "onnxruntime LIB files: ${onnxruntime_lib_libs}")

# Copy all DLLs to the build directory
foreach(dll_file ${onnxruntime_lib_files})
  file(COPY ${dll_file} DESTINATION ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
endforeach()

# Install DLL files
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
install(FILES ${onnxruntime_lib_files} DESTINATION bin)

# Optional Python support
if(ENABLE_PYTHON)
  install(FILES ${onnxruntime_lib_files} DESTINATION python/csrc)
endif()
