project(vad_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
file(GLOB_RECURSE SOURCES 
    "utils/*.cc"
    "vad/*.cc"
    "include/*.cc"
)
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/vad/test-sliding-window-bit.cc")


# Library
if(BUILD_SHARED_LIBS STREQUAL "ON")
    add_library(vad_filter_onnx SHARED ${SOURCES})
else()
    add_library(vad_filter_onnx STATIC ${SOURCES})
endif()
add_library(vad_filter_onnx::vad_filter_onnx ALIAS vad_filter_onnx)

target_compile_features(vad_filter_onnx PUBLIC cxx_std_20)
target_compile_definitions(vad_filter_onnx PUBLIC _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})

target_include_directories(vad_filter_onnx PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(BUILD_SHARED_LIBS STREQUAL "ON")
    target_link_libraries(vad_filter_onnx PRIVATE onnxruntime)
else()
    target_link_libraries(vad_filter_onnx PUBLIC onnxruntime)
endif()

if(VAD_FILTER_ONNX_BUILD_TESTS)
    # Test
    add_executable(test-sliding-window-bit vad/test-sliding-window-bit.cc)
    target_link_libraries(test-sliding-window-bit PRIVATE vad_filter_onnx)
endif()

# Installation
install(TARGETS vad_filter_onnx
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
