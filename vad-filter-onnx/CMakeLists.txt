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
add_library(vad_filter_onnx STATIC ${SOURCES})

target_include_directories(vad_filter_onnx PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(vad_filter_onnx 
    PRIVATE 
    onnxruntime
)

# Test
add_executable(test-sliding-window-bit vad/test-sliding-window-bit.cc)
target_link_libraries(test-sliding-window-bit PRIVATE vad_filter_onnx)

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

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()


