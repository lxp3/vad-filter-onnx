cmake_minimum_required(VERSION 3.10)
project(vad_filter_onnx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Source files
file(GLOB_RECURSE SOURCES 
    "utils/*.cc"
    "vad/*.cc"
    "../include/*.cc"
)
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/vad/test-sliding-window-bit.cc")


# Library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    onnxruntime
)

# Test
add_executable(test-sliding-window-bit vad/test-sliding-window-bit.cc)
target_link_libraries(test-sliding-window-bit PRIVATE ${PROJECT_NAME})


