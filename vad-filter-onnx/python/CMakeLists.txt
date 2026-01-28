include(pybind11)

# Define the python module
pybind11_add_module(vad_filter_onnx_python binding.cc)

# Link to the main library
target_link_libraries(vad_filter_onnx_python PRIVATE vad_filter_onnx)

# Include directories
target_include_directories(vad_filter_onnx_python PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../vad
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
)

# Set properties
set_target_properties(vad_filter_onnx_python PROPERTIES 
    OUTPUT_NAME "vad_filter_onnx"
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

# Copy onnxruntime.dll to the same directory as the pyd file
if(WIN32)
    add_custom_command(TARGET vad_filter_onnx_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:vad_filter_onnx_python>/"
    )
endif()
