# Copyright (c)  2022-2023  Xiaomi Corporation
function(download_onnxruntime)
  include(FetchContent)

  message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
  message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
  
  if(CMAKE_SYSTEM_NAME STREQUAL Linux AND CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
    # Linux x86_64 GPU
    include(onnxruntime-linux-x86_64-gpu)
  elseif(WIN32)
    # Windows x64 GPU
    include(onnxruntime-win-x64-gpu)
  else()
    message(FATAL_ERROR "只支持 Linux x86_64 和 Windows x64 系统")
  endif()
  
  message(STATUS "ONNXRUNTIME_SRC_DIR: ${ONNXRUNTIME_SRC_DIR}")

  # 将onnxruntime的include路径添加到全局包含路径
  if(ONNXRUNTIME_SRC_DIR)
    message(STATUS "Adding onnxruntime include directory: ${ONNXRUNTIME_SRC_DIR}/include")
    include_directories(${ONNXRUNTIME_SRC_DIR}/include)
  endif()
  
  set(ONNXRUNTIME_SRC_DIR ${ONNXRUNTIME_SRC_DIR} PARENT_SCOPE)
endfunction()

# Check if onnxruntime is already configured by previous subprojects (like sherpa-onnx)
if(onnxruntime_SOURCE_DIR)
  set(ONNXRUNTIME_SRC_DIR "${onnxruntime_SOURCE_DIR}")
  message(STATUS "Reusing existing onnxruntime from: ${ONNXRUNTIME_SRC_DIR}")
elseif(ONNXRUNTIME_SRC_DIR AND EXISTS "${ONNXRUNTIME_SRC_DIR}")
  message(STATUS "Using user specified ONNXRUNTIME_SRC_DIR: ${ONNXRUNTIME_SRC_DIR}")
else()
  # 如果没有指定或者目录不存在，则下载
  message(STATUS "ONNXRUNTIME_SRC_DIR [${ONNXRUNTIME_SRC_DIR}] 不存在或未指定，下载预编译版本")
  download_onnxruntime()
endif()

# 确保在函数外部也能访问onnxruntime的头文件和库
if(ONNXRUNTIME_SRC_DIR)
  # 设置全局包含路径
  include_directories(${ONNXRUNTIME_SRC_DIR}/include)
  
  # 设置库文件路径变量，供CMakeLists.txt使用
  if(WIN32)
    # Windows系统
    file(GLOB ONNXRUNTIME_LIBS "${ONNXRUNTIME_SRC_DIR}/lib/*.lib")
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_SRC_DIR}/lib/*.dll")
    
    # 将库添加到全局链接库列表
    message(STATUS "Adding ONNX Runtime libraries to global link libraries")
    link_libraries(${ONNXRUNTIME_LIBS})
    
    # 同时设置 ONNXRUNTIME_LIBRARIES 变量以保持兼容性
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBS})
    
    # 设置DLL路径供安装使用
    set(ONNXRUNTIME_DLLS ${ONNXRUNTIME_DLLS} CACHE STRING "ONNX Runtime DLLs" FORCE)
  else()
    # Linux系统
    file(GLOB ONNXRUNTIME_LIBS "${ONNXRUNTIME_SRC_DIR}/lib/*.so")
    if(NOT ONNXRUNTIME_LIBS)
      file(GLOB ONNXRUNTIME_LIBS "${ONNXRUNTIME_SRC_DIR}/lib/*.a")
    endif()
    
    # 将库添加到全局链接库列表
    message(STATUS "Adding ONNX Runtime libraries to global link libraries")
    link_libraries(${ONNXRUNTIME_LIBS})

    # 同时设置 ONNXRUNTIME_LIBRARIES 变量以保持兼容性
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBS})
  endif()
  
  message(STATUS "ONNXRUNTIME include: ${ONNXRUNTIME_SRC_DIR}/include")
  message(STATUS "ONNXRUNTIME libs: ${ONNXRUNTIME_LIBS}")
  if(WIN32)
    message(STATUS "ONNXRUNTIME DLLs: ${ONNXRUNTIME_DLLS}")
  endif()
endif()
