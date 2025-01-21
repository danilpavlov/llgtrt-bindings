# python_bindings/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(trtllm_c_python)

set(TRTLLM_DIR "/workspaces/llgtrt-bindings/TensorRT-LLM")

set(TRTLLM_INCLUDE_DIR "${TRTLLM_DIR}/cpp/include")
set(TRTLLM_INCLUDE_COMMON_DIR "${TRTLLM_DIR}/cpp")

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)
enable_language(CUDA)
# find_package(TensorRT REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

pybind11_add_module(trtllm_c_python
    bindings.cpp
    ../main.cpp
    ../logits.cpp
    ../mask_logits.cu
)

target_include_directories(trtllm_c_python PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${TRTLLM_INCLUDE_DIR}
    ${TRTLLM_INCLUDE_COMMON_DIR}
    ../
)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")


target_link_libraries(trtllm_c_python PRIVATE
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    # Add any other necessary libraries
)

#add_library(trtllm_c_python SHARED bindings.cpp ../tlc.cpp)

set_target_properties(trtllm_c_python PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)