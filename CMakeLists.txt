cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(CellTree CXX)

set(CMAKE_CXX_STANDARD 14)

message(STATUS"Build type: ${CMAKE_BUILD_TYPE}")

set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} "/home/jitao/nvlab/kittool")
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/jitao/clib/libtorch_gpu/libtorch")
# find_package(Torch REQUIRED)

# set(CMAKE_CUDA_COMPILER "/home/jitao/nvlab/kittool/bin/nvcc")

set(Torch_DIR "/home/jitao/clib/libtorch_gpu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})


include_directories(${PROJECT_SOURCE_DIR}/include)
# include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
include_directories("/home/jitao/clib/eigen-3.4.0/")
# include_directories(/home/jitao/clib/libtorch/include/torch/csrc/api/include)

# find_package (Eigen3 3.4 REQUIRED NO_MODULE)


aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)
add_library(ENTITY STATIC ${SRC_DIR})


add_executable(CellTree Exp.cpp ${SRC_DIR})
# add_executable(CellTree_Debug Exp.cpp ${SRC_DIR})
add_executable(CellTree_split SplitData.cpp ${SRC_DIR})
add_executable(test_c test.cpp ${SRC_DIR})


target_link_libraries(CellTree "${TORCH_LIBRARIES}" ENTITY)
# target_link_libraries(CellTree_Debug "${TORCH_LIBRARIES}" ENTITY)
target_link_libraries(CellTree_split "${TORCH_LIBRARIES}" ENTITY)
target_link_libraries(test_c "${TORCH_LIBRARIES}" ENTITY)
