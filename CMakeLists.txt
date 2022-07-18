cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(HIndex CXX)

set(CMAKE_CXX_STANDARD 14)

message(STATUS"Build type: ${CMAKE_BUILD_TYPE}")

set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} "/nvlab/kittool")


set(Torch_DIR "/libtorch_gpu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("/eigen-3.4.0/")



aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)
add_library(ENTITY STATIC ${SRC_DIR})


add_executable(HIndex Exp.cpp ${SRC_DIR})
add_executable(DBTree SplitData.cpp ${SRC_DIR})
add_executable(test_c test.cpp ${SRC_DIR})


target_link_libraries(HIndex "${TORCH_LIBRARIES}" ENTITY)
target_link_libraries(DBTree "${TORCH_LIBRARIES}" ENTITY)
target_link_libraries(test_c "${TORCH_LIBRARIES}" ENTITY)


add_executable(remove_exp removeExp.cpp ${SRC_DIR})
target_link_libraries(remove_exp "${TORCH_LIBRARIES}" ENTITY)
