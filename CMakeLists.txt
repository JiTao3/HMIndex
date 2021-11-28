cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(CellTree CXX)

set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/jitao/clib/libtorch")
find_package(Torch REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
include_directories("/home/jitao/clib/eigen-3.4.0/")
# include_directories(/home/jitao/clib/libtorch/include/torch/csrc/api/include)

# find_package (Eigen3 3.4 REQUIRED NO_MODULE)


aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)
add_library(ENTITY STATIC ${SRC_DIR})



add_executable(CellTree Exp.cpp ${SRC_DIR})
# add_executable(CellTree_Debug Exp.cpp ${SRC_DIR})
# add_executable(test_c test.cpp ${SRC_DIR})
# add_executable(CellTree_split SplitData.cpp ${SRC_DIR})


target_link_libraries(CellTree "${TORCH_LIBRARIES}" ENTITY)
# target_link_libraries(CellTree_Debug "${TORCH_LIBRARIES}" ENTITY)
# target_link_libraries(CellTree_split "${TORCH_LIBRARIES}" ENTITY)
# target_link_libraries(test_c "${TORCH_LIBRARIES}" ENTITY)
