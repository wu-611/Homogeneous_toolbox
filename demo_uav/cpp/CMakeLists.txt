cmake_minimum_required(VERSION 3.14)
project(demo_uav_homogeneous_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

# 依赖 hcs_toolbox (header-only, 相对路径)
set(HCS_TOOLBOX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../hcs_toolbox_cpp)
add_library(hcs_toolbox INTERFACE)
target_include_directories(hcs_toolbox INTERFACE ${HCS_TOOLBOX_DIR}/include)
target_link_libraries(hcs_toolbox INTERFACE Eigen3::Eigen)

add_executable(demo_uav_homogeneous_control
    demo_uav_homogeneous_control.cpp
)
target_link_libraries(demo_uav_homogeneous_control PRIVATE hcs_toolbox)

# 输出到 build/bin/
set_target_properties(demo_uav_homogeneous_control PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
