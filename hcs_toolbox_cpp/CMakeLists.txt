cmake_minimum_required(VERSION 3.14)
project(hcs_toolbox_cpp VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Eigen3 dependency
# ============================================================
find_package(Eigen3 REQUIRED)
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

# ============================================================
# Toolbox is header-only
# ============================================================
add_library(hcs_toolbox INTERFACE)
target_include_directories(hcs_toolbox INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(hcs_toolbox INTERFACE Eigen3::Eigen)

# ============================================================
# Demo executables
# ============================================================
function(add_hcs_demo name src_file)
    add_executable(${name} ${src_file})
    target_link_libraries(${name} PRIVATE hcs_toolbox)
    # Set output to build/bin/
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_hcs_demo(demo_lpc2hpc                src/demo_lpc2hpc.cpp)
add_hcs_demo(demo_lpic2hpic              src/demo_lpic2hpic.cpp)
add_hcs_demo(demo_lpc_hpc_distance_square src/demo_lpc_hpc_distance_square.cpp)
add_hcs_demo(demo_uav_homogeneous_control src/demo_uav_homogeneous_control.cpp)

# ============================================================
# Install
# ============================================================
install(TARGETS demo_lpc2hpc demo_lpic2hpic demo_lpc_hpc_distance_square
         demo_uav_homogeneous_control
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/
    DESTINATION include
)
install(PROGRAMS
    scripts/plot_demo_lpc2hpc.py
    scripts/plot_demo_lpic2hpic.py
    scripts/plot_demo_distance_square.py
    scripts/plot_uav_comparison.py
    DESTINATION bin
)
