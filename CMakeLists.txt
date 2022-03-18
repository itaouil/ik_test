# Accepted CMake range versions
cmake_minimum_required(VERSION 3.1...3.22)

# Project statement
project(agile_ik_test
        DESCRIPTION "Perform FK and IK on a 3DOF manipulator"
        LANGUAGES CXX)

# Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# find python libraries
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})

# Set compiler options
set(CMAKE_CXX_STANDARD 11)
add_compile_options(-Wall -Wextra -Wpedantic)

# Find Eigen library
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# populate matplotlib repository
include(FetchContent)
FetchContent_Declare(
        matplotlib
        GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
        GIT_TAG        f23347fca25219d1c42cbb91608b5556814bf572
)
FetchContent_GetProperties(matplotlib)
if(NOT matplotlib_POPULATED)
  FetchContent_Populate(matplotlib)
endif()
include_directories(SYSTEM ${matplotlib_SOURCE_DIR})

# Create executable and link Eigen's exported CMake target
add_executable(agile_ik_task src/agile_ik_task.cpp)
target_link_libraries (agile_ik_task
        PRIVATE
        ${PYTHON_LIBRARIES}
        Python3::NumPy
        Eigen3::Eigen)