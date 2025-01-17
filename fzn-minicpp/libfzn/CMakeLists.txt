cmake_minimum_required(VERSION 3.22)
project(libfzn)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CXX)

# Sources
set(LIBFZN_SRC Parser.cpp Printer.cpp)

# Library
add_library(libfzn ${LIBFZN_SRC})
set_target_properties(libfzn PROPERTIES PREFIX "")

# Includes
if(PROJECT_IS_TOP_LEVEL)
    target_include_directories(libfzn PRIVATE ${PROJECT_SOURCE_DIR})
else()
    cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PROJECT_PARENT_DIR)
    target_include_directories(libfzn INTERFACE ${PROJECT_PARENT_DIR})
endif()

# Demo
add_executable(libfzn-demo EXCLUDE_FROM_ALL Demo.cpp)
target_link_libraries(libfzn-demo libfzn)