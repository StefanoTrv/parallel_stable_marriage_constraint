cmake_minimum_required(VERSION 3.22)
project(libminicpp)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CXX)

# Sources
set(LIBMINICPP_SRC
    acstr.cpp
    bitset.cpp
    constraint.cpp
    domain.cpp
    domain.cpp
    fail.cpp
    intvar.cpp
    lex.cpp
    mallocWatch.cpp
    matching.cpp
    regular.cpp
    RuntimeMonitor.cpp
    search.cpp
    solver.cpp
    store.cpp
    table.cpp
    trailable.cpp
    trail.cpp
)

# Library
add_library(libminicpp ${LIBMINICPP_SRC})
set_target_properties(libminicpp PROPERTIES PREFIX "")

# Includes
if(PROJECT_IS_TOP_LEVEL)
    target_include_directories(libminicpp PRIVATE ${PROJECT_SOURCE_DIR})
else()
    cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PROJECT_PARENT_DIR)
    target_include_directories(libminicpp INTERFACE ${PROJECT_PARENT_DIR})
endif()

# Demo
add_executable(minicpp-demo EXCLUDE_FROM_ALL Demo.cpp)
target_link_libraries(minicpp-demo libminicpp)