cmake_minimum_required(VERSION 3.0)
project(shxm C CXX)

set(SKIP_GLSLANG_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)

add_subdirectory(glslang)

include_directories(glslang/glslang/Public)

add_executable(check
    interface.cpp
    driver.c
    )

target_link_libraries(check glslang)
