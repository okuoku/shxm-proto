cmake_minimum_required(VERSION 3.0)
project(shxm C CXX)

set(SKIP_GLSLANG_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)

add_subdirectory(glslang)

include_directories(glslang/glslang/Public glslang
    yuniframe/include
    yuniframe/shxm/src
    )

add_definitions(-DSOURCEPATH=\"${CMAKE_CURRENT_LIST_DIR}\")

add_executable(check
    # glslang standalone
    glslang/StandAlone/ResourceLimits.cpp
    glslang/StandAlone/resource_limits_c.cpp
    # main
    yuniframe/shxm/src/shxm_glslang.cpp
    yuniframe/shxm/src/shxm_obj.c
    yuniframe/shxm/src/shxm_linker.c
    yuniframe/shxm/src/shxm_spirv_reader.c
    driver.c
    )

target_link_libraries(check glslang SPIRV)
