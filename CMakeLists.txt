cmake_minimum_required(VERSION 3.0)
project(shxm C CXX)

set(SPIRV_WERROR OFF CACHE BOOL "" FORCE) # For SPIRV-Tools

set(SKIP_GLSLANG_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
add_subdirectory(glslang)

set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
add_subdirectory(SPIRV-Cross)

set(SPIRV-Headers_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/SPIRV-Headers)
set(SKIP_SPIRV_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
add_subdirectory(SPIRV-Tools)

include_directories(glslang/glslang/Public glslang
    yuniframe/include
    yuniframe/shxm/src
    SPIRV-Cross
    SPIRV-Tools/include
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
    yuniframe/shxm/src/shxm_spirv_decomp.c
    yuniframe/shxm/src/shxm_patcher.c
    yuniframe/shxm/src/shxm_util_buf.c
    yuniframe/shxm/src/shxm_spvopt.cpp

    # Test driver
    driver.c
    )

target_link_libraries(check glslang SPIRV
    spirv-cross-c
    SPIRV-Tools
    SPIRV-Tools-opt
    )
