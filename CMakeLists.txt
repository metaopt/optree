# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cmake_minimum_required(VERSION 3.11)  # for FetchContent
project(optree LANGUAGES CXX)

include(FetchContent)
set(PYBIND11_VERSION v2.10.4)
set(ABSEIL_CPP_VERSION 20230125.3)
set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third-party")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)  # for likely/unlikely attributes
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC
set(CMAKE_CXX_VISIBILITY_PRESET hidden)  # -fvisibility=hidden

if(MSVC)
    string(
        APPEND CMAKE_CXX_FLAGS
        " /Zc:preprocessor"
        " /experimental:external /external:anglebrackets /external:W0"
        " /Wall /Wv:19.34"  # Visual Studio 2022 version 17.4
        # Suppress following warnings
        " /wd4365"  # conversion from 'type_1' to 'type_2', signed/unsigned mismatch
        " /wd4514"  # unreferenced inline function has been removed
        " /wd4710"  # function not inlined
        " /wd4711"  # function selected for inline expansion
        " /wd4820"  # bytes padding added after construct 'member_name'
        " /wd4868"  # compiler may not enforce left-to-right evaluation order in braced initializer list
        " /wd5045"  # compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
        " /wd5262"  # use [[fallthrough]] when a break statement is intentionally omitted between cases
        " /wd5264"  # 'const' variable is not used
    )
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " /Zi")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2 /Ob2")
else()
    string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g -Og")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
endif()

if(NOT DEFINED OPTREE_CXX_WERROR AND NOT "$ENV{OPTREE_CXX_WERROR}" STREQUAL "")
    set(OPTREE_CXX_WERROR "$ENV{OPTREE_CXX_WERROR}")
endif()

if(OPTREE_CXX_WERROR)
    message(WARNING "Treats all compiler warnings as errors. Set `OPTREE_CXX_WERROR=OFF` to disable this.")

    if(MSVC)
        string(APPEND CMAKE_CXX_FLAGS " /WX")
    else()
        string(APPEND CMAKE_CXX_FLAGS " -Werror -Wno-error=attributes -Wno-error=redundant-move")
    endif()
endif()

string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_PREFIX_SIZE)
add_definitions("-DSOURCE_PATH_PREFIX_SIZE=${SOURCE_PATH_PREFIX_SIZE}")

function(system)
    set(options STRIP)
    set(oneValueArgs OUTPUT_VARIABLE ERROR_VARIABLE WORKING_DIRECTORY)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(
        SYSTEM
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        "${ARGN}"
    )

    if(NOT DEFINED SYSTEM_WORKING_DIRECTORY)
        set(SYSTEM_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif()

    execute_process(
        COMMAND ${SYSTEM_COMMAND}
        OUTPUT_VARIABLE STDOUT
        ERROR_VARIABLE STDERR
        WORKING_DIRECTORY "${SYSTEM_WORKING_DIRECTORY}"
    )

    if("${SYSTEM_STRIP}")
        string(STRIP "${STDOUT}" STDOUT)
        string(STRIP "${STDERR}" STDERR)
    endif()

    set("${SYSTEM_OUTPUT_VARIABLE}" "${STDOUT}" PARENT_SCOPE)

    if(DEFINED SYSTEM_ERROR_VARIABLE)
        set("${SYSTEM_ERROR_VARIABLE}" "${STDERR}" PARENT_SCOPE)
    endif()
endfunction()

if(NOT DEFINED PYTHON_EXECUTABLE)
    if(WIN32)
        set(PYTHON_EXECUTABLE "python.exe")
    else()
        set(PYTHON_EXECUTABLE "python")
    endif()
endif()

if(UNIX)
    system(
        STRIP OUTPUT_VARIABLE PYTHON_EXECUTABLE
        COMMAND bash -c "type -P '${PYTHON_EXECUTABLE}'"
    )
endif()

system(
    STRIP OUTPUT_VARIABLE PYTHON_VERSION
    COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('platform').python_version())"
)

message(STATUS "Use Python version: ${PYTHON_VERSION}")
message(STATUS "Use Python executable: \"${PYTHON_EXECUTABLE}\"")

if(NOT DEFINED PYTHON_INCLUDE_DIR)
    message(STATUS "Auto detecting Python include directory...")
    system(
        STRIP OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('platinclude'))"
    )
endif()

if("${PYTHON_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "Python include directory not found")
else()
    message(STATUS "Detected Python include directory: \"${PYTHON_INCLUDE_DIR}\"")
    include_directories("${PYTHON_INCLUDE_DIR}")
endif()

# Include pybind11
set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}")

if(NOT DEFINED PYBIND11_CMAKE_DIR)
    message(STATUS "Auto detecting pybind11 CMake directory...")
    system(
        STRIP OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --cmakedir
    )
endif()

if("${PYBIND11_CMAKE_DIR}" STREQUAL "")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG "${PYBIND11_VERSION}"
        GIT_SHALLOW TRUE
        SOURCE_DIR "${THIRD_PARTY_DIR}/pybind11"
        BINARY_DIR "${THIRD_PARTY_DIR}/.cmake/pybind11/build"
        STAMP_DIR "${THIRD_PARTY_DIR}/.cmake/pybind11/stamp"
    )
    FetchContent_GetProperties(pybind11)

    if(NOT pybind11_POPULATED)
        message(STATUS "Populating Git repository pybind11@${PYBIND11_VERSION} to third-party/pybind11...")
        FetchContent_MakeAvailable(pybind11)
    endif()
else()
    message(STATUS "Detected Pybind11 CMake directory: \"${PYBIND11_CMAKE_DIR}\"")
    find_package(pybind11 CONFIG PATHS "${PYBIND11_CMAKE_DIR}")
endif()

# Include abseil-cpp
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
FetchContent_Declare(
    abseilcpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG "${ABSEIL_CPP_VERSION}"
    GIT_SHALLOW TRUE
    SOURCE_DIR "${THIRD_PARTY_DIR}/abseil-cpp"
    BINARY_DIR "${THIRD_PARTY_DIR}/.cmake/abseil-cpp/build"
    STAMP_DIR "${THIRD_PARTY_DIR}/.cmake/abseil-cpp/stamp"
)
FetchContent_GetProperties(abseilcpp)

if(NOT abseilcpp_POPULATED)
    message(STATUS "Populating Git repository abseil-cpp@${ABSEIL_CPP_VERSION} to third-party/abseil-cpp...")
    FetchContent_MakeAvailable(abseilcpp)
endif()

include_directories("${CMAKE_SOURCE_DIR}")
add_subdirectory(src)
