# Copyright 2022 MetaOPT Team. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.4)
project(optree LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)  # for likely/unlikely attributes
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC
set(CMAKE_CXX_VISIBILITY_PRESET hidden)  # -fvisibility=hidden

if(MSVC)
    string(APPEND CMAKE_CXX_FLAGS " /Wall")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " /Zi")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2 /Ob2")
else()
    string(APPEND CMAKE_CXX_FLAGS " -Wall")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g -Og")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
endif()

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
    include_directories(${PYTHON_INCLUDE_DIR})
endif()

set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}")

if(NOT DEFINED PYBIND11_CMAKE_DIR)
    message(STATUS "Auto detecting pybind11 CMake directory...")
    system(
        STRIP OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --cmakedir
    )
endif()

if("${PYBIND11_CMAKE_DIR}" STREQUAL "")
    add_subdirectory(third_party/pybind11)
    include_directories(third_party/pybind11/include)
else()
    message(STATUS "Detected Pybind11 CMake directory: \"${PYBIND11_CMAKE_DIR}\"")
    find_package(pybind11 CONFIG PATHS "${PYBIND11_CMAKE_DIR}")
endif()

set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)

include_directories("${CMAKE_SOURCE_DIR}")
add_subdirectory(src)
