# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.18)
project(optree LANGUAGES CXX)

include(FetchContent)

set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third-party")

set(pybind11_MINIMUM_VERSION 2.12)  # for pybind11::gil_safe_call_once_and_store
if(NOT DEFINED pybind11_VERSION AND NOT "$ENV{pybind11_VERSION}" STREQUAL "")
    set(pybind11_VERSION "$ENV{pybind11_VERSION}")
endif()
if(NOT pybind11_VERSION)
    set(pybind11_VERSION stable)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(NOT DEFINED CMAKE_CXX_STANDARD AND NOT "$ENV{CMAKE_CXX_STANDARD}" STREQUAL "")
    set(CMAKE_CXX_STANDARD "$ENV{CMAKE_CXX_STANDARD}")
endif()
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)  # for likely/unlikely attributes
endif()
if (CMAKE_CXX_STANDARD VERSION_LESS 17)
    message(FATAL_ERROR "C++17 or higher is required")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Use C++ standard: C++${CMAKE_CXX_STANDARD}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC
set(CMAKE_CXX_VISIBILITY_PRESET hidden)  # -fvisibility=hidden

string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_CXX_FLAGS_DEBUG}" CMAKE_CXX_FLAGS_DEBUG)
string(STRIP "${CMAKE_CXX_FLAGS_RELEASE}" CMAKE_CXX_FLAGS_RELEASE)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI AND NOT "$ENV{_GLIBCXX_USE_CXX11_ABI}" STREQUAL "")
        set(_GLIBCXX_USE_CXX11_ABI "$ENV{_GLIBCXX_USE_CXX11_ABI}")
    endif()
    if(_GLIBCXX_USE_CXX11_ABI)
        message(STATUS "Use _GLIBCXX_USE_CXX11_ABI: ${_GLIBCXX_USE_CXX11_ABI}")
        add_definitions("-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}")  # use C++11 ABI
    endif()
endif()

if(MSVC)
    string(
        APPEND CMAKE_CXX_FLAGS
        " /EHsc /bigobj"
        " /Zc:preprocessor"
        " /experimental:external /external:anglebrackets /external:W0"
        " /Wall /Wv:19.40"  # Visual Studio 2022 version 17.10
        # Suppress following warnings
        " /wd4365"  # conversion from 'type_1' to 'type_2', signed/unsigned mismatch
        " /wd4514"  # unreferenced inline function has been removed
        " /wd4710"  # function not inlined
        " /wd4711"  # function selected for inline expansion
        " /wd4714"  # function marked as forceinline not inlined
        " /wd4820"  # bytes padding added after construct 'member_name'
        " /wd4868"  # compiler may not enforce left-to-right evaluation order in braced initializer list
        " /wd5045"  # compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
        " /wd5262"  # use [[fallthrough]] when a break statement is intentionally omitted between cases
        " /wd5264"  # 'const' variable is not used
    )
    string(
        APPEND CMAKE_CXX_FLAGS_DEBUG
        " /wd4702"  # unreachable code
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

string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_CXX_FLAGS_DEBUG}" CMAKE_CXX_FLAGS_DEBUG)
string(STRIP "${CMAKE_CXX_FLAGS_RELEASE}" CMAKE_CXX_FLAGS_RELEASE)
message(STATUS "CXX flags: \"${CMAKE_CXX_FLAGS}\"")
message(STATUS "CXX flags (Debug): \"${CMAKE_CXX_FLAGS_DEBUG}\"")
message(STATUS "CXX flags (Release): \"${CMAKE_CXX_FLAGS_RELEASE}\"")

if(MSVC AND NOT "$ENV{VSCMD_ARG_TGT_ARCH}" STREQUAL "")
    message(STATUS "Use VSCMD_ARG_TGT_ARCH: \"$ENV{VSCMD_ARG_TGT_ARCH}\"")
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

if(NOT DEFINED Python_EXECUTABLE)
    if(WIN32)
        set(Python_EXECUTABLE "python.exe")
    else()
        set(Python_EXECUTABLE "python")
    endif()
endif()

if(UNIX)
    system(
        STRIP OUTPUT_VARIABLE Python_EXECUTABLE
        COMMAND bash -c "type -P '${Python_EXECUTABLE}'"
    )
endif()

system(
    STRIP OUTPUT_VARIABLE Python_VERSION
    COMMAND "${Python_EXECUTABLE}" -c "print('.'.join(map(str, __import__('sys').version_info[:3])))"
)

message(STATUS "Use Python version: ${Python_VERSION}")
message(STATUS "Use Python executable: \"${Python_EXECUTABLE}\"")

if(NOT DEFINED Python_INCLUDE_DIR)
    message(STATUS "Auto detecting Python include directory...")
    system(
        STRIP OUTPUT_VARIABLE Python_INCLUDE_DIR
        COMMAND "${Python_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('platinclude'))"
    )
endif()

if("${Python_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "Python include directory not found")
else()
    message(STATUS "Detected Python include directory: \"${Python_INCLUDE_DIR}\"")
    include_directories("${Python_INCLUDE_DIR}")
endif()

if(DEFINED Python_EXTRA_INCLUDE_DIRS)
    message(STATUS "Use Python_EXTRA_INCLUDE_DIRS: \"${Python_EXTRA_INCLUDE_DIRS}\"")
    foreach(Python_EXTRA_INCLUDE_DIR IN LISTS Python_EXTRA_INCLUDE_DIRS)
        include_directories("${Python_EXTRA_INCLUDE_DIR}")
    endforeach()
endif()
if(DEFINED Python_EXTRA_LIBRARY_DIRS)
    message(STATUS "Use Python_EXTRA_LIBRARY_DIRS: \"${Python_EXTRA_LIBRARY_DIRS}\"")
    list(PREPEND CMAKE_PREFIX_PATH "${Python_EXTRA_LIBRARY_DIRS}")
    foreach(Python_EXTRA_LIBRARY_DIR IN LISTS Python_EXTRA_LIBRARY_DIRS)
        link_directories("${Python_EXTRA_LIBRARY_DIR}")
    endforeach()
endif()
if(DEFINED Python_EXTRA_LIBRARIES)
    message(STATUS "Use Python_EXTRA_LIBRARIES: \"${Python_EXTRA_LIBRARIES}\"")
endif()

# Include pybind11
set(PYBIND11_PYTHON_VERSION "${Python_VERSION}")
set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_PYTHONLIBS_OVERWRITE OFF)

if(NOT DEFINED pybind11_DIR)
    message(STATUS "Auto detecting pybind11 CMake directory...")
    system(
        STRIP OUTPUT_VARIABLE pybind11_DIR
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    )
endif()

if("${pybind11_DIR}" STREQUAL "")
    find_package(pybind11 "${pybind11_MINIMUM_VERSION}" CONFIG)
    if(pybind11_FOUND)
        message(STATUS "Detected pybind11 CMake directory: \"${pybind11_DIR}\"")
    else()
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG "${pybind11_VERSION}"
            GIT_SHALLOW TRUE
            SOURCE_DIR "${THIRD_PARTY_DIR}/pybind11"
            BINARY_DIR "${THIRD_PARTY_DIR}/.cmake/pybind11/build"
            STAMP_DIR "${THIRD_PARTY_DIR}/.cmake/pybind11/stamp"
        )
        FetchContent_GetProperties(pybind11)

        if(NOT pybind11_POPULATED)
            message(STATUS "Populating Git repository pybind11@${pybind11_VERSION} to third-party/pybind11...")
            FetchContent_MakeAvailable(pybind11)
        endif()
    endif()
else()
    message(STATUS "Detected Pybind11 CMake directory: \"${pybind11_DIR}\"")
    list(PREPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
    find_package(pybind11 "${pybind11_MINIMUM_VERSION}" CONFIG REQUIRED)
endif()

set(SETUPTOOLS_EXT_SUFFIX "$ENV{SETUPTOOLS_EXT_SUFFIX}")
if(SETUPTOOLS_EXT_SUFFIX)
    message(STATUS "Use SETUPTOOLS_EXT_SUFFIX: \"${SETUPTOOLS_EXT_SUFFIX}\"")
    if(NOT "${SETUPTOOLS_EXT_SUFFIX}" STREQUAL "${PYTHON_MODULE_EXTENSION}")
        message(STATUS "Overwrite PYTHON_MODULE_EXTENSION: "
                       "\"${PYTHON_MODULE_EXTENSION}\" -> \"${SETUPTOOLS_EXT_SUFFIX}\"")
        set(PYTHON_MODULE_EXTENSION "$ENV{SETUPTOOLS_EXT_SUFFIX}")
    endif()
endif()
message(STATUS "Use PYTHON_MODULE_EXTENSION: \"${PYTHON_MODULE_EXTENSION}\"")

include_directories("${CMAKE_SOURCE_DIR}/include")
add_subdirectory(src)
