# Copyright 2022-2026 MetaOPT Team. All Rights Reserved.
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

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third-party")

macro(setdefault_ifndef varname)  # setdefault_ifndef(<varname> [<default>])
    if(NOT DEFINED "${varname}" AND NOT "$ENV{${varname}}" STREQUAL "")
        set("${varname}" "$ENV{${varname}}")
    endif()
    if("${${varname}}" STREQUAL "" AND NOT "${ARGN}" STREQUAL "")
        set("${varname}" "${ARGN}")
    endif()
endmacro()

set(pybind11_MINIMUM_VERSION 2.12)  # for pybind11::gil_safe_call_once_and_store
setdefault_ifndef(pybind11_VERSION stable)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

setdefault_ifndef(CMAKE_CXX_STANDARD 20)  # for likely/unlikely attributes
if(CMAKE_CXX_STANDARD VERSION_LESS 17)
    message(FATAL_ERROR "C++17 or higher is required")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Use C++ standard: C++${CMAKE_CXX_STANDARD}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC
set(CMAKE_CXX_VISIBILITY_PRESET hidden)  # -fvisibility=hidden

string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_CXX_FLAGS_DEBUG}" CMAKE_CXX_FLAGS_DEBUG)
string(STRIP "${CMAKE_CXX_FLAGS_RELEASE}" CMAKE_CXX_FLAGS_RELEASE)

setdefault_ifndef(_GLIBCXX_USE_CXX11_ABI 1)
if(NOT "${_GLIBCXX_USE_CXX11_ABI}" STREQUAL "")
    message(STATUS "Use _GLIBCXX_USE_CXX11_ABI: ${_GLIBCXX_USE_CXX11_ABI}")
    add_definitions("-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}")
endif()

if(MSVC)
    string(
        APPEND CMAKE_CXX_FLAGS
        " /EHsc /bigobj"
        " /Zc:preprocessor"
        " /experimental:external /external:anglebrackets /external:W0"
        # https://learn.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warnings-by-compiler-version
        " /Wall /Wv:19.45"  # Visual Studio 2022 version 17.15
        # Suppress following warnings
        " /wd4127"  # conditional expression is constant
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
        " /Zi"
    )
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2 /Ob2")

    # Force the pre-VS-2022-17.10 `std::mutex` layout on MSVC builds so the constructor's
    # `_Mtx_init_in_situ` call and the `_Mtx_lock` body both bind to whichever `msvcp140.dll` the
    # Windows loader resolves at runtime. Without this, if another wheel in the process preloads an
    # older `msvcp140.dll` (e.g., pyarrow's bundled 14.28.29334.0), the older `_Mtx_lock` interprets
    # the newer constexpr-zero-init mutex layout as the old ConcRT-backed object and dereferences a
    # null vptr, crashing inside pybind11's `internals_pp_manager` during `PyInit__C` before any
    # optree code runs. `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` is Microsoft's documented escape
    # hatch for this newer-toolset / older-redistributable ABI mismatch:
    #
    #   https://github.com/microsoft/STL/releases/tag/vs-2022-17.10
    #
    # See also: https://github.com/metaopt/optree/issues/278.
    #
    # Remove once optree no longer ships wheels that may be loaded alongside an `msvcp140.dll` older
    # than the toolset used to build the wheel.
    #
    # | CMake variable       | environment variable | mutex workaround |
    # |----------------------|----------------------|------------------|
    # | defined as empty     | (whatever)           | disabled         |
    # | defined as non-empty | (whatever)           | enabled          |
    # |   defined as "ON"    | (whatever)           | enabled          |
    # |   defined as "OFF"   | (whatever)           | enabled          |
    # | not defined          | defined as empty     | disabled         |
    # | not defined          | defined as non-empty | enabled          |
    # | not defined          |   defined as "ON"    | enabled          |
    # | not defined          |   defined as "OFF"   | enabled          |
    # | not defined          | not defined          | enabled          |
    if (DEFINED _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR OR DEFINED ENV{_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR})
        setdefault_ifndef(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    else()
        setdefault_ifndef(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR 1)
    endif()
    if(NOT "${_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR}" STREQUAL "")
        message(STATUS "Use _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR: enabled")
        add_definitions("-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
    else()
        message(STATUS "_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR is disabled, using default MSVC mutex layout.")
    endif()
else()
    string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g -Og")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
endif()

setdefault_ifndef(OPTREE_CXX_WERROR OFF)
if(OPTREE_CXX_WERROR)
    message(
        AUTHOR_WARNING
        "Treat all compiler warnings as errors. Set `OPTREE_CXX_WERROR=OFF` to disable this."
    )
    if(MSVC)
        string(APPEND CMAKE_CXX_FLAGS " /WX")
    else()
        string(APPEND CMAKE_CXX_FLAGS " -Werror -Wno-error=attributes -Wno-error=redundant-move")
    endif()
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_CXX_FLAGS_DEBUG}" CMAKE_CXX_FLAGS_DEBUG)
string(STRIP "${CMAKE_CXX_FLAGS_RELEASE}" CMAKE_CXX_FLAGS_RELEASE)
message(STATUS "CXX flags: \"${CMAKE_CXX_FLAGS}\"")
message(STATUS "CXX flags (Debug): \"${CMAKE_CXX_FLAGS_DEBUG}\"")
message(STATUS "CXX flags (Release): \"${CMAKE_CXX_FLAGS_RELEASE}\"")
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    string(STRIP "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}" "CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}")
    message(STATUS "CXX flags (${CMAKE_BUILD_TYPE}): \"${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}\"")
endif()

if(NOT DEFINED "CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE_UPPER}")
    set("CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE_UPPER}" "${CMAKE_BINARY_DIR}/lib")
endif()
message(STATUS "Library output directory (${CMAKE_BUILD_TYPE}): "
               "\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE_UPPER}}\"")

if(MSVC AND NOT "$ENV{VSCMD_ARG_TGT_ARCH}" STREQUAL "")
    message(STATUS "Use VSCMD_ARG_TGT_ARCH: \"$ENV{VSCMD_ARG_TGT_ARCH}\"")
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

if(DEFINED Python_ROOT_DIR)
    message(STATUS "Use Python_ROOT_DIR: \"${Python_ROOT_DIR}\"")
endif()

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
    if(NOT EXISTS "${Python_INCLUDE_DIR}/Python.h")
        message(WARNING "Python.h not found in \"${Python_INCLUDE_DIR}\"")
        if(DEFINED Python_EXTRA_INCLUDE_DIRS)
            message(STATUS "Looking for Python.h in Python_EXTRA_INCLUDE_DIRS: "
                           "\"${Python_EXTRA_INCLUDE_DIRS}\"")
            foreach(Python_EXTRA_INCLUDE_DIR IN LISTS Python_EXTRA_INCLUDE_DIRS)
                if(EXISTS "${Python_EXTRA_INCLUDE_DIR}/Python.h")
                    set(Python_INCLUDE_DIR "${Python_EXTRA_INCLUDE_DIR}")
                    message(STATUS "Detected Python.h in \"${Python_INCLUDE_DIR}\"")
                    break()
                endif()
            endforeach()
            if(NOT EXISTS "${Python_INCLUDE_DIR}/Python.h")
                message(WARNING "Python.h not found in Python_EXTRA_INCLUDE_DIRS")
            endif()
        endif()
    endif()
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
        message(STATUS "Populating Git repository pybind11@${pybind11_VERSION} to third-party/pybind11...")
        FetchContent_MakeAvailable(pybind11)
    endif()
else()
    message(STATUS "Detected pybind11 CMake directory: \"${pybind11_DIR}\"")
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

foreach(
    varname IN ITEMS
    CMAKE_SYSTEM_NAME
    CMAKE_GENERATOR_PLATFORM
    CMAKE_OSX_SYSROOT
    CMAKE_OSX_DEPLOYMENT_TARGET
    CMAKE_OSX_ARCHITECTURES
    Python_ROOT_DIR
    Python_INCLUDE_DIR
    Python_INCLUDE_DIRS
    Python_LIBRARY
    Python_LIBRARIES
    PYTHON_MODULE_DEBUG_POSTFIX
    PYTHON_MODULE_EXTENSION
    PYTHON_IS_DEBUG
    SETUPTOOLS_EXT_SUFFIX
)
    if(NOT "${${varname}}" STREQUAL "")
        message(STATUS "Use ${varname}: \"${${varname}}\"")
    endif()
endforeach()

include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory("${PROJECT_SOURCE_DIR}/src")
