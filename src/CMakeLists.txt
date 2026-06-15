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

set(
    optree_csrc
    optree.cpp
    registry.cpp
    treespec/constructors.cpp
    treespec/treespec.cpp
    treespec/flatten.cpp
    treespec/unflatten.cpp
    treespec/traversal.cpp
    treespec/serialization.cpp
    treespec/hashing.cpp
    treespec/richcomparison.cpp
    treespec/gc.cpp
)

pybind11_add_module(_C MODULE "${optree_csrc}")

# Free-threaded (PEP 703) builds must compile the extension with `Py_GIL_DISABLED` in order to make
# the `PyObject` / `PyTypeObject` header layout match the free-threaded interpreter it loads into.
#
# `Py_GIL_DISABLED` is not defined for extensions automatically: CPython 3.14+ dropped it from the
# static Windows `pyconfig.h` (python/cpython#133966; its comment says "a build backend will define
# it with the value"). CMake's `FindPython` does add it via `Python_DEFINITIONS` for a free-threaded
# RELEASE interpreter, but misses a free-threaded DEBUG one -- an upstream CMake bug. It files the
# debug import library `python{X}{Y}t_d.lib` in the release slot (`Python_LIBRARY_RELEASE`), while
# its free-threaded test only matches a `...t.lib` suffix, so the `_d` debug suffix slips through.
# Probe the interpreter directly instead -- authoritative for release/debug and GIL/free-threaded.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "print('ON' if __import__('sysconfig').get_config_var('Py_GIL_DISABLED') else 'OFF')"
    OUTPUT_VARIABLE OPTREE_Py_GIL_DISABLED
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(OPTREE_Py_GIL_DISABLED)
    message(STATUS "Free-threaded Python detected: defining Py_GIL_DISABLED=1 for target _C")
    target_compile_definitions(_C PRIVATE Py_GIL_DISABLED=1)
endif()

if(DEFINED Python_EXTRA_LIBRARIES)
    target_link_libraries(_C PRIVATE ${Python_EXTRA_LIBRARIES})
endif()
