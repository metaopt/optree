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

if(MSVC)
    # Force the pre-VS-2022-17.10 `std::mutex` layout on the `_C` target so the constructor's
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
    target_compile_definitions(_C PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

if(DEFINED Python_EXTRA_LIBRARIES)
    target_link_libraries(_C PRIVATE ${Python_EXTRA_LIBRARIES})
endif()
