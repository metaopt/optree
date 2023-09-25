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

set(
    optree_csrc
    optree.cpp
    registry.cpp
    treespec/treespec.cpp
    treespec/flatten.cpp
    treespec/unflatten.cpp
    treespec/traversal.cpp
)

pybind11_add_module(_C MODULE THIN_LTO "${optree_csrc}")

target_link_libraries(
    _C
    PUBLIC
    absl::flat_hash_map
    absl::str_format
)
