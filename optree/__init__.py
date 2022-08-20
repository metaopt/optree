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
"""OpTree: Optimized PyTree."""

from optree.ops import (
    all_leaves,
    build_tree,
    tree_all,
    tree_any,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_reduce,
    tree_structure,
    tree_transpose,
    tree_unflatten,
    treedef_children,
    treedef_is_leaf,
    treedef_is_strict_leaf,
    treedef_tuple,
)
from optree.registry import (
    AttributeKeyPathEntry,
    GetitemKeyPathEntry,
    Partial,
    PyTreeDef,
    register_keypaths,
    register_pytree_node,
    register_pytree_node_class,
)
from optree.version import __version__


__all__ = [
    'AttributeKeyPathEntry',
    'GetitemKeyPathEntry',
    'Partial',
    'PyTreeDef',
    'all_leaves',
    'build_tree',
    'register_keypaths',
    'register_pytree_node',
    'register_pytree_node_class',
    'tree_all',
    'tree_any',
    'tree_flatten',
    'tree_leaves',
    'tree_map',
    'tree_reduce',
    'tree_structure',
    'tree_transpose',
    'tree_unflatten',
    'treedef_children',
    'treedef_is_leaf',
    'treedef_is_strict_leaf',
    'treedef_tuple',
]
