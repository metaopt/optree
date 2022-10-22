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
"""OpTree: Optimized PyTree Utilities."""

from optree.ops import (
    NONE_IS_LEAF,
    NONE_IS_NODE,
    all_leaves,
    prefix_errors,
    tree_all,
    tree_any,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_reduce,
    tree_replace_nones,
    tree_structure,
    tree_transpose,
    tree_unflatten,
    treespec_children,
    treespec_is_leaf,
    treespec_is_strict_leaf,
    treespec_leaf,
    treespec_none,
    treespec_tuple,
)
from optree.registry import (
    AttributeKeyPathEntry,
    GetitemKeyPathEntry,
    Partial,
    register_keypaths,
    register_pytree_node,
    register_pytree_node_class,
)
from optree.typing import CustomTreeNode, PyTree, PyTreeDef, PyTreeSpec, PyTreeTypeVar
from optree.version import __version__


__all__ = [
    # Typing
    'PyTreeSpec',
    'PyTreeDef',
    'PyTree',
    'PyTreeTypeVar',
    'CustomTreeNode',
    # Tree operations
    'NONE_IS_NODE',
    'NONE_IS_LEAF',
    'tree_flatten',
    'tree_unflatten',
    'tree_leaves',
    'tree_structure',
    'all_leaves',
    'tree_map',
    'tree_reduce',
    'tree_transpose',
    'tree_replace_nones',
    'tree_all',
    'tree_any',
    'treespec_children',
    'treespec_is_leaf',
    'treespec_is_strict_leaf',
    'treespec_leaf',
    'treespec_none',
    'treespec_tuple',
    'prefix_errors',
    # Registry
    'register_pytree_node',
    'register_pytree_node_class',
    'Partial',
    'register_keypaths',
    'AttributeKeyPathEntry',
    'GetitemKeyPathEntry',
]
