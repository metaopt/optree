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
"""OpTree: Optimized PyTree Utilities."""

from optree import integration, typing
from optree.ops import (
    MAX_RECURSION_DEPTH,
    NONE_IS_LEAF,
    NONE_IS_NODE,
    all_leaves,
    broadcast_common,
    broadcast_prefix,
    prefix_errors,
    tree_all,
    tree_any,
    tree_broadcast_common,
    tree_broadcast_map,
    tree_broadcast_map_with_path,
    tree_broadcast_prefix,
    tree_flatten,
    tree_flatten_one_level,
    tree_flatten_with_path,
    tree_is_leaf,
    tree_leaves,
    tree_map,
    tree_map_,
    tree_map_with_path,
    tree_map_with_path_,
    tree_max,
    tree_min,
    tree_paths,
    tree_reduce,
    tree_replace_nones,
    tree_structure,
    tree_sum,
    tree_transpose,
    tree_unflatten,
    treespec_child,
    treespec_children,
    treespec_entries,
    treespec_entry,
    treespec_is_leaf,
    treespec_is_prefix,
    treespec_is_strict_leaf,
    treespec_is_suffix,
    treespec_leaf,
    treespec_none,
    treespec_paths,
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
from optree.typing import (
    CustomTreeNode,
    FlattenFunc,
    PyTree,
    PyTreeDef,
    PyTreeKind,
    PyTreeSpec,
    PyTreeTypeVar,
    UnflattenFunc,
    is_namedtuple,
    is_namedtuple_class,
    is_structseq,
    is_structseq_class,
    namedtuple_fields,
    structseq_fields,
)
from optree.version import __version__


__all__ = [
    # Tree operations
    'MAX_RECURSION_DEPTH',
    'NONE_IS_NODE',
    'NONE_IS_LEAF',
    'tree_flatten',
    'tree_flatten_with_path',
    'tree_unflatten',
    'tree_leaves',
    'tree_structure',
    'tree_paths',
    'tree_is_leaf',
    'all_leaves',
    'tree_map',
    'tree_map_',
    'tree_map_with_path',
    'tree_map_with_path_',
    'tree_replace_nones',
    'tree_transpose',
    'tree_broadcast_prefix',
    'broadcast_prefix',
    'tree_broadcast_common',
    'broadcast_common',
    'tree_broadcast_map',
    'tree_broadcast_map_with_path',
    'tree_reduce',
    'tree_sum',
    'tree_max',
    'tree_min',
    'tree_all',
    'tree_any',
    'tree_flatten_one_level',
    'prefix_errors',
    'treespec_is_prefix',
    'treespec_is_suffix',
    'treespec_paths',
    'treespec_entries',
    'treespec_entry',
    'treespec_children',
    'treespec_child',
    'treespec_is_leaf',
    'treespec_is_strict_leaf',
    'treespec_leaf',
    'treespec_none',
    'treespec_tuple',
    # Registry
    'register_pytree_node',
    'register_pytree_node_class',
    'Partial',
    'register_keypaths',
    'AttributeKeyPathEntry',
    'GetitemKeyPathEntry',
    # Typing
    'PyTreeSpec',
    'PyTreeDef',
    'PyTreeKind',
    'PyTree',
    'PyTreeTypeVar',
    'CustomTreeNode',
    'FlattenFunc',
    'UnflattenFunc',
    'is_namedtuple',
    'is_namedtuple_class',
    'namedtuple_fields',
    'is_structseq',
    'is_structseq_class',
    'structseq_fields',
]

MAX_RECURSION_DEPTH: int = MAX_RECURSION_DEPTH  # 2000
"""Maximum recursion depth for pytree traversal. It is 2000.

This limit prevents infinite recursion from causing an overflow of the C stack
and crashing Python.
"""
NONE_IS_NODE: bool = NONE_IS_NODE  # literal constant
"""Literal constant that treats :data:`None` as a pytree non-leaf node."""
NONE_IS_LEAF: bool = NONE_IS_LEAF  # literal constant
"""Literal constant that treats :data:`None` as a pytree leaf node."""
