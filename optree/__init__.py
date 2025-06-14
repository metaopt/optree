# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
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

from optree import accessors, dataclasses, functools, integrations, pytree, treespec, typing
from optree.accessors import (
    AutoEntry,
    DataclassEntry,
    FlattenedEntry,
    GetAttrEntry,
    GetItemEntry,
    MappingEntry,
    NamedTupleEntry,
    PyTreeAccessor,
    PyTreeEntry,
    SequenceEntry,
    StructSequenceEntry,
)
from optree.ops import (
    MAX_RECURSION_DEPTH,
    NONE_IS_LEAF,
    NONE_IS_NODE,
    all_leaves,
    broadcast_common,
    broadcast_prefix,
    prefix_errors,
    tree_accessors,
    tree_all,
    tree_any,
    tree_broadcast_common,
    tree_broadcast_map,
    tree_broadcast_map_with_accessor,
    tree_broadcast_map_with_path,
    tree_broadcast_prefix,
    tree_flatten,
    tree_flatten_one_level,
    tree_flatten_with_accessor,
    tree_flatten_with_path,
    tree_is_leaf,
    tree_iter,
    tree_leaves,
    tree_map,
    tree_map_,
    tree_map_with_accessor,
    tree_map_with_accessor_,
    tree_map_with_path,
    tree_map_with_path_,
    tree_max,
    tree_min,
    tree_partition,
    tree_paths,
    tree_reduce,
    tree_replace_nones,
    tree_structure,
    tree_sum,
    tree_transpose,
    tree_transpose_map,
    tree_transpose_map_with_accessor,
    tree_transpose_map_with_path,
    tree_unflatten,
    treespec_accessors,
    treespec_child,
    treespec_children,
    treespec_defaultdict,
    treespec_deque,
    treespec_dict,
    treespec_entries,
    treespec_entry,
    treespec_from_collection,
    treespec_is_leaf,
    treespec_is_one_level,
    treespec_is_prefix,
    treespec_is_strict_leaf,
    treespec_is_suffix,
    treespec_leaf,
    treespec_list,
    treespec_namedtuple,
    treespec_none,
    treespec_one_level,
    treespec_ordereddict,
    treespec_paths,
    treespec_structseq,
    treespec_transform,
    treespec_tuple,
)
from optree.registry import (
    dict_insertion_ordered,
    register_pytree_node,
    register_pytree_node_class,
    unregister_pytree_node,
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
    is_namedtuple_instance,
    is_structseq,
    is_structseq_class,
    is_structseq_instance,
    namedtuple_fields,
    structseq_fields,
)
from optree.version import __version__ as __version__  # pylint: disable=useless-import-alias


__all__ = [
    # Tree operations
    'MAX_RECURSION_DEPTH',
    'NONE_IS_NODE',
    'NONE_IS_LEAF',
    'tree_flatten',
    'tree_flatten_with_path',
    'tree_flatten_with_accessor',
    'tree_unflatten',
    'tree_iter',
    'tree_leaves',
    'tree_structure',
    'tree_paths',
    'tree_accessors',
    'tree_is_leaf',
    'all_leaves',
    'tree_map',
    'tree_map_',
    'tree_map_with_path',
    'tree_map_with_path_',
    'tree_map_with_accessor',
    'tree_map_with_accessor_',
    'tree_replace_nones',
    'tree_partition',
    'tree_transpose',
    'tree_transpose_map',
    'tree_transpose_map_with_path',
    'tree_transpose_map_with_accessor',
    'tree_broadcast_prefix',
    'broadcast_prefix',
    'tree_broadcast_common',
    'broadcast_common',
    'tree_broadcast_map',
    'tree_broadcast_map_with_path',
    'tree_broadcast_map_with_accessor',
    'tree_reduce',
    'tree_sum',
    'tree_max',
    'tree_min',
    'tree_all',
    'tree_any',
    'tree_flatten_one_level',
    'prefix_errors',
    'treespec_paths',
    'treespec_accessors',
    'treespec_entries',
    'treespec_entry',
    'treespec_children',
    'treespec_child',
    'treespec_one_level',
    'treespec_transform',
    'treespec_is_leaf',
    'treespec_is_strict_leaf',
    'treespec_is_one_level',
    'treespec_is_prefix',
    'treespec_is_suffix',
    'treespec_leaf',
    'treespec_none',
    'treespec_tuple',
    'treespec_list',
    'treespec_dict',
    'treespec_namedtuple',
    'treespec_ordereddict',
    'treespec_defaultdict',
    'treespec_deque',
    'treespec_structseq',
    'treespec_from_collection',
    # Accessor
    'PyTreeEntry',
    'GetAttrEntry',
    'GetItemEntry',
    'FlattenedEntry',
    'AutoEntry',
    'SequenceEntry',
    'MappingEntry',
    'NamedTupleEntry',
    'StructSequenceEntry',
    'DataclassEntry',
    'PyTreeAccessor',
    # Registry
    'register_pytree_node',
    'register_pytree_node_class',
    'unregister_pytree_node',
    'dict_insertion_ordered',
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
    'is_namedtuple_instance',
    'namedtuple_fields',
    'is_structseq',
    'is_structseq_class',
    'is_structseq_instance',
    'structseq_fields',
]

MAX_RECURSION_DEPTH: int = MAX_RECURSION_DEPTH
"""Maximum recursion depth for pytree traversal.

This limit prevents infinite recursion from causing an overflow of the C stack
and crashing Python.
"""
NONE_IS_NODE: bool = NONE_IS_NODE  # literal constant
"""Literal constant that treats :data:`None` as a pytree non-leaf node."""
NONE_IS_LEAF: bool = NONE_IS_LEAF  # literal constant
"""Literal constant that treats :data:`None` as a pytree leaf node."""


# pylint: disable-next=fixme
# TODO: remove this function in version 0.18.0
def __getattr__(name: str, /) -> object:  # pragma: no cover
    """Get an attribute from the module."""
    if name == 'accessor':
        global accessor  # pylint: disable=global-statement

        import optree.accessor as accessor  # pylint: disable=import-outside-toplevel

        return accessor  # type: ignore[name-defined]
    if name == 'integration':
        global integration  # pylint: disable=global-statement

        import optree.integration as integration  # pylint: disable=import-outside-toplevel

        return integration  # type: ignore[name-defined]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
