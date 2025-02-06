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

# pylint: disable=too-many-lines

from __future__ import annotations

import functools
import itertools
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Generic,
    Iterable,
    overload,
)

import optree._C as _C
from optree.accessor import PyTreeAccessor
from optree.typing import NamedTuple, T


if TYPE_CHECKING:
    import builtins

    from optree.accessor import PyTreeEntry
    from optree.typing import MetaData, PyTree, PyTreeKind, PyTreeSpec, S, U


__all__ = [
    'flatten',
    'flatten_with_path',
    'flatten_with_accessor',
    'unflatten',
    'iter',
    'leaves',
    'structure',
    'paths',
    'accessors',
    'is_leaf',
    'map',
    'map_',
    'map_with_path',
    'map_with_path_',
    'map_with_accessor',
    'map_with_accessor_',
    'replace_nones',
    'transpose',
    'transpose_map',
    'transpose_map_with_path',
    'transpose_map_with_accessor',
    'broadcast_map',
    'broadcast_map_with_path',
    'broadcast_map_with_accessor',
    'reduce',
    'sum',
    'max',
    'min',
    'all',
    'any',
    'flatten_one_level',
]

MAX_RECURSION_DEPTH: int = _C.MAX_RECURSION_DEPTH  # 1000
"""Maximum recursion depth for pytree traversal. It is 1000.

This limit prevents infinite recursion from causing an overflow of the C stack
and crashing Python.
"""
NONE_IS_NODE: bool = False  # literal constant
"""Literal constant that treats :data:`None` as a pytree non-leaf node."""
NONE_IS_LEAF: bool = True  # literal constant
"""Literal constant that treats :data:`None` as a pytree leaf node."""


def flatten(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[T], PyTreeSpec]:
    """Flatten a pytree.

    See also :func:`flatten_with_path` and :func:`unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> pytree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten(pytree)  # doctest: +IGNORE_WHITESPACE
    (
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> tree_flatten(pytree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [1, 2, 3, 4, None, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    )
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*))
    >>> tree_flatten(None)
    ([], PyTreeSpec(None))
    >>> tree_flatten(None, none_is_leaf=True)
    ([None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> pytree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten(pytree)  # doctest: +IGNORE_WHITESPACE
    (
        [2, 3, 4, 1, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}))
    )
    >>> tree_flatten(pytree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [2, 3, 4, 1, None, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}), NoneIsLeaf)
    )

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the
        second element is a treespec representing the structure of the pytree.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf, namespace)


def tree_flatten_with_path(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[tuple[Any, ...]], list[T], PyTreeSpec]:
    """Flatten a pytree and additionally record the paths.

    See also :func:`tree_flatten`, :func:`tree_paths`, and :func:`treespec_paths`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten_with_path(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('d',)],
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> tree_flatten_with_path(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('c',), ('d',)],
        [1, 2, 3, 4, None, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    )
    >>> tree_flatten_with_path(1)
    ([()], [1], PyTreeSpec(*))
    >>> tree_flatten_with_path(None)
    ([], [], PyTreeSpec(None))
    >>> tree_flatten_with_path(None, none_is_leaf=True)
    ([()], [None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten_with_path(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [('b', 0), ('b', 1, 0), ('b', 1, 1), ('a',), ('d',)],
        [2, 3, 4, 1, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}))
    )
    >>> tree_flatten_with_path(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [('b', 0), ('b', 1, 0), ('b', 1, 1), ('a',), ('c',), ('d',)],
        [2, 3, 4, 1, None, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}), NoneIsLeaf)
    )

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A triple ``(paths, leaves, treespec)``. The first element is a list of the paths to the leaf
        values, while each path is a tuple of the index or keys. The second element is a list of
        leaf values and the last element is a treespec representing the structure of the pytree.
    """
    return _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)


def tree_flatten_with_accessor(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[PyTreeAccessor], list[T], PyTreeSpec]:
    """Flatten a pytree and additionally record the accessors.

    See also :func:`tree_flatten`, :func:`tree_accessors`, and :func:`treespec_accessors`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten_with_accessor(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [
            PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
            PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
            PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
            PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
        ],
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> tree_flatten_with_accessor(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [
            PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
            PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
            PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
            PyTreeAccessor(*['c'], (MappingEntry(key='c', type=<class 'dict'>),)),
            PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
        ],
        [1, 2, 3, 4, None, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    )
    >>> tree_flatten_with_accessor(1)
    ([PyTreeAccessor(*, ())], [1], PyTreeSpec(*))
    >>> tree_flatten_with_accessor(None)
    ([], [], PyTreeSpec(None))
    >>> tree_flatten_with_accessor(None, none_is_leaf=True)
    ([PyTreeAccessor(*, ())], [None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten_with_accessor(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [
            PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
            PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
            PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'collections.OrderedDict'>),)),
            PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'collections.OrderedDict'>),))
        ],
        [2, 3, 4, 1, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}))
    )
    >>> tree_flatten_with_accessor(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [
            PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
            PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
            PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'collections.OrderedDict'>),)),
            PyTreeAccessor(*['c'], (MappingEntry(key='c', type=<class 'collections.OrderedDict'>),)),
            PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'collections.OrderedDict'>),))
        ],
        [2, 3, 4, 1, None, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}), NoneIsLeaf)
    )

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A triple ``(accessors, leaves, treespec)``. The first element is a list of accessors to the
        leaf values. The second element is a list of leaf values and the last element is a treespec
        representing the structure of the pytree.
    """  # pylint: disable=line-too-long
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    return treespec.accessors(), leaves, treespec


def tree_unflatten(treespec: PyTreeSpec, leaves: Iterable[T]) -> PyTree[T]:
    """Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> leaves, treespec = tree_flatten(tree)
    >>> tree == tree_unflatten(treespec, leaves)
    True

    Args:
        treespec (PyTreeSpec): The treespec to reconstruct.
        leaves (iterable): The list of leaves to use for reconstruction. The list must match the
            number of leaves of the treespec.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by
        ``treespec``.
    """
    return treespec.unflatten(leaves)


def tree_iter(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> Iterable[T]:
    """Get an iterator over the leaves of a pytree.

    See also :func:`tree_flatten` and :func:`tree_leaves`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> list(tree_iter(tree))
    [1, 2, 3, 4, 5]
    >>> list(tree_iter(tree, none_is_leaf=True))
    [1, 2, 3, 4, None, 5]
    >>> list(tree_iter(1))
    [1]
    >>> list(tree_iter(None))
    []
    >>> list(tree_iter(None, none_is_leaf=True))
    [None]

    Args:
        tree (pytree): A pytree to iterate over.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        An iterator over the leaf values.
    """
    return _C.PyTreeIter(tree, is_leaf, none_is_leaf, namespace)


def tree_leaves(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[T]:
    """Get the leaves of a pytree.

    See also :func:`tree_flatten` and :func:`tree_iter`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_leaves(tree)
    [1, 2, 3, 4, 5]
    >>> tree_leaves(tree, none_is_leaf=True)
    [1, 2, 3, 4, None, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    []
    >>> tree_leaves(None, none_is_leaf=True)
    [None]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A list of leaf values.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf, namespace)[0]


def tree_structure(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Get the treespec for a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_structure(tree)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    >>> tree_structure(tree, none_is_leaf=True)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    >>> tree_structure(1)
    PyTreeSpec(*)
    >>> tree_structure(None)
    PyTreeSpec(None)
    >>> tree_structure(None, none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec object representing the structure of the pytree.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf, namespace)[1]


def tree_paths(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[tuple[Any, ...]]:
    """Get the path entries to the leaves of a pytree.

    See also :func:`tree_flatten`, :func:`tree_flatten_with_path`, and :func:`treespec_paths`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_paths(tree)
    [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('d',)]
    >>> tree_paths(tree, none_is_leaf=True)
    [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('c',), ('d',)]
    >>> tree_paths(1)
    [()]
    >>> tree_paths(None)
    []
    >>> tree_paths(None, none_is_leaf=True)
    [()]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A list of the paths to the leaf values, while each path is a tuple of the index or keys.
    """
    return _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)[0]


def tree_accessors(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[PyTreeAccessor]:
    """Get the accessors to the leaves of a pytree.

    See also :func:`tree_flatten`, :func:`tree_flatten_with_accessor`, and
    :func:`treespec_accessors`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_accessors(tree)  # doctest: +IGNORE_WHITESPACE
    [
        PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
        PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
        PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
        PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
        PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
    ]
    >>> tree_accessors(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    [
        PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
        PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
        PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
        PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
        PyTreeAccessor(*['c'], (MappingEntry(key='c', type=<class 'dict'>),)),
        PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
    ]
    >>> tree_accessors(1)
    [PyTreeAccessor(*, ())]
    >>> tree_accessors(None)
    []
    >>> tree_accessors(None, none_is_leaf=True)
    [PyTreeAccessor(*, ())]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A list of accessors to the leaf values.
    """  # pylint: disable=line-too-long
    return _C.flatten(tree, is_leaf, none_is_leaf, namespace)[1].accessors()


def tree_is_leaf(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether the given object is a leaf node.

    See also :func:`tree_flatten`, :func:`tree_leaves`, and :func:`all_leaves`.

    >>> tree_is_leaf(1)
    True
    >>> tree_is_leaf(None)
    False
    >>> tree_is_leaf(None, none_is_leaf=True)
    True
    >>> tree_is_leaf({'a': 1, 'b': (2, 3)})
    False

    Args:
        tree (pytree): A pytree to check if it is a leaf node.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than a leaf. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A boolean indicating if the given object is a leaf node.
    """
    return _C.is_leaf(tree, is_leaf, none_is_leaf, namespace)  # type: ignore[arg-type]


def tree_map(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`, :func:`tree_map_with_path`, :func:`tree_map_with_path_`,
    and :func:`tree_broadcast_map`.

    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (43, 65), 'z': None}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': None}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None}, none_is_leaf=True)
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:

    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, *flat_args))


def tree_map_(
    func: Callable[..., Any],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`, :func:`tree_map_with_path`, and :func:`tree_map_with_path_`.

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.
    """
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, *flat_args), maxlen=0)  # consume and exhaust the iterable
    return tree


def tree_map_with_path(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree.

    See also :func:`tree_map`, :func:`tree_map_`, and :func:`tree_map_with_path_`.

    >>> tree_map_with_path(lambda p, x: (len(p), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> tree_map_with_path(lambda p, x: x + len(p), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> tree_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}})
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: None}}
    >>> tree_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}}, none_is_leaf=True)
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: ('z', 1.5)}}

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(p, x, *xs)`` where ``(p, x)`` are the path and value at the corresponding leaf in
        ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    paths, leaves, treespec = _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, paths, *flat_args))


def tree_map_with_path_(
    func: Callable[..., Any],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:
    """Like :func:`tree_map_with_path`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`, :func:`tree_map_`, and :func:`tree_map_with_path`.

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(p, x, *xs)`` (not the return value) where ``(p, x)`` are the path and value at the
        corresponding leaf in ``tree`` and ``xs`` is the tuple of values at values at corresponding
        nodes in ``rests``.
    """
    paths, leaves, treespec = _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, paths, *flat_args), maxlen=0)  # consume and exhaust the iterable
    return tree


def tree_map_with_accessor(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree accessors to produce a new pytree.

    See also :func:`tree_map`, :func:`tree_map_`, and :func:`tree_map_with_accessor_`.

    >>> tree_map_with_accessor(lambda a, x: f'{a.codify("tree")} = {x!r}', {'x': 7, 'y': (42, 64)})
    {'x': "tree['x'] = 7", 'y': ("tree['y'][0] = 42", "tree['y'][1] = 64")}
    >>> tree_map_with_accessor(lambda a, x: x + len(a), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> tree_map_with_accessor(  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
    ...     lambda a, x: a,
    ...     {'x': 7, 'y': (42, 64), 'z': {1.5: None}},
    ... )
    {
        'x': PyTreeAccessor(*['x'], ...),
        'y': (
            PyTreeAccessor(*['y'][0], ...),
            PyTreeAccessor(*['y'][1], ...)
        ),
        'z': {1.5: None}
    }
    >>> tree_map_with_accessor(  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
    ...     lambda a, x: a,
    ...     {'x': 7, 'y': (42, 64), 'z': {1.5: None}},
    ...     none_is_leaf=True,
    ... )
    {
        'x': PyTreeAccessor(*['x'], ...),
        'y': (
            PyTreeAccessor(*['y'][0], ...),
            PyTreeAccessor(*['y'][1], ...)
        ),
        'z': {
            1.5: PyTreeAccessor(*['z'][1.5], ...)
        }
    }

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra accessors.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(a, x, *xs)`` where ``(a, x)`` are the accessor and value at the corresponding leaf in
        ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, treespec.accessors(), *flat_args))


def tree_map_with_accessor_(
    func: Callable[..., Any],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:
    """Like :func:`tree_map_with_accessor`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`, :func:`tree_map_`, and :func:`tree_map_with_accessor`.

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra accessors.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(a, x, *xs)`` (not the return value) where ``(a, x)`` are the accessor and value at
        the corresponding leaf in ``tree`` and ``xs`` is the tuple of values at values at
        corresponding nodes in ``rests``.
    """
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, treespec.accessors(), *flat_args), maxlen=0)  # consume and exhaust the iterable
    return tree


def tree_replace_nones(sentinel: Any, tree: PyTree[T] | None, /, namespace: str = '') -> PyTree[T]:
    """Replace :data:`None` in ``tree`` with ``sentinel``.

    See also :func:`tree_flatten` and :func:`tree_map`.

    >>> tree_replace_nones(0, {'a': 1, 'b': None, 'c': (2, None)})
    {'a': 1, 'b': 0, 'c': (2, 0)}
    >>> tree_replace_nones(0, None)
    0

    Args:
        sentinel (object): The value to replace :data:`None` with.
        tree (pytree): A pytree to be transformed.
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with :data:`None` replaced.
    """
    if tree is None:
        return sentinel
    return tree_map(
        lambda x: x if x is not None else sentinel,
        tree,
        none_is_leaf=True,
        namespace=namespace,
    )


def tree_transpose(
    outer_treespec: PyTreeSpec,
    inner_treespec: PyTreeSpec,
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
) -> PyTree[T]:  # PyTree[PyTree[T]]
    """Transform a tree having tree structure (outer, inner) into one having structure (inner, outer).

    See also :func:`tree_flatten`, :func:`tree_structure`, and :func:`tree_transpose_map`.

    >>> outer_treespec = tree_structure({'a': 1, 'b': 2, 'c': (3, 4)})
    >>> outer_treespec
    PyTreeSpec({'a': *, 'b': *, 'c': (*, *)})
    >>> inner_treespec = tree_structure((1, 2))
    >>> inner_treespec
    PyTreeSpec((*, *))
    >>> tree = {'a': (1, 2), 'b': (3, 4), 'c': ((5, 6), (7, 8))}
    >>> tree_transpose(outer_treespec, inner_treespec, tree)
    ({'a': 1, 'b': 3, 'c': (5, 7)}, {'a': 2, 'b': 4, 'c': (6, 8)})

    For performance reasons, this function is only checks for the number of leaves in the input
    pytree, not the structure. The result is only enumerated up to the original order of leaves in
    ``tree``, then transpose depends on the number of leaves in structure (inner, outer). The caller
    is responsible for ensuring that the input pytree has a prefix structure of ``outer_treespec``
    followed by a prefix structure of ``inner_treespec``. Otherwise, the result may be incorrect.

    >>> tree_transpose(outer_treespec, inner_treespec, list(range(1, 9)))
    ({'a': 1, 'b': 3, 'c': (5, 7)}, {'a': 2, 'b': 4, 'c': (6, 8)})

    Args:
        outer_treespec (PyTreeSpec): A treespec object representing the outer structure of the pytree.
        inner_treespec (PyTreeSpec): A treespec object representing the inner structure of the pytree.
        tree (pytree): A pytree to be transposed.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.

    Returns:
        A new pytree with the same structure as ``inner_treespec`` but with the value at each leaf
        has the same structure as ``outer_treespec``.
    """
    if outer_treespec.none_is_leaf != inner_treespec.none_is_leaf:
        raise ValueError('Tree structures must have the same none_is_leaf value.')
    outer_size = outer_treespec.num_leaves
    inner_size = inner_treespec.num_leaves
    if outer_size == 0 or inner_size == 0:
        raise ValueError('Tree structures must have at least one leaf.')
    if (
        outer_treespec.namespace
        and inner_treespec.namespace
        and outer_treespec.namespace != inner_treespec.namespace
    ):
        raise ValueError(
            f'Tree structures must have the same namespace, '
            f'got {outer_treespec.namespace!r} vs. {inner_treespec.namespace!r}.',
        )

    leaves, treespec = tree_flatten(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=outer_treespec.none_is_leaf,
        namespace=outer_treespec.namespace or inner_treespec.namespace,
    )
    if treespec.num_leaves != outer_size * inner_size:
        expected_treespec = outer_treespec.compose(inner_treespec)
        raise TypeError(f'Tree structure mismatch; expected: {expected_treespec}, got: {treespec}.')

    grouped = [
        leaves[offset : offset + inner_size]
        for offset in range(0, outer_size * inner_size, inner_size)
    ]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)  # type: ignore[arg-type]


def tree_transpose_map(
    func: Callable[..., PyTree[U]],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    inner_treespec: PyTreeSpec | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:  # PyTree[PyTree[U]]
    """Map a multi-input function over pytree args to produce a new pytree with transposed structure.

    See also :func:`tree_map`, :func:`tree_map_with_path`, and :func:`tree_transpose`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> tree_transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': 2 * x},
    ...     tree,
    ... )
    {
        'identity': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
        'double': {'b': (4, [6, 8]), 'a': 2, 'c': (10, 12)}
    }
    >>> tree_transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': (x, x)},
    ...     tree,
    ... )
    {
        'identity': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
        'double': (
            {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
            {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
        )
    }
    >>> tree_transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': (x, x)},
    ...     tree,
    ...     inner_treespec=tree_structure({'identity': 0, 'double': 0}),
    ... )
    {
        'identity': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
        'double': {'b': ((2, 2), [(3, 3), (4, 4)]), 'a': (1, 1), 'c': ((5, 5), (6, 6))}
    }

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        inner_treespec (PyTreeSpec, optional): The treespec object representing the inner structure
            of the result pytree. If not specified, the inner structure is inferred from the result
            of the function ``func`` on the first leaf. (default: :data:`None`)
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new nested pytree with the same structure as ``inner_treespec`` but with the value at each
        leaf has the same structure as ``tree``. The subtree at each leaf is given by the result of
        function ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and
        ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, outer_treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    if outer_treespec.num_leaves == 0:
        raise ValueError(f'The outer structure must have at least one leaf. Got: {outer_treespec}.')
    flat_args = [leaves] + [outer_treespec.flatten_up_to(r) for r in rests]
    outputs = list(map(func, *flat_args))

    if inner_treespec is None:
        inner_treespec = tree_structure(
            outputs[0],
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    if inner_treespec.num_leaves == 0:
        raise ValueError(f'The inner structure must have at least one leaf. Got: {inner_treespec}.')

    grouped = [inner_treespec.flatten_up_to(o) for o in outputs]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)  # type: ignore[arg-type]


def tree_transpose_map_with_path(
    func: Callable[..., PyTree[U]],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    inner_treespec: PyTreeSpec | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:  # PyTree[PyTree[U]]
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree with transposed structure.

    See also :func:`tree_map_with_path`, :func:`tree_transpose_map`, and :func:`tree_transpose`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> tree_transpose_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x: {'depth': len(p), 'value': x},
    ...     tree,
    ... )
    {
        'depth': {'b': (2, [3, 3]), 'a': 1, 'c': (2, 2)},
        'value': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    }
    >>> tree_transpose_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x: {'path': p, 'value': x},
    ...     tree,
    ...     inner_treespec=tree_structure({'path': 0, 'value': 0}),
    ... )
    {
        'path': {
            'b': (('b', 0), [('b', 1, 0), ('b', 1, 1)]),
            'a': ('a',),
            'c': (('c', 0), ('c', 1))
        },
        'value': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    }

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        inner_treespec (PyTreeSpec, optional): The treespec object representing the inner structure
            of the result pytree. If not specified, the inner structure is inferred from the result
            of the function ``func`` on the first leaf. (default: :data:`None`)
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new nested pytree with the same structure as ``inner_treespec`` but with the value at each
        leaf has the same structure as ``tree``. The subtree at each leaf is given by the result of
        function ``func(p, x, *xs)`` where ``(p, x)`` are the path and value at the corresponding
        leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """  # pylint: disable=line-too-long
    paths, leaves, outer_treespec = _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)
    if outer_treespec.num_leaves == 0:
        raise ValueError(f'The outer structure must have at least one leaf. Got: {outer_treespec}.')
    flat_args = [leaves] + [outer_treespec.flatten_up_to(r) for r in rests]
    outputs = list(map(func, paths, *flat_args))

    if inner_treespec is None:
        inner_treespec = tree_structure(
            outputs[0],
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    if inner_treespec.num_leaves == 0:
        raise ValueError(f'The inner structure must have at least one leaf. Got: {inner_treespec}.')

    grouped = [inner_treespec.flatten_up_to(o) for o in outputs]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)  # type: ignore[arg-type]


def tree_transpose_map_with_accessor(
    func: Callable[..., PyTree[U]],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    inner_treespec: PyTreeSpec | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:  # PyTree[PyTree[U]]
    """Map a multi-input function over pytree args as well as the tree accessors to produce a new pytree with transposed structure.

    See also :func:`tree_map_with_accessor`, :func:`tree_transpose_map`, and :func:`tree_transpose`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> tree_transpose_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: {'depth': len(a), 'code': a.codify('tree'), 'value': x},
    ...     tree,
    ... )
    {
        'depth': {
            'b': (2, [3, 3]),
            'a': 1,
            'c': (2, 2)
        },
        'code': {
            'b': ("tree['b'][0]", ["tree['b'][1][0]", "tree['b'][1][1]"]),
            'a': "tree['a']",
            'c': ("tree['c'][0]", "tree['c'][1]")
        },
        'value': {
            'b': (2, [3, 4]),
            'a': 1,
            'c': (5, 6)
        }
    }
    >>> tree_transpose_map_with_accessor(  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
    ...     lambda a, x: {'path': a.path, 'accessor': a, 'value': x},
    ...     tree,
    ...     inner_treespec=tree_structure({'path': 0, 'accessor': 0, 'value': 0}),
    ... )
    {
        'path': {
            'b': (('b', 0), [('b', 1, 0), ('b', 1, 1)]),
            'a': ('a',),
            'c': (('c', 0), ('c', 1))
        },
        'accessor': {
            'b': (
                PyTreeAccessor(*['b'][0], ...),
                [
                    PyTreeAccessor(*['b'][1][0], ...),
                    PyTreeAccessor(*['b'][1][1], ...)
                ]
            ),
            'a': PyTreeAccessor(*['a'], ...),
            'c': (
                PyTreeAccessor(*['c'][0], ...),
                PyTreeAccessor(*['c'][1], ...)
            )
        },
        'value': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    }

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra accessors.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        inner_treespec (PyTreeSpec, optional): The treespec object representing the inner structure
            of the result pytree. If not specified, the inner structure is inferred from the result
            of the function ``func`` on the first leaf. (default: :data:`None`)
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new nested pytree with the same structure as ``inner_treespec`` but with the value at each
        leaf has the same structure as ``tree``. The subtree at each leaf is given by the result of
        function ``func(a, x, *xs)`` where ``(a, x)`` are the accessor and value at the corresponding
        leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """  # pylint: disable=line-too-long
    leaves, outer_treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    if outer_treespec.num_leaves == 0:
        raise ValueError(f'The outer structure must have at least one leaf. Got: {outer_treespec}.')
    flat_args = [leaves] + [outer_treespec.flatten_up_to(r) for r in rests]
    outputs = list(map(func, outer_treespec.accessors(), *flat_args))

    if inner_treespec is None:
        inner_treespec = tree_structure(
            outputs[0],
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    if inner_treespec.num_leaves == 0:
        raise ValueError(f'The inner structure must have at least one leaf. Got: {inner_treespec}.')

    grouped = [inner_treespec.flatten_up_to(o) for o in outputs]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)  # type: ignore[arg-type]


def tree_broadcast_prefix(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:  # PyTree[PyTree[T]]
    """Return a pytree of same structure of ``full_tree`` with broadcasted subtrees in ``prefix_tree``.

    See also :func:`broadcast_prefix`, :func:`tree_broadcast_common`, and :func:`treespec_is_prefix`.

    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be
    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.

    This function returns a pytree with the same size as ``full_tree``. The leaves are replicated
    from ``prefix_tree``. The number of replicas is determined by the corresponding subtree in
    ``full_tree``.

    >>> tree_broadcast_prefix(1, [2, 3, 4])
    [1, 1, 1]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, 6])
    [1, 2, 3]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4; list: [4, 5, 6, 7].
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, (6, 7)])
    [1, 2, (3, 3)]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}])
    [1, 2, {'a': 3, 'b': 3, 'c': (None, 3)}]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}], none_is_leaf=True)
    [1, 2, {'a': 3, 'b': 3, 'c': (3, 3)}]

    Args:
        prefix_tree (pytree): A pytree with the prefix structure of ``full_tree``.
        full_tree (pytree): A pytree with the suffix structure of ``prefix_tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A pytree of same structure of ``full_tree`` with broadcasted subtrees in ``prefix_tree``.
    """

    def broadcast_leaves(x: T, subtree: PyTree[S]) -> PyTree[T]:
        subtreespec = tree_structure(
            subtree,
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        return subtreespec.unflatten(itertools.repeat(x, subtreespec.num_leaves))

    # If prefix_tree is not a tree prefix of full_tree, this code can raise a ValueError;
    # use prefix_errors to find disagreements and raise more precise error messages.
    # errors = prefix_errors(
    #     prefix_tree,
    #     full_tree,
    #     is_leaf=is_leaf,
    #     none_is_leaf=none_is_leaf,
    #     namespace=namespace,
    # )
    return tree_map(
        broadcast_leaves,  # type: ignore[arg-type]
        prefix_tree,
        full_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def tree_broadcast_common(
    tree: PyTree[T],
    other_tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[PyTree[T], PyTree[T]]:
    """Return two pytrees of common suffix structure of ``tree`` and ``other_tree`` with broadcasted subtrees.

    See also :func:`broadcast_common`, :func:`tree_broadcast_prefix`, and :func:`treespec_is_prefix`.

    If a ``suffix_tree`` is a suffix of a ``tree``, this means the ``suffix_tree`` can be
    constructed by replacing the leaves of ``tree`` with appropriate **subtrees**.

    This function returns two pytrees with the same structure. The tree structure is the common
    suffix structure of ``tree`` and ``other_tree``. The leaves are replicated from ``tree`` and
    ``other_tree``. The number of replicas is determined by the corresponding subtree in the suffix
    structure.

    >>> tree_broadcast_common(1, [2, 3, 4])
    ([1, 1, 1], [2, 3, 4])
    >>> tree_broadcast_common([1, 2, 3], [4, 5, 6])
    ([1, 2, 3], [4, 5, 6])
    >>> tree_broadcast_common([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4.
    >>> tree_broadcast_common([1, (2, 3), 4], [5, 6, (7, 8)])
    ([1, (2, 3), (4, 4)], [5, (6, 6), (7, 8)])
    >>> tree_broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}])
    ([1, {'a': (2, 3)}, {'a': 4, 'b': 4, 'c': (None, 4)}],
     [5, {'a': (6, 6)}, {'a': 7, 'b': 8, 'c': (None, 9)}])
    >>> tree_broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}], none_is_leaf=True)
    ([1, {'a': (2, 3)}, {'a': 4, 'b': 4, 'c': (4, 4)}],
     [5, {'a': (6, 6)}, {'a': 7, 'b': 8, 'c': (None, 9)}])
    >>> tree_broadcast_common([1, None], [None, 2])
    ([None, None], [None, None])
    >>> tree_broadcast_common([1, None], [None, 2], none_is_leaf=True)
    ([1, None], [None, 2])

    Args:
        tree (pytree): A pytree has a common suffix structure of ``other_tree``.
        other_tree (pytree): A pytree has a common suffix structure of ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        Two pytrees of common suffix structure of ``tree`` and ``other_tree`` with broadcasted subtrees.
    """
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    other_leaves, other_treespec = _C.flatten(other_tree, is_leaf, none_is_leaf, namespace)
    common_suffix_treespec = treespec.broadcast_to_common_suffix(other_treespec)

    sentinel: T = object()  # type: ignore[assignment]
    common_suffix_tree: PyTree[T] = common_suffix_treespec.unflatten(
        itertools.repeat(sentinel, common_suffix_treespec.num_leaves),
    )

    def broadcast_leaves(x: T, subtree: PyTree[T]) -> PyTree[T]:
        subtreespec = tree_structure(
            subtree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        return subtreespec.unflatten(itertools.repeat(x, subtreespec.num_leaves))

    broadcasted_tree: PyTree[T] = treespec.unflatten(
        map(
            broadcast_leaves,  # type: ignore[arg-type]
            leaves,
            treespec.flatten_up_to(common_suffix_tree),
        ),
    )
    other_broadcasted_tree: PyTree[T] = other_treespec.unflatten(
        map(
            broadcast_leaves,  # type: ignore[arg-type]
            other_leaves,
            other_treespec.flatten_up_to(common_suffix_tree),
        ),
    )
    return broadcasted_tree, other_broadcasted_tree


def _tree_broadcast_common(
    tree: PyTree[T],
    /,
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[PyTree[T], ...]:
    if not rests:
        return (tree,)
    if len(rests) == 1:
        return tree_broadcast_common(
            tree,
            rests[0],
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )

    broadcasted_tree = tree
    broadcasted_rests = list(rests)
    for _ in range(2):
        for i, rest in enumerate(rests):
            broadcasted_tree, broadcasted_rests[i] = tree_broadcast_common(
                broadcasted_tree,
                rest,
                is_leaf=is_leaf,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )

    return (broadcasted_tree, *broadcasted_rests)


# pylint: disable-next=too-many-locals
def tree_broadcast_map(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_broadcast_map_with_path`, :func:`tree_map`, :func:`tree_map_`,
    and :func:`tree_map_with_path`.

    If only one input is provided, this function is the same as :func:`tree_map`:

    >>> tree_broadcast_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_broadcast_map(lambda x: x + 1, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (43, 65), 'z': None}
    >>> tree_broadcast_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': None}
    >>> tree_broadcast_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None}, none_is_leaf=True)
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, all input trees will be broadcasted to the common suffix structure
    of all inputs:

    >>> tree_broadcast_map(lambda x, y: x * y, [5, 6, (3, 4)], [{'a': 7, 'b': 9}, [1, 2], 8])
    [{'a': 35, 'b': 45}, [6, 12], (24, 32)]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, they should have a common suffix structure with
            each other and with ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the structure as the common suffix structure of ``tree`` and ``rests`` but
        with the value at each leaf given by ``func(x, *xs)`` where ``x`` is the value at the
        corresponding leaf (may be broadcasted) in ``tree`` and ``xs`` is the tuple of values at
        corresponding leaves (may be broadcasted) in ``rests``.
    """
    return tree_map(
        func,
        *_tree_broadcast_common(
            tree,
            *rests,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


# pylint: disable-next=too-many-locals
def tree_broadcast_map_with_path(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree.

    See also :func:`tree_broadcast_map`, :func:`tree_map`, :func:`tree_map_`,
    and :func:`tree_map_with_path`.

    If only one input is provided, this function is the same as :func:`tree_map`:

    >>> tree_broadcast_map_with_path(lambda p, x: (len(p), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> tree_broadcast_map_with_path(lambda p, x: x + len(p), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> tree_broadcast_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}})
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: None}}
    >>> tree_broadcast_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}}, none_is_leaf=True)
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: ('z', 1.5)}}

    If multiple inputs are given, all input trees will be broadcasted to the common suffix structure
    of all inputs:

    >>> tree_broadcast_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x, y: (p, x * y),
    ...     [5, 6, (3, 4)],
    ...     [{'a': 7, 'b': 9}, [1, 2], 8],
    ... )
    [
        {'a': ((0, 'a'), 35), 'b': ((0, 'b'), 45)},
        [((1, 0), 6), ((1, 1), 12)],
        (((2, 0), 24), ((2, 1), 32))
    ]

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, they should have a common suffix structure with
            each other and with ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the structure as the common suffix structure of ``tree`` and ``rests`` but
        with the value at each leaf given by ``func(p, x, *xs)`` where ``(p, x)`` are the path and
        value at the corresponding leaf (may be broadcasted) in and ``xs`` is the tuple of values at
        corresponding leaves (may be broadcasted) in ``rests``.
    """
    return tree_map_with_path(
        func,
        *_tree_broadcast_common(
            tree,
            *rests,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def tree_broadcast_map_with_accessor(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree accessors to produce a new pytree.

    See also :func:`tree_broadcast_map`, :func:`tree_map`, :func:`tree_map_`,
    and :func:`tree_map_with_accessor`.

    If only one input is provided, this function is the same as :func:`tree_map`:

    >>> tree_broadcast_map_with_accessor(lambda a, x: (len(a), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> tree_broadcast_map_with_accessor(lambda a, x: x + len(a), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> tree_broadcast_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: a.codify('tree'),
    ...     {'x': 7, 'y': (42, 64), 'z': {1.5: None}},
    ... )
    {
        'x': "tree['x']",
        'y': ("tree['y'][0]", "tree['y'][1]"),
        'z': {1.5: None}
    }
    >>> tree_broadcast_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: a.codify('tree'),
    ...     {'x': 7, 'y': (42, 64), 'z': {1.5: None}},
    ...     none_is_leaf=True,
    ... )
    {
        'x': "tree['x']",
        'y': ("tree['y'][0]", "tree['y'][1]"),
        'z': {1.5: "tree['z'][1.5]"}
    }

    If multiple inputs are given, all input trees will be broadcasted to the common suffix structure
    of all inputs:

    >>> tree_broadcast_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x, y: f'{a.codify("tree")} = {x * y}',
    ...     [5, 6, (3, 4)],
    ...     [{'a': 7, 'b': 9}, [1, 2], 8],
    ... )
    [
        {'a': "tree[0]['a'] = 35", 'b': "tree[0]['b'] = 45"},
        ['tree[1][0] = 6', 'tree[1][1] = 12'],
        ('tree[2][0] = 24', 'tree[2][1] = 32')
    ]

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra accessors.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, they should have a common suffix structure with
            each other and with ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the structure as the common suffix structure of ``tree`` and ``rests`` but
        with the value at each leaf given by ``func(a, x, *xs)`` where ``(a, x)`` are the accessor
        and value at the corresponding leaf (may be broadcasted) in and ``xs`` is the tuple of
        values at corresponding leaves (may be broadcasted) in ``rests``.
    """
    return tree_map_with_accessor(
        func,
        *_tree_broadcast_common(
            tree,
            *rests,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


# pylint: disable-next=missing-class-docstring,too-few-public-methods
class MissingSentinel:  # pragma: no cover
    __slots__: ClassVar[tuple[()]] = ()

    def __repr__(self) -> str:
        return '<MISSING>'


__MISSING: T = MissingSentinel()  # type: ignore[valid-type]
del MissingSentinel


@overload
def tree_reduce(
    func: Callable[[T, T], T],
    tree: PyTree[T],
    /,
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


@overload
def tree_reduce(
    func: Callable[[T, S], T],
    tree: PyTree[S],
    /,
    initial: T = __MISSING,
    *,
    is_leaf: Callable[[S], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


def tree_reduce(
    func: Callable[[T, S], T],
    tree: PyTree[S],
    /,
    initial: T = __MISSING,
    *,
    is_leaf: Callable[[S], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T:
    """Traversal through a pytree and reduce the leaves in left-to-right depth-first order.

    See also :func:`tree_leaves` and :func:`tree_sum`.

    >>> tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, 3)})
    6
    >>> tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node with arity 0 by default
    6
    >>> tree_reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3})
    3
    >>> tree_reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    None

    Args:
        func (callable): A function that takes two arguments and returns a value of the same type.
        tree (pytree): A pytree to be traversed.
        initial (object, optional): An initial value to be used for the reduction. If not provided,
            the first leaf value is used as the initial value.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The result of reducing the leaves of the pytree using ``func``.
    """  # pylint: disable=line-too-long
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if initial is __MISSING:
        return functools.reduce(func, leaves)  # type: ignore[arg-type,return-value]
    return functools.reduce(func, leaves, initial)


def tree_sum(
    tree: PyTree[T],
    /,
    start: T = 0,  # type: ignore[assignment]
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T:
    """Sum ``start`` and leaf values in ``tree`` in left-to-right depth-first order and return the total.

    See also :func:`tree_leaves` and :func:`tree_reduce`.

    >>> tree_sum({'x': 1, 'y': (2, 3)})
    6
    >>> tree_sum({'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node with arity 0 by default
    6
    >>> tree_sum({'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
    >>> tree_sum({'x': 'a', 'y': ('b', None), 'z': 'c'}, start='')
    'abc'
    >>> tree_sum({'x': [1], 'y': ([2], [None]), 'z': [3]}, start=[], is_leaf=lambda x: isinstance(x, list))
    [1, 2, None, 3]

    Args:
        tree (pytree): A pytree to be traversed.
        start (object, optional): An initial value to be used for the sum. (default: :data:`0`)
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The total sum of ``start`` and leaf values in ``tree``.
    """
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    # sum() rejects string values for `start` parameter
    if isinstance(start, str):
        return ''.join([start, *leaves])  # type: ignore[list-item,return-value]
    if isinstance(start, (bytes, bytearray)):
        return b''.join([start, *leaves])  # type: ignore[list-item,return-value]
    return sum(leaves, start)  # type: ignore[call-overload]


@overload
def tree_max(
    tree: PyTree[T],
    /,
    *,
    is_leaf: Callable[[T], bool] | None = None,
    key: Callable[[T], Any] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


@overload
def tree_max(
    tree: PyTree[T],
    /,
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


def tree_max(
    tree: PyTree[T],
    /,
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T:
    """Return the maximum leaf value in ``tree``.

    See also :func:`tree_leaves` and :func:`tree_min`.

    >>> tree_max({})
    Traceback (most recent call last):
        ...
    ValueError: max() iterable argument is empty
    >>> tree_max({}, default=0)
    0
    >>> tree_max({'x': 0, 'y': (2, 1)})
    2
    >>> tree_max({'x': 0, 'y': (2, 1)}, key=lambda x: -x)
    0
    >>> tree_max({'a': None})  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: max() iterable argument is empty
    >>> tree_max({'a': None}, default=0)  # `None` is a non-leaf node with arity 0 by default
    0
    >>> tree_max({'a': None}, none_is_leaf=True)
    None
    >>> tree_max(None)  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: max() iterable argument is empty
    >>> tree_max(None, default=0)
    0
    >>> tree_max(None, none_is_leaf=True)
    None

    Args:
        tree (pytree): A pytree to be traversed.
        default (object, optional): The default value to return if ``tree`` is empty. If the ``tree``
            is empty and ``default`` is not specified, raise a :exc:`ValueError`.
        key (callable or None, optional): An one argument ordering function like that used for
            :meth:`list.sort`.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The maximum leaf value in ``tree``.
    """
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if default is __MISSING:
        return max(leaves, key=key)  # type: ignore[type-var,arg-type]
    return max(leaves, default=default, key=key)  # type: ignore[type-var,arg-type]


@overload
def tree_min(
    tree: PyTree[T],
    /,
    *,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


@overload
def tree_min(
    tree: PyTree[T],
    /,
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


def tree_min(
    tree: PyTree[T],
    /,
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T:
    """Return the minimum leaf value in ``tree``.

    See also :func:`tree_leaves` and :func:`tree_max`.

    >>> tree_min({})
    Traceback (most recent call last):
        ...
    ValueError: min() iterable argument is empty
    >>> tree_min({}, default=0)
    0
    >>> tree_min({'x': 0, 'y': (2, 1)})
    0
    >>> tree_min({'x': 0, 'y': (2, 1)}, key=lambda x: -x)
    2
    >>> tree_min({'a': None})  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: min() iterable argument is empty
    >>> tree_min({'a': None}, default=0)  # `None` is a non-leaf node with arity 0 by default
    0
    >>> tree_min({'a': None}, none_is_leaf=True)
    None
    >>> tree_min(None)  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: min() iterable argument is empty
    >>> tree_min(None, default=0)
    0
    >>> tree_min(None, none_is_leaf=True)
    None

    Args:
        tree (pytree): A pytree to be traversed.
        default (object, optional): The default value to return if ``tree`` is empty. If the ``tree``
            is empty and ``default`` is not specified, raise a :exc:`ValueError`.
        key (callable or None, optional): An one argument ordering function like that used for
            :meth:`list.sort`.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The minimum leaf value in ``tree``.
    """
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if default is __MISSING:
        return min(leaves, key=key)  # type: ignore[type-var,arg-type]
    return min(leaves, default=default, key=key)  # type: ignore[type-var,arg-type]


def tree_all(
    tree: PyTree[T],
    /,
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether all leaves in ``tree`` are true (or if ``tree`` is empty).

    See also :func:`tree_leaves` and :func:`tree_any`.

    >>> tree_all({})
    True
    >>> tree_all({'x': 1, 'y': (2, 3)})
    True
    >>> tree_all({'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node by default
    True
    >>> tree_all({'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    False
    >>> tree_all(None)  # `None` is a non-leaf node by default
    True
    >>> tree_all(None, none_is_leaf=True)
    False

    Args:
        tree (pytree): A pytree to be traversed.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        :data:`True` if all leaves in ``tree`` are true, or if ``tree`` is empty.
        Otherwise, :data:`False`.
    """
    return all(
        tree_iter(
            tree,  # type: ignore[arg-type]
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
    )


def tree_any(
    tree: PyTree[T],
    /,
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether all leaves in ``tree`` are true (or :data:`False` if ``tree`` is empty).

    See also :func:`tree_leaves` and :func:`tree_all`.

    >>> tree_any({})
    False
    >>> tree_any({'x': 0, 'y': (2, 0)})
    True
    >>> tree_any({'a': None})  # `None` is a non-leaf node with arity 0 by default
    False
    >>> tree_any({'a': None}, none_is_leaf=True)  # `None` is evaluated as false
    False
    >>> tree_any(None)  # `None` is a non-leaf node with arity 0 by default
    False
    >>> tree_any(None, none_is_leaf=True)  # `None` is evaluated as false
    False

    Args:
        tree (pytree): A pytree to be traversed.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        :data:`True` if any leaves in ``tree`` are true, otherwise, :data:`False`. If ``tree`` is
        empty, return :data:`False`.
    """
    return any(
        tree_iter(
            tree,  # type: ignore[arg-type]
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
    )


class FlattenOneLevelOutput(NamedTuple, Generic[T]):
    """The output of :func:`tree_flatten_one_level`."""

    children: list[PyTree[T]]
    """A list of one-level children of the pytree node."""

    metadata: MetaData
    """The metadata used to reconstruct the pytree node."""

    entries: tuple[Any, ...]
    """A tuple of path entries to the children."""

    unflatten_func: Callable[[MetaData, list[PyTree[T]]], PyTree[T]]
    """A function that can be used to unflatten the metadata and children back to the pytree node."""


# Subclass the namedtuple class to allow assigning new attributes.
class FlattenOneLevelOutputEx(FlattenOneLevelOutput[T]):
    """The output of :func:`tree_flatten_one_level`."""

    type: builtins.type[Collection[T]]
    """The type of the pytree node."""

    path_entry_type: builtins.type[PyTreeEntry]
    """The type of the path entry for the pytree node."""

    kind: PyTreeKind
    """The kind of the pytree node."""


def tree_flatten_one_level(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> FlattenOneLevelOutputEx[T]:
    """Flatten the pytree one level, returning a 4-tuple of children, metadata, path entries, and an unflatten function.

    See also :func:`tree_flatten`, :func:`tree_flatten_with_path`.

    >>> children, metadata, entries, unflatten_func = tree_flatten_one_level({'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5})
    >>> children, metadata, entries
    ([1, (2, [3, 4]), None, 5], ['a', 'b', 'c', 'd'], ('a', 'b', 'c', 'd'))
    >>> unflatten_func(metadata, children)
    {'a': 1, 'b': (2, [3, 4]), 'c': None, 'd': 5}
    >>> children, metadata, entries, unflatten_func = tree_flatten_one_level([{'a': 1, 'b': (2, 3)}, (4, 5)])
    >>> children, metadata, entries
    ([{'a': 1, 'b': (2, 3)}, (4, 5)], None, (0, 1))
    >>> unflatten_func(metadata, children)
    [{'a': 1, 'b': (2, 3)}, (4, 5)]

    Args:
        tree (pytree): A pytree to be traversed.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A 4-tuple ``(children, metadata, entries, unflatten_func)``. The first element is a list of
        one-level children of the pytree node. The second element is the metadata used to
        reconstruct the pytree node. The third element is a tuple of path entries to the children.
        The fourth element is a function that can be used to unflatten the metadata and
        children back to the pytree node.
    """  # pylint: disable=line-too-long
    node_type = type(tree)
    if (tree is None and none_is_leaf) or (is_leaf is not None and is_leaf(tree)):  # type: ignore[unreachable,arg-type]
        raise ValueError(f'Cannot flatten leaf-type: {node_type} (node: {tree!r}).')

    from optree.registry import register_pytree_node  # pylint: disable=import-outside-toplevel

    handler = register_pytree_node.get(node_type, namespace=namespace)
    if handler is None:
        raise ValueError(f'Cannot flatten leaf-type: {node_type} (node: {tree!r}).')

    flattened = tuple(handler.flatten_func(tree))
    if len(flattened) == 2:
        flattened = (*flattened, None)
    elif len(flattened) != 3:
        raise RuntimeError(
            f'PyTree custom flatten function for type {node_type} should return a 2- or 3-tuple, '
            f'got {len(flattened)}.',
        )
    flattened: tuple[Iterable[PyTree[T]], MetaData, Iterable[Any] | None]
    children, metadata, entries = flattened
    children = list(children)
    entries = tuple(range(len(children)) if entries is None else entries)
    if len(children) != len(entries):
        raise RuntimeError(
            f'PyTree custom flatten function for type {node_type} returned inconsistent '
            f'number of children ({len(children)}) and number of entries ({len(entries)}).',
        )

    output = FlattenOneLevelOutputEx(
        children=children,
        metadata=metadata,
        entries=entries,
        unflatten_func=handler.unflatten_func,  # type: ignore[arg-type]
    )
    output.type = node_type  # type: ignore[assignment]
    output.path_entry_type = handler.path_entry_type
    output.kind = handler.kind
    return output
