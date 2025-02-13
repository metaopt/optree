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
"""Utilities for working with ``PyTree``.

The :mod:`optree.pytree` namespace contains aliases of utilities from ``optree.tree_``.

>>> import optree.pytree as pt
...
...
 >>> import optree.pytree as pytree
>>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
>>> leaves, treespec = pytree.flatten(tree)
>>> leaves, treespec  # doctest: +IGNORE_WHITESPACE
(
    [1, 2, 3, 4, 5],
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
)
>>> tree == pytree.unflatten(treespec, leaves)
True

.. versionadded:: 0.14.1
"""

# pylint: disable=too-many-lines

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    overload,
)

import optree.ops as ops
from optree.ops import __MISSING


if TYPE_CHECKING:
    from ops import FlattenOneLevelOutputEx
    from optree.accessor import PyTreeAccessor
    from optree.typing import PyTree, PyTreeSpec, S, T, U


__all__ = [
    'flatten',
    'unflatten',
    'flatten_one_level',
    'flatten_with_path',
    'flatten_with_accessor',
    'leaves',
    'structure',
    'paths',
    'accessors',
    'is_leaf',
    'max',
    'min',
    'all',
    'any',
    'iter',
    'sum',
    'reduce',
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
    'broadcast_prefix',
    'broadcast_common',
]


def flatten(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[T], PyTreeSpec]:
    """Flatten a pytree.

    See also :func:`pytree.flatten_with_path` and :func:`pytree.unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> pytree.flatten(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> pytree.flatten(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [1, 2, 3, 4, None, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    )
    >>> pytree.flatten(1)
    ([1], PyTreeSpec(*))
    >>> pytree.flatten(None)
    ([], PyTreeSpec(None))
    >>> pytree.flatten(None, none_is_leaf=True)
    ([None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> pytree.flatten(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [2, 3, 4, 1, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}))
    )
    >>> pytree.flatten(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_flatten(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


def flatten_with_path(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[tuple[Any, ...]], list[T], PyTreeSpec]:
    """Flatten a pytree and additionally record the paths.

    See also :func:`pytree.flatten`, :func:`pytree.paths`, and :func:`treespec.paths`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> pytree.flatten_with_path(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('d',)],
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> pytree.flatten_with_path(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('c',), ('d',)],
        [1, 2, 3, 4, None, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    )
    >>> pytree.flatten_with_path(1)
    ([()], [1], PyTreeSpec(*))
    >>> pytree.flatten_with_path(None)
    ([], [], PyTreeSpec(None))
    >>> pytree.flatten_with_path(None, none_is_leaf=True)
    ([()], [None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> pytree.flatten_with_path(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [('b', 0), ('b', 1, 0), ('b', 1, 1), ('a',), ('d',)],
        [2, 3, 4, 1, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}))
    )
    >>> pytree.flatten_with_path(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
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

    .. versionadded:: 0.14.1    leaf values and the last element is a treespec representing the structure of the pytree.
    """
    return ops.tree_flatten_with_path(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def flatten_with_accessor(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[PyTreeAccessor], list[T], PyTreeSpec]:
    """Flatten a pytree and additionally record the accessors.

    See also :func:`pytree.flatten`, :func:`pytree.accessors`, and :func:`treespec.accessors`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> pytree.flatten_with_accessor(tree)  # doctest: +IGNORE_WHITESPACE
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
    >>> pytree.flatten_with_accessor(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
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
    >>> pytree.flatten_with_accessor(1)
    ([PyTreeAccessor(*, ())], [1], PyTreeSpec(*))
    >>> pytree.flatten_with_accessor(None)
    ([], [], PyTreeSpec(None))
    >>> pytree.flatten_with_accessor(None, none_is_leaf=True)
    ([PyTreeAccessor(*, ())], [None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> pytree.flatten_with_accessor(tree)  # doctest: +IGNORE_WHITESPACE
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
    >>> pytree.flatten_with_accessor(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
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

    .. versionadded:: 0.14.1
    """  # pylint: disable=line-too-long
    return ops.tree_flatten_with_accessor(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def unflatten(treespec: PyTreeSpec, leaves: Iterable[T]) -> PyTree[T]:
    """Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`pytree.flatten`.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> leaves, treespec = pytree.flatten(tree)
    >>> tree == pytree.unflatten(treespec, leaves)
    True

    Args:
        treespec (PyTreeSpec): The treespec to reconstruct.
        leaves (iterable): The list of leaves to use for reconstruction. The list must match the
            number of leaves of the treespec.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by ``treespec``.

    .. versionadded:: 0.14.1
    """
    return ops.tree_unflatten(treespec, leaves)


def iter(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> Iterable[T]:
    """Get an iterator over the leaves of a pytree.

    See also :func:`pytree.flatten` and :func:`pytree.leaves`.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> list(pytree.iter(tree))
    [1, 2, 3, 4, 5]
    >>> list(pytree.iter(tree, none_is_leaf=True))
    [1, 2, 3, 4, None, 5]
    >>> list(pytree.iter(1))
    [1]
    >>> list(pytree.iter(None))
    []
    >>> list(pytree.iter(None, none_is_leaf=True))
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_iter(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


def leaves(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[T]:
    """Get the leaves of a pytree.

    See also :func:`pytree.flatten` and :func:`pytree.iter`.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> pytree.leaves(tree)
    [1, 2, 3, 4, 5]
    >>> pytree.leaves(tree, none_is_leaf=True)
    [1, 2, 3, 4, None, 5]
    >>> pytree.leaves(1)
    [1]
    >>> pytree.leaves(None)
    []
    >>> pytree.leaves(None, none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


def structure(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Get the treespec for a pytree.

    See also :func:`pytree.flatten`.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> pytree.structure(tree)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    >>> pytree.structure(tree, none_is_leaf=True)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    >>> pytree.structure(1)
    PyTreeSpec(*)
    >>> pytree.structure(None)
    PyTreeSpec(None)
    >>> pytree.structure(None, none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_structure(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


def paths(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[tuple[Any, ...]]:
    """Get the path entries to the leaves of a pytree.

    See also :func:`pytree.flatten`, :func:`pytree.flatten_with_path`, and :func:`treespec.paths`.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> pytree.paths(tree)
    [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('d',)]
    >>> pytree.paths(tree, none_is_leaf=True)
    [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('c',), ('d',)]
    >>> pytree.paths(1)
    [()]
    >>> pytree.paths(None)
    []
    >>> pytree.paths(None, none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_paths(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


def accessors(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[PyTreeAccessor]:
    """Get the accessors to the leaves of a pytree.

    See also :func:`pytree.flatten`, :func:`pytree.flatten_with_accessor`, and
    :func:`treespec.accessors`.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> pytree.accessors(tree)  # doctest: +IGNORE_WHITESPACE
    [
        PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
        PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
        PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
        PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
        PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
    ]
    >>> pytree.accessors(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    [
        PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
        PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
        PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
        PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
        PyTreeAccessor(*['c'], (MappingEntry(key='c', type=<class 'dict'>),)),
        PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
    ]
    >>> pytree.accessors(1)
    [PyTreeAccessor(*, ())]
    >>> pytree.accessors(None)
    []
    >>> pytree.accessors(None, none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """  # pylint: disable=line-too-long
    return ops.tree_accessors(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


def is_leaf(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether the given object is a leaf node.

    See also :func:`pytree.flatten`, :func:`pytree.leaves`, and :func:`all_leaves`.

    >>> import optree.pytree as pytree
    >>> pytree.is_leaf(1)
    True
    >>> pytree.is_leaf(None)
    False
    >>> pytree.is_leaf(None, none_is_leaf=True)
    True
    >>> pytree.is_leaf({'a': 1, 'b': (2, 3)})
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_is_leaf(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


def map(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`pytree.map_`, :func:`pytree.map_with_path`, :func:`pytree.map_with_path_`,
    and :func:`pytree.broadcast_map`.

    >>> import optree.pytree as pytree
    >>> pytree.map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> pytree.map(lambda x: x + 1, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (43, 65), 'z': None}
    >>> pytree.map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': None}
    >>> pytree.map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None}, none_is_leaf=True)
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:

    >>> pytree.map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_map(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def map_(
    func: Callable[..., Any],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:
    """Like :func:`pytree.map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`pytree.map`, :func:`pytree.map_with_path`, and :func:`pytree.map_with_path_`.

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

    .. versionadded:: 0.14.1
    """
    return ops.tree_map_(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def map_with_path(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree.

    See also :func:`pytree.map`, :func:`pytree.map_`, and :func:`pytree.map_with_path_`.

    >>> import optree.pytree as pytree
    >>> pytree.map_with_path(lambda p, x: (len(p), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> pytree.map_with_path(lambda p, x: x + len(p), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> pytree.map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}})
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: None}}
    >>> pytree.map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}}, none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_map_with_path(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def map_with_path_(
    func: Callable[..., Any],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:
    """Like :func:`pytree.map_with_path`, but do an inplace call on each leaf and return the original tree.

    See also :func:`pytree.map`, :func:`pytree.map_`, and :func:`pytree.map_with_path`.

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

    .. versionadded:: 0.14.1
    """
    return ops.tree_map_with_path_(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def map_with_accessor(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree accessors to produce a new pytree.

    See also :func:`pytree.map`, :func:`pytree.map_`, and :func:`pytree.map_with_accessor_`.

    >>> import optree.pytree as pytree
    >>> pytree.map_with_accessor(lambda a, x: f'{a.codify("tree")} = {x!r}', {'x': 7, 'y': (42, 64)})
    {'x': "tree['x'] = 7", 'y': ("tree['y'][0] = 42", "tree['y'][1] = 64")}
    >>> pytree.map_with_accessor(lambda a, x: x + len(a), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> pytree.map_with_accessor(  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
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
    >>> pytree.map_with_accessor(  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_map_with_accessor(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def map_with_accessor_(
    func: Callable[..., Any],
    tree: PyTree[T],
    /,
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:
    """Like :func:`pytree.map_with_accessor`, but do an inplace call on each leaf and return the original tree.

    See also :func:`pytree.map`, :func:`pytree.map_`, and :func:`pytree.map_with_accessor`.

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

    .. versionadded:: 0.14.1
    """
    return ops.tree_map_with_accessor_(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def replace_nones(sentinel: Any, tree: PyTree[T] | None, /, namespace: str = '') -> PyTree[T]:
    """Replace :data:`None` in ``tree`` with ``sentinel``.

    See also :func:`pytree.flatten` and :func:`pytree.map`.

    >>> import optree.pytree as pytree
    >>> pytree.replace_nones(0, {'a': 1, 'b': None, 'c': (2, None)})
    {'a': 1, 'b': 0, 'c': (2, 0)}
    >>> pytree.replace_nones(0, None)
    0

    Args:
        sentinel (object): The value to replace :data:`None` with.
        tree (pytree): A pytree to be transformed.
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with :data:`None` replaced.

    .. versionadded:: 0.14.1
    """
    return ops.tree_replace_nones(sentinel, tree, namespace=namespace)


def transpose(
    outer_treespec: PyTreeSpec,
    inner_treespec: PyTreeSpec,
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
) -> PyTree[T]:  # PyTree[PyTree[T]]
    """Transform a tree having tree structure (outer, inner) into one having structure (inner, outer).

    See also :func:`pytree.flatten`, :func:`pytree.structure`, and :func:`pytree.transpose_map`.

    >>> import optree.pytree as pytree
    >>> outer_treespec = pytree.structure({'a': 1, 'b': 2, 'c': (3, 4)})
    >>> outer_treespec
    PyTreeSpec({'a': *, 'b': *, 'c': (*, *)})
    >>> inner_treespec = pytree.structure((1, 2))
    >>> inner_treespec
    PyTreeSpec((*, *))
    >>> tree = {'a': (1, 2), 'b': (3, 4), 'c': ((5, 6), (7, 8))}
    >>> pytree.transpose(outer_treespec, inner_treespec, tree)
    ({'a': 1, 'b': 3, 'c': (5, 7)}, {'a': 2, 'b': 4, 'c': (6, 8)})

    For performance reasons, this function is only checks for the number of leaves in the input
    pytree, not the structure. The result is only enumerated up to the original order of leaves in
    ``tree``, then transpose depends on the number of leaves in structure (inner, outer). The caller
    is responsible for ensuring that the input pytree has a prefix structure of ``outer_treespec``
    followed by a prefix structure of ``inner_treespec``. Otherwise, the result may be incorrect.

    >>> pytree.transpose(outer_treespec, inner_treespec, list(range(1, 9)))
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_transpose(outer_treespec, inner_treespec, tree, is_leaf=is_leaf)


def transpose_map(
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

    See also :func:`pytree.map`, :func:`pytree.map_with_path`, and :func:`pytree.transpose`.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> pytree.transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': 2 * x},
    ...     tree,
    ... )
    {
        'identity': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
        'double': {'b': (4, [6, 8]), 'a': 2, 'c': (10, 12)}
    }
    >>> pytree.transpose_map(  # doctest: +IGNORE_WHITESPACE
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
    >>> pytree.transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': (x, x)},
    ...     tree,
    ...     inner_treespec=pytree.structure({'identity': 0, 'double': 0}),
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_transpose_map(
        func,
        tree,
        *rests,
        inner_treespec=inner_treespec,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def transpose_map_with_path(
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

    See also :func:`pytree.map_with_path`, :func:`pytree.transpose_map`, and :func:`pytree.transpose`.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> pytree.transpose_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x: {'depth': len(p), 'value': x},
    ...     tree,
    ... )
    {
        'depth': {'b': (2, [3, 3]), 'a': 1, 'c': (2, 2)},
        'value': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    }
    >>> pytree.transpose_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x: {'path': p, 'value': x},
    ...     tree,
    ...     inner_treespec=pytree.structure({'path': 0, 'value': 0}),
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

    .. versionadded:: 0.14.1
    """  # pylint: disable=line-too-long
    return ops.tree_transpose_map_with_path(
        func,
        tree,
        *rests,
        inner_treespec=inner_treespec,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def transpose_map_with_accessor(
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

    See also :func:`pytree.map_with_accessor`, :func:`pytree.transpose_map`, and :func:`pytree.transpose`.

    >>> import optree.pytree as pytree
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> pytree.transpose_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
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
    >>> pytree.transpose_map_with_accessor(  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
    ...     lambda a, x: {'path': a.path, 'accessor': a, 'value': x},
    ...     tree,
    ...     inner_treespec=pytree.structure({'path': 0, 'accessor': 0, 'value': 0}),
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

    .. versionadded:: 0.14.1
    """  # pylint: disable=line-too-long
    return ops.tree_transpose_map_with_accessor(
        func,
        tree,
        *rests,
        inner_treespec=inner_treespec,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def broadcast_prefix(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:  # PyTree[PyTree[T]]
    """Return a pytree of same structure of ``full_tree`` with broadcasted subtrees in ``prefix_tree``.

    See also :func:`broadcast_prefix`, :func:`pytree.broadcast_common`, and :func:`treespec.is_prefix`.

    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be
    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.

    This function returns a pytree with the same size as ``full_tree``. The leaves are replicated
    from ``prefix_tree``. The number of replicas is determined by the corresponding subtree in
    ``full_tree``.

    >>> import optree.pytree as pytree
    >>> pytree.broadcast_prefix(1, [2, 3, 4])
    [1, 1, 1]
    >>> pytree.broadcast_prefix([1, 2, 3], [4, 5, 6])
    [1, 2, 3]
    >>> pytree.broadcast_prefix([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4; list: [4, 5, 6, 7].
    >>> pytree.broadcast_prefix([1, 2, 3], [4, 5, (6, 7)])
    [1, 2, (3, 3)]
    >>> pytree.broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}])
    [1, 2, {'a': 3, 'b': 3, 'c': (None, 3)}]
    >>> pytree.broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}], none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_broadcast_prefix(
        prefix_tree,
        full_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def broadcast_common(
    tree: PyTree[T],
    other_tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[PyTree[T], PyTree[T]]:
    """Return two pytrees of common suffix structure of ``tree`` and ``other_tree`` with broadcasted subtrees.

    See also :func:`broadcast_common`, :func:`pytree.broadcast_prefix`, and :func:`treespec.is_prefix`.

    If a ``suffix_tree`` is a suffix of a ``tree``, this means the ``suffix_tree`` can be
    constructed by replacing the leaves of ``tree`` with appropriate **subtrees**.

    This function returns two pytrees with the same structure. The tree structure is the common
    suffix structure of ``tree`` and ``other_tree``. The leaves are replicated from ``tree`` and
    ``other_tree``. The number of replicas is determined by the corresponding subtree in the suffix
    structure.

    >>> import optree.pytree as pytree
    >>> pytree.broadcast_common(1, [2, 3, 4])
    ([1, 1, 1], [2, 3, 4])
    >>> pytree.broadcast_common([1, 2, 3], [4, 5, 6])
    ([1, 2, 3], [4, 5, 6])
    >>> pytree.broadcast_common([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4.
    >>> pytree.broadcast_common([1, (2, 3), 4], [5, 6, (7, 8)])
    ([1, (2, 3), (4, 4)], [5, (6, 6), (7, 8)])
    >>> pytree.broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}])
    ([1, {'a': (2, 3)}, {'a': 4, 'b': 4, 'c': (None, 4)}],
     [5, {'a': (6, 6)}, {'a': 7, 'b': 8, 'c': (None, 9)}])
    >>> pytree.broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}], none_is_leaf=True)
    ([1, {'a': (2, 3)}, {'a': 4, 'b': 4, 'c': (4, 4)}],
     [5, {'a': (6, 6)}, {'a': 7, 'b': 8, 'c': (None, 9)}])
    >>> pytree.broadcast_common([1, None], [None, 2])
    ([None, None], [None, None])
    >>> pytree.broadcast_common([1, None], [None, 2], none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_broadcast_common(
        tree,
        other_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


# pylint: disable-next=too-many-locals
def broadcast_map(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`pytree.broadcast_map_with_path`, :func:`pytree.map`, :func:`pytree.map_`,
    and :func:`pytree.map_with_path`.

    If only one input is provided, this function is the same as :func:`pytree.map`:

    >>> import optree.pytree as pytree
    >>> pytree.broadcast_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> pytree.broadcast_map(lambda x: x + 1, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (43, 65), 'z': None}
    >>> pytree.broadcast_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': None}
    >>> pytree.broadcast_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None}, none_is_leaf=True)
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, all input trees will be broadcasted to the common suffix structure
    of all inputs:

    >>> pytree.broadcast_map(lambda x, y: x * y, [5, 6, (3, 4)], [{'a': 7, 'b': 9}, [1, 2], 8])
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_broadcast_map(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


# pylint: disable-next=too-many-locals
def broadcast_map_with_path(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree.

    See also :func:`pytree.broadcast_map`, :func:`pytree.map`, :func:`pytree.map_`,
    and :func:`pytree.map_with_path`.

    If only one input is provided, this function is the same as :func:`pytree.map`:

    >>> import optree.pytree as pytree
    >>> pytree.broadcast_map_with_path(lambda p, x: (len(p), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> pytree.broadcast_map_with_path(lambda p, x: x + len(p), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> pytree.broadcast_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}})
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: None}}
    >>> pytree.broadcast_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}}, none_is_leaf=True)
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: ('z', 1.5)}}

    If multiple inputs are given, all input trees will be broadcasted to the common suffix structure
    of all inputs:

    >>> pytree.broadcast_map_with_path(  # doctest: +IGNORE_WHITESPACE
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_broadcast_map_with_path(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def broadcast_map_with_accessor(
    func: Callable[..., U],
    tree: PyTree[T],
    /,
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree accessors to produce a new pytree.

    See also :func:`pytree.broadcast_map`, :func:`pytree.map`, :func:`pytree.map_`,
    and :func:`pytree.map_with_accessor`.

    If only one input is provided, this function is the same as :func:`pytree.map`:

    >>> import optree.pytree as pytree
    >>> pytree.broadcast_map_with_accessor(lambda a, x: (len(a), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> pytree.broadcast_map_with_accessor(lambda a, x: x + len(a), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> pytree.broadcast_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: a.codify('tree'),
    ...     {'x': 7, 'y': (42, 64), 'z': {1.5: None}},
    ... )
    {
        'x': "tree['x']",
        'y': ("tree['y'][0]", "tree['y'][1]"),
        'z': {1.5: None}
    }
    >>> pytree.broadcast_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
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

    >>> pytree.broadcast_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_broadcast_map_with_accessor(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@overload
def reduce(
    func: Callable[[T, T], T],
    tree: PyTree[T],
    /,
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


@overload
def reduce(
    func: Callable[[T, S], T],
    tree: PyTree[S],
    /,
    initial: T = __MISSING,
    *,
    is_leaf: Callable[[S], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


def reduce(  # pyright: ignore[reportInconsistentOverload]
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

    See also :func:`pytree.leaves` and :func:`pytree.sum`.

    >>> import optree.pytree as pytree
    >>> pytree.reduce(lambda x, y: x + y, {'x': 1, 'y': (2, 3)})
    6
    >>> pytree.reduce(lambda x, y: x + y, {'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node with arity 0 by default
    6
    >>> pytree.reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3})
    3
    >>> pytree.reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """  # pylint: disable=line-too-long
    return ops.tree_reduce(
        func,
        tree,
        initial=initial,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def sum(
    tree: PyTree[T],
    /,
    start: T = 0,  # type: ignore[assignment]
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T:
    """Sum ``start`` and leaf values in ``tree`` in left-to-right depth-first order and return the total.

    See also :func:`pytree.leaves` and :func:`pytree.reduce`.

    >>> import optree.pytree as pytree
    >>> pytree.sum({'x': 1, 'y': (2, 3)})
    6
    >>> pytree.sum({'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node with arity 0 by default
    6
    >>> pytree.sum({'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
    >>> pytree.sum({'x': 'a', 'y': ('b', None), 'z': 'c'}, start='')
    'abc'
    >>> pytree.sum({'x': [1], 'y': ([2], [None]), 'z': [3]}, start=[], is_leaf=lambda x: isinstance(x, list))
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_sum(
        tree,
        start=start,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@overload
def max(
    tree: PyTree[T],
    /,
    *,
    is_leaf: Callable[[T], bool] | None = None,
    key: Callable[[T], Any] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


@overload
def max(
    tree: PyTree[T],
    /,
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


def max(  # pyright: ignore[reportInconsistentOverload]
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

    See also :func:`pytree.leaves` and :func:`pytree.min`.

    >>> import optree.pytree as pytree
    >>> pytree.max({})
    Traceback (most recent call last):
        ...
    ValueError: max() iterable argument is empty
    >>> pytree.max({}, default=0)
    0
    >>> pytree.max({'x': 0, 'y': (2, 1)})
    2
    >>> pytree.max({'x': 0, 'y': (2, 1)}, key=lambda x: -x)
    0
    >>> pytree.max({'a': None})  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: max() iterable argument is empty
    >>> pytree.max({'a': None}, default=0)  # `None` is a non-leaf node with arity 0 by default
    0
    >>> pytree.max({'a': None}, none_is_leaf=True)
    None
    >>> pytree.max(None)  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: max() iterable argument is empty
    >>> pytree.max(None, default=0)
    0
    >>> pytree.max(None, none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_max(
        tree,
        default=default,
        key=key,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@overload
def min(
    tree: PyTree[T],
    /,
    *,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


@overload
def min(
    tree: PyTree[T],
    /,
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


def min(  # pyright: ignore[reportInconsistentOverload]
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

    See also :func:`pytree.leaves` and :func:`pytree.max`.

    >>> import optree.pytree as pytree
    >>> pytree.min({})
    Traceback (most recent call last):
        ...
    ValueError: min() iterable argument is empty
    >>> pytree.min({}, default=0)
    0
    >>> pytree.min({'x': 0, 'y': (2, 1)})
    0
    >>> pytree.min({'x': 0, 'y': (2, 1)}, key=lambda x: -x)
    2
    >>> pytree.min({'a': None})  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: min() iterable argument is empty
    >>> pytree.min({'a': None}, default=0)  # `None` is a non-leaf node with arity 0 by default
    0
    >>> pytree.min({'a': None}, none_is_leaf=True)
    None
    >>> pytree.min(None)  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: min() iterable argument is empty
    >>> pytree.min(None, default=0)
    0
    >>> pytree.min(None, none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_min(
        tree,
        default=default,
        key=key,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def all(
    tree: PyTree[T],
    /,
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether all leaves in ``tree`` are true (or if ``tree`` is empty).

    See also :func:`pytree.leaves` and :func:`pytree.any`.

    >>> import optree.pytree as pytree
    >>> pytree.all({})
    True
    >>> pytree.all({'x': 1, 'y': (2, 3)})
    True
    >>> pytree.all({'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node by default
    True
    >>> pytree.all({'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    False
    >>> pytree.all(None)  # `None` is a non-leaf node by default
    True
    >>> pytree.all(None, none_is_leaf=True)
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_all(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def any(
    tree: PyTree[T],
    /,
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether all leaves in ``tree`` are true (or :data:`False` if ``tree`` is empty).

    See also :func:`pytree.leaves` and :func:`pytree.all`.

    >>> import optree.pytree as pytree
    >>> pytree.any({})
    False
    >>> pytree.any({'x': 0, 'y': (2, 0)})
    True
    >>> pytree.any({'a': None})  # `None` is a non-leaf node with arity 0 by default
    False
    >>> pytree.any({'a': None}, none_is_leaf=True)  # `None` is evaluated as false
    False
    >>> pytree.any(None)  # `None` is a non-leaf node with arity 0 by default
    False
    >>> pytree.any(None, none_is_leaf=True)  # `None` is evaluated as false
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

    .. versionadded:: 0.14.1
    """
    return ops.tree_any(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def flatten_one_level(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> FlattenOneLevelOutputEx[T]:
    """Flatten the pytree one level, returning a 4-tuple of children, metadata, path entries, and an unflatten function.

    See also :func:`pytree.flatten`, :func:`pytree.flatten_with_path`.

    >>> import optree.pytree as pytree
    >>> children, metadata, entries, unflatten_func = pytree.flatten_one_level({'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5})
    >>> children, metadata, entries
    ([1, (2, [3, 4]), None, 5], ['a', 'b', 'c', 'd'], ('a', 'b', 'c', 'd'))
    >>> unflatten_func(metadata, children)
    {'a': 1, 'b': (2, [3, 4]), 'c': None, 'd': 5}
    >>> children, metadata, entries, unflatten_func = pytree.flatten_one_level([{'a': 1, 'b': (2, 3)}, (4, 5)])
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

    .. versionadded:: 0.14.1
    """  # pylint: disable=line-too-long
    return ops.tree_flatten_one_level(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
