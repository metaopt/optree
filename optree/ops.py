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
# mypy: disable-error-code=union-attr
# pyright: reportOptionalMemberAccess=false

from __future__ import annotations

import difflib
import itertools
import textwrap
from collections import OrderedDict, defaultdict, deque
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
)

import optree._C as _C
from optree import tree
from optree.accessor import PyTreeAccessor
from optree.typing import NamedTuple, T, is_namedtuple_instance, is_structseq_instance


if TYPE_CHECKING:
    from optree.typing import PyTree, PyTreeSpec, S
    from optree.typing import structseq as StructSequence  # noqa: N812


__all__ = [
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
    'prefix_errors',
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


tree_flatten = tree.flatten
tree_flatten.__doc__ = tree.flatten.__doc__.replace('tree.flatten', 'tree_flatten')

tree_flatten_with_path = tree.flatten_with_path
tree_flatten_with_path.__doc__ = tree.flatten_with_path.__doc__.replace(
    'tree.flatten_with_path',
    'tree_flatten_with_path',
)

tree_flatten_with_accessor = tree.flatten_with_accessor
tree_flatten_with_accessor.__doc__ = tree.flatten_with_accessor.__doc__.replace(
    'tree.flatten_with_accessor',
    'tree_flatten_with_accessor',
)
tree_unflatten = tree.unflatten
tree_unflatten.__doc__ = tree.unflatten.__doc__.replace('tree.unflatten', 'tree_unflatten')

tree_iter = tree.iter
tree_iter.__doc__ = tree.iter.__doc__.replace('tree.iter', 'tree_iter')

tree_leaves = tree.leaves
tree_leaves.__doc__ = tree.leaves.__doc__.replace('tree.leaves', 'tree_leaves')

tree_structure = tree.structure
tree_structure.__doc__ = tree.structure.__doc__.replace('tree.structure', 'tree_structure')

tree_paths = tree.paths
tree_paths.__doc__ = tree.paths.__doc__.replace('tree.paths', 'tree_paths')

tree_accessors = tree.accessors
tree_accessors.__doc__ = tree.accessors.__doc__.replace('tree.accessors', 'tree_accessors')

tree_is_leaf = tree.is_leaf
tree_is_leaf.__doc__ = tree.is_leaf.__doc__.replace('tree.is_leaf', 'tree_is_leaf')

tree_map = tree.map
tree_map.__doc__ = tree.map.__doc__.replace('tree.map', 'tree_map')

tree_map_ = tree.map_
tree_map_.__doc__ = tree.map_.__doc__.replace('tree.map_', 'tree_map_')


def all_leaves(
    iterable: Iterable[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether all elements in the given iterable are all leaves.

    See also :func:`tree_flatten`, :func:`tree_leaves`, and :func:`tree_is_leaf`.

    >>> tree = {'a': [1, 2, 3]}
    >>> all_leaves(tree_leaves(tree))
    True
    >>> all_leaves([tree])
    False
    >>> all_leaves([1, 2, None, 3])
    False
    >>> all_leaves([1, 2, None, 3], none_is_leaf=True)
    True

    Note that this function iterates and checks the elements in the input iterable object, which
    uses the :func:`iter` function. For dictionaries, ``iter(d)`` for a dictionary ``d`` iterates
    the keys of the dictionary, not the values.

    >>> list({'a': 1, 'b': (2, 3)})
    ['a', 'b']
    >>> all_leaves({'a': 1, 'b': (2, 3)})
    True

    This function is useful in advanced cases. For example, if a library allows arbitrary map
    operations on a flat list of leaves it may want to check if the result is still a flat list
    of leaves.

    Args:
        iterable (iterable): A iterable of leaves.
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
        A boolean indicating if all elements in the input iterable are leaves.
    """
    return _C.all_leaves(iterable, is_leaf, none_is_leaf, namespace)


tree_map_with_path = tree.map_with_path
tree_map_with_path.__doc__ = tree.map_with_path.__doc__.replace(
    'tree.map_with_path',
    'tree_map_with_path',
)

tree_map_with_path_ = tree.map_with_path_
tree_map_with_path_.__doc__ = tree.map_with_path_.__doc__.replace(
    'tree.map_with_path_',
    'tree_map_with_path_',
)

tree_map_with_accessor = tree.map_with_accessor
tree_map_with_accessor.__doc__ = tree.map_with_accessor.__doc__.replace(
    'tree.map_with_accessor',
    'tree_map_with_accessor',
)

tree_map_with_accessor_ = tree.map_with_accessor_
tree_map_with_accessor_.__doc__ = tree.map_with_accessor_.__doc__.replace(
    'tree.map_with_accessor_',
    'tree_map_with_accessor_',
)

tree_replace_nones = tree.replace_nones
tree_replace_nones.__doc__ = tree.replace_nones.__doc__.replace(
    'tree.replace_nones',
    'tree_replace_nones',
)

tree_transpose = tree.transpose
tree_transpose.__doc__ = tree.transpose.__doc__.replace('tree.transpose', 'tree_transpose')

tree_transpose_map = tree.transpose_map
tree_transpose_map.__doc__ = tree.transpose_map.__doc__.replace(
    'tree.transpose_map',
    'tree_transpose_map',
)

tree_transpose_map_with_path = tree.transpose_map_with_path
tree_transpose_map_with_path.__doc__ = tree.transpose_map_with_path.__doc__.replace(
    'tree.transpose_map_with_path',
    'tree_transpose_map_with_path',
)

tree_transpose_map_with_accessor = tree.transpose_map_with_accessor
tree_transpose_map_with_accessor.__doc__ = tree.transpose_map_with_accessor.__doc__.replace(
    'tree.transpose_map_with_accessor',
    'tree_transpose_map_with_accessor',
)

tree_broadcast_prefix = tree.broadcast_prefix
tree_broadcast_prefix.__doc__ = tree.broadcast_prefix.__doc__.replace(
    'tree.broadcast_prefix',
    'tree_broadcast_prefix',
)


def broadcast_prefix(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[T]:
    """Return a list of broadcasted leaves in ``prefix_tree`` to match the number of leaves in ``full_tree``.

    See also :func:`tree_broadcast_prefix`, :func:`broadcast_common`, and :func:`treespec_is_prefix`.

    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be
    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.

    This function returns a list of leaves with the same size as ``full_tree``. The leaves are
    replicated from ``prefix_tree``. The number of replicas is determined by the corresponding
    subtree in ``full_tree``.

    >>> broadcast_prefix(1, [2, 3, 4])
    [1, 1, 1]
    >>> broadcast_prefix([1, 2, 3], [4, 5, 6])
    [1, 2, 3]
    >>> broadcast_prefix([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4; list: [4, 5, 6, 7].
    >>> broadcast_prefix([1, 2, 3], [4, 5, (6, 7)])
    [1, 2, 3, 3]
    >>> broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}])
    [1, 2, 3, 3, 3]
    >>> broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}], none_is_leaf=True)
    [1, 2, 3, 3, 3, 3]

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
        A list of leaves in ``prefix_tree`` broadcasted to match the number of leaves in ``full_tree``.
    """
    result: list[T] = []

    def add_leaves(x: T, subtree: PyTree[S]) -> None:
        subtreespec = tree_structure(
            subtree,
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        result.extend(itertools.repeat(x, subtreespec.num_leaves))

    # If prefix_tree is not a tree prefix of full_tree, this code can raise a ValueError;
    # use prefix_errors to find disagreements and raise more precise error messages.
    # errors = prefix_errors(
    #     prefix_tree,
    #     full_tree,
    #     is_leaf=is_leaf,
    #     none_is_leaf=none_is_leaf,
    #     namespace=namespace,
    # )
    tree_map_(
        add_leaves,
        prefix_tree,
        full_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    return result


tree_broadcast_common = tree.broadcast_common
tree_broadcast_common.__doc__ = tree.broadcast_common.__doc__.replace(
    'tree.broadcast_common',
    'tree_broadcast_common',
)


def broadcast_common(
    tree: PyTree[T],
    other_tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[T], list[T]]:
    """Return two lists of leaves in ``tree`` and ``other_tree`` broadcasted to match the number of leaves in the common suffix structure.

    See also :func:`tree_broadcast_common`, :func:`broadcast_prefix`, and :func:`treespec_is_prefix`.

    If a ``suffix_tree`` is a suffix of a ``tree``, this means the ``suffix_tree`` can be
    constructed by replacing the leaves of ``tree`` with appropriate **subtrees**.

    This function returns two pytrees with the same structure. The tree structure is the common
    suffix structure of ``tree`` and ``other_tree``. The leaves are replicated from ``tree`` and
    ``other_tree``. The number of replicas is determined by the corresponding subtree in the suffix
    structure.

    >>> broadcast_common(1, [2, 3, 4])
    ([1, 1, 1], [2, 3, 4])
    >>> broadcast_common([1, 2, 3], [4, 5, 6])
    ([1, 2, 3], [4, 5, 6])
    >>> broadcast_common([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4.
    >>> broadcast_common([1, (2, 3), 4], [5, 6, (7, 8)])
    ([1, 2, 3, 4, 4], [5, 6, 6, 7, 8])
    >>> broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}])
    ([1, 2, 3, 4, 4, 4], [5, 6, 6, 7, 8, 9])
    >>> broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}], none_is_leaf=True)
    ([1, 2, 3, 4, 4, 4, 4], [5, 6, 6, 7, 8, None, 9])
    >>> broadcast_common([1, None], [None, 2])
    ([], [])
    >>> broadcast_common([1, None], [None, 2], none_is_leaf=True)
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
        Two lists of leaves in ``tree`` and ``other_tree`` broadcasted to match the number of leaves
        in the common suffix structure.
    """  # pylint: disable=line-too-long
    broadcasted_tree, other_broadcasted_tree = tree_broadcast_common(
        tree,
        other_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )

    broadcasted_leaves: list[T] = []
    other_broadcasted_leaves: list[T] = []

    def add_leaves(x: T, y: T) -> None:
        broadcasted_leaves.append(x)
        other_broadcasted_leaves.append(y)

    tree_map_(
        add_leaves,
        broadcasted_tree,
        other_broadcasted_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    return broadcasted_leaves, other_broadcasted_leaves


tree_broadcast_map = tree.broadcast_map
tree_broadcast_map.__doc__ = tree.broadcast_map.__doc__.replace(
    'tree.broadcast_map',
    'tree_broadcast_map',
)

tree_broadcast_map_with_path = tree.broadcast_map_with_path
tree_broadcast_map_with_path.__doc__ = tree.broadcast_map_with_path.__doc__.replace(
    'tree.broadcast_map_with_path',
    'tree_broadcast_map_with_path',
)

tree_broadcast_map_with_accessor = tree.broadcast_map_with_accessor
tree_broadcast_map_with_accessor.__doc__ = tree.broadcast_map_with_accessor.__doc__.replace(
    'tree.broadcast_map_with_accessor',
    'tree_broadcast_map_with_accessor',
)

tree_reduce = tree.reduce
tree_reduce.__doc__ = tree.reduce.__doc__.replace('tree.reduce', 'tree_reduce')

tree_sum = tree.sum
tree_sum.__doc__ = tree.sum.__doc__.replace('tree.sum', 'tree_sum')

tree_max = tree.max
tree_max.__doc__ = tree.max.__doc__.replace('tree.max', 'tree_max')

tree_min = tree.min
tree_min.__doc__ = tree.min.__doc__.replace('tree.min', 'tree_min')

tree_all = tree.all
tree_all.__doc__ = tree.all.__doc__.replace('tree.all', 'tree_all')

tree_any = tree.any
tree_any.__doc__ = tree.any.__doc__.replace('tree.any', 'tree_any')

tree_flatten_one_level = tree.flatten_one_level
tree_flatten_one_level.__doc__ = tree.flatten_one_level.__doc__.replace(
    'tree.flatten_one_level',
    'tree_flatten_one_level',
)


def treespec_paths(treespec: PyTreeSpec, /) -> list[tuple[Any, ...]]:
    """Return a list of paths to the leaves of a treespec.

    See also :func:`tree_flatten_with_path`, :func:`tree_paths`, and :meth:`PyTreeSpec.paths`.

    >>> treespec = tree_structure({'b': 3, 'a': (0, [1, 2]), 'c': (4, None)})
    >>> treespec
    PyTreeSpec({'a': (*, [*, *]), 'b': *, 'c': (*, None)})
    >>> treespec_paths(treespec)
    [('a', 0), ('a', 1, 0), ('a', 1, 1), ('b',), ('c', 0)]
    """
    return treespec.paths()


def treespec_accessors(treespec: PyTreeSpec, /) -> list[PyTreeAccessor]:
    """Return a list of accessors to the leaves of a treespec.

    See also :func:`tree_flatten_with_accessor`, :func:`tree_accessors`,
    and :meth:`PyTreeSpec.accessors`.

    >>> treespec = tree_structure({'b': 3, 'a': (0, [1, 2]), 'c': (4, None)})
    >>> treespec
    PyTreeSpec({'a': (*, [*, *]), 'b': *, 'c': (*, None)})
    >>> treespec_accessors(treespec)  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
    [
        PyTreeAccessor(*['a'][0], ...),
        PyTreeAccessor(*['a'][1][0], ...),
        PyTreeAccessor(*['a'][1][1], ...),
        PyTreeAccessor(*['b'], ...),
        PyTreeAccessor(*['c'][0], ...)
    ]
    >>> treespec_accessors(treespec_leaf())
    [PyTreeAccessor(*, ())]
    >>> treespec_accessors(treespec_none())
    []
    """
    return treespec.accessors()


def treespec_entries(treespec: PyTreeSpec, /) -> list[Any]:
    """Return a list of one-level entries of a treespec to its children.

    See also :func:`treespec_entry`, :func:`treespec_paths`, :func:`treespec_children`,
    and :meth:`PyTreeSpec.entries`.

    >>> treespec = tree_structure({'b': 3, 'a': (0, [1, 2]), 'c': (4, None)})
    >>> treespec
    PyTreeSpec({'a': (*, [*, *]), 'b': *, 'c': (*, None)})
    >>> treespec_entries(treespec)
    ['a', 'b', 'c']
    """
    return treespec.entries()


def treespec_entry(treespec: PyTreeSpec, index: int, /) -> Any:
    """Return the entry of a treespec at the given index.

    See also :func:`treespec_entries`, :func:`treespec_children`, and :meth:`PyTreeSpec.entry`.
    """
    return treespec.entry(index)


def treespec_children(treespec: PyTreeSpec, /) -> list[PyTreeSpec]:
    """Return a list of treespecs for the children of a treespec.

    See also :func:`treespec_child`, :func:`treespec_paths`, :func:`treespec_entries`,
    :func:`treespec_one_level`, and :meth:`PyTreeSpec.children`.

    >>> treespec = tree_structure({'b': 3, 'a': (0, [1, 2]), 'c': (4, None)})
    >>> treespec
    PyTreeSpec({'a': (*, [*, *]), 'b': *, 'c': (*, None)})
    >>> treespec_children(treespec)
    [PyTreeSpec((*, [*, *])), PyTreeSpec(*), PyTreeSpec((*, None))]
    """
    return treespec.children()


def treespec_child(treespec: PyTreeSpec, index: int, /) -> PyTreeSpec:
    """Return the treespec of the child of a treespec at the given index.

    See also :func:`treespec_children`, :func:`treespec_entries`, and :meth:`PyTreeSpec.child`.
    """
    return treespec.child(index)


def treespec_one_level(treespec: PyTreeSpec, /) -> PyTreeSpec | None:
    """Return the one-level tree structure of the treespec or :data:`None` if the treespec is a leaf.

    See also :func:`treespec_children`, :func:`treespec_is_one_level`, and :meth:`PyTreeSpec.one_level`.

    >>> treespec = tree_structure({'b': 3, 'a': (0, [1, 2]), 'c': (4, None)})
    >>> treespec
    PyTreeSpec({'a': (*, [*, *]), 'b': *, 'c': (*, None)})
    >>> treespec_one_level(treespec)
    PyTreeSpec({'a': *, 'b': *, 'c': *})
    """
    return treespec.one_level()


def treespec_transform(
    treespec: PyTreeSpec,
    /,
    f_node: Callable[[PyTreeSpec], PyTreeSpec] | None = None,
    f_leaf: Callable[[PyTreeSpec], PyTreeSpec] | None = None,
) -> PyTreeSpec:
    """Transform a treespec by applying functions to its nodes and leaves.

    See also :func:`treespec_children`, :func:`treespec_is_leaf`, and :meth:`PyTreeSpec.transform`.

    >>> treespec = tree_structure({'b': 3, 'a': (0, [1, 2]), 'c': (4, None)})
    >>> treespec
    PyTreeSpec({'a': (*, [*, *]), 'b': *, 'c': (*, None)})
    >>> treespec_transform(treespec, lambda spec: treespec_dict(zip(spec.entries(), spec.children())))
    PyTreeSpec({'a': {0: *, 1: {0: *, 1: *}}, 'b': *, 'c': {0: *, 1: {}}})
    >>> treespec_transform(
    ...     treespec,
    ...     lambda spec: (
    ...         treespec_ordereddict(zip(spec.entries(), spec.children()))
    ...         if spec.type is dict
    ...         else spec
    ...     ),
    ... )
    PyTreeSpec(OrderedDict({'a': (*, [*, *]), 'b': *, 'c': (*, None)}))
    >>> treespec_transform(
    ...     treespec,
    ...     lambda spec: (
    ...         treespec_ordereddict(tree_unflatten(spec, spec.children()))
    ...         if spec.type is dict
    ...         else spec
    ...     ),
    ... )
    PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': (*, None)}))
    >>> treespec_transform(treespec, lambda spec: treespec_tuple(spec.children()))
    PyTreeSpec(((*, (*, *)), *, (*, ())))
    >>> treespec_transform(
    ...     treespec,
    ...     lambda spec: (
    ...         treespec_list(spec.children())
    ...         if spec.type is tuple
    ...         else spec
    ...     ),
    ... )
    PyTreeSpec({'a': [*, [*, *]], 'b': *, 'c': [*, None]})
    >>> treespec_transform(treespec, None, lambda spec: tree_structure((1, [2])))
    PyTreeSpec({'a': ((*, [*]), [(*, [*]), (*, [*])]), 'b': (*, [*]), 'c': ((*, [*]), None)})
    """
    return treespec.transform(f_node, f_leaf)


def treespec_is_leaf(treespec: PyTreeSpec, /, *, strict: bool = True) -> bool:
    """Return whether the treespec is a leaf that has no children.

    See also :func:`treespec_is_strict_leaf` and :meth:`PyTreeSpec.is_leaf`.

    This function is equivalent to ``treespec.is_leaf(strict=strict)``. If ``strict=False``, it will
    return :data:`True` if and only if the treespec represents a strict leaf. If ``strict=False``,
    it will return :data:`True` if the treespec represents a strict leaf or :data:`None` or an empty
    container (e.g., an empty tuple).

    >>> treespec_is_leaf(tree_structure(1))
    True
    >>> treespec_is_leaf(tree_structure((1, 2)))
    False
    >>> treespec_is_leaf(tree_structure(None))
    False
    >>> treespec_is_leaf(tree_structure(None), strict=False)
    True
    >>> treespec_is_leaf(tree_structure(None, none_is_leaf=False))
    False
    >>> treespec_is_leaf(tree_structure(None, none_is_leaf=True))
    True
    >>> treespec_is_leaf(tree_structure(()))
    False
    >>> treespec_is_leaf(tree_structure(()), strict=False)
    True
    >>> treespec_is_leaf(tree_structure([]))
    False
    >>> treespec_is_leaf(tree_structure([]), strict=False)
    True

    Args:
        treespec (PyTreeSpec): A treespec.
        strict (bool, optional): Whether not to treat :data:`None` or an empty
            container (e.g., an empty tuple) as a leaf. (default: :data:`True`)

    Returns:
        :data:`True` if the treespec represents a leaf that has no children, otherwise, :data:`False`.
    """
    if strict:
        return treespec.num_nodes == 1 and treespec.num_leaves == 1
    return treespec.num_nodes == 1


def treespec_is_strict_leaf(treespec: PyTreeSpec, /) -> bool:
    """Return whether the treespec is a strict leaf.

    See also :func:`treespec_is_leaf` and :meth:`PyTreeSpec.is_leaf`.

    This function respects the ``none_is_leaf`` setting in the treespec. It is equivalent to
    ``treespec.is_leaf(strict=True)``. It will return :data:`True` if and only if the treespec
    represents a strict leaf.

    >>> treespec_is_strict_leaf(tree_structure(1))
    True
    >>> treespec_is_strict_leaf(tree_structure((1, 2)))
    False
    >>> treespec_is_strict_leaf(tree_structure(None))
    False
    >>> treespec_is_strict_leaf(tree_structure(None, none_is_leaf=False))
    False
    >>> treespec_is_strict_leaf(tree_structure(None, none_is_leaf=True))
    True
    >>> treespec_is_strict_leaf(tree_structure(()))
    False
    >>> treespec_is_strict_leaf(tree_structure([]))
    False

    Args:
        treespec (PyTreeSpec): A treespec.

    Returns:
        :data:`True` if the treespec represents a strict leaf, otherwise, :data:`False`.
    """
    return treespec.num_nodes == 1 and treespec.num_leaves == 1


def treespec_is_one_level(treespec: PyTreeSpec, /) -> bool:
    """Return whether the treespec is a one-level tree structure.

    See also :func:`treespec_is_leaf`, :func:`treespec_one_level`, and :meth:`PyTreeSpec.is_one_level`.

    >>> treespec_is_one_level(tree_structure(1))
    False
    >>> treespec_is_one_level(tree_structure((1, 2)))
    True
    >>> treespec_is_one_level(tree_structure({'a': 1, 'b': 2, 'c': 3}))
    True
    >>> treespec_is_one_level(tree_structure({'a': 1, 'b': (2, 3), 'c': 4}))
    False
    >>> treespec_is_one_level(tree_structure(None))
    True
    """
    return (
        treespec.num_nodes == treespec.num_children + 1
        and treespec.num_leaves == treespec.num_children
    )


def treespec_is_prefix(
    treespec: PyTreeSpec,
    other_treespec: PyTreeSpec,
    /,
    *,
    strict: bool = False,
) -> bool:
    """Return whether ``treespec`` is a prefix of ``other_treespec``.

    See also :func:`treespec_is_prefix` and :meth:`PyTreeSpec.is_prefix`.
    """
    return treespec.is_prefix(other_treespec, strict=strict)


def treespec_is_suffix(
    treespec: PyTreeSpec,
    other_treespec: PyTreeSpec,
    /,
    *,
    strict: bool = False,
) -> bool:
    """Return whether ``treespec`` is a suffix of ``other_treespec``.

    See also :func:`treespec_is_suffix` :meth:`PyTreeSpec.is_suffix`.
    """
    return treespec.is_suffix(other_treespec, strict=strict)


def treespec_leaf(
    *,
    none_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec:
    """Make a treespec representing a leaf node.

    See also :func:`tree_structure`, :func:`treespec_none`, and :func:`treespec_tuple`.

    >>> treespec_leaf()
    PyTreeSpec(*)
    >>> treespec_leaf(none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec_leaf(none_is_leaf=False) == treespec_leaf(none_is_leaf=True)
    False
    >>> treespec_leaf() == tree_structure(1)
    True
    >>> treespec_leaf(none_is_leaf=True) == tree_structure(1, none_is_leaf=True)
    True
    >>> treespec_leaf(none_is_leaf=True) == tree_structure(None, none_is_leaf=True)
    True
    >>> treespec_leaf(none_is_leaf=True) == tree_structure(None, none_is_leaf=False)
    False
    >>> treespec_leaf(none_is_leaf=True) == treespec_none(none_is_leaf=True)
    True
    >>> treespec_leaf(none_is_leaf=True) == treespec_none(none_is_leaf=False)
    False
    >>> treespec_leaf(none_is_leaf=False) == treespec_none(none_is_leaf=True)
    False
    >>> treespec_leaf(none_is_leaf=False) == treespec_none(none_is_leaf=False)
    False

    Args:
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a leaf node.
    """
    return _C.make_leaf(
        none_is_leaf,
        namespace,  # unused
    )


def treespec_none(
    *,
    none_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec:
    """Make a treespec representing a :data:`None` node.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_tuple`.

    >>> treespec_none()
    PyTreeSpec(None)
    >>> treespec_none(none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec_none(none_is_leaf=False) == treespec_none(none_is_leaf=True)
    False
    >>> treespec_none() == tree_structure(None)
    True
    >>> treespec_none() == tree_structure(1)
    False
    >>> treespec_none(none_is_leaf=True) == tree_structure(1, none_is_leaf=True)
    True
    >>> treespec_none(none_is_leaf=True) == tree_structure(None, none_is_leaf=True)
    True
    >>> treespec_none(none_is_leaf=True) == tree_structure(None, none_is_leaf=False)
    False
    >>> treespec_none(none_is_leaf=True) == treespec_leaf(none_is_leaf=True)
    True
    >>> treespec_none(none_is_leaf=False) == treespec_leaf(none_is_leaf=True)
    False
    >>> treespec_none(none_is_leaf=True) == treespec_leaf(none_is_leaf=False)
    False
    >>> treespec_none(none_is_leaf=False) == treespec_leaf(none_is_leaf=False)
    False

    Args:
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a :data:`None` node.
    """
    return _C.make_none(
        none_is_leaf,
        namespace,  # unused
    )


def treespec_tuple(
    iterable: Iterable[PyTreeSpec] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a tuple treespec from an iterable of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_tuple([treespec_leaf(), treespec_leaf()])
    PyTreeSpec((*, *))
    >>> treespec_tuple([treespec_leaf(), treespec_leaf(), treespec_none()])
    PyTreeSpec((*, *, None))
    >>> treespec_tuple()
    PyTreeSpec(())
    >>> treespec_tuple([treespec_leaf(), treespec_tuple([treespec_leaf(), treespec_leaf()])])
    PyTreeSpec((*, (*, *)))
    >>> treespec_tuple([treespec_leaf(), tree_structure({'a': 1, 'b': 2})])
    PyTreeSpec((*, {'a': *, 'b': *}))
    >>> treespec_tuple([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a tuple node with the given children.
    """
    return _C.make_from_collection(
        tuple(iterable),
        none_is_leaf,
        namespace,
    )


def treespec_list(
    iterable: Iterable[PyTreeSpec] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a list treespec from an iterable of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_list([treespec_leaf(), treespec_leaf()])
    PyTreeSpec([*, *])
    >>> treespec_list([treespec_leaf(), treespec_leaf(), treespec_none()])
    PyTreeSpec([*, *, None])
    >>> treespec_list()
    PyTreeSpec([])
    >>> treespec_list([treespec_leaf(), treespec_tuple([treespec_leaf(), treespec_leaf()])])
    PyTreeSpec([*, (*, *)])
    >>> treespec_list([treespec_leaf(), tree_structure({'a': 1, 'b': 2})])
    PyTreeSpec([*, {'a': *, 'b': *}])
    >>> treespec_list([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a list node with the given children.
    """
    return _C.make_from_collection(
        list(iterable),
        none_is_leaf,
        namespace,
    )


def treespec_dict(
    mapping: Mapping[Any, PyTreeSpec] | Iterable[tuple[Any, PyTreeSpec]] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    """Make a dict treespec from a dict of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_dict({'a': treespec_leaf(), 'b': treespec_leaf()})
    PyTreeSpec({'a': *, 'b': *})
    >>> treespec_dict([('b', treespec_leaf()), ('c', treespec_leaf()), ('a', treespec_none())])
    PyTreeSpec({'a': None, 'b': *, 'c': *})
    >>> treespec_dict()
    PyTreeSpec({})
    >>> treespec_dict(a=treespec_leaf(), b=treespec_tuple([treespec_leaf(), treespec_leaf()]))
    PyTreeSpec({'a': *, 'b': (*, *)})
    >>> treespec_dict({'a': treespec_leaf(), 'b': tree_structure([1, 2])})
    PyTreeSpec({'a': *, 'b': [*, *]})
    >>> treespec_dict({'a': treespec_leaf(), 'b': tree_structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)
        **kwargs (PyTreeSpec, optional): Additional child treespecs to add to the mapping.

    Returns:
        A treespec representing a dict node with the given children.
    """
    return _C.make_from_collection(
        dict(mapping, **kwargs),
        none_is_leaf,
        namespace,
    )


def treespec_namedtuple(
    namedtuple: NamedTuple[PyTreeSpec],  # type: ignore[type-arg]
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a namedtuple treespec from a namedtuple of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> from collections import namedtuple
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=treespec_leaf()))
    PyTreeSpec(Point(x=*, y=*))
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=treespec_tuple([treespec_leaf(), treespec_leaf()])))
    PyTreeSpec(Point(x=*, y=(*, *)))
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=tree_structure([1, 2])))
    PyTreeSpec(Point(x=*, y=[*, *]))
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=tree_structure([1, 2], none_is_leaf=True)))
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        namedtuple (namedtuple of PyTreeSpec): A namedtuple of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a dict node with the given children.
    """
    if not is_namedtuple_instance(namedtuple):
        raise ValueError(f'Expected a namedtuple of PyTreeSpec(s), got {namedtuple!r}.')
    return _C.make_from_collection(
        namedtuple,
        none_is_leaf,
        namespace,
    )


def treespec_ordereddict(
    mapping: Mapping[Any, PyTreeSpec] | Iterable[tuple[Any, PyTreeSpec]] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    """Make an OrderedDict treespec from an OrderedDict of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_ordereddict({'a': treespec_leaf(), 'b': treespec_leaf()})
    PyTreeSpec(OrderedDict({'a': *, 'b': *}))
    >>> treespec_ordereddict([('b', treespec_leaf()), ('c', treespec_leaf()), ('a', treespec_none())])
    PyTreeSpec(OrderedDict({'b': *, 'c': *, 'a': None}))
    >>> treespec_ordereddict()
    PyTreeSpec(OrderedDict())
    >>> treespec_ordereddict(a=treespec_leaf(), b=treespec_tuple([treespec_leaf(), treespec_leaf()]))
    PyTreeSpec(OrderedDict({'a': *, 'b': (*, *)}))
    >>> treespec_ordereddict({'a': treespec_leaf(), 'b': tree_structure([1, 2])})
    PyTreeSpec(OrderedDict({'a': *, 'b': [*, *]}))
    >>> treespec_ordereddict({'a': treespec_leaf(), 'b': tree_structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)
        **kwargs (PyTreeSpec, optional): Additional child treespecs to add to the mapping.

    Returns:
        A treespec representing an OrderedDict node with the given children.
    """
    return _C.make_from_collection(
        OrderedDict(mapping, **kwargs),
        none_is_leaf,
        namespace,
    )


def treespec_defaultdict(
    default_factory: Callable[[], Any] | None = None,
    mapping: Mapping[Any, PyTreeSpec] | Iterable[tuple[Any, PyTreeSpec]] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    """Make a defaultdict treespec from a defaultdict of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_defaultdict(int, {'a': treespec_leaf(), 'b': treespec_leaf()})
    PyTreeSpec(defaultdict(<class 'int'>, {'a': *, 'b': *}))
    >>> treespec_defaultdict(int, [('b', treespec_leaf()), ('c', treespec_leaf()), ('a', treespec_none())])
    PyTreeSpec(defaultdict(<class 'int'>, {'a': None, 'b': *, 'c': *}))
    >>> treespec_defaultdict()
    PyTreeSpec(defaultdict(None, {}))
    >>> treespec_defaultdict(int)
    PyTreeSpec(defaultdict(<class 'int'>, {}))
    >>> treespec_defaultdict(int, a=treespec_leaf(), b=treespec_tuple([treespec_leaf(), treespec_leaf()]))
    PyTreeSpec(defaultdict(<class 'int'>, {'a': *, 'b': (*, *)}))
    >>> treespec_defaultdict(int, {'a': treespec_leaf(), 'b': tree_structure([1, 2])})
    PyTreeSpec(defaultdict(<class 'int'>, {'a': *, 'b': [*, *]}))
    >>> treespec_defaultdict(int, {'a': treespec_leaf(), 'b': tree_structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        default_factory (callable or None, optional): A factory function that will be used to create
            a missing value. (default: :data:`None`)
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)
        **kwargs (PyTreeSpec, optional): Additional child treespecs to add to the mapping.

    Returns:
        A treespec representing a defaultdict node with the given children.
    """
    return _C.make_from_collection(
        defaultdict(default_factory, mapping, **kwargs),
        none_is_leaf,
        namespace,
    )


def treespec_deque(
    iterable: Iterable[PyTreeSpec] = (),
    /,
    maxlen: int | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a deque treespec from a deque of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_deque([treespec_leaf(), treespec_leaf()])
    PyTreeSpec(deque([*, *]))
    >>> treespec_deque([treespec_leaf(), treespec_leaf(), treespec_none()], maxlen=5)
    PyTreeSpec(deque([*, *, None], maxlen=5))
    >>> treespec_deque()
    PyTreeSpec(deque([]))
    >>> treespec_deque([treespec_leaf(), treespec_tuple([treespec_leaf(), treespec_leaf()])])
    PyTreeSpec(deque([*, (*, *)]))
    >>> treespec_deque([treespec_leaf(), tree_structure({'a': 1, 'b': 2})], maxlen=5)
    PyTreeSpec(deque([*, {'a': *, 'b': *}], maxlen=5))
    >>> treespec_deque([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)], maxlen=5)
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        maxlen (int or None, optional): The maximum size of a deque or :data:`None` if unbounded.
            (default: :data:`None`)
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a deque node with the given children.
    """
    return _C.make_from_collection(
        deque(iterable, maxlen=maxlen),
        none_is_leaf,
        namespace,
    )


def treespec_structseq(
    structseq: StructSequence[PyTreeSpec],
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a PyStructSequence treespec from a PyStructSequence of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    Args:
        structseq (PyStructSequence of PyTreeSpec): A PyStructSequence of child treespecs. They must
            have the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a PyStructSequence node with the given children.
    """
    if not is_structseq_instance(structseq):
        raise ValueError(f'Expected a PyStructSequence of PyTreeSpec(s), got {structseq!r}.')
    return _C.make_from_collection(
        structseq,
        none_is_leaf,
        namespace,
    )


def treespec_from_collection(
    collection: Collection[PyTreeSpec],
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a treespec from a collection of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_from_collection(None)
    PyTreeSpec(None)
    >>> treespec_from_collection(None, none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec_from_collection(object())
    PyTreeSpec(*)
    >>> treespec_from_collection([treespec_leaf(), treespec_none()])
    PyTreeSpec([*, None])
    >>> treespec_from_collection({'a': treespec_leaf(), 'b': treespec_none()})
    PyTreeSpec({'a': *, 'b': None})
    >>> treespec_from_collection(deque([treespec_leaf(), tree_structure({'a': 1, 'b': 2})], maxlen=5))
    PyTreeSpec(deque([*, {'a': *, 'b': *}], maxlen=5))
    >>> treespec_from_collection({'a': treespec_leaf(), 'b': (treespec_leaf(), treespec_none())})
    Traceback (most recent call last):
        ...
    ValueError: Expected a(n) dict of PyTreeSpec(s), got {'a': PyTreeSpec(*), 'b': (PyTreeSpec(*), PyTreeSpec(None))}.
    >>> treespec_from_collection([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.


    Args:
        collection (collection of PyTreeSpec): A collection of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing the same structure of the collection with the given children.
    """
    return _C.make_from_collection(collection, none_is_leaf, namespace)


STANDARD_DICT_TYPES: frozenset[type] = frozenset({dict, OrderedDict, defaultdict})


def prefix_errors(  # noqa: C901
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[Callable[[str], ValueError]]:
    """Return a list of errors that would be raised by :func:`broadcast_prefix`."""

    def helper(  # pylint: disable=too-many-locals
        accessor: PyTreeAccessor,
        prefix_subtree: PyTree[T],
        full_subtree: PyTree[S],
    ) -> Iterable[Callable[[str], ValueError]]:
        # A leaf is a valid prefix of any tree
        if tree_is_leaf(
            prefix_subtree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ):
            return

        # The subtrees may disagree because their roots are of different types:
        prefix_tree_type = type(prefix_subtree)
        full_tree_type = type(full_subtree)
        both_standard_dict = (
            prefix_tree_type in STANDARD_DICT_TYPES and full_tree_type in STANDARD_DICT_TYPES
        )
        both_deque = prefix_tree_type is deque and full_tree_type is deque  # type: ignore[comparison-overlap]
        if (
            prefix_tree_type is not full_tree_type
            and not both_standard_dict  # special handling for dictionary types
        ):
            yield lambda name: ValueError(
                f'pytree structure error: different types at key path\n'
                f'    {accessor.codify(name) if accessor else name + " tree root"}\n'
                f'At that key path, the prefix pytree {name} has a subtree of type\n'
                f'    {type(prefix_subtree)}\n'
                f'but at the same key path the full pytree has a subtree of different type\n'
                f'    {type(full_subtree)}.',
            )
            return  # don't look for more errors in this subtree

        # Or they may disagree if their roots have different numbers of children (note that because both
        # prefix_tree and full_tree have the same type at this point, and because prefix_tree is not a
        # leaf, each can be flattened once):
        prefix_tree_one_level_output = (
            prefix_tree_children,
            prefix_tree_metadata,
            prefix_tree_entries,
            _,
        ) = tree_flatten_one_level(
            prefix_subtree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        full_tree_one_level_output = (
            full_tree_children,
            full_tree_metadata,
            full_tree_entries,
            _,
        ) = tree_flatten_one_level(
            full_subtree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        # Special handling for dictionary types
        if both_standard_dict:
            prefix_tree_keys: list[Any] = (
                prefix_tree_metadata  # type: ignore[assignment]
                if prefix_tree_type is not defaultdict  # type: ignore[comparison-overlap]
                else prefix_tree_metadata[1]  # type: ignore[index]
            )
            full_tree_keys: list[Any] = (
                full_tree_metadata  # type: ignore[assignment]
                if full_tree_type is not defaultdict  # type: ignore[comparison-overlap]
                else full_tree_metadata[1]  # type: ignore[index]
            )
            prefix_tree_keys_set = set(prefix_tree_keys)
            full_tree_keys_set = set(full_tree_keys)
            if prefix_tree_keys_set != full_tree_keys_set:
                missing_keys = sorted(prefix_tree_keys_set.difference(full_tree_keys_set))
                extra_keys = sorted(full_tree_keys_set.difference(prefix_tree_keys_set))
                key_difference = ''
                if missing_keys:
                    key_difference += f'\nmissing key(s):\n    {missing_keys}'
                if extra_keys:
                    key_difference += f'\nextra key(s):\n    {extra_keys}'
                yield lambda name: ValueError(
                    f'pytree structure error: different pytree keys at key path\n'
                    f'    {accessor.codify(name) if accessor else name + " tree root"}\n'
                    f'At that key path, the prefix pytree {name} has a subtree of type\n'
                    f'    {prefix_tree_type}\n'
                    f'with {len(prefix_tree_keys)} key(s)\n'
                    f'    {prefix_tree_keys}\n'
                    f'but at the same key path the full pytree has a subtree of type\n'
                    f'    {full_tree_type}\n'
                    f'but with {len(full_tree_keys)} key(s)\n'
                    f'    {full_tree_keys}{key_difference}',
                )
                return  # don't look for more errors in this subtree

            # If the keys agree, we should ensure that the children are in the same order:
            full_tree_children = [full_subtree[k] for k in prefix_tree_keys]  # type: ignore[misc]

        if len(prefix_tree_children) != len(full_tree_children):
            yield lambda name: ValueError(
                f'pytree structure error: different numbers of pytree children at key path\n'
                f'    {accessor.codify(name) if accessor else name + " tree root"}\n'
                f'At that key path, the prefix pytree {name} has a subtree of type\n'
                f'    {prefix_tree_type}\n'
                f'with {len(prefix_tree_children)} children, '
                f'but at the same key path the full pytree has a subtree of the same '
                f'type but with {len(full_tree_children)} children.',
            )
            return  # don't look for more errors in this subtree

        # Or they may disagree if their roots have different pytree metadata:
        if (
            prefix_tree_metadata != full_tree_metadata
            and (not both_deque)  # ignore maxlen mismatch for deque
            and (
                # Special handling for dictionary types already done in the keys check above
                not both_standard_dict
            )
        ):
            prefix_tree_metadata_repr = repr(prefix_tree_metadata)
            full_tree_metadata_repr = repr(full_tree_metadata)
            metadata_diff = textwrap.indent(
                '\n'.join(
                    difflib.ndiff(
                        prefix_tree_metadata_repr.splitlines(),
                        full_tree_metadata_repr.splitlines(),
                    ),
                ),
                prefix='    ',
            )
            yield lambda name: ValueError(
                f'pytree structure error: different pytree metadata at key path\n'
                f'    {accessor.codify(name) if accessor else name + " tree root"}\n'
                f'At that key path, the prefix pytree {name} has a subtree of type\n'
                f'    {prefix_tree_type}\n'
                f'with metadata\n'
                f'    {prefix_tree_metadata_repr}\n'
                f'but at the same key path the full pytree has a subtree of the same '
                f'type but with metadata\n'
                f'    {full_tree_metadata_repr}\n'
                f'so the diff in the metadata at these pytree nodes is\n'
                f'{metadata_diff}',
            )
            return  # don't look for more errors in this subtree

        # If the root types and numbers of children agree, there must be an error in a subtree,
        # so recurse:
        entries = [
            prefix_tree_one_level_output.path_entry_type(
                e,
                prefix_tree_type,
                prefix_tree_one_level_output.kind,
            )
            for e in prefix_tree_entries
        ]
        entries_ = [
            full_tree_one_level_output.path_entry_type(
                e,
                full_tree_type,
                full_tree_one_level_output.kind,
            )
            for e in full_tree_entries
        ]
        assert (
            both_standard_dict  # special handling for dictionary types already done in the keys check above
            or entries == entries_
        ), f'equal pytree nodes gave different keys: {entries} and {entries_}'
        # pylint: disable-next=invalid-name
        for e, t1, t2 in zip(entries, prefix_tree_children, full_tree_children):
            yield from helper(accessor + e, t1, t2)

    return list(helper(PyTreeAccessor(), prefix_tree, full_tree))
