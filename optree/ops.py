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

import difflib
import functools
import textwrap
from typing import Any, Callable, Optional, cast, overload

import optree._C as _C
from optree.registry import (
    AttributeKeyPathEntry,
    FlattenedKeyPathEntry,
    KeyPath,
    KeyPathEntry,
    register_keypaths,
    register_pytree_node,
)
from optree.typing import (
    AuxData,
    Children,
    Iterable,
    List,
    NamedTuple,
    PyTree,
    PyTreeSpec,
    S,
    T,
    Tuple,
    U,
    is_namedtuple,
)


__all__ = [
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
]


NONE_IS_NODE: bool = False  # literal constant
NONE_IS_LEAF: bool = True  # literal constant


def tree_flatten(
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> Tuple[List[T], PyTreeSpec]:
    """Flattens a pytree.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten(tree)
    ([1, 2, 3, 4, 5], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *}))
    >>> tree_flatten(tree, none_is_leaf=True)
    ([1, 2, 3, 4, None, 5], PyTreeSpec(NoneIsLeaf, {'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}))
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*))
    >>> tree_flatten(None)
    ([], PyTreeSpec(None))
    >>> tree_flatten(None, none_is_leaf=True)
    ([None], PyTreeSpec(NoneIsLeaf, *))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    ([2, 3, 4, 1, 5], PyTreeSpec(OrderedDict([('b', (*, [*, *])), ('a', *), ('c', None), ('d', *)])))

    Args:
        tree: A pytree to flatten.
        is_leaf: An optionally specified function that will be called at each flattening step. It
            should return a boolean, with :data:`True` stopping the traversal and the whole subtree
            being treated as a leaf, and :data:`False` indicating the flattening should traverse the
            current object.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`, :data:`None` is a
            non-leaf node with arity 0. Thus :data:`None` is contained in the treespec rather than
            in the leaves list. (default: :data:`False`)

    Returns:
        A pair where the first element is a list of leaf values and the second element is a treespec
        representing the structure of the pytree.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf)


def tree_unflatten(treespec: PyTreeSpec, leaves: Iterable[T]) -> PyTree[T]:
    """Reconstructs a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    >>> leaves, treespec = tree_flatten(tree)
    >>> tree == tree_unflatten(treespec, leaves)
    True

    Args:
        treespec: The treespec to reconstruct.
        leaves: The list of leaves to use for reconstruction. The list must match the number of
            leaves of the treespec.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by
        ``treespec``.
    """
    return treespec.unflatten(leaves)


def tree_leaves(
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> List[T]:
    """Gets the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_leaves(tree)
    [1, 2, 3, 4, 5]
    >>> tree_leaves(tree)
    [1, 2, 3, 4, None, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    []
    >>> tree_leaves(None, none_is_leaf=True)
    [None]

    Args:
        tree: A pytree to flatten.
        is_leaf: An optionally specified function that will be called at each flattening step. It
            should return a boolean, with :data:`True` stopping the traversal and the whole subtree
            being treated as a leaf, and :data:`False` indicating the flattening should traverse the
            current object.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`, :data:`None` is a
            non-leaf node with arity 0. Thus :data:`None` is contained in the treespec rather than
            in the leaves list. (default: :data:`False`)

    Returns:
        A list of leaf values.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf)[0]


def tree_structure(
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> PyTreeSpec:
    """Gets the treespec for a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_structure(tree)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    >>> tree_structure(tree, none_is_leaf=True)
    PyTreeSpec(NoneIsLeaf, {'a': *, 'b': (*, [*, *]), 'c': *, 'd': *})
    >>> tree_structure(1)
    PyTreeSpec(*)
    >>> tree_structure(None)
    PyTreeSpec(None)
    >>> tree_structure(None, none_is_leaf=True)
    PyTreeSpec(NoneIsLeaf, *)

    Args:
        tree: A pytree to flatten.
        is_leaf: An optionally specified function that will be called at each flattening step. It
            should return a boolean, with :data:`True` stopping the traversal and the whole subtree
            being treated as a leaf, and :data:`False` indicating the flattening should traverse the
            current object.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`, :data:`None` is a
            non-leaf node with arity 0. Thus :data:`None` is contained in the treespec rather than
            in the leaves list. (default: :data:`False`)

    Returns:
        A treespec object representing the structure of the pytree.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf)[1]


def all_leaves(
    iterable: Iterable[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> bool:
    """Tests whether all elements in the given iterable are all leaves.

    See also :func:`tree_flatten` and :func:`tree_leaves`.

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
        iterable: A iterable of leaves.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`, :data:`None` is a
            non-leaf node with arity 0. Thus :data:`None` is contained in the treespec rather than
            a leaf. (default: :data:`False`)

    Returns:
        A boolean indicating if all elements in the input iterable are leaves.
    """
    if is_leaf is None:
        return _C.all_leaves(iterable, none_is_leaf)

    nodes = list(iterable)
    if all(map(is_leaf, nodes)):
        return True
    return nodes == tree_leaves(nodes, is_leaf, none_is_leaf=none_is_leaf)  # type: ignore[arg-type]


def tree_map(
    func: Callable[..., U],
    tree: PyTree[T],
    *rests: PyTree[S],
    is_leaf: Optional[Callable[[T], bool]] = None,
    none_is_leaf: bool = False,
) -> PyTree[U]:
    """Maps a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_flatten`, :func:`tree_leaves`, and :func:`tree_unflatten`.

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
        func: A function that takes ``1 + len(rests)`` arguments, to be applied at the corresponding
            leaves of the pytrees.
        tree: A pytree to be mapped over, with each leaf providing the first positional argument to
            function ``func``.
        rests: A tuple of pytrees, each of which has the same structure as ``tree`` or has ``tree``
            as a prefix.
        is_leaf: An optionally specified function that will be called at each flattening step. It
            should return a boolean, with :data:`True` stopping the traversal and the whole subtree
            being treated as a leaf, and :data:`False` indicating the flattening should traverse the
            current object.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`, :data:`None` is a
            non-leaf node with arity 0. Thus :data:`None` is contained in the treespec rather than
            in the leaves list and :data:`None` will be remain in the result pytree. (default:
            :data:`False`)

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, treespec = tree_flatten(tree, is_leaf, none_is_leaf=none_is_leaf)
    arglists = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    results = map(func, *arglists)
    return treespec.unflatten(results)


__INITIAL_MISSING: T = object()  # type: ignore[valid-type]


@overload
def tree_reduce(
    func: Callable[[T, T], T],
    tree: PyTree[T],
    *,
    is_leaf: Optional[Callable[[T], bool]] = None,
    none_is_leaf: bool = False,
) -> T:  # pragma: no cover
    ...


@overload
def tree_reduce(
    func: Callable[[T, T], T],
    tree: PyTree[T],
    initial: T = __INITIAL_MISSING,
    *,
    is_leaf: Optional[Callable[[T], bool]] = None,
    none_is_leaf: bool = False,
) -> T:  # pragma: no cover
    ...


def tree_reduce(  # type: ignore[misc]
    func: Callable[[T, T], T],
    tree: PyTree[T],
    initial: T = __INITIAL_MISSING,
    *,
    is_leaf: Optional[Callable[[T], bool]] = None,
    none_is_leaf: bool = False,
) -> T:
    """Traversals through a pytree and reduces the leaves.

    See also :func:`tree_leaves`.

    >>> tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, 3)})
    6
    >>> tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, None), 'z': 3})
    6
    >>> tree_reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3})
    3
    >>> tree_reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    None

    Args:
        func: A function that takes two arguments and returns a value of the same type.
        tree: A pytree to be traversed.
        initial: An initial value to be used for the reduction. If not provided, the first leaf
            value is used as the initial value.
        is_leaf: An optionally specified function that will be called at each flattening step. It
            should return a boolean, with :data:`True` stopping the traversal and the whole subtree
            being treated as a leaf, and :data:`False` indicating the flattening should traverse the
            current object.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`, :data:`None` is a
            non-leaf node with arity 0. Thus :data:`None` is contained in the treespec rather than
            in the leaves list. (default: :data:`False`)

    Returns:
        The result of reducing the leaves of the pytree using ``func``.
    """
    if initial is __INITIAL_MISSING:
        return functools.reduce(func, tree_leaves(tree, is_leaf, none_is_leaf=none_is_leaf))

    return functools.reduce(func, tree_leaves(tree, is_leaf, none_is_leaf=none_is_leaf), initial)


def tree_transpose(
    outer_treespec: PyTreeSpec,
    inner_treespec: PyTreeSpec,
    tree: PyTree[T],
) -> PyTree[PyTree[T]]:
    """Transforms a tree having tree structure (outer, inner) into one having structure (inner, outer).

    See also :func:`tree_flatten` and :func:`tree_unflatten`.

    >>> outer_treespec = tree_structure({'a': 1, 'b': 2, 'c': (3, 4)})
    >>> outer_treespec
    PyTreeSpec({'a': *, 'b': *, 'c': (*, *)})
    >>> inner_treespec = tree_structure((1, 2))
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
        outer_treespec: A treespec object representing the outer structure of the pytree.
        inner_treespec: A treespec object representing the inner structure of the pytree.
        tree: A pytree to be transposed.

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
    leaves, treespec = tree_flatten(tree, none_is_leaf=outer_treespec.none_is_leaf)
    if treespec.num_leaves != inner_size * outer_size:
        expected_treespec = outer_treespec.compose(inner_treespec)
        raise TypeError(f'Tree structure mismatch:\n{treespec}\n != \n{expected_treespec}')
    iter_leaves = iter(leaves)
    grouped = [
        [next(iter_leaves) for _ in range(inner_size)]
        for __ in range(outer_size)
    ]  # fmt: skip
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)  # type: ignore[arg-type]


def tree_replace_nones(sentinel: Any, tree: Optional[PyTree[T]]) -> PyTree[T]:
    """Replaces :data:`None` in ``tree`` with ``sentinel``.

    See also :func:`tree_flatten` and :func:`tree_map`.

    >>> tree_replace_nones(0, {'a': 1, 'b': None, 'c': (2, None)})
    {'a': 1, 'b': 0, 'c': (2, 0)}
    >>> tree_replace_nones(0, None)
    0
    """
    if tree is None:
        return sentinel
    return tree_map(lambda x: x if x is not None else sentinel, tree, none_is_leaf=True)


def tree_all(
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> bool:
    """Tests whether all leaves in ``tree`` are true (or if ``tree`` is empty).

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
        tree: A pytree to be traversed.
        is_leaf: An optionally specified function that will be called at each flattening step. It
            should return a boolean, with :data:`True` stopping the traversal and the whole subtree
            being treated as a leaf, and :data:`False` indicating the flattening should traverse the
            current object.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`, :data:`None` is a
            non-leaf node with arity 0. Thus :data:`None` is contained in the treespec rather than
            in the leaves list. (default: :data:`False`)

    Returns:
        :data:`True` if all leaves in ``tree`` are true, or if ``tree`` is empty.
        Otherwise, :data:`False`.
    """
    return all(tree_leaves(tree, is_leaf, none_is_leaf=none_is_leaf))  # type: ignore[arg-type]


def tree_any(
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> bool:
    """Tests whether all leaves in ``tree`` are true (or :data:`False` if ``tree`` is empty).

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
        tree: A pytree to be traversed.
        is_leaf: An optionally specified function that will be called at each flattening step. It
            should return a boolean, with :data:`True` stopping the traversal and the whole subtree
            being treated as a leaf, and :data:`False` indicating the flattening should traverse the
            current object.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`, :data:`None` is a
            non-leaf node with arity 0. Thus :data:`None` is contained in the treespec rather than
            in the leaves list. (default: :data:`False`)

    Returns:
        :data:`True` if any leaves in ``tree`` are true, otherwise, :data:`False`. If ``tree`` is
        empty, return :data:`False`.
    """
    return any(tree_leaves(tree, is_leaf, none_is_leaf=none_is_leaf))  # type: ignore[arg-type]


def treespec_children(treespec: PyTreeSpec) -> List[PyTreeSpec]:
    """Returns a list of treespecs for the children of a treespec."""
    return treespec.children()


def treespec_is_leaf(treespec: PyTreeSpec) -> bool:
    """Returns whether the treespec is a leaf.

    This function does not check whether the treespec set ``none_is_leaf``.

    >>> treespec_is_leaf(tree_structure(1))
    True
    >>> treespec_is_leaf(tree_structure((1, 2)))
    False
    >>> treespec_is_leaf(tree_structure(None))
    True
    >>> treespec_is_leaf(tree_structure(None, none_is_leaf=True))
    True
    """
    return treespec.num_nodes == 1


def treespec_is_strict_leaf(treespec: PyTreeSpec) -> bool:
    """Returns whether the treespec is a strict leaf.

    >>> treespec_is_strict_leaf(tree_structure(1))
    True
    >>> treespec_is_strict_leaf(tree_structure((1, 2)))
    False
    >>> treespec_is_strict_leaf(tree_structure(None))
    False
    >>> treespec_is_strict_leaf(tree_structure(None, none_is_leaf=True))
    True
    """
    return treespec.num_nodes == 1 and treespec.num_leaves == 1


def treespec_leaf(*, none_is_leaf: bool = False) -> PyTreeSpec:
    """Makes a treespec representing a leaf node.

    See also :func:`tree_structure`, :func:`treespec_none`, and `func`:`treespec_tuple`.

    >>> treespec_leaf()
    PyTreeSpec(*)
    >>> treespec_leaf(none_is_leaf=True)
    PyTreeSpec(NoneIsLeaf, *)
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
    """
    return _C.leaf(none_is_leaf)


def treespec_none(*, none_is_leaf: bool = False) -> PyTreeSpec:
    """Makes a treespec representing a :data:`None` node.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and `func`:`treespec_tuple`.

    >>> treespec_none()
    PyTreeSpec(None)
    >>> treespec_none(none_is_leaf=True)
    PyTreeSpec(NoneIsLeaf, *)
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
    """
    return _C.none(none_is_leaf)


def treespec_tuple(
    treespecs: Iterable[PyTreeSpec] = (), *, none_is_leaf: bool = False
) -> PyTreeSpec:
    """Makes a tuple treespec from a list of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and `func`:`treespec_none`.

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
    """
    return _C.tuple(list(treespecs), none_is_leaf)


def broadcast_prefix(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> List[T]:
    """Returns a list of broadcasted leaves in ``prefix_tree`` to match the number of leaves in ``full_tree``."""
    # If prefix_tree is not a tree prefix of full_tree, this code can raise a ValueError;
    # use prefix_errors to find disagreements and raise more precise error messages.
    result: List[T] = []

    def num_leaves(tree: PyTree[U]) -> int:
        return tree_structure(tree, none_is_leaf=none_is_leaf).num_leaves

    def add_leaves(x: T, subtree: PyTree[S]) -> None:
        result.extend([x] * num_leaves(subtree))

    tree_map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf)
    return result


def flatten_one_level(
    tree: PyTree[T], *, none_is_leaf: bool = False
) -> Tuple[Children[T], AuxData]:
    """Flattens the pytree one level, returning a tuple of children and auxiliary data."""
    if tree is None:
        if none_is_leaf:  # type: ignore[unreachable]
            raise ValueError('Cannot flatten None')
        return (), None

    node_type = type(tree)
    handler = register_pytree_node.get(node_type)  # type: ignore[attr-defined]
    if handler:
        children, aux_data = handler.to_iter(tree)
        return list(children), aux_data

    if is_namedtuple(tree):
        return list(cast(NamedTuple, tree)), node_type

    raise ValueError(f'Cannot tree-flatten type: {node_type}.')


def prefix_errors(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> List[Callable[[str], ValueError]]:
    """Returns a list of errors that would be raised by :func:`broadcast_prefix`."""
    return list(
        _prefix_error(
            KeyPath(),
            prefix_tree,
            full_tree,
            is_leaf,
            none_is_leaf=none_is_leaf,
        )
    )


# pylint: disable-next=too-many-locals
def _prefix_error(
    key_path: KeyPath,
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> Iterable[Callable[[str], ValueError]]:
    # A leaf is a valid prefix of any tree
    if treespec_is_strict_leaf(
        tree_structure(prefix_tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf)
    ):
        return

    # The subtrees may disagree because their roots are of different types:
    if type(prefix_tree) is not type(full_tree):
        yield lambda name: ValueError(
            f'pytree structure error: different types at key path\n'
            f'    {{name}}{key_path.pprint()}\n'
            f'At that key path, the prefix pytree {{name}} has a subtree of type\n'
            f'    {type(prefix_tree)}\n'
            f'but at the same key path the full pytree has a subtree of different type\n'
            f'    {type(full_tree)}.'.format(name=name)
        )
        return  # don't look for more errors in this subtree

    # Or they may disagree if their roots have different numbers of children (note that because both
    # prefix_tree and full_tree have the same type at this point, and because prefix_tree is not a
    # leaf, each can be flattened once):
    prefix_tree_children, prefix_tree_meta = flatten_one_level(prefix_tree)
    full_tree_children, full_tree_meta = flatten_one_level(full_tree)
    if len(prefix_tree_children) != len(full_tree_children):
        yield lambda name: ValueError(
            f'pytree structure error: different numbers of pytree children at key path\n'
            f'    {{name}}{key_path.pprint()}\n'
            f'At that key path, the prefix pytree {{name}} has a subtree of type\n'
            f'    {type(prefix_tree)}\n'
            f'with {len(prefix_tree_children)} children, '
            f'but at the same key path the full pytree has a subtree of the same '
            f'type but with {len(full_tree_children)} children.'.format(name=name)
        )
        return  # don't look for more errors in this subtree

    # Or they may disagree if their roots have different pytree metadata:
    if prefix_tree_meta != full_tree_meta:
        prefix_tree_meta_str = str(prefix_tree_meta)
        full_tree_meta_str = str(full_tree_meta)
        metadata_diff = textwrap.indent(
            '\n'.join(
                difflib.ndiff(prefix_tree_meta_str.splitlines(), full_tree_meta_str.splitlines())
            ),
            prefix='    ',
        )
        yield lambda name: ValueError(
            f'pytree structure error: different pytree metadata at key path\n'
            f'    {{name}}{key_path.pprint()}\n'
            f'At that key path, the prefix pytree {{name}} has a subtree of type\n'
            f'    {type(prefix_tree)}\n'
            f'with metadata\n'
            f'    {prefix_tree_meta_str}\n'
            f'but at the same key path the full pytree has a subtree of the same '
            f'type but with metadata\n'
            f'    {full_tree_meta_str}\n'
            f'so the diff in the metadata at these pytree nodes is\n'
            f'{metadata_diff}'.format(name=name)
        )
        return  # don't look for more errors in this subtree

    # If the root types and numbers of children agree, there must be an error in a subtree,
    # so recurse:
    keys = _child_keys(prefix_tree)
    keys_ = _child_keys(full_tree)
    assert keys == keys_, f'equal pytree nodes gave differing keys: {keys} and {keys_}'
    # pylint: disable-next=invalid-name
    for k, t1, t2 in zip(keys, prefix_tree_children, full_tree_children):
        yield from _prefix_error(key_path + k, cast(PyTree[T], t1), cast(PyTree[S], t2))


def _child_keys(tree: PyTree[T], *, none_is_leaf: bool = False) -> List[KeyPathEntry]:
    assert not treespec_is_strict_leaf(tree_structure(tree, none_is_leaf=none_is_leaf))

    handler = register_keypaths.get(type(tree))  # type: ignore[attr-defined]
    if handler:
        return handler(tree)

    if is_namedtuple(tree):
        # handle namedtuple as a special case, based on heuristic
        return list(map(AttributeKeyPathEntry, cast(NamedTuple, tree)._fields))

    num_children = len(treespec_children(tree_structure(tree, none_is_leaf=none_is_leaf)))
    return list(map(FlattenedKeyPathEntry, range(num_children)))
