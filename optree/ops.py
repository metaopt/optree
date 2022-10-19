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

# pylint: disable=missing-class-docstring,missing-function-docstring

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
    'all_leaves',
    'tree_all',
    'tree_any',
    'tree_flatten',
    'tree_leaves',
    'tree_map',
    'tree_reduce',
    'tree_structure',
    'tree_transpose',
    'tree_unflatten',
    'treespec_children',
    'treespec_is_leaf',
    'treespec_is_strict_leaf',
    'treespec_tuple',
]


NONE_IS_NODE: bool = False  # literal constant
NONE_IS_LEAF: bool = True  # literal constant


def treespec_children(treespec: PyTreeSpec) -> List[PyTreeSpec]:
    """Return a list of treespecs for the children of a treespec."""
    return treespec.children()


def treespec_is_leaf(treespec: PyTreeSpec) -> bool:
    """Return whether the treespec is a leaf."""
    return treespec.num_nodes == 1


def treespec_is_strict_leaf(treespec: PyTreeSpec) -> bool:
    """Return whether the treespec is a strict leaf."""
    return treespec.num_nodes == 1 and treespec.num_leaves == 1


def treespec_tuple(treespecs: Iterable[PyTreeSpec], *, none_is_leaf: bool = False) -> PyTreeSpec:
    """Make a tuple treespec from a list of child treespecs."""
    return _C.tuple(list(treespecs), none_is_leaf)


def tree_flatten(
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> Tuple[List[T], PyTreeSpec]:
    """Flatten a pytree.

    The flattening order (i.e. the order of elements in the output list)
    is deterministic, corresponding to a left-to-right depth-first tree
    traversal.

    Args:
        tree: a pytree to flatten.
        is_leaf: an optionally specified function that will be called at each
            flattening step. It should return a boolean, with true stopping the
            traversal and the whole subtree being treated as a leaf, and false
            indicating the flattening should traverse the current object.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a node with arity 0. (default: :data:`False`)

    Returns:
        A pair where the first element is a list of leaf values and the second
        element is a treespec representing the structure of the flattened tree.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf)


def tree_unflatten(treespec: PyTreeSpec, leaves: Iterable[T]) -> PyTree[T]:
    """Reconstructs a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    Args:
        treespec: the treespec to reconstruct
        leaves: the list of leaves to use for reconstruction. The list must match
            the leaves of the treespec.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure
        described by ``treespec``.
    """
    return treespec.unflatten(leaves)


def tree_leaves(
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> List[T]:
    """Get the leaves of a pytree."""
    return _C.flatten(tree, is_leaf, none_is_leaf)[0]


def tree_structure(
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> PyTreeSpec:
    """Get the treespec for a pytree."""
    return _C.flatten(tree, is_leaf, none_is_leaf)[1]


def all_leaves(
    iterable: Iterable[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    *,
    none_is_leaf: bool = False,
) -> bool:
    """Test whether all elements in the given iterable are all leaves.

    >>> tree = {"a": [1, 2, 3]}
    >>> assert all_leaves(optree.tree_leaves(tree))
    >>> assert not all_leaves([tree])

    This function is useful in advanced cases, for example if a library allows
    arbitrary map operations on a flat list of leaves it may want to check if
    the result is still a flat list of leaves.

    Args:
        iterable: Iterable of leaves.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a node with arity 0. (default: :data:`False`)

    Returns:
        A boolean indicating if all elements in the input are leaves.
    """
    if is_leaf is None:
        return _C.all_leaves(iterable, none_is_leaf)

    nodes = list(iterable)
    return nodes == tree_leaves(nodes, is_leaf, none_is_leaf=none_is_leaf)  # type: ignore[arg-type]


def tree_map(
    func: Callable[..., U],
    tree: PyTree[T],
    *rest: PyTree[S],
    is_leaf: Optional[Callable[[T], bool]] = None,
    none_is_leaf: bool = False,
) -> PyTree[U]:
    """Map a multi-input function over pytree args to produce a new pytree.

    Args:
        f: function that takes ``1 + len(rest)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree: a pytree to be mapped over, with each leaf providing the first
            positional argument to ``f``.
        rest: a tuple of pytrees, each of which has the same structure as ``tree``
            or has ``tree`` as a prefix.
        is_leaf: an optionally specified function that will be called at each
            flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        none_is_leaf: Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a node with arity 0. (default: :data:`False`)

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each
        leaf given by ``f(x, *xs)`` where ``x`` is the value at the corresponding
        leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes
        in ``rest``.

    Examples:
        >>> import optree
        >>> optree.tree_map(lambda x: x + 1, {"x": 7, "y": 42})
        {'x': 8, 'y': 43}

        If multiple inputs are passed, the structure of the tree is taken from
        the first input; subsequent inputs need only have ``tree`` as a prefix:

        >>> optree.tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
        [[5, 7, 9], [6, 1, 2]]
    """
    leaves, treespec = tree_flatten(tree, is_leaf, none_is_leaf=none_is_leaf)
    arglists = [leaves] + [treespec.flatten_up_to(r) for r in rest]
    results = map(func, *arglists)
    return treespec.unflatten(results)


def tree_transpose(
    outer_treespec: PyTreeSpec,
    inner_treespec: PyTreeSpec,
    tree: PyTree[T],
) -> PyTree[PyTree[T]]:
    """Transform a tree having tree structure (outer, inner) into one having structure (inner, outer)."""
    if outer_treespec.none_is_leaf != inner_treespec.none_is_leaf:
        raise ValueError('Tree structures must have the same none_is_leaf value.')
    leaves, treespec = tree_flatten(tree, none_is_leaf=outer_treespec.none_is_leaf)
    inner_size = inner_treespec.num_leaves
    outer_size = outer_treespec.num_leaves
    if treespec.num_leaves != inner_size * outer_size:
        expected_treespec = outer_treespec.compose(inner_treespec)
        raise TypeError(f'Tree structure mismatch:\n{treespec}\n != \n{expected_treespec}')
    iter_leaves = iter(leaves)
    grouped = [
        [next(iter_leaves) for _ in range(inner_size)]
        for __ in range(outer_size)
    ]  # fmt: skip
    transposed = zip(*grouped)
    subtrees = map(functools.partial(tree_unflatten, outer_treespec), transposed)
    return tree_unflatten(inner_treespec, cast(Iterable[PyTree[T]], subtrees))


def _replace_nones(sentinel: Any, tree: Optional[PyTree[T]]) -> PyTree[T]:
    """Replace :data:`None` in ``tree`` with ``sentinel``."""
    if tree is None:
        return sentinel

    handler = register_pytree_node.get(type(tree))  # type: ignore[attr-defined]
    if handler:
        children, metadata = handler.to_iter(tree)
        proc_children: List[PyTree] = [_replace_nones(sentinel, child) for child in children]
        return handler.from_iter(metadata, proc_children)

    if is_namedtuple(tree):
        # handle namedtuple as a special case, based on heuristic
        proc_children = [_replace_nones(sentinel, child) for child in cast(NamedTuple, tree)]
        return type(tree)(*proc_children)

    return tree


__INITIAL_MISSING: T = object()  # type: ignore[valid-type]


@overload
def tree_reduce(
    func: Callable[[T, T], T],
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
) -> T:
    ...


@overload
def tree_reduce(
    func: Callable[[T, T], T],
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    initial: T = __INITIAL_MISSING,
) -> T:
    ...


def tree_reduce(
    func: Callable[[T, T], T],
    tree: PyTree[T],
    is_leaf: Optional[Callable[[T], bool]] = None,
    initial: T = __INITIAL_MISSING,
) -> T:
    """Traversals through a pytree and reduces the leaves."""
    if initial is __INITIAL_MISSING:
        return functools.reduce(func, tree_leaves(tree, is_leaf))

    return functools.reduce(func, tree_leaves(tree, is_leaf), initial)


def tree_all(tree: PyTree[T], is_leaf: Optional[Callable[[T], bool]] = None) -> bool:
    """Test whether all leaves in the tree are true."""
    return all(tree_leaves(tree, is_leaf))  # type: ignore[arg-type]


def tree_any(tree: PyTree[T], is_leaf: Optional[Callable[[T], bool]] = None) -> bool:
    """Test whether any leaves in the tree are true."""
    return any(tree_leaves(tree, is_leaf))  # type: ignore[arg-type]


def broadcast_prefix(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Optional[Callable[[T], bool]] = None,
) -> List[T]:
    """Return a list of broadcasted leaves in ``prefix_tree`` to match the number of leaves in ``full_tree``."""
    # If prefix_tree is not a tree prefix of full_tree, this code can raise a
    # ValueError; use prefix_errors to find disagreements and raise more precise
    # error messages.
    result: List[T] = []

    def num_leaves(tree: PyTree[U]) -> int:
        return tree_structure(tree).num_leaves

    def add_leaves(x: T, subtree: PyTree[S]) -> None:
        result.extend([x] * num_leaves(subtree))

    tree_map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
    return result


def flatten_one_level(tree: PyTree[T]) -> Tuple[Children[T], AuxData]:
    """Flatten the pytree one level, returning a tuple of children and auxiliary data."""
    handler = register_pytree_node.get(type(tree))  # type: ignore[attr-defined]
    if handler:
        children, meta = handler.to_iter(tree)
        return list(children), meta

    if is_namedtuple(tree):
        return list(cast(NamedTuple, tree)), None

    raise ValueError(f"Can't tree-flatten type: {type(tree)}.")


def prefix_errors(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Optional[Callable[[T], bool]] = None,
) -> List[Callable[[str], ValueError]]:
    """Return a list of errors that would be raised by :func:`broadcast_prefix`."""
    return list(_prefix_error(KeyPath(), prefix_tree, full_tree, is_leaf))


# pylint: disable-next=too-many-locals
def _prefix_error(
    key_path: KeyPath,
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Optional[Callable[[T], bool]] = None,
) -> Iterable[Callable[[str], ValueError]]:
    # A leaf is a valid prefix of any tree:
    if treespec_is_strict_leaf(tree_structure(prefix_tree, is_leaf=is_leaf)):
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

    # Or they may disagree if their roots have different numbers of children (note
    # that because both prefix_tree and full_tree have the same type at this
    # point, and because prefix_tree is not a leaf, each can be flattened once):
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

    # If the root types and numbers of children agree, there must be an error
    # in a subtree, so recurse:
    keys = _child_keys(prefix_tree)
    keys_ = _child_keys(full_tree)
    assert keys == keys_, f'equal pytree nodes gave differing keys: {keys} and {keys_}'
    # pylint: disable-next=invalid-name
    for k, t1, t2 in zip(keys, prefix_tree_children, full_tree_children):
        yield from _prefix_error(key_path + k, cast(PyTree[T], t1), cast(PyTree[S], t2))


def _child_keys(tree: PyTree[T]) -> List[KeyPathEntry]:
    # pylint: disable-next=import-outside-toplevel

    assert not treespec_is_strict_leaf(tree_structure(tree))

    handler = register_keypaths.get(type(tree))  # type: ignore[attr-defined]
    if handler:
        return handler(tree)

    if is_namedtuple(tree):
        # handle namedtuple as a special case, based on heuristic
        return list(map(AttributeKeyPathEntry, cast(NamedTuple, tree)._fields))

    num_children = len(treespec_children(tree_structure(tree)))
    return list(map(FlattenedKeyPathEntry, range(num_children)))
