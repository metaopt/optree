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
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    overload,
)

import optree._optree as pytree
from optree.typing import PyTree, PyTreeDef


__all__ = [
    'all_leaves',
    'build_tree',
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

if TYPE_CHECKING:
    from optree.registry import KeyPath, KeyPathEntry


def treedef_children(treedef: PyTreeDef) -> List[PyTreeDef]:
    return treedef.children()


def treedef_is_leaf(treedef: PyTreeDef) -> bool:
    return treedef.num_nodes == 1


def treedef_is_strict_leaf(treedef: PyTreeDef) -> bool:
    return treedef.num_nodes == 1 and treedef.num_leaves == 1


def treedef_tuple(treedefs: Iterable[PyTreeDef]) -> PyTreeDef:
    """Makes a tuple treedef from a list of child treedefs."""
    return pytree.tuple(list(treedefs))


def tree_flatten(
    tree: PyTree, is_leaf: Optional[Callable[[Any], bool]] = None
) -> Tuple[List[Any], PyTreeDef]:
    """Flattens a pytree.

    The flattening order (i.e. the order of elements in the output list)
    is deterministic, corresponding to a left-to-right depth-first tree
    traversal.

    Args:
        tree: a pytree to flatten.
        is_leaf: an optionally specified function that will be called at each
            flattening step. It should return a boolean, with true stopping the
            traversal and the whole subtree being treated as a leaf, and false
            indicating the flattening should traverse the current object.
    Returns:
        A pair where the first element is a list of leaf values and the second
        element is a treedef representing the structure of the flattened tree.
    """
    return pytree.flatten(tree, is_leaf)


def tree_unflatten(treedef: PyTreeDef, leaves: Iterable[Any]) -> PyTree:
    """Reconstructs a pytree from the treedef and the leaves.

    The inverse of :func:`tree_flatten`.

    Args:
        treedef: the treedef to reconstruct
        leaves: the list of leaves to use for reconstruction. The list must match
            the leaves of the treedef.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure
        described by ``treedef``.
    """
    return treedef.unflatten(leaves)


def tree_leaves(tree: PyTree, is_leaf: Optional[Callable[[Any], bool]] = None) -> List[Any]:
    """Gets the leaves of a pytree."""
    return pytree.flatten(tree, is_leaf)[0]


def tree_structure(tree: PyTree, is_leaf: Optional[Callable[[Any], bool]] = None) -> PyTreeDef:
    """Gets the treedef for a pytree."""
    return pytree.flatten(tree, is_leaf)[1]


def all_leaves(iterable: Iterable[Any], is_leaf: Optional[Callable[[Any], bool]] = None) -> bool:
    """Tests whether all elements in the given iterable are all leaves.

    >>> tree = {"a": [1, 2, 3]}
    >>> assert all_leaves(optree.tree_leaves(tree))
    >>> assert not all_leaves([tree])

    This function is useful in advanced cases, for example if a library allows
    arbitrary map operations on a flat list of leaves it may want to check if
    the result is still a flat list of leaves.

    Args:
        iterable: Iterable of leaves.

    Returns:
        A boolean indicating if all elements in the input are leaves.
    """
    if is_leaf is None:
        return pytree.all_leaves(iterable)

    lst = list(iterable)
    return lst == tree_leaves(lst, is_leaf)


def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rest: PyTree,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree:
    """Maps a multi-input function over pytree args to produce a new pytree.

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
    leaves, treedef = tree_flatten(tree, is_leaf)
    # pylint: disable-next=redefined-outer-name
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(func(*xs) for xs in zip(*all_leaves))


def build_tree(treedef: PyTreeDef, xs):
    return treedef.from_iterable_tree(xs)


def tree_transpose(
    outer_treedef: PyTreeDef, inner_treedef: PyTreeDef, pytree_to_transpose: PyTree
) -> PyTree:
    """Transform a tree having tree structure (outer, inner) into one having
    structure (inner, outer).
    """
    flat, treedef = tree_flatten(pytree_to_transpose)
    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves
    if treedef.num_leaves != (inner_size * outer_size):
        expected_treedef = outer_treedef.compose(inner_treedef)
        raise TypeError(f'Mismatch\n{treedef}\n != \n{expected_treedef}')
    iter_flat = iter(flat)
    lol = [
        [next(iter_flat) for _ in range(inner_size)]
        for __ in range(outer_size)
    ]  # fmt: skip
    transposed_lol = zip(*lol)
    subtrees = map(partial(tree_unflatten, outer_treedef), transposed_lol)
    return tree_unflatten(inner_treedef, subtrees)


def _replace_nones(sentinel: Any, tree: Optional[PyTree]) -> PyTree:
    """Replaces ``None`` in ``tree`` with ``sentinel``."""
    if tree is None:
        return sentinel

    from optree.registry import register_pytree_node  # pylint: disable=import-outside-toplevel

    handler = register_pytree_node.get(type(tree))  # type: ignore[attr-defined]
    if handler:
        children, metadata = handler.to_iter(tree)
        proc_children: List[PyTree] = [_replace_nones(sentinel, child) for child in children]
        return handler.from_iter(metadata, proc_children)
    if isinstance(tree, tuple) and hasattr(tree, '_fields'):
        # handle namedtuple as a special case, based on heuristic
        children = iter(tree)
        proc_children = [_replace_nones(sentinel, child) for child in children]
        return type(tree)(*proc_children)  # type: ignore[arg-type,return-value]

    return tree


T = TypeVar('T')
__NO_INITIALIZER = object()


@overload
def tree_reduce(func: Callable[[T, T], T], tree: PyTree) -> T:
    ...


@overload
def tree_reduce(func: Callable[[T, T], T], tree: PyTree, initializer: T) -> T:
    ...


def tree_reduce(func: Callable[[T, T], T], tree: PyTree, initializer: T = __NO_INITIALIZER) -> T:
    if initializer is __NO_INITIALIZER:
        return functools.reduce(func, tree_leaves(tree))

    return functools.reduce(func, tree_leaves(tree), initializer)


def tree_all(tree: PyTree) -> bool:
    return all(tree_leaves(tree))


def tree_any(tree: PyTree) -> bool:
    return any(tree_leaves(tree))


def broadcast_prefix(
    prefix_tree: PyTree, full_tree: PyTree, is_leaf: Optional[Callable[[Any], bool]] = None
) -> List[Any]:
    # If prefix_tree is not a tree prefix of full_tree, this code can raise a
    # ValueError; use prefix_errors to find disagreements and raise more precise
    # error messages.
    result = []

    def num_leaves(tree):
        return tree_structure(tree).num_leaves

    def add_leaves(x, subtree):
        return result.extend([x] * num_leaves(subtree))

    tree_map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
    return result


def flatten_one_level(tree: PyTree) -> Tuple[List[Any], Hashable]:
    from optree.registry import register_pytree_node  # pylint: disable=import-outside-toplevel

    handler = register_pytree_node.get(type(tree))  # type: ignore[attr-defined]
    if handler:
        children, meta = handler.to_iter(tree)
        return list(children), meta
    if isinstance(tree, tuple) and hasattr(tree, '_fields'):
        return list(tree), None
    raise ValueError(f"can't tree-flatten type: {type(tree)}")


def prefix_errors(
    prefix_tree: PyTree,
    full_tree: PyTree,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> List[Callable[[str], ValueError]]:
    from optree.registry import KeyPath  # pylint: disable=import-outside-toplevel

    return list(_prefix_error(KeyPath(), prefix_tree, full_tree, is_leaf))


# pylint: disable-next=too-many-locals
def _prefix_error(
    key_path: 'KeyPath',
    prefix_tree: PyTree,
    full_tree: PyTree,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Iterable[Callable[[str], ValueError]]:
    # A leaf is a valid prefix of any tree:
    if treedef_is_strict_leaf(tree_structure(prefix_tree, is_leaf=is_leaf)):
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
        yield from _prefix_error(key_path + k, t1, t2)


def _child_keys(tree: PyTree) -> List['KeyPathEntry']:
    # pylint: disable-next=import-outside-toplevel
    from optree.registry import AttributeKeyPathEntry, FlattenedKeyPathEntry, register_keypaths

    assert not treedef_is_strict_leaf(tree_structure(tree))

    handler = register_keypaths.get(type(tree))  # type: ignore[attr-defined]
    if handler:
        return handler(tree)

    if isinstance(tree, tuple) and hasattr(tree, '_fields'):
        # handle namedtuple as a special case, based on heuristic
        return [AttributeKeyPathEntry(s) for s in tree._fields]

    num_children = len(treedef_children(tree_structure(tree)))
    return [FlattenedKeyPathEntry(i) for i in range(num_children)]
