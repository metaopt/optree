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

# pylint: disable=missing-function-docstring,invalid-name

import functools
import itertools
from collections import OrderedDict

import pytest

import optree

# pylint: disable-next=wrong-import-order
from helpers import LEAVES, TREE_STRINGS, TREES, CustomTuple, FlatCache, parametrize


def dummy_func(*args, **kwargs):  # pylint: disable=unused-argument
    return


def is_tuple(t):
    return isinstance(t, tuple)


def is_list(l):
    return isinstance(l, list)


def is_none(n):
    return n is None


def always(o):  # pylint: disable=unused-argument
    return True


def never(o):  # pylint: disable=unused-argument
    return False


@parametrize(tree=list(TREES + LEAVES), none_is_leaf=[False, True])
def test_round_trip(tree, none_is_leaf):
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)
    actual = optree.tree_unflatten(treespec, leaves)
    assert actual == tree


@parametrize(tree=list(TREES + LEAVES), none_is_leaf=[False, True])
def test_round_trip_with_flatten_up_to(tree, none_is_leaf):
    _, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)
    leaves = treespec.flatten_up_to(tree)
    actual = optree.tree_unflatten(treespec, leaves)
    assert actual == tree


@parametrize(
    tree=[
        [0, ((1, 2), 3, (4, (5, 6, 7))), 8, 9],
        [0, ((1, 2), 3, (4, (5, 6, 7))), 8, 9],
        [0, ((1, (2, 3)), (4, (5, 6, 7))), 8, 9],
        [0, {'d': (4, (5, 6, 7)), 'c': ((1, {'b': 3, 'a': 2})), 'e': [8, 9]}],
        [0, {1: (4, (5, 6, 7)), 1.1: ((1, {'b': 3, 'a': 2})), 'c': [8, 9]}],
        [0, OrderedDict([(1, (1, (2, 3, 4))), (1.1, ((5, {'b': 7, 'a': 6}))), ('c', [8, 9])])],
    ],
    none_is_leaf=[False, True],
)
def test_flatten_order(tree, none_is_leaf):
    flat, _ = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)

    assert flat == list(range(10))


def test_flatten_up_to():
    _, treespec = optree.tree_flatten([(1, 2), None, CustomTuple(foo=3, bar=7)])
    subtrees = treespec.flatten_up_to(
        [({'foo': 7}, (3, 4)), None, CustomTuple(foo=(11, 9), bar=None)]
    )
    assert subtrees == [{'foo': 7}, (3, 4), (11, 9), None]


def test_flatten_up_to_none_is_leaf():
    _, treespec = optree.tree_flatten([(1, 2), None, CustomTuple(foo=3, bar=7)], none_is_leaf=True)
    subtrees = treespec.flatten_up_to(
        [({'foo': 7}, (3, 4)), None, CustomTuple(foo=(11, 9), bar=None)]
    )
    assert subtrees == [{'foo': 7}, (3, 4), None, (11, 9), None]


@parametrize(
    leaves_fn=[
        optree.tree_leaves,
        lambda tree, is_leaf: optree.tree_flatten(tree, is_leaf)[0],
    ]
)
def test_flatten_is_leaf(leaves_fn):
    x = [(1, 2), (3, 4), (5, 6)]
    leaves = leaves_fn(x, is_leaf=lambda t: False)
    assert leaves == [1, 2, 3, 4, 5, 6]
    leaves = leaves_fn(x, is_leaf=lambda t: isinstance(t, tuple))
    assert leaves == x
    leaves = leaves_fn(x, is_leaf=lambda t: isinstance(t, list))
    assert leaves == [x]
    leaves = leaves_fn(x, is_leaf=lambda t: True)
    assert leaves == [x]

    y = [[[(1,)], [[(2,)], {'a': (3,)}]]]
    leaves = leaves_fn(y, is_leaf=lambda t: isinstance(t, tuple))
    assert leaves == [(1,), (2,), (3,)]

    z = [(1, 2), (3, 4), None, (5, None)]
    leaves = leaves_fn(z, is_leaf=lambda t: t is None)
    assert leaves == [1, 2, 3, 4, None, 5, None]


@parametrize(
    structure_fn=[
        optree.tree_structure,
        lambda tree, is_leaf: optree.tree_flatten(tree, is_leaf)[1],
    ]
)
def test_structure_is_leaf(structure_fn):
    x = [(1, 2), (3, 4), (5, 6)]
    treespec = structure_fn(x, is_leaf=lambda t: False)
    assert treespec.num_leaves == 6
    treespec = structure_fn(x, is_leaf=lambda t: isinstance(t, tuple))
    assert treespec.num_leaves == 3
    treespec = structure_fn(x, is_leaf=lambda t: isinstance(t, list))
    assert treespec.num_leaves == 1
    treespec = structure_fn(x, is_leaf=lambda t: True)
    assert treespec.num_leaves == 1

    y = [[[(1,)], [[(2,)], {'a': (3,)}]]]
    treespec = structure_fn(y, is_leaf=lambda t: isinstance(t, tuple))
    assert treespec.num_leaves == 3


@parametrize(
    tree=TREES,
    is_leaf=[
        is_tuple,
        is_list,
        is_none,
        always,
        never,
    ],
    none_is_leaf=[False, True],
)
def test_round_trip_is_leaf(tree, is_leaf, none_is_leaf):
    subtrees, treespec = optree.tree_flatten(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf)
    actual = optree.tree_unflatten(treespec, subtrees)
    assert actual == tree


@parametrize(tree=TREES, none_is_leaf=[False, True])
def test_all_leaves_with_trees(tree, none_is_leaf):
    leaves = optree.tree_leaves(tree, none_is_leaf=none_is_leaf)
    assert optree.all_leaves(leaves, none_is_leaf=none_is_leaf)
    if [tree] != leaves:
        assert not optree.all_leaves([tree], none_is_leaf=none_is_leaf)


@parametrize(leaf=LEAVES, none_is_leaf=[False, True])
def test_all_leaves_with_leaves(leaf, none_is_leaf):
    assert optree.all_leaves([leaf], none_is_leaf=none_is_leaf)


@parametrize(
    tree=TREES,
    is_leaf=[
        is_tuple,
        is_none,
        always,
        never,
    ],
    none_is_leaf=[False, True],
)
def test_all_leaves_with_is_leaf(tree, is_leaf, none_is_leaf):
    leaves = optree.tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf)
    assert optree.all_leaves(leaves, is_leaf=is_leaf, none_is_leaf=none_is_leaf)


def test_tree_map():
    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y)
    assert out == (((1, [3]), (2, None), None), ((3, {'foo': 'bar'}), (4, 7), (5, [5, 6])))


def test_tree_map_none_is_leaf():
    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (((1, [3]), (2, None), (None, 4)), ((3, {'foo': 'bar'}), (4, 7), (5, [5, 6])))


def test_tree_map_with_is_leaf_none():
    x = ((1, 2, None), (3, 4, 5))
    out = optree.tree_map(lambda *xs: tuple(xs), x, none_is_leaf=False)
    assert out == (((1,), (2,), None), ((3,), (4,), (5,)))
    out = optree.tree_map(lambda *xs: tuple(xs), x, none_is_leaf=True)
    assert out == (((1,), (2,), (None,)), ((3,), (4,), (5,)))
    out = optree.tree_map(lambda *xs: tuple(xs), x, is_leaf=lambda x: x is None)
    assert out == (((1,), (2,), (None,)), ((3,), (4,), (5,)))


def test_tree_map_with_is_leaf():
    x = ((1, 2, None), [3, 4, 5])
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y, is_leaf=lambda n: isinstance(n, list))
    assert out == (((1, [3]), (2, None), None), (([3, 4, 5], ({'foo': 'bar'}, 7, [5, 6]))))


def test_tree_map_with_is_leaf_none_is_leaf():
    x = ((1, 2, None), [3, 4, 5])
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))
    out = optree.tree_map(
        lambda *xs: tuple(xs), x, y, is_leaf=lambda n: isinstance(n, list), none_is_leaf=True
    )
    assert out == (((1, [3]), (2, None), (None, 4)), (([3, 4, 5], ({'foo': 'bar'}, 7, [5, 6]))))


def test_tree_reduce():
    assert optree.tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, 3)}) == 6
    assert optree.tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, None), 'z': 3}) == 6
    assert optree.tree_reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3}) == 3
    assert (
        optree.tree_reduce(
            lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True
        )
        is None
    )
    assert (
        optree.tree_reduce(
            lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3}, False, none_is_leaf=True
        )
        is False
    )


@parametrize(
    tree=TREES,
    inner_tree=[
        None,
        ['*', '*', '*'],
        ['*', '*', None],
        {'a': '*', 'b': None},
        {'a': '*', 'b': ('*', '*')},
    ],
    none_is_leaf=[False, True],
)
def test_compose(tree, inner_tree, none_is_leaf):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf)
    inner_treespec = optree.tree_structure(inner_tree, none_is_leaf=none_is_leaf)
    composed_treespec = treespec.compose(inner_treespec)
    expected_leaves = treespec.num_leaves * inner_treespec.num_leaves
    assert composed_treespec.num_leaves == treespec.num_leaves * inner_treespec.num_leaves
    expected_nodes = (treespec.num_nodes - treespec.num_leaves) + (
        inner_treespec.num_nodes * treespec.num_leaves
    )
    assert composed_treespec.num_nodes == expected_nodes
    leaves = [1] * expected_leaves
    composed = optree.tree_unflatten(composed_treespec, leaves)
    assert leaves == optree.tree_leaves(composed, none_is_leaf=none_is_leaf)
    try:
        assert composed_treespec == optree.tree_structure(composed, none_is_leaf=none_is_leaf)
    except AssertionError as ex:
        if 'CustomTreeNode(' not in str(ex):
            raise


@parametrize(tree=TREES)
def test_transpose(tree):
    outer_treespec = optree.tree_structure(tree)
    inner_treespec = optree.tree_structure([1, 1, 1])
    nested = optree.tree_map(lambda x: [x, x, x], tree)
    if outer_treespec.num_leaves == 0:
        with pytest.raises(ValueError, match='Tree structures must have at least one leaf.'):
            optree.tree_transpose(outer_treespec, inner_treespec, nested)
        return
    with pytest.raises(ValueError, match='Tree structures must have the same none_is_leaf value.'):
        optree.tree_transpose(
            outer_treespec, optree.tree_structure([1, 1, 1], none_is_leaf=True), nested
        )
    actual = optree.tree_transpose(outer_treespec, inner_treespec, nested)
    assert actual == [tree, tree, tree]


def test_transpose_mismatch_outer():
    tree = {'a': [1, 2], 'b': [3, 4]}
    outer_treespec = optree.tree_structure({'a': 1, 'b': 2, 'c': 3})
    inner_treespec = optree.tree_structure([1, 2])
    with pytest.raises(TypeError, match='mismatch'):
        optree.tree_transpose(outer_treespec, inner_treespec, tree)


def test_transpose_mismatch_inner():
    tree = {'a': [1, 2], 'b': [3, 4]}
    outer_treespec = optree.tree_structure({'a': 1, 'b': 2})
    inner_treespec = optree.tree_structure([1, 2, 3])
    with pytest.raises(TypeError, match='mismatch'):
        optree.tree_transpose(outer_treespec, inner_treespec, tree)


def test_transpose_with_custom_object():
    outer_treespec = optree.tree_structure(FlatCache({'a': 1, 'b': 2}))
    inner_treespec = optree.tree_structure([1, 2])
    expected = [FlatCache({'a': 3, 'b': 5}), FlatCache({'a': 4, 'b': 6})]
    actual = optree.tree_transpose(
        outer_treespec, inner_treespec, FlatCache({'a': [3, 4], 'b': [5, 6]})
    )
    assert actual == expected


def test_tree_all():
    assert optree.tree_all({})
    assert optree.tree_all({'x': 1, 'y': (2, 3)})
    assert optree.tree_all({'x': 1, 'y': (2, None), 'z': 3})
    assert not optree.tree_all({'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    assert optree.tree_all(None)
    assert not optree.tree_all(None, none_is_leaf=True)


def test_tree_any():
    assert not optree.tree_any({})
    assert optree.tree_any({'x': 0, 'y': (2, 0)})
    assert not optree.tree_any({'a': None})
    assert not optree.tree_any({'a': None}, none_is_leaf=True)
    assert not optree.tree_any(None)
    assert not optree.tree_any(None, none_is_leaf=True)


def test_tree_replace_nones():
    sentinel = object()
    assert optree.tree_replace_nones(sentinel, {'a': 1, 'b': None, 'c': (2, None)}) == {
        'a': 1,
        'b': sentinel,
        'c': (2, sentinel),
    }
    assert optree.tree_replace_nones(sentinel, None) == sentinel


@parametrize(
    tree=[
        optree.Partial(dummy_func),
        optree.Partial(dummy_func, 1, 2),
        optree.Partial(dummy_func, x='a'),
        optree.Partial(dummy_func, 1, 2, 3, x=4, y=5),
        optree.Partial(dummy_func, 1, None, x=4, y=5, z=None),
    ],
    none_is_leaf=[False, True],
)
def test_partial_round_trip(tree, none_is_leaf):
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)
    actual = optree.tree_unflatten(treespec, leaves)
    assert actual.func == tree.func
    assert actual.args == tree.args
    assert actual.keywords == tree.keywords


def test_partial_does_not_merge_with_other_partials():
    def f(a, b, c):  # pylint: disable=unused-argument
        pass

    g = functools.partial(f, 2)
    h = optree.Partial(g, 3)
    assert h.args == (3,)


def test_partial_func_attribute_has_stable_hash():
    fun = functools.partial(print, 1)
    p1 = optree.Partial(fun, 2)
    p2 = optree.Partial(fun, 2)
    assert p1.func == fun  # pylint: disable=comparison-with-callable
    assert fun == p1.func  # pylint: disable=comparison-with-callable
    assert p1.func == p2.func
    assert hash(p1.func) == hash(p2.func)
