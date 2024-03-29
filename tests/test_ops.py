# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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

import copy
import functools
import itertools
import pickle
import re
from collections import OrderedDict, defaultdict, deque

import pytest

import optree

# pylint: disable-next=wrong-import-order
from helpers import (
    LEAVES,
    TREE_PATHS,
    TREES,
    Counter,
    CustomTuple,
    FlatCache,
    MyAnotherDict,
    parametrize,
)


def dummy_func(*args, **kwargs):  # pylint: disable=unused-argument
    return


dummy_partial_func = functools.partial(dummy_func, a=1)


def is_tuple(tup):
    return isinstance(tup, tuple)


def is_list(lst):
    return isinstance(lst, list)


def is_none(none):
    return none is None


def always(obj):  # pylint: disable=unused-argument
    return True


def never(obj):  # pylint: disable=unused-argument
    return False


def test_max_depth():
    lst = [1]
    for _ in range(optree.MAX_RECURSION_DEPTH - 1):
        lst = [lst]
    optree.tree_flatten(lst)
    optree.tree_flatten_with_path(lst)

    lst = [lst]
    with pytest.raises(
        RecursionError,
        match='Maximum recursion depth exceeded during flattening the tree.',
    ):
        optree.tree_flatten(lst)
    with pytest.raises(
        RecursionError,
        match='Maximum recursion depth exceeded during flattening the tree.',
    ):
        optree.tree_flatten_with_path(lst)


@parametrize(
    tree=list(TREES + LEAVES),
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_round_trip(tree, none_is_leaf, namespace):
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    actual = optree.tree_unflatten(treespec, leaves)
    assert actual == tree


@parametrize(
    tree=list(TREES + LEAVES),
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_round_trip_with_flatten_up_to(tree, none_is_leaf, namespace):
    _, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    leaves = treespec.flatten_up_to(tree)
    actual = optree.tree_unflatten(treespec, leaves)
    assert actual == tree


@parametrize(
    tree=[
        [0, ((1, 2), 3, (4, (5, 6, 7))), 8, 9],
        [0, ((1, 2), 3, (4, (5, 6, 7))), 8, 9],
        [0, ((1, (2, 3)), (4, (5, 6, 7))), 8, 9],
        [0, {'d': (4, (5, 6, 7)), 'c': (1, {'b': 3, 'a': 2}), 'e': [8, 9]}],
        [0, {1: (4, (5, 6, 7)), 1.1: (1, {'b': 3, 'a': 2}), 'c': [8, 9]}],
        [0, OrderedDict([(1, (1, (2, 3, 4))), (1.1, ((5, {'b': 7, 'a': 6}))), ('c', [8, 9])])],
    ],
    none_is_leaf=[False, True],
)
def test_flatten_order(tree, none_is_leaf):
    flat, _ = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)

    assert flat == list(range(10))


def test_flatten_dict_order():
    assert optree.tree_leaves({'a': 1, 2: 2}) == [2, 1]
    assert optree.tree_leaves({'a': 1, 2: 2, 3.0: 3}) == [3, 2, 1]
    assert optree.tree_leaves({2: 2, 3.0: 3}) == [2, 3]
    assert list(optree.tree_iter({'a': 1, 2: 2})) == [2, 1]
    assert list(optree.tree_iter({'a': 1, 2: 2, 3.0: 3})) == [3, 2, 1]
    assert list(optree.tree_iter({2: 2, 3.0: 3})) == [2, 3]

    sorted_treespec = optree.tree_structure({'a': 1, 'b': 2, 'c': {'e': 3, 'f': None, 'g': 4}})

    tree = {'b': 2, 'a': 1, 'c': {'f': None, 'e': 3, 'g': 4}}
    leaves, treespec = optree.tree_flatten(tree)
    assert treespec == sorted_treespec
    assert leaves == [1, 2, 3, 4]
    assert str(treespec) == r"PyTreeSpec({'a': *, 'b': *, 'c': {'e': *, 'f': None, 'g': *}})"
    restored_tree = optree.tree_unflatten(treespec, leaves)
    assert list(restored_tree) == ['b', 'a', 'c']

    restored_treespec = pickle.loads(pickle.dumps(treespec))
    assert restored_treespec == treespec
    assert restored_treespec == sorted_treespec
    assert str(restored_treespec) == str(treespec)
    assert str(restored_treespec) == str(sorted_treespec)
    restored_tree = optree.tree_unflatten(restored_treespec, leaves)
    assert list(restored_tree) == ['b', 'a', 'c']


@parametrize(
    tree=list(TREES + LEAVES),
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_tree_unflatten_mismatch_number_of_leaves(tree, none_is_leaf, namespace):
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    if len(leaves) > 0:
        with pytest.raises(ValueError, match='Too few leaves for PyTreeSpec.'):
            optree.tree_unflatten(treespec, leaves[:-1])
    with pytest.raises(ValueError, match='Too many leaves for PyTreeSpec.'):
        optree.tree_unflatten(treespec, (*leaves, 0))


@parametrize(
    tree=list(TREES + LEAVES),
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_tree_iter(tree, none_is_leaf, namespace):
    leaves = optree.tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    it = optree.tree_iter(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    assert iter(it) is it
    assert list(it) == leaves
    with pytest.raises(StopIteration):
        next(it)


def test_walk():
    tree = {'b': 2, 'a': 1, 'c': {'f': None, 'e': 3, 'g': 4}}
    #          tree
    #        /  |  \
    #     ---   |   ---
    #    /      |      \
    # ('a')   ('b')   ('c')
    #   1       2     / | \
    #              ---  |  ---
    #             /     |     \
    #          ('e')  ('f')  ('g')
    #            3     None    4
    #                   |
    #                   X
    #

    def get_functions():
        nodes_visited = []
        node_data_visited = []
        leaves_visited = []

        def f_node(node, node_data):
            nodes_visited.append(node)
            node_data_visited.append(node_data)
            return copy.deepcopy(nodes_visited), None

        def f_leaf(leaf):
            leaves_visited.append(leaf)
            return copy.deepcopy(leaves_visited)

        return f_node, f_leaf, nodes_visited, node_data_visited, leaves_visited

    leaves, treespec = optree.tree_flatten(tree)

    f_node, f_leaf, nodes_visited, node_data_visited, leaves_visited = get_functions()
    with pytest.raises(ValueError, match='Too few leaves for PyTreeSpec.'):
        treespec.walk(f_node, f_leaf, leaves[:-1])

    f_node, f_leaf, nodes_visited, node_data_visited, leaves_visited = get_functions()
    with pytest.raises(ValueError, match='Too many leaves for PyTreeSpec.'):
        treespec.walk(f_node, f_leaf, (*leaves, 0))

    f_node, f_leaf, nodes_visited, node_data_visited, leaves_visited = get_functions()
    output = treespec.walk(f_node, f_leaf, leaves)
    assert leaves_visited == [1, 2, 3, 4]
    assert nodes_visited == [
        (),
        ([1, 2, 3], ([()], None), [1, 2, 3, 4]),
        ([1], [1, 2], ([(), ([1, 2, 3], ([()], None), [1, 2, 3, 4])], None)),
    ]
    assert node_data_visited == [None, ['e', 'f', 'g'], ['a', 'b', 'c']]
    assert output == (
        [
            (),
            ([1, 2, 3], ([()], None), [1, 2, 3, 4]),
            ([1], [1, 2], ([(), ([1, 2, 3], ([()], None), [1, 2, 3, 4])], None)),
        ],
        None,
    )

    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=True)

    f_node, f_leaf, nodes_visited, node_data_visited, leaves_visited = get_functions()
    with pytest.raises(ValueError, match='Too few leaves for PyTreeSpec.'):
        treespec.walk(f_node, f_leaf, leaves[:-1])

    f_node, f_leaf, nodes_visited, node_data_visited, leaves_visited = get_functions()
    with pytest.raises(ValueError, match='Too many leaves for PyTreeSpec.'):
        treespec.walk(f_node, f_leaf, (*leaves, 0))

    f_node, f_leaf, nodes_visited, node_data_visited, leaves_visited = get_functions()
    output = treespec.walk(f_node, f_leaf, leaves)
    assert leaves_visited == [1, 2, 3, None, 4]
    assert nodes_visited == [
        ([1, 2, 3], [1, 2, 3, None], [1, 2, 3, None, 4]),
        ([1], [1, 2], ([([1, 2, 3], [1, 2, 3, None], [1, 2, 3, None, 4])], None)),
    ]
    assert node_data_visited == [['e', 'f', 'g'], ['a', 'b', 'c']]
    assert output == (
        [
            ([1, 2, 3], [1, 2, 3, None], [1, 2, 3, None, 4]),
            ([1], [1, 2], ([([1, 2, 3], [1, 2, 3, None], [1, 2, 3, None, 4])], None)),
        ],
        None,
    )


def test_flatten_up_to():
    _, treespec = optree.tree_flatten([(1, 2), None, CustomTuple(foo=3, bar=7)])
    subtrees = treespec.flatten_up_to(
        [({'foo': 7}, (3, 4)), None, CustomTuple(foo=(11, 9), bar=None)],
    )
    assert subtrees == [{'foo': 7}, (3, 4), (11, 9), None]


def test_flatten_up_to_none_is_leaf():
    _, treespec = optree.tree_flatten([(1, 2), None, CustomTuple(foo=3, bar=7)], none_is_leaf=True)
    subtrees = treespec.flatten_up_to(
        [({'foo': 7}, (3, 4)), None, CustomTuple(foo=(11, 9), bar=None)],
    )
    assert subtrees == [{'foo': 7}, (3, 4), None, (11, 9), None]


@parametrize(
    leaves_fn=[
        optree.tree_leaves,
        lambda tree, is_leaf: list(optree.tree_iter(tree, is_leaf)),
        lambda tree, is_leaf: optree.tree_flatten(tree, is_leaf)[0],
        lambda tree, is_leaf: optree.tree_flatten_with_path(tree, is_leaf)[1],
    ],
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
        lambda tree, is_leaf: optree.tree_flatten_with_path(tree, is_leaf)[2],
    ],
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
    data=list(
        itertools.chain(
            zip(TREES, TREE_PATHS[False], itertools.repeat(False)),
            zip(TREES, TREE_PATHS[True], itertools.repeat(True)),
        ),
    ),
)
def test_paths(data):
    tree, expected_paths, none_is_leaf = data
    expected_leaves, expected_treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)
    paths, leaves, treespec = optree.tree_flatten_with_path(tree, none_is_leaf=none_is_leaf)
    treespec_paths = optree.treespec_paths(treespec)
    assert len(paths) == len(leaves)
    assert leaves == expected_leaves
    assert treespec == expected_treespec
    assert paths == expected_paths
    assert len(treespec_paths) == len(leaves)
    assert treespec_paths == expected_paths
    paths = optree.tree_paths(tree, none_is_leaf=none_is_leaf)
    assert paths == expected_paths


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
    namespace=['', 'undefined', 'namespace'],
)
def test_paths_with_is_leaf(tree, is_leaf, none_is_leaf, namespace):
    expected_leaves, expected_treespec = optree.tree_flatten(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    paths, leaves, treespec = optree.tree_flatten_with_path(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    treespec_paths = optree.treespec_paths(treespec)
    assert len(paths) == len(leaves)
    assert leaves == expected_leaves
    assert treespec == expected_treespec
    assert len(treespec_paths) == len(leaves)
    assert paths == treespec_paths
    paths = optree.tree_paths(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    assert len(paths) == len(leaves)
    assert paths == treespec_paths


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
    namespace=['', 'undefined', 'namespace'],
)
def test_round_trip_is_leaf(tree, is_leaf, none_is_leaf, namespace):
    subtrees, treespec = optree.tree_flatten(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    actual = optree.tree_unflatten(treespec, subtrees)
    assert actual == tree


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_tree_is_leaf_with_trees(tree, none_is_leaf, namespace):
    leaves = optree.tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    for leaf in leaves:
        assert optree.tree_is_leaf(leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if [tree] != leaves:
        assert not optree.tree_is_leaf(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    else:
        assert optree.tree_is_leaf(tree, none_is_leaf=none_is_leaf, namespace=namespace)


@parametrize(
    leaf=LEAVES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_tree_is_leaf_with_leaves(leaf, none_is_leaf, namespace):
    assert optree.tree_is_leaf(leaf, none_is_leaf=none_is_leaf, namespace=namespace)


@parametrize(
    tree=TREES,
    is_leaf=[
        is_tuple,
        is_none,
        always,
        never,
    ],
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_tree_is_leaf_with_is_leaf(tree, is_leaf, none_is_leaf, namespace):
    leaves = optree.tree_leaves(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    for leaf in leaves:
        assert optree.tree_is_leaf(
            leaf,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    if [tree] != leaves:
        assert not optree.tree_is_leaf(
            tree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    else:
        assert optree.tree_is_leaf(
            tree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_all_leaves_with_trees(tree, none_is_leaf, namespace):
    leaves = optree.tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    assert optree.all_leaves(leaves, none_is_leaf=none_is_leaf, namespace=namespace)
    if [tree] != leaves:
        assert not optree.all_leaves([tree], none_is_leaf=none_is_leaf, namespace=namespace)


@parametrize(
    leaf=LEAVES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_all_leaves_with_leaves(leaf, none_is_leaf, namespace):
    assert optree.all_leaves([leaf], none_is_leaf=none_is_leaf, namespace=namespace)


@parametrize(
    tree=TREES,
    is_leaf=[
        is_tuple,
        is_none,
        always,
        never,
    ],
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_all_leaves_with_is_leaf(tree, is_leaf, none_is_leaf, namespace):
    leaves = optree.tree_leaves(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    assert optree.all_leaves(
        leaves,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def test_tree_map():
    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y)
    assert out == (((1, [6]), (2, None), None), ((3, {'foo': 'bar'}), (4, 7), (5, [8, 9])))

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 0]))
    with pytest.raises(ValueError, match=re.escape('Expected None, got 7.')):
        optree.tree_map(lambda *xs: tuple(xs), x, y)


def test_tree_map_with_path():
    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map_with_path(lambda *xs: tuple(xs), x, y)
    assert out == (
        (((0, 0), 1, [6]), ((0, 1), 2, None), None),
        (((1, 0), 3, {'foo': 'bar'}), ((1, 1), 4, 7), ((1, 2), 5, [8, 9])),
    )

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 0]))
    with pytest.raises(ValueError, match=re.escape('Expected None, got 7.')):
        optree.tree_map_with_path(lambda *xs: tuple(xs), x, y)


def test_tree_broadcast_map():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_broadcast_map(lambda *xs: tuple(xs), x, y)
    assert out == (
        ([(1, 7)], None, None),
        ({'foo': (3, 'bar')}, ((4, 8), [(5, 8)]), [(6, 9), (6, 10)]),
    )

    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, 8), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map(lambda *xs: tuple(xs), x, y)
    assert out == (
        ([(1, 7)], None, None),
        ({'foo': (3, 'bar')}, ((4, 9), [(5, 9)]), [(6, 10), (6, 11)]),
    )


def test_tree_broadcast_map_with_path():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_broadcast_map_with_path(lambda *xs: tuple(xs), x, y)
    assert out == (
        ([((0, 0, 0), 1, 7)], None, None),
        (
            {'foo': ((1, 0, 'foo'), 3, 'bar')},
            (((1, 1, 0), 4, 8), [((1, 1, 1, 0), 5, 8)]),
            [((1, 2, 0), 6, 9), ((1, 2, 1), 6, 10)],
        ),
    )

    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, 8), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map_with_path(lambda *xs: tuple(xs), x, y)
    assert out == (
        ([((0, 0, 0), 1, 7)], None, None),
        (
            {'foo': ((1, 0, 'foo'), 3, 'bar')},
            (((1, 1, 0), 4, 9), [((1, 1, 1, 0), 5, 9)]),
            [((1, 2, 0), 6, 10), ((1, 2, 1), 6, 11)],
        ),
    )


def test_tree_transpose_map():
    with pytest.raises(
        ValueError,
        match=r'The outer structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map(lambda a, b: {'a': a, 'b': b}, (), ())
    with pytest.raises(
        ValueError,
        match=r'The inner structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map(lambda a, b: None, (1,), (2,))
    with pytest.raises(
        ValueError,
        match=r'The inner structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map(lambda a, b: (), (1,), (2,))

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_transpose_map(lambda a, b: {'a': a, 'b': b}, x, y)
    assert out == {
        'a': ((1, 2, None), (3, 4, 5)),
        'b': ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    with pytest.raises(ValueError, match=re.escape('Expected an instance of list, got None.')):
        optree.tree_transpose_map(lambda a, b: {'a': a, 'b': b}, x, y)
    out = optree.tree_transpose_map(
        lambda a, b: {'a': a, 'b': b},
        x,
        y,
        inner_treespec=optree.tree_structure({'a': 0, 'b': 1}),
    )
    assert out == {
        'a': ((1, 2, None), (3, 4, 5)),
        'b': (([6], None, None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], 7), ({'foo': 'bar'}, 8, [9, 0]))
    with pytest.raises(ValueError, match=re.escape('Expected None, got 7.')):
        optree.tree_transpose_map(lambda a, b: {'a': a, 'b': b}, x, y)


def test_tree_transpose_map_with_path():
    with pytest.raises(
        ValueError,
        match=r'The outer structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_path(lambda p, a, b: {'p': p, 'a': a, 'b': b}, (), ())
    with pytest.raises(
        ValueError,
        match=r'The inner structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_path(lambda p, a, b: None, (1,), (2,))
    with pytest.raises(
        ValueError,
        match=r'The inner structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_path(lambda p, a, b: (), (1,), (2,))

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_transpose_map_with_path(lambda p, a, b: {'d': len(p), 'a': a, 'b': b}, x, y)
    assert out == {
        'd': ((2, 2, None), (2, 2, 2)),
        'a': ((1, 2, None), (3, 4, 5)),
        'b': ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    with pytest.raises(ValueError, match=re.escape('Expected an instance of list, got None.')):
        optree.tree_transpose_map_with_path(lambda p, a, b: {'p': p, 'a': a, 'b': b}, x, y)
    out = optree.tree_transpose_map_with_path(
        lambda p, a, b: {'p': p, 'a': a, 'b': b},
        x,
        y,
        inner_treespec=optree.tree_structure({'p': 0, 'a': 1, 'b': 2}),
    )
    assert out == {
        'p': (((0, 0), (0, 1), None), ((1, 0), (1, 1), (1, 2))),
        'a': ((1, 2, None), (3, 4, 5)),
        'b': (([6], None, None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], 7), ({'foo': 'bar'}, 8, [9, 0]))
    with pytest.raises(ValueError, match=re.escape('Expected None, got 7.')):
        optree.tree_transpose_map(
            lambda p, a, b: {'p': p, 'a': a, 'b': b},
            x,
            y,
            inner_treespec=optree.tree_structure({'p': 0, 'a': 1, 'b': 2}),
        )


def test_tree_map_none_is_leaf():
    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (((1, [6]), (2, None), (None, None)), ((3, {'foo': 'bar'}), (4, 7), (5, [8, 9])))

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 0]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (((1, [6]), (2, None), (None, 7)), ((3, {'foo': 'bar'}), (4, 8), (5, [9, 0])))


def test_tree_map_with_path_none_is_leaf():
    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map_with_path(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        (((0, 0), 1, [6]), ((0, 1), 2, None), ((0, 2), None, None)),
        (((1, 0), 3, {'foo': 'bar'}), ((1, 1), 4, 7), ((1, 2), 5, [8, 9])),
    )

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 0]))
    out = optree.tree_map_with_path(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        (((0, 0), 1, [6]), ((0, 1), 2, None), ((0, 2), None, 7)),
        (((1, 0), 3, {'foo': 'bar'}), ((1, 1), 4, 8), ((1, 2), 5, [9, 0])),
    )


def test_tree_broadcast_map_none_is_leaf():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        ([(1, 7)], (2, None), (None, None)),
        ({'foo': (3, 'bar')}, ((4, 9), [(5, 9)]), [(6, 10), (6, 11)]),
    )

    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, 8), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        ([(1, 7)], (2, None), (None, 8)),
        ({'foo': (3, 'bar')}, ((4, 9), [(5, 9)]), [(6, 10), (6, 11)]),
    )


def test_tree_broadcast_map_with_path_none_is_leaf():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map_with_path(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        ([((0, 0, 0), 1, 7)], ((0, 1), 2, None), ((0, 2), None, None)),
        (
            {'foo': ((1, 0, 'foo'), 3, 'bar')},
            (((1, 1, 0), 4, 9), [((1, 1, 1, 0), 5, 9)]),
            [((1, 2, 0), 6, 10), ((1, 2, 1), 6, 11)],
        ),
    )

    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, 8), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map_with_path(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        ([((0, 0, 0), 1, 7)], ((0, 1), 2, None), ((0, 2), None, 8)),
        (
            {'foo': ((1, 0, 'foo'), 3, 'bar')},
            (((1, 1, 0), 4, 9), [((1, 1, 1, 0), 5, 9)]),
            [((1, 2, 0), 6, 10), ((1, 2, 1), 6, 11)],
        ),
    )


def test_tree_transpose_map_none_is_leaf():
    with pytest.raises(
        ValueError,
        match=r'The outer structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map(lambda a, b: {'a': a, 'b': b}, (), (), none_is_leaf=True)
    out = optree.tree_transpose_map(lambda a, b: None, (1,), (2,), none_is_leaf=True)
    assert out == (None,)
    with pytest.raises(
        ValueError,
        match=r'The inner structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map(lambda a, b: (), (1,), (2,), none_is_leaf=True)

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_transpose_map(lambda a, b: {'a': a, 'b': b}, x, y, none_is_leaf=True)
    assert out == {
        'a': ((1, 2, None), (3, 4, 5)),
        'b': ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    with pytest.raises(ValueError, match=re.escape('Expected an instance of list, got None.')):
        optree.tree_transpose_map(lambda a, b: {'a': a, 'b': b}, x, y, none_is_leaf=True)
    out = optree.tree_transpose_map(
        lambda a, b: {'a': a, 'b': b},
        x,
        y,
        inner_treespec=optree.tree_structure({'a': 0, 'b': 1}),
        none_is_leaf=True,
    )
    assert out == {
        'a': ((1, 2, None), (3, 4, 5)),
        'b': (([6], None, None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], 7), ({'foo': 'bar'}, 8, [9, 0]))
    out = optree.tree_transpose_map(
        lambda a, b: {'a': a, 'b': b},
        x,
        y,
        none_is_leaf=True,
    )
    assert out == {
        'a': ((1, 2, None), (3, 4, 5)),
        'b': ((6, [None], 7), ({'foo': 'bar'}, 8, [9, 0])),
    }


def test_tree_transpose_map_with_path_none_is_leaf():
    with pytest.raises(
        ValueError,
        match=r'The outer structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_path(
            lambda p, a, b: {'p': p, 'a': a, 'b': b},
            (),
            (),
            none_is_leaf=True,
        )
    out = optree.tree_transpose_map_with_path(
        lambda p, a, b: None,
        (1,),
        (2,),
        none_is_leaf=True,
    )
    assert out == (None,)
    with pytest.raises(
        ValueError,
        match=r'The inner structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_path(
            lambda p, a, b: (),
            (1,),
            (2,),
            none_is_leaf=True,
        )

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_transpose_map_with_path(
        lambda p, a, b: {'d': len(p), 'a': a, 'b': b},
        x,
        y,
        none_is_leaf=True,
    )
    assert out == {
        'd': ((2, 2, 2), (2, 2, 2)),
        'a': ((1, 2, None), (3, 4, 5)),
        'b': ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    with pytest.raises(ValueError, match=re.escape('Expected an instance of list, got None.')):
        optree.tree_transpose_map_with_path(
            lambda p, a, b: {'p': p, 'a': a, 'b': b},
            x,
            y,
            none_is_leaf=True,
        )
    out = optree.tree_transpose_map_with_path(
        lambda p, a, b: {'p': p, 'a': a, 'b': b},
        x,
        y,
        inner_treespec=optree.tree_structure({'p': 0, 'a': 1, 'b': 2}),
        none_is_leaf=True,
    )
    assert out == {
        'p': (((0, 0), (0, 1), (0, 2)), ((1, 0), (1, 1), (1, 2))),
        'a': ((1, 2, None), (3, 4, 5)),
        'b': (([6], None, None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], 7), ({'foo': 'bar'}, 8, [9, 0]))
    out = optree.tree_transpose_map_with_path(
        lambda p, a, b: {'p': p, 'a': a, 'b': b},
        x,
        y,
        inner_treespec=optree.tree_structure({'p': 0, 'a': 1, 'b': 2}),
        none_is_leaf=True,
    )
    assert out == {
        'p': (((0, 0), (0, 1), (0, 2)), ((1, 0), (1, 1), (1, 2))),
        'a': ((1, 2, None), (3, 4, 5)),
        'b': ((6, [None], 7), ({'foo': 'bar'}, 8, [9, 0])),
    }


def test_tree_map_key_order():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    leaves = []

    def add_leaves(x):
        leaves.append(x)
        return x

    for tree_map in (optree.tree_map, optree.tree_broadcast_map):
        leaves.clear()
        mapped = tree_map(add_leaves, tree)
        assert mapped == tree
        assert list(mapped.keys()) == list(tree.keys())
        assert list(mapped.values()) == list(tree.values())
        assert list(mapped.items()) == list(tree.items())
        assert leaves == [1, 2, 3, 4]


def test_tree_map_with_path_key_order():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    paths = []
    leaves = []

    def add_leaves(p, x):
        paths.append(p)
        leaves.append(x)
        return p, x

    for tree_map_with_path in (optree.tree_map_with_path, optree.tree_broadcast_map_with_path):
        paths.clear()
        leaves.clear()
        mapped = tree_map_with_path(add_leaves, tree)
        expected = {
            'b': (('b',), 2),
            'a': (('a',), 1),
            'c': (('c',), 3),
            'd': None,
            'e': (('e',), 4),
        }
        assert mapped == expected
        assert list(mapped.keys()) == list(expected.keys())
        assert list(mapped.values()) == list(expected.values())
        assert list(mapped.items()) == list(expected.items())
        assert paths == [('a',), ('b',), ('c',), ('e',)]
        assert leaves == [1, 2, 3, 4]


def test_tree_map_key_order_none_is_leaf():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    leaves = []

    def add_leaves(x):
        leaves.append(x)
        return x

    for tree_map in (optree.tree_map, optree.tree_broadcast_map):
        leaves.clear()
        mapped = tree_map(add_leaves, tree, none_is_leaf=True)
        assert mapped == tree
        assert list(mapped.keys()) == list(tree.keys())
        assert list(mapped.values()) == list(tree.values())
        assert list(mapped.items()) == list(tree.items())
        assert leaves == [1, 2, 3, None, 4]


def test_tree_map_with_path_key_order_none_is_leaf():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    paths = []
    leaves = []

    def add_leaves(p, x):
        paths.append(p)
        leaves.append(x)
        return p, x

    for tree_map_with_path in (optree.tree_map_with_path, optree.tree_broadcast_map_with_path):
        paths.clear()
        leaves.clear()
        mapped = tree_map_with_path(add_leaves, tree, none_is_leaf=True)
        expected = {
            'b': (('b',), 2),
            'a': (('a',), 1),
            'c': (('c',), 3),
            'd': (('d',), None),
            'e': (('e',), 4),
        }
        assert mapped == expected
        assert list(mapped.keys()) == list(expected.keys())
        assert list(mapped.values()) == list(expected.values())
        assert list(mapped.items()) == list(expected.items())
        assert paths == [('a',), ('b',), ('c',), ('d',), ('e',)]
        assert leaves == [1, 2, 3, None, 4]


def test_tree_map_with_is_leaf_none():
    x = ((1, 2, None), (3, 4, 5))
    for tree_map in (optree.tree_map, optree.tree_broadcast_map):
        out = tree_map(lambda *xs: tuple(xs), x, none_is_leaf=False)
        assert out == (((1,), (2,), None), ((3,), (4,), (5,)))
        out = tree_map(lambda *xs: tuple(xs), x, none_is_leaf=True)
        assert out == (((1,), (2,), (None,)), ((3,), (4,), (5,)))
        out = tree_map(lambda *xs: tuple(xs), x, is_leaf=lambda x: x is None)
        assert out == (((1,), (2,), (None,)), ((3,), (4,), (5,)))


def test_tree_map_with_is_leaf():
    x = ((1, 2, None), [3, 4, 5])
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y, is_leaf=lambda n: isinstance(n, list))
    assert out == (((1, [6]), (2, None), None), (([3, 4, 5], ({'foo': 'bar'}, 7, [8, 9]))))

    x = ((1, 2, None), [3, 4, 5])
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 0]))
    with pytest.raises(ValueError, match=re.escape('Expected None, got 7.')):
        optree.tree_map(lambda *xs: tuple(xs), x, y, is_leaf=lambda n: isinstance(n, list))


def test_tree_map_with_is_leaf_none_is_leaf():
    x = ((1, 2, None), [3, 4, 5])
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map(
        lambda *xs: tuple(xs),
        x,
        y,
        is_leaf=lambda n: isinstance(n, list),
        none_is_leaf=True,
    )
    assert out == (((1, [6]), (2, None), (None, None)), (([3, 4, 5], ({'foo': 'bar'}, 7, [8, 9]))))

    x = ((1, 2, None), [3, 4, 5])
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 0]))
    out = optree.tree_map(
        lambda *xs: tuple(xs),
        x,
        y,
        is_leaf=lambda n: isinstance(n, list),
        none_is_leaf=True,
    )
    assert out == (((1, [6]), (2, None), (None, 7)), (([3, 4, 5], ({'foo': 'bar'}, 8, [9, 0]))))


def test_tree_broadcast_map_with_is_leaf():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_broadcast_map(
        lambda *xs: tuple(xs),
        x,
        y,
        is_leaf=lambda n: isinstance(n, list),
    )
    assert out == (
        ((1, [7]), None, None),
        ({'foo': (3, 'bar')}, ((4, 8), ([5], 8)), (6, [9, 10])),
    )

    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, 8), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map(
        lambda *xs: tuple(xs),
        x,
        y,
        is_leaf=lambda n: isinstance(n, list),
    )
    assert out == (
        ((1, [7]), None, None),
        ({'foo': (3, 'bar')}, ((4, 9), ([5], 9)), (6, [10, 11])),
    )


def test_tree_broadcast_map_with_is_leaf_none_is_leaf():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map(
        lambda *xs: tuple(xs),
        x,
        y,
        is_leaf=lambda n: isinstance(n, list),
        none_is_leaf=True,
    )
    assert out == (
        ((1, [7]), (2, None), (None, None)),
        ({'foo': (3, 'bar')}, ((4, 9), ([5], 9)), (6, [10, 11])),
    )

    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, 8), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map(
        lambda *xs: tuple(xs),
        x,
        y,
        is_leaf=lambda n: isinstance(n, list),
        none_is_leaf=True,
    )
    assert out == (
        ((1, [7]), (2, None), (None, 8)),
        ({'foo': (3, 'bar')}, ((4, 9), ([5], 9)), (6, [10, 11])),
    )


def test_tree_map_ignore_return():
    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, None), ({'foo': 'bar'}, 7, [5, 6]))

    def fn1(*xs):
        leaves.append(xs)
        return 0

    leaves = []
    out = optree.tree_map_(fn1, x, y)
    assert out is x
    assert x == ((1, 2, None), (3, 4, 5))
    assert leaves == [(1, [3]), (2, None), (3, {'foo': 'bar'}), (4, 7), (5, [5, 6])]

    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))

    def fn2(*xs):
        leaves.append(xs)
        return 0

    leaves = []
    with pytest.raises(ValueError, match=re.escape('Expected None, got 4.')):
        optree.tree_map_(fn2, x, y)


def test_tree_map_with_path_ignore_return():
    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, None), ({'foo': 'bar'}, 7, [5, 6]))

    def fn1(p, *xs):
        if p[1] >= 1:
            leaves.append(xs)
        return 0

    leaves = []
    out = optree.tree_map_with_path_(fn1, x, y)
    assert out is x
    assert x == ((1, 2, None), (3, 4, 5))
    assert leaves == [(2, None), (4, 7), (5, [5, 6])]

    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))

    def fn2(p, *xs):
        if p[1] >= 1:
            leaves.append(xs)
        return 0

    leaves = []
    with pytest.raises(ValueError, match=re.escape('Expected None, got 4.')):
        optree.tree_map_with_path_(fn2, x, y)


def test_tree_map_with_path_ignore_return_none_is_leaf():
    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, None), ({'foo': 'bar'}, 7, [5, 6]))

    def fn1(p, *xs):
        if p[1] >= 1:
            leaves.append(xs)
        return 0

    leaves = []
    out = optree.tree_map_with_path_(fn1, x, y, none_is_leaf=True)
    assert out is x
    assert x == ((1, 2, None), (3, 4, 5))
    assert leaves == [(2, None), (None, None), (4, 7), (5, [5, 6])]

    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))

    def fn2(p, *xs):
        if p[1] >= 1:
            leaves.append(xs)
        return 0

    leaves = []
    out = optree.tree_map_with_path_(fn2, x, y, none_is_leaf=True)
    assert out is x
    assert x == ((1, 2, None), (3, 4, 5))
    assert leaves == [(2, None), (None, 4), (4, 7), (5, [5, 6])]


def test_tree_map_inplace():
    x = ((Counter(1), Counter(2), None), (Counter(3), Counter(4), Counter(5)))
    y = ((3, 0, None), (5, 7, 6))

    def fn1_(xi, yi):
        xi.increment(yi)

    out = optree.tree_map_(fn1_, x, y)
    assert out is x
    assert x == ((Counter(4), Counter(2), None), (Counter(8), Counter(11), Counter(11)))

    x = ((Counter(1), Counter(2), None), (Counter(3), Counter(4), Counter(5)))
    y = ((3, 0, 4), (5, 7, 6))

    def fn2_(xi, yi):
        xi.increment(yi)

    with pytest.raises(ValueError, match=re.escape('Expected None, got 4.')):
        optree.tree_map_(fn2_, x, y)


def test_tree_map_with_path_inplace():
    x = ((Counter(1), Counter(2), None), (Counter(3), Counter(4), Counter(5)))
    y = ((3, 0, None), (5, 7, 6))

    def fn1_(p, xi, yi):
        xi.increment(yi * (1 + sum(p)))

    out = optree.tree_map_with_path_(fn1_, x, y)
    assert out is x
    assert x == ((Counter(4), Counter(2), None), (Counter(13), Counter(25), Counter(29)))

    x = ((Counter(1), Counter(2), None), (Counter(3), Counter(4), Counter(5)))
    y = ((3, 0, 4), (5, 7, 6))

    def fn2_(p, xi, yi):
        xi.increment(yi * (1 + sum(p)))

    with pytest.raises(ValueError, match=re.escape('Expected None, got 4.')):
        optree.tree_map_with_path_(fn2_, x, y)


def test_tree_map_inplace_key_order():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    expected = tree.copy()
    leaves = []

    def add_leaves(x):
        leaves.append(x)
        return x

    mapped = optree.tree_map_(add_leaves, tree)
    assert mapped is tree
    assert mapped == expected
    assert list(mapped.keys()) == list(expected.keys())
    assert list(mapped.values()) == list(expected.values())
    assert list(mapped.items()) == list(expected.items())
    assert leaves == [1, 2, 3, 4]


def test_tree_map_with_path_inplace_key_order():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    expected = tree.copy()
    paths = []
    leaves = []

    def add_leaves(p, x):
        paths.append(p)
        leaves.append(x)
        return p, x

    mapped = optree.tree_map_with_path_(add_leaves, tree)
    assert mapped is tree
    assert mapped == expected
    assert list(mapped.keys()) == list(expected.keys())
    assert list(mapped.values()) == list(expected.values())
    assert list(mapped.items()) == list(expected.items())
    assert paths == [('a',), ('b',), ('c',), ('e',)]
    assert leaves == [1, 2, 3, 4]


def test_tree_map_inplace_key_order_none_is_leaf():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    expected = tree.copy()
    leaves = []

    def add_leaves(x):
        leaves.append(x)
        return x

    mapped = optree.tree_map_(add_leaves, tree, none_is_leaf=True)
    assert mapped is tree
    assert mapped == expected
    assert list(mapped.keys()) == list(expected.keys())
    assert list(mapped.values()) == list(expected.values())
    assert list(mapped.items()) == list(expected.items())
    assert leaves == [1, 2, 3, None, 4]


def test_tree_map_with_path_inplace_key_order_none_is_leaf():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    expected = tree.copy()
    paths = []
    leaves = []

    def add_leaves(p, x):
        paths.append(p)
        leaves.append(x)
        return p, x

    mapped = optree.tree_map_with_path_(add_leaves, tree, none_is_leaf=True)
    assert mapped is tree
    assert mapped == expected
    assert list(mapped.keys()) == list(expected.keys())
    assert list(mapped.values()) == list(expected.values())
    assert list(mapped.items()) == list(expected.items())
    assert paths == [('a',), ('b',), ('c',), ('d',), ('e',)]
    assert leaves == [1, 2, 3, None, 4]


def test_tree_replace_nones():
    sentinel = object()
    assert optree.tree_replace_nones(sentinel, {'a': 1, 'b': None, 'c': (2, None)}) == {
        'a': 1,
        'b': sentinel,
        'c': (2, sentinel),
    }
    assert optree.tree_replace_nones(sentinel, None) == sentinel


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_tree_transpose(tree, none_is_leaf, namespace):
    outer_treespec = optree.tree_structure(
        tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    inner_treespec = optree.tree_structure(
        [1, 1, 1],
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    nested = optree.tree_map(
        lambda x: [x, x, x],
        tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    if outer_treespec.num_leaves == 0:
        with pytest.raises(ValueError, match='Tree structures must have at least one leaf.'):
            optree.tree_transpose(outer_treespec, inner_treespec, nested)
        return
    with pytest.raises(ValueError, match='Tree structures must have the same none_is_leaf value.'):
        optree.tree_transpose(
            outer_treespec,
            optree.tree_structure(
                [1, 1, 1],
                none_is_leaf=not none_is_leaf,
                namespace=namespace,
            ),
            nested,
        )
    actual = optree.tree_transpose(outer_treespec, inner_treespec, nested)
    assert actual == [tree, tree, tree]


def test_tree_transpose_mismatch_outer():
    tree = {'a': [1, 2], 'b': [3, 4]}
    outer_treespec = optree.tree_structure({'a': 1, 'b': 2, 'c': 3})
    inner_treespec = optree.tree_structure([1, 2])
    with pytest.raises(TypeError, match='mismatch'):
        optree.tree_transpose(outer_treespec, inner_treespec, tree)


def test_tree_transpose_mismatch_inner():
    tree = {'a': [1, 2], 'b': [3, 4]}
    outer_treespec = optree.tree_structure({'a': 1, 'b': 2})
    inner_treespec = optree.tree_structure([1, 2, 3])
    with pytest.raises(TypeError, match='mismatch'):
        optree.tree_transpose(outer_treespec, inner_treespec, tree)


def test_tree_transpose_with_custom_object():
    outer_treespec = optree.tree_structure(FlatCache({'a': 1, 'b': 2}))
    inner_treespec = optree.tree_structure([1, 2])
    expected = [FlatCache({'a': 3, 'b': 5}), FlatCache({'a': 4, 'b': 6})]
    actual = optree.tree_transpose(
        outer_treespec,
        inner_treespec,
        FlatCache({'a': [3, 4], 'b': [5, 6]}),
    )
    assert actual == expected


def test_tree_transpose_with_custom_namespace():
    outer_treespec = optree.tree_structure(MyAnotherDict({'a': 1, 'b': 2}), namespace='namespace')
    inner_treespec = optree.tree_structure(
        MyAnotherDict({'c': 1, 'd': 2, 'e': 3}),
        namespace='namespace',
    )
    nested = MyAnotherDict(
        {
            'a': MyAnotherDict({'c': 1, 'd': 2, 'e': 3}),
            'b': MyAnotherDict({'c': 4, 'd': 5, 'e': 6}),
        },
    )
    actual = optree.tree_transpose(outer_treespec, inner_treespec, nested)
    assert actual == MyAnotherDict(
        {
            'c': MyAnotherDict({'a': 1, 'b': 4}),
            'd': MyAnotherDict({'a': 2, 'b': 5}),
            'e': MyAnotherDict({'a': 3, 'b': 6}),
        },
    )


def test_tree_transpose_mismatch_namespace():
    @optree.register_pytree_node_class(namespace='subnamespace')
    class MyExtraDict(MyAnotherDict):
        pass

    outer_treespec = optree.tree_structure(MyAnotherDict({'a': 1, 'b': 2}), namespace='namespace')
    inner_treespec = optree.tree_structure(
        MyExtraDict({'c': 1, 'd': 2, 'e': 3}),
        namespace='subnamespace',
    )
    nested = MyAnotherDict(
        {
            'a': MyExtraDict({'c': 1, 'd': 2, 'e': 3}),
            'b': MyExtraDict({'c': 4, 'd': 5, 'e': 6}),
        },
    )
    with pytest.raises(ValueError, match='Tree structures must have the same namespace.'):
        optree.tree_transpose(outer_treespec, inner_treespec, nested)

    optree.register_pytree_node_class(MyExtraDict, namespace='namespace')
    inner_treespec = optree.tree_structure(
        MyExtraDict({'c': 1, 'd': 2, 'e': 3}),
        namespace='namespace',
    )
    actual = optree.tree_transpose(outer_treespec, inner_treespec, nested)
    assert actual == MyExtraDict(
        {
            'c': MyAnotherDict({'a': 1, 'b': 4}),
            'd': MyAnotherDict({'a': 2, 'b': 5}),
            'e': MyAnotherDict({'a': 3, 'b': 6}),
        },
    )


def test_tree_broadcast_prefix():
    assert optree.tree_broadcast_prefix(1, [2, 3, 4]) == [1, 1, 1]
    assert optree.tree_broadcast_prefix([1, 2, 3], [4, 5, 6]) == [1, 2, 3]
    with pytest.raises(
        ValueError,
        match=re.escape('list arity mismatch; expected: 3, got: 4; list: [4, 5, 6, 7].'),
    ):
        optree.tree_broadcast_prefix([1, 2, 3], [4, 5, 6, 7])
    assert optree.tree_broadcast_prefix([1, 2, 3], [4, 5, (6, 7)]) == [1, 2, (3, 3)]
    assert optree.tree_broadcast_prefix(
        [1, 2, 3],
        [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}],
    ) == [1, 2, {'a': 3, 'b': 3, 'c': (None, 3)}]
    assert optree.tree_broadcast_prefix(
        [1, 2, 3],
        [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}],
        none_is_leaf=True,
    ) == [1, 2, {'a': 3, 'b': 3, 'c': (3, 3)}]
    assert optree.tree_broadcast_prefix(
        [1, OrderedDict(b=3, c=4, a=2)],
        [(5, 6), {'c': (None, 9), 'a': 7, 'b': 8}],
    ) == [(1, 1), OrderedDict(b=3, c=(None, 4), a=2)]
    assert optree.tree_broadcast_prefix(
        [1, OrderedDict(b=3, c=4, a=2)],
        [(5, 6), {'c': (None, 9), 'a': 7, 'b': 8}],
        none_is_leaf=True,
    ) == [(1, 1), OrderedDict(b=3, c=(4, 4), a=2)]
    assert optree.tree_broadcast_prefix(
        [1, {'c': 4, 'b': 3, 'a': 2}],
        [(5, 6), OrderedDict(b=8, c=(None, 9), a=7)],
    ) == [(1, 1), {'c': (None, 4), 'b': 3, 'a': 2}]
    assert optree.tree_broadcast_prefix(
        [1, {'c': 4, 'b': 3, 'a': 2}],
        [(5, 6), OrderedDict(b=8, c=(None, 9), a=7)],
        none_is_leaf=True,
    ) == [(1, 1), {'c': (4, 4), 'b': 3, 'a': 2}]


def test_broadcast_prefix():
    assert optree.broadcast_prefix(1, [2, 3, 4]) == [1, 1, 1]
    assert optree.broadcast_prefix([1, 2, 3], [4, 5, 6]) == [1, 2, 3]
    with pytest.raises(
        ValueError,
        match=re.escape('list arity mismatch; expected: 3, got: 4; list: [4, 5, 6, 7].'),
    ):
        optree.broadcast_prefix([1, 2, 3], [4, 5, 6, 7])
    assert optree.broadcast_prefix([1, 2, 3], [4, 5, (6, 7)]) == [1, 2, 3, 3]
    assert optree.broadcast_prefix(
        [1, 2, 3],
        [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}],
    ) == [1, 2, 3, 3, 3]
    assert optree.broadcast_prefix(
        [1, 2, 3],
        [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}],
        none_is_leaf=True,
    ) == [1, 2, 3, 3, 3, 3]
    assert optree.broadcast_prefix(
        [1, OrderedDict(b=3, c=4, a=2)],
        [(5, 6), {'c': (None, 9), 'a': 7, 'b': 8}],
    ) == [1, 1, 3, 4, 2]
    assert optree.broadcast_prefix(
        [1, OrderedDict(b=3, c=4, a=2)],
        [(5, 6), {'c': (None, 9), 'a': 7, 'b': 8}],
        none_is_leaf=True,
    ) == [1, 1, 3, 4, 4, 2]
    assert optree.broadcast_prefix(
        [1, {'c': 4, 'b': 3, 'a': 2}],
        [(5, 6), OrderedDict(b=8, c=(None, 9), a=7)],
    ) == [1, 1, 2, 3, 4]
    assert optree.broadcast_prefix(
        [1, {'c': 4, 'b': 3, 'a': 2}],
        [(5, 6), OrderedDict(b=8, c=(None, 9), a=7)],
        none_is_leaf=True,
    ) == [1, 1, 2, 3, 4, 4]


def test_tree_broadcast_common():
    assert optree.tree_broadcast_common(1, [2, 3, 4]) == ([1, 1, 1], [2, 3, 4])
    assert optree.tree_broadcast_common([1, 2, 3], [4, 5, 6]) == ([1, 2, 3], [4, 5, 6])
    with pytest.raises(
        ValueError,
        match=re.escape('list arity mismatch; expected: 3, got: 4.'),
    ):
        optree.tree_broadcast_common([1, 2, 3], [1, 2, 3, 4])
    assert optree.tree_broadcast_common([1, 2, 3], [4, 5, (6, 7)]) == (
        [1, 2, (3, 3)],
        [4, 5, (6, 7)],
    )
    assert optree.tree_broadcast_common([1, (2, 3), 4], [5, 6, (7, 8)]) == (
        [1, (2, 3), (4, 4)],
        [5, (6, 6), (7, 8)],
    )
    assert optree.tree_broadcast_common(
        [1, (2, 3), 4],
        [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}],
    ) == (
        [1, (2, 3), {'a': 4, 'b': 4, 'c': (None, 4)}],
        [5, (6, 6), {'a': 7, 'b': 8, 'c': (None, 9)}],
    )
    assert optree.tree_broadcast_common(
        [1, OrderedDict(foo=2, bar=3), 4],
        [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}],
    ) == (
        [1, OrderedDict(foo=2, bar=3), {'a': 4, 'b': 4, 'c': (None, 4)}],
        [5, OrderedDict(foo=6, bar=6), {'a': 7, 'b': 8, 'c': (None, 9)}],
    )
    assert optree.tree_broadcast_common(
        [1, OrderedDict(foo=2, bar=3), 4],
        [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}],
        none_is_leaf=True,
    ) == (
        [1, OrderedDict(foo=2, bar=3), {'a': 4, 'b': 4, 'c': (4, 4)}],
        [5, OrderedDict(foo=6, bar=6), {'a': 7, 'b': 8, 'c': (None, 9)}],
    )
    assert optree.tree_broadcast_common(
        [1, OrderedDict(foo=2, bar=3), 4],
        [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}],
        none_is_leaf=True,
    ) == (
        [1, OrderedDict(foo=2, bar=3), {'a': 4, 'b': 4, 'c': (4, 4)}],
        [5, OrderedDict(foo=6, bar=6), {'a': 7, 'b': 8, 'c': (None, 9)}],
    )
    assert optree.tree_broadcast_common(
        [1, OrderedDict(b=4, c=5, a=(2, 3))],
        [(6, 7), {'c': (None, 0), 'a': 8, 'b': 9}],
    ) == (
        [(1, 1), OrderedDict(b=4, c=(None, 5), a=(2, 3))],
        [(6, 7), {'c': (None, 0), 'a': (8, 8), 'b': 9}],
    )
    assert optree.tree_broadcast_common(
        [1, OrderedDict(b=4, c=5, a=(2, 3))],
        [(6, 7), {'c': (None, 0), 'a': 8, 'b': 9}],
        none_is_leaf=True,
    ) == (
        [(1, 1), OrderedDict(b=4, c=(5, 5), a=(2, 3))],
        [(6, 7), {'c': (None, 0), 'a': (8, 8), 'b': 9}],
    )
    assert optree.tree_broadcast_common(
        [1, {'c': (None, 4), 'b': 3, 'a': 2}],
        [(5, 6), OrderedDict(b=9, c=0, a=(7, 8))],
    ) == (
        [(1, 1), {'c': (None, 4), 'b': 3, 'a': (2, 2)}],
        [(5, 6), OrderedDict(b=9, c=(None, 0), a=(7, 8))],
    )
    assert optree.tree_broadcast_common(
        [1, {'b': 3, 'a': 2, 'c': (None, 4)}],
        [(5, 6), OrderedDict(b=9, c=0, a=(7, 8))],
        none_is_leaf=True,
    ) == (
        [(1, 1), {'c': (None, 4), 'b': 3, 'a': (2, 2)}],
        [(5, 6), OrderedDict(b=9, c=(0, 0), a=(7, 8))],
    )


def test_broadcast_common():
    assert optree.broadcast_common(1, [2, 3, 4]) == ([1, 1, 1], [2, 3, 4])
    assert optree.broadcast_common([1, 2, 3], [4, 5, 6]) == ([1, 2, 3], [4, 5, 6])
    with pytest.raises(
        ValueError,
        match=re.escape('list arity mismatch; expected: 3, got: 4.'),
    ):
        optree.broadcast_common([1, 2, 3], [1, 2, 3, 4])
    assert optree.broadcast_common([1, 2, 3], [4, 5, (6, 7)]) == (
        [1, 2, 3, 3],
        [4, 5, 6, 7],
    )
    assert optree.broadcast_common([1, (2, 3), 4], [5, 6, (7, 8)]) == (
        [1, 2, 3, 4, 4],
        [5, 6, 6, 7, 8],
    )
    assert optree.broadcast_common(
        [1, (2, 3), 4],
        [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}],
    ) == (
        [1, 2, 3, 4, 4, 4],
        [5, 6, 6, 7, 8, 9],
    )
    assert optree.broadcast_common(
        [1, (2, 3), 4],
        [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}],
        none_is_leaf=True,
    ) == (
        [1, 2, 3, 4, 4, 4, 4],
        [5, 6, 6, 7, 8, None, 9],
    )
    assert optree.broadcast_common(
        [1, OrderedDict(foo=2, bar=3), 4],
        [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}],
    ) == (
        [1, 2, 3, 4, 4, 4],
        [5, 6, 6, 7, 8, 9],
    )
    assert optree.broadcast_common(
        [1, OrderedDict(foo=2, bar=3), 4],
        [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}],
        none_is_leaf=True,
    ) == (
        [1, 2, 3, 4, 4, 4, 4],
        [5, 6, 6, 7, 8, None, 9],
    )
    assert optree.broadcast_common(
        [1, OrderedDict(b=4, c=5, a=(2, 3))],
        [(6, 7), {'c': (None, 0), 'a': 8, 'b': 9}],
    ) == (
        [1, 1, 4, 5, 2, 3],
        [6, 7, 9, 0, 8, 8],
    )
    assert optree.broadcast_common(
        [1, OrderedDict(b=4, c=5, a=(2, 3))],
        [(6, 7), {'c': (None, 0), 'a': 8, 'b': 9}],
        none_is_leaf=True,
    ) == (
        [1, 1, 4, 5, 5, 2, 3],
        [6, 7, 9, None, 0, 8, 8],
    )
    assert optree.broadcast_common(
        [1, {'c': (None, 4), 'b': 3, 'a': 2}],
        [(5, 6), OrderedDict(b=9, c=0, a=(7, 8))],
    ) == (
        [1, 1, 2, 2, 3, 4],
        [5, 6, 7, 8, 9, 0],
    )
    assert optree.broadcast_common(
        [1, {'b': 3, 'a': 2, 'c': (None, 4)}],
        [(5, 6), OrderedDict(b=9, c=0, a=(7, 8))],
        none_is_leaf=True,
    ) == (
        [1, 1, 2, 2, 3, None, 4],
        [5, 6, 7, 8, 9, 0, 0],
    )


def test_tree_reduce():
    assert optree.tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, 3)}) == 6
    assert optree.tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, None), 'z': 3}) == 6
    assert optree.tree_reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3}) == 3
    assert (
        optree.tree_reduce(
            lambda x, y: x and y,
            {'x': 1, 'y': (2, None), 'z': 3},
            none_is_leaf=True,
        )
        is None
    )
    assert (
        optree.tree_reduce(
            lambda x, y: x and y,
            {'x': 1, 'y': (2, None), 'z': 3},
            False,
            none_is_leaf=True,
        )
        is False
    )


def test_tree_sum():
    assert optree.tree_sum({'x': 1, 'y': (2, 3)}) == 6
    assert optree.tree_sum({'x': 1, 'y': (2, None), 'z': 3}) == 6
    with pytest.raises(
        TypeError,
        match=re.escape("unsupported operand type(s) for +: 'int' and 'NoneType'"),
    ):
        optree.tree_sum({'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    assert optree.tree_sum({'x': 'a', 'y': ('b', None), 'z': 'c'}, start='') == 'abc'
    assert optree.tree_sum({'x': b'a', 'y': (b'b', None), 'z': b'c'}, start=b'') == b'abc'
    assert optree.tree_sum(
        {'x': [1], 'y': ([2], [None]), 'z': [3]},
        start=[],
        is_leaf=lambda x: isinstance(x, list),
    ) == [1, 2, None, 3]


def test_tree_max():
    with pytest.raises(ValueError, match='empty'):
        optree.tree_max({})
    assert optree.tree_max({}, default=0) == 0
    assert optree.tree_max({'x': 0, 'y': (2, 1)}) == 2
    assert optree.tree_max({'x': 0, 'y': (2, 1)}, key=lambda x: -x) == 0
    with pytest.raises(ValueError, match='empty'):
        optree.tree_max({'a': None})
    assert optree.tree_max({'a': None}, default=0) == 0
    assert optree.tree_max({'a': None}, none_is_leaf=True) is None
    with pytest.raises(ValueError, match='empty'):
        optree.tree_max(None)
    assert optree.tree_max(None, default=0) == 0
    assert optree.tree_max(None, none_is_leaf=True) is None
    assert optree.tree_max(None, default=0, key=lambda x: -x) == 0
    with pytest.raises(TypeError, match=re.escape("bad operand type for unary -: 'NoneType'")):
        assert optree.tree_max(None, default=0, key=lambda x: -x, none_is_leaf=True) is None


def test_tree_min():
    with pytest.raises(ValueError, match='empty'):
        optree.tree_min({})
    assert optree.tree_min({}, default=0) == 0
    assert optree.tree_min({'x': 0, 'y': (2, 1)}) == 0
    assert optree.tree_min({'x': 0, 'y': (2, 1)}, key=lambda x: -x) == 2
    with pytest.raises(ValueError, match='empty'):
        optree.tree_min({'a': None})
    assert optree.tree_min({'a': None}, default=0) == 0
    assert optree.tree_min({'a': None}, none_is_leaf=True) is None
    with pytest.raises(ValueError, match='empty'):
        optree.tree_min(None)
    assert optree.tree_min(None, default=0) == 0
    assert optree.tree_min(None, none_is_leaf=True) is None
    assert optree.tree_min(None, default=0, key=lambda x: -x) == 0
    with pytest.raises(TypeError, match=re.escape("bad operand type for unary -: 'NoneType'")):
        assert optree.tree_min(None, default=0, key=lambda x: -x, none_is_leaf=True) is None


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


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_tree_flatten_one_level(tree, none_is_leaf, namespace):  # noqa: C901
    stack = [tree]
    actual_leaves = []
    expected_leaves = optree.tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    while stack:
        node = stack.pop()
        counter = itertools.count()
        expected_children, one_level_treespec = optree.tree_flatten(
            node,
            is_leaf=lambda x: next(counter) > 0,  # noqa: B023
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        node_type = type(node)
        if one_level_treespec.is_leaf():
            assert expected_children == [node]
            assert one_level_treespec.kind == optree.PyTreeKind.LEAF
            with pytest.raises(
                ValueError,
                match=re.escape(f'Cannot flatten leaf-type: {node_type} (node: {node!r}).'),
            ):
                optree.tree_flatten_one_level(node, none_is_leaf=none_is_leaf, namespace=namespace)
            actual_leaves.append(node)
        else:
            children, metadata, entries, unflatten_func = optree.tree_flatten_one_level(
                node,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            assert children == expected_children
            if node_type in (type(None), tuple, list):
                assert metadata is None
                if node_type is tuple:
                    assert one_level_treespec.kind == optree.PyTreeKind.TUPLE
                elif node_type is list:
                    assert one_level_treespec.kind == optree.PyTreeKind.LIST
                else:
                    assert one_level_treespec.kind == optree.PyTreeKind.NONE
            elif node_type is dict:
                assert metadata == sorted(node.keys())
                assert one_level_treespec.kind == optree.PyTreeKind.DICT
            elif node_type is OrderedDict:
                assert metadata == list(node.keys())
                assert one_level_treespec.kind == optree.PyTreeKind.ORDEREDDICT
            elif node_type is defaultdict:
                assert metadata == (node.default_factory, sorted(node.keys()))
                assert one_level_treespec.kind == optree.PyTreeKind.DEFAULTDICT
            elif node_type is deque:
                assert metadata == node.maxlen
                assert one_level_treespec.kind == optree.PyTreeKind.DEQUE
            elif optree.is_structseq(node):
                assert optree.is_structseq_class(node_type)
                assert isinstance(node, optree.typing.structseq)
                assert issubclass(node_type, optree.typing.structseq)
                assert metadata is node_type
                assert one_level_treespec.kind == optree.PyTreeKind.STRUCTSEQUENCE
            elif optree.is_namedtuple(node):
                assert optree.is_namedtuple_class(node_type)
                assert metadata is node_type
                assert one_level_treespec.kind == optree.PyTreeKind.NAMEDTUPLE
            else:
                assert one_level_treespec.kind == optree.PyTreeKind.CUSTOM
            assert len(entries) == len(children)
            if hasattr(node, '__getitem__'):
                for child, entry in zip(children, entries):
                    assert node[entry] is child

            assert unflatten_func(metadata, children) == node
            if node_type is type(None):
                assert unflatten_func(metadata, []) is None
                with pytest.raises(ValueError, match=re.escape('Expected no children.')):
                    unflatten_func(metadata, range(1))

            stack.extend(reversed(children))

    assert actual_leaves == expected_leaves


@parametrize(
    tree=[
        optree.Partial(dummy_func),
        optree.Partial(dummy_func, 1, 2),
        optree.Partial(dummy_func, x='a'),
        optree.Partial(dummy_func, 1, 2, 3, x=4, y=5),
        optree.Partial(dummy_func, 1, None, x=4, y=5, z=None),
        optree.Partial(dummy_partial_func, 1, 2, 3, x=4, y=5),
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
    def f(a=None, b=None, c=None):
        return a, b, c

    g = functools.partial(f, 2)
    h = optree.Partial(g, 3)
    assert h.args == (3,)
    assert g() == (2, None, None)
    assert h() == (2, 3, None)


def test_partial_func_attribute_has_stable_hash():
    fun = functools.partial(print, 1)
    p1 = optree.Partial(fun, 2)
    p2 = optree.Partial(fun, 2)
    assert p1.func == fun  # pylint: disable=comparison-with-callable
    assert fun == p1.func  # pylint: disable=comparison-with-callable
    assert p1.func == p2.func
    assert hash(p1.func) == hash(p2.func)
