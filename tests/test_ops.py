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

# pylint: disable=missing-function-docstring,invalid-name,wrong-import-order

import copy
import functools
import itertools
import operator
import os
import pickle
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict, deque

import pytest

import optree
from helpers import (
    GLOBAL_NAMESPACE,
    IS_LEAF_FUNCTIONS,
    LEAVES,
    TEST_ROOT,
    TREE_ACCESSORS,
    TREE_PATHS,
    TREES,
    Counter,
    CustomTuple,
    FlatCache,
    MyAnotherDict,
    always,
    assert_equal_type_and_value,
    is_list,
    is_none,
    is_tuple,
    never,
    parametrize,
)


def test_import_no_warnings():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(('PYTHON', 'PYTEST', 'COV_'))
    }
    assert (
        subprocess.check_output(
            [
                sys.executable,
                '-W',
                'always',
                '-W',
                'error',
                '-c',
                'import optree',
            ],
            stderr=subprocess.STDOUT,
            text=True,
            cwd=TEST_ROOT,
            env=env,
        )
        == ''
    )


def test_max_depth():
    lst = [1]
    for _ in range(optree.MAX_RECURSION_DEPTH - 1):
        lst = [lst]
    list(optree.tree_iter(lst))
    optree.tree_flatten(lst)
    optree.tree_flatten_with_path(lst)
    optree.tree_flatten_with_accessor(lst)

    lst = [lst]
    with pytest.raises(
        RecursionError,
        match=re.escape('Maximum recursion depth exceeded during flattening the tree.'),
    ):
        list(optree.tree_iter(lst))
    with pytest.raises(
        RecursionError,
        match=re.escape('Maximum recursion depth exceeded during flattening the tree.'),
    ):
        optree.tree_flatten(lst)
    with pytest.raises(
        RecursionError,
        match=re.escape('Maximum recursion depth exceeded during flattening the tree.'),
    ):
        optree.tree_flatten_with_path(lst)
    with pytest.raises(
        RecursionError,
        match=re.escape('Maximum recursion depth exceeded during flattening the tree.'),
    ):
        optree.tree_flatten_with_accessor(lst)


@parametrize(
    tree=list(TREES + LEAVES),
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_round_trip(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        leaves, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        actual = optree.tree_unflatten(treespec, leaves)
        assert actual == tree


@parametrize(
    tree=list(TREES + LEAVES),
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_round_trip_with_flatten_up_to(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        _, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        leaves = treespec.flatten_up_to(tree)
        actual = optree.tree_unflatten(treespec, leaves)
        assert actual == tree
        assert leaves == [accessor(tree) for accessor in optree.treespec_accessors(treespec)]


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
        with pytest.raises(ValueError, match='Too few leaves for PyTreeSpec'):
            optree.tree_unflatten(treespec, leaves[:-1])
    with pytest.raises(ValueError, match='Too many leaves for PyTreeSpec'):
        optree.tree_unflatten(treespec, (*leaves, 0))


@parametrize(
    tree=list(TREES + LEAVES),
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_iter(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        leaves = optree.tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        it = optree.tree_iter(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        assert iter(it) is it
        assert list(it) == leaves
        with pytest.raises(StopIteration):
            next(it)


def test_traverse():
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

    def unflatten_node(node):
        return node

    def get_functions():
        nodes_visited = []
        leaves_visited = []

        def f_node(node):
            nodes_visited.append(node)
            return copy.deepcopy(nodes_visited), None

        def f_leaf(leaf):
            leaves_visited.append(leaf)
            return copy.deepcopy(leaves_visited)

        return f_node, f_leaf, leaves_visited, nodes_visited

    leaves, treespec = optree.tree_flatten(tree)

    f_node, f_leaf, *_ = get_functions()
    with pytest.raises(ValueError, match=re.escape('Too few leaves for PyTreeSpec.')):
        treespec.traverse(leaves[:-1], f_node, f_leaf)

    f_node, f_leaf, *_ = get_functions()
    with pytest.raises(ValueError, match=re.escape('Too many leaves for PyTreeSpec.')):
        treespec.traverse((*leaves, 0), f_node, f_leaf)

    f_node, f_leaf, leaves_visited, nodes_visited = get_functions()
    output = treespec.traverse(leaves, f_node, f_leaf)
    assert leaves_visited == [1, 2, 3, 4]
    assert nodes_visited == [
        None,
        {'f': ([None], None), 'e': [1, 2, 3], 'g': [1, 2, 3, 4]},
        {
            'b': [1, 2],
            'a': [1],
            'c': ([None, {'f': ([None], None), 'e': [1, 2, 3], 'g': [1, 2, 3, 4]}], None),
        },
    ]
    assert output == (
        [
            None,
            {'f': ([None], None), 'e': [1, 2, 3], 'g': [1, 2, 3, 4]},
            {
                'b': [1, 2],
                'a': [1],
                'c': ([None, {'f': ([None], None), 'e': [1, 2, 3], 'g': [1, 2, 3, 4]}], None),
            },
        ],
        None,
    )

    assert treespec.traverse(leaves) == tree
    assert treespec.traverse(leaves, unflatten_node, None) == tree
    assert treespec.traverse(leaves, None, lambda x: x + 1) == optree.tree_map(
        lambda x: x + 1,
        tree,
    )

    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=True)

    f_node, f_leaf, *_ = get_functions()
    with pytest.raises(ValueError, match=re.escape('Too few leaves for PyTreeSpec.')):
        treespec.traverse(leaves[:-1], f_node, f_leaf)

    f_node, f_leaf, *_ = get_functions()
    with pytest.raises(ValueError, match=re.escape('Too many leaves for PyTreeSpec.')):
        treespec.traverse((*leaves, 0), f_node, f_leaf)

    f_node, f_leaf, leaves_visited, nodes_visited = get_functions()
    output = treespec.traverse(leaves, f_node, f_leaf)
    assert leaves_visited == [1, 2, 3, None, 4]
    assert nodes_visited == [
        {'f': [1, 2, 3, None], 'e': [1, 2, 3], 'g': [1, 2, 3, None, 4]},
        {
            'b': [1, 2],
            'a': [1],
            'c': ([{'f': [1, 2, 3, None], 'e': [1, 2, 3], 'g': [1, 2, 3, None, 4]}], None),
        },
    ]
    assert output == (
        [
            {'f': [1, 2, 3, None], 'e': [1, 2, 3], 'g': [1, 2, 3, None, 4]},
            {
                'b': [1, 2],
                'a': [1],
                'c': ([{'f': [1, 2, 3, None], 'e': [1, 2, 3], 'g': [1, 2, 3, None, 4]}], None),
            },
        ],
        None,
    )

    assert treespec.traverse(leaves) == tree
    assert treespec.traverse(leaves, unflatten_node, None) == tree
    assert treespec.traverse(leaves, None, lambda x: (x,)) == optree.tree_map(
        lambda x: (x,),
        tree,
        none_is_leaf=True,
    )


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

    def unflatten_node(node_type, node_data, children):
        return optree.register_pytree_node.get(node_type).unflatten_func(node_data, children)

    def get_functions():
        node_types_visited = []
        node_data_visited = []
        nodes_visited = []
        leaves_visited = []

        def f_node(node_type, node_data, node):
            node_types_visited.append(node_type)
            node_data_visited.append(node_data)
            nodes_visited.append(node)
            return copy.deepcopy(nodes_visited), None

        def f_leaf(leaf):
            leaves_visited.append(leaf)
            return copy.deepcopy(leaves_visited)

        return f_node, f_leaf, leaves_visited, nodes_visited, node_data_visited, node_types_visited

    leaves, treespec = optree.tree_flatten(tree)

    f_node, f_leaf, *_ = get_functions()
    with pytest.raises(ValueError, match=re.escape('Too few leaves for PyTreeSpec.')):
        treespec.walk(leaves[:-1], f_node, f_leaf)

    f_node, f_leaf, *_ = get_functions()
    with pytest.raises(ValueError, match=re.escape('Too many leaves for PyTreeSpec.')):
        treespec.walk((*leaves, 0), f_node, f_leaf)

    (
        f_node,
        f_leaf,
        leaves_visited,
        nodes_visited,
        node_data_visited,
        node_types_visited,
    ) = get_functions()
    output = treespec.walk(leaves, f_node, f_leaf)
    assert leaves_visited == [1, 2, 3, 4]
    assert nodes_visited == [
        (),
        ([1, 2, 3], ([()], None), [1, 2, 3, 4]),
        ([1], [1, 2], ([(), ([1, 2, 3], ([()], None), [1, 2, 3, 4])], None)),
    ]
    assert node_data_visited == [None, ['e', 'f', 'g'], ['a', 'b', 'c']]
    assert node_types_visited == [type(None), dict, dict]
    assert output == (
        [
            (),
            ([1, 2, 3], ([()], None), [1, 2, 3, 4]),
            ([1], [1, 2], ([(), ([1, 2, 3], ([()], None), [1, 2, 3, 4])], None)),
        ],
        None,
    )

    assert treespec.walk(leaves) == tree
    assert treespec.walk(leaves, unflatten_node, None) == tree
    assert treespec.walk(leaves, None, lambda x: x + 1) == optree.tree_map(lambda x: x + 1, tree)

    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=True)

    f_node, f_leaf, *_ = get_functions()
    with pytest.raises(ValueError, match=re.escape('Too few leaves for PyTreeSpec.')):
        treespec.walk(leaves[:-1], f_node, f_leaf)

    f_node, f_leaf, *_ = get_functions()
    with pytest.raises(ValueError, match=re.escape('Too many leaves for PyTreeSpec.')):
        treespec.walk((*leaves, 0), f_node, f_leaf)

    (
        f_node,
        f_leaf,
        leaves_visited,
        nodes_visited,
        node_data_visited,
        node_types_visited,
    ) = get_functions()
    output = treespec.walk(leaves, f_node, f_leaf)
    assert leaves_visited == [1, 2, 3, None, 4]
    assert nodes_visited == [
        ([1, 2, 3], [1, 2, 3, None], [1, 2, 3, None, 4]),
        ([1], [1, 2], ([([1, 2, 3], [1, 2, 3, None], [1, 2, 3, None, 4])], None)),
    ]
    assert node_data_visited == [['e', 'f', 'g'], ['a', 'b', 'c']]
    assert node_types_visited == [dict, dict]
    assert output == (
        [
            ([1, 2, 3], [1, 2, 3, None], [1, 2, 3, None, 4]),
            ([1], [1, 2], ([([1, 2, 3], [1, 2, 3, None], [1, 2, 3, None, 4])], None)),
        ],
        None,
    )

    assert treespec.walk(leaves) == tree
    assert treespec.walk(leaves, unflatten_node, None) == tree
    assert treespec.walk(leaves, None, lambda x: (x,)) == optree.tree_map(
        lambda x: (x,),
        tree,
        none_is_leaf=True,
    )


def test_flatten_up_to():
    treespec = optree.tree_structure([(1, 2), None, CustomTuple(foo=3, bar=7)])
    tree = [({'foo': 7}, (3, 4)), None, CustomTuple(foo=(11, 9), bar=None)]
    subtrees = treespec.flatten_up_to(tree)
    assert subtrees == [{'foo': 7}, (3, 4), (11, 9), None]
    assert subtrees == [accessor(tree) for accessor in optree.treespec_accessors(treespec)]


def test_flatten_up_to_none_is_leaf():
    treespec = optree.tree_structure([(1, 2), None, CustomTuple(foo=3, bar=7)], none_is_leaf=True)
    tree = [({'foo': 7}, (3, 4)), None, CustomTuple(foo=(11, 9), bar=None)]
    subtrees = treespec.flatten_up_to(tree)
    assert subtrees == [{'foo': 7}, (3, 4), None, (11, 9), None]
    assert subtrees == [accessor(tree) for accessor in optree.treespec_accessors(treespec)]


@parametrize(
    leaves_fn=[
        optree.tree_leaves,
        lambda tree, is_leaf: list(optree.tree_iter(tree, is_leaf)),
        lambda tree, is_leaf: optree.tree_flatten(tree, is_leaf)[0],
        lambda tree, is_leaf: optree.tree_flatten_with_path(tree, is_leaf)[1],
        lambda tree, is_leaf: optree.tree_flatten_with_accessor(tree, is_leaf)[1],
    ],
)
def test_flatten_is_leaf(leaves_fn):
    x = [(1, 2), (3, 4), (5, 6)]
    leaves = leaves_fn(x, is_leaf=never)
    assert leaves == [1, 2, 3, 4, 5, 6]
    leaves = leaves_fn(x, is_leaf=is_tuple)
    assert leaves == x
    leaves = leaves_fn(x, is_leaf=is_list)
    assert leaves == [x]
    leaves = leaves_fn(x, is_leaf=always)
    assert leaves == [x]

    y = [[[(1,)], [[(2,)], {'a': (3,)}]]]
    leaves = leaves_fn(y, is_leaf=is_tuple)
    assert leaves == [(1,), (2,), (3,)]

    z = [(1, 2), (3, 4), None, (5, None)]
    leaves = leaves_fn(z, is_leaf=is_none)
    assert leaves == [1, 2, 3, 4, None, 5, None]


@parametrize(
    structure_fn=[
        optree.tree_structure,
        lambda tree, is_leaf: optree.tree_flatten(tree, is_leaf)[1],
        lambda tree, is_leaf: optree.tree_flatten_with_path(tree, is_leaf)[2],
        lambda tree, is_leaf: optree.tree_flatten_with_accessor(tree, is_leaf)[2],
    ],
)
def test_structure_is_leaf(structure_fn):
    x = [(1, 2), (3, 4), (5, 6)]
    treespec = structure_fn(x, is_leaf=never)
    assert treespec.num_leaves == 6
    treespec = structure_fn(x, is_leaf=is_tuple)
    assert treespec.num_leaves == 3
    treespec = structure_fn(x, is_leaf=is_list)
    assert treespec.num_leaves == 1
    treespec = structure_fn(x, is_leaf=always)
    assert treespec.num_leaves == 1

    y = [[[(1,)], [[(2,)], {'a': (3,)}]]]
    treespec = structure_fn(y, is_leaf=is_tuple)
    assert treespec.num_leaves == 3


@parametrize(
    data=list(
        itertools.chain(
            zip(TREES, TREE_PATHS[False], TREE_ACCESSORS[False], itertools.repeat(False)),
            zip(TREES, TREE_PATHS[True], TREE_ACCESSORS[True], itertools.repeat(True)),
        ),
    ),
)
def test_paths_and_accessors(data):
    tree, expected_paths, expected_accessors, none_is_leaf = data
    expected_leaves, expected_treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)
    paths, leaves, treespec = optree.tree_flatten_with_path(tree, none_is_leaf=none_is_leaf)
    accessors, other_leaves, other_treespec = optree.tree_flatten_with_accessor(
        tree,
        none_is_leaf=none_is_leaf,
    )
    assert len(paths) == len(leaves)
    assert len(accessors) == len(leaves)
    assert leaves == expected_leaves
    assert treespec == expected_treespec
    assert other_leaves == expected_leaves
    assert other_treespec == expected_treespec
    assert paths == expected_paths
    assert accessors == expected_accessors
    for leaf, accessor, path in zip(leaves, accessors, paths):
        assert isinstance(accessor, optree.PyTreeAccessor)
        assert isinstance(path, tuple)
        assert len(accessor) == len(path)
        assert all(
            isinstance(e, optree.PyTreeEntry)
            and isinstance(e.type, type)
            and isinstance(e.kind, optree.PyTreeKind)
            for e in accessor
        )
        assert accessor.path == path
        assert tuple(e.entry for e in accessor) == path
        assert accessor(tree) == leaf
        if all(e.__class__.codify is not optree.PyTreeEntry.codify for e in accessor):
            # pylint: disable-next=eval-used
            assert eval(accessor.codify('__tree'), {'__tree': tree}, {}) == leaf
            # pylint: disable-next=eval-used
            assert eval(f'lambda __tree: {accessor.codify("__tree")}', {}, {})(tree) == leaf
        else:
            assert 'flat index' in accessor.codify('')

    assert optree.treespec_paths(treespec) == expected_paths
    assert optree.treespec_accessors(treespec) == expected_accessors
    assert optree.tree_paths(tree, none_is_leaf=none_is_leaf) == expected_paths
    assert optree.tree_accessors(tree, none_is_leaf=none_is_leaf) == expected_accessors


@parametrize(
    tree=TREES,
    is_leaf=IS_LEAF_FUNCTIONS,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_paths_and_accessors_with_is_leaf(
    tree,
    is_leaf,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
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
        accessors, other_leaves, other_treespec = optree.tree_flatten_with_accessor(
            tree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert len(paths) == len(leaves)
        assert len(accessors) == len(leaves)
        assert leaves == expected_leaves
        assert treespec == expected_treespec
        assert other_leaves == expected_leaves
        assert other_treespec == expected_treespec
        for leaf, accessor, path in zip(leaves, accessors, paths):
            assert isinstance(accessor, optree.PyTreeAccessor)
            assert isinstance(path, tuple)
            assert len(accessor) == len(path)
            assert all(
                isinstance(e, optree.PyTreeEntry)
                and isinstance(e.type, type)
                and isinstance(e.kind, optree.PyTreeKind)
                for e in accessor
            )
            assert accessor.path == path
            assert tuple(e.entry for e in accessor) == path
            assert accessor(tree) == leaf
            if all(e.__class__.codify is not optree.PyTreeEntry.codify for e in accessor):
                # pylint: disable-next=eval-used
                assert eval(accessor.codify('__tree'), {'__tree': tree}, {}) == leaf
                # pylint: disable-next=eval-used
                assert eval(f'lambda __tree: {accessor.codify("__tree")}', {}, {})(tree) == leaf
            else:
                assert 'flat index' in accessor.codify('')

        assert optree.treespec_paths(treespec) == paths
        assert optree.treespec_accessors(treespec) == accessors
        assert (
            optree.tree_paths(
                tree,
                is_leaf=is_leaf,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            == paths
        )
        assert (
            optree.tree_accessors(
                tree,
                is_leaf=is_leaf,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            == accessors
        )


@parametrize(
    tree=TREES,
    is_leaf=IS_LEAF_FUNCTIONS,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_round_trip_is_leaf(
    tree,
    is_leaf,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_is_leaf_with_trees(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_is_leaf_with_leaves(
    leaf,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        assert optree.tree_is_leaf(leaf, none_is_leaf=none_is_leaf, namespace=namespace)


@parametrize(
    tree=TREES,
    is_leaf=IS_LEAF_FUNCTIONS,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_is_leaf_with_is_leaf(
    tree,
    is_leaf,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_all_leaves_with_trees(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        leaves = optree.tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        assert optree.all_leaves(leaves, none_is_leaf=none_is_leaf, namespace=namespace)
        if [tree] != leaves:
            assert not optree.all_leaves([tree], none_is_leaf=none_is_leaf, namespace=namespace)


@parametrize(
    leaf=LEAVES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_all_leaves_with_leaves(
    leaf,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        assert optree.all_leaves([leaf], none_is_leaf=none_is_leaf, namespace=namespace)


@parametrize(
    tree=TREES,
    is_leaf=IS_LEAF_FUNCTIONS,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_all_leaves_with_is_leaf(
    tree,
    is_leaf,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
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
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 10]))
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
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 10]))
    with pytest.raises(ValueError, match=re.escape('Expected None, got 7.')):
        optree.tree_map_with_path(lambda *xs: tuple(xs), x, y)


def test_tree_map_with_accessor():
    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map_with_accessor(lambda *xs: tuple(xs), x, y)
    assert out == (
        (
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                1,
                [6],
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                2,
                None,
            ),
            None,
        ),
        (
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                3,
                {'foo': 'bar'},
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                4,
                7,
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                5,
                [8, 9],
            ),
        ),
    )

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 10]))
    with pytest.raises(ValueError, match=re.escape('Expected None, got 7.')):
        optree.tree_map_with_accessor(lambda *xs: tuple(xs), x, y)


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

    tree1 = [(1, 2, 3), 4, 5, OrderedDict([('y', 7), ('x', 6)])]
    tree2 = [8, [9, 10, 11], 12, {'x': 13, 'y': 14}]
    tree3 = 15
    tree4 = [16, 17, {'a': 18, 'b': 19, 'c': 20}, 21]
    out = optree.tree_broadcast_map(
        lambda *args: functools.reduce(operator.mul, args, 1),
        tree1,
        tree2,
        tree3,
        tree4,
    )
    assert out == [
        (1920, 3840, 5760),
        [9180, 10200, 11220],
        {'a': 16200, 'b': 17100, 'c': 18000},
        OrderedDict([('y', 30870), ('x', 24570)]),
    ]
    for trees in itertools.permutations([tree1, tree2, tree3, tree4], 4):
        new_out = optree.tree_broadcast_map(
            lambda *args: functools.reduce(operator.mul, args, 1),
            *trees,
        )
        assert new_out == out
        if trees.index(tree1) < trees.index(tree2):
            assert type(new_out[-1]) is OrderedDict
        else:
            assert type(new_out[-1]) is dict


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


def test_tree_broadcast_map_with_accessor():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_broadcast_map_with_accessor(lambda *xs: tuple(xs), x, y)
    assert out == (
        (
            [
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    1,
                    7,
                ),
            ],
            None,
            None,
        ),
        (
            {
                'foo': (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.MappingEntry('foo', dict, optree.PyTreeKind.DICT),
                        ),
                    ),
                    3,
                    'bar',
                ),
            },
            (
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        ),
                    ),
                    4,
                    8,
                ),
                [
                    (
                        optree.PyTreeAccessor(
                            (
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                            ),
                        ),
                        5,
                        8,
                    ),
                ],
            ),
            [
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    6,
                    9,
                ),
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    6,
                    10,
                ),
            ],
        ),
    )

    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, 8), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map_with_accessor(lambda *xs: tuple(xs), x, y)
    assert out == (
        (
            [
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    1,
                    7,
                ),
            ],
            None,
            None,
        ),
        (
            {
                'foo': (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.MappingEntry('foo', dict, optree.PyTreeKind.DICT),
                        ),
                    ),
                    3,
                    'bar',
                ),
            },
            (
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        ),
                    ),
                    4,
                    9,
                ),
                [
                    (
                        optree.PyTreeAccessor(
                            (
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                            ),
                        ),
                        5,
                        9,
                    ),
                ],
            ),
            [
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    6,
                    10,
                ),
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    6,
                    11,
                ),
            ],
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
        optree.tree_transpose_map_with_path(
            lambda p, a, b: {'p': p, 'a': a, 'b': b},
            x,
            y,
            inner_treespec=optree.tree_structure({'p': 0, 'a': 1, 'b': 2}),
        )


def test_tree_transpose_map_with_accessor():
    with pytest.raises(
        ValueError,
        match=r'The outer structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_accessor(lambda p, u, v: {'p': p, 'u': u, 'v': v}, (), ())
    with pytest.raises(
        ValueError,
        match=r'The inner structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_accessor(lambda a, u, v: None, (1,), (2,))
    with pytest.raises(
        ValueError,
        match=r'The inner structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_accessor(lambda a, u, v: (), (1,), (2,))

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_transpose_map_with_accessor(
        lambda a, u, v: {'d': len(a), 'u': u, 'v': v},
        x,
        y,
    )
    assert out == {
        'd': ((2, 2, None), (2, 2, 2)),
        'u': ((1, 2, None), (3, 4, 5)),
        'v': ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    with pytest.raises(ValueError, match=re.escape('Expected an instance of list, got None.')):
        optree.tree_transpose_map_with_accessor(lambda a, u, v: {'a': a, 'u': u, 'v': v}, x, y)
    out = optree.tree_transpose_map_with_accessor(
        lambda a, u, v: {'a': a, 'u': u, 'v': v},
        x,
        y,
        inner_treespec=optree.tree_structure({'a': 0, 'u': 1, 'v': 2}),
    )
    assert out == {
        'a': (
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                None,
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
            ),
        ),
        'u': ((1, 2, None), (3, 4, 5)),
        'v': (([6], None, None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], 7), ({'foo': 'bar'}, 8, [9, 0]))
    with pytest.raises(ValueError, match=re.escape('Expected None, got 7.')):
        optree.tree_transpose_map_with_accessor(
            lambda a, u, v: {'a': a, 'u': u, 'v': v},
            x,
            y,
            inner_treespec=optree.tree_structure({'a': 0, 'u': 1, 'v': 2}),
        )


def test_tree_map_none_is_leaf():
    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (((1, [6]), (2, None), (None, None)), ((3, {'foo': 'bar'}), (4, 7), (5, [8, 9])))

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (((1, [6]), (2, None), (None, 7)), ((3, {'foo': 'bar'}), (4, 8), (5, [9, 10])))


def test_tree_map_with_path_none_is_leaf():
    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map_with_path(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        (((0, 0), 1, [6]), ((0, 1), 2, None), ((0, 2), None, None)),
        (((1, 0), 3, {'foo': 'bar'}), ((1, 1), 4, 7), ((1, 2), 5, [8, 9])),
    )

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_map_with_path(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        (((0, 0), 1, [6]), ((0, 1), 2, None), ((0, 2), None, 7)),
        (((1, 0), 3, {'foo': 'bar'}), ((1, 1), 4, 8), ((1, 2), 5, [9, 10])),
    )


def test_tree_map_with_accessor_none_is_leaf():
    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map_with_accessor(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        (
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                1,
                [6],
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                2,
                None,
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                None,
                None,
            ),
        ),
        (
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                3,
                {'foo': 'bar'},
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                4,
                7,
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                5,
                [8, 9],
            ),
        ),
    )

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_map_with_accessor(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        (
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                1,
                [6],
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                2,
                None,
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                None,
                7,
            ),
        ),
        (
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                3,
                {'foo': 'bar'},
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                4,
                8,
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                5,
                [9, 10],
            ),
        ),
    )


def test_tree_broadcast_map_none_is_leaf():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_broadcast_map(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        ([(1, 7)], (2, None), (None, None)),
        ({'foo': (3, 'bar')}, ((4, 8), [(5, 8)]), [(6, 9), (6, 10)]),
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
    y = (([7], None, None), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_broadcast_map_with_path(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        ([((0, 0, 0), 1, 7)], ((0, 1), 2, None), ((0, 2), None, None)),
        (
            {'foo': ((1, 0, 'foo'), 3, 'bar')},
            (((1, 1, 0), 4, 8), [((1, 1, 1, 0), 5, 8)]),
            [((1, 2, 0), 6, 9), ((1, 2, 1), 6, 10)],
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


def test_tree_broadcast_map_with_accessor_none_is_leaf():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_broadcast_map_with_accessor(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        (
            [
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    1,
                    7,
                ),
            ],
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                2,
                None,
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                None,
                None,
            ),
        ),
        (
            {
                'foo': (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.MappingEntry('foo', dict, optree.PyTreeKind.DICT),
                        ),
                    ),
                    3,
                    'bar',
                ),
            },
            (
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        ),
                    ),
                    4,
                    8,
                ),
                [
                    (
                        optree.PyTreeAccessor(
                            (
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                            ),
                        ),
                        5,
                        8,
                    ),
                ],
            ),
            [
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    6,
                    9,
                ),
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    6,
                    10,
                ),
            ],
        ),
    )

    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, 8), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map_with_accessor(lambda *xs: tuple(xs), x, y, none_is_leaf=True)
    assert out == (
        (
            [
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    1,
                    7,
                ),
            ],
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                2,
                None,
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                None,
                8,
            ),
        ),
        (
            {
                'foo': (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                            optree.MappingEntry('foo', dict, optree.PyTreeKind.DICT),
                        ),
                    ),
                    3,
                    'bar',
                ),
            },
            (
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        ),
                    ),
                    4,
                    9,
                ),
                [
                    (
                        optree.PyTreeAccessor(
                            (
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                            ),
                        ),
                        5,
                        9,
                    ),
                ],
            ),
            [
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    6,
                    10,
                ),
                (
                    optree.PyTreeAccessor(
                        (
                            optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                            optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                        ),
                    ),
                    6,
                    11,
                ),
            ],
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


def test_tree_transpose_map_with_accessor_none_is_leaf():
    with pytest.raises(
        ValueError,
        match=r'The outer structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_accessor(
            lambda a, u, v: {'a': a, 'u': u, 'v': v},
            (),
            (),
            none_is_leaf=True,
        )
    out = optree.tree_transpose_map_with_accessor(
        lambda a, u, v: None,
        (1,),
        (2,),
        none_is_leaf=True,
    )
    assert out == (None,)
    with pytest.raises(
        ValueError,
        match=r'The inner structure must have at least one leaf\. Got: .*\.',
    ):
        optree.tree_transpose_map_with_accessor(
            lambda a, u, v: (),
            (1,),
            (2,),
            none_is_leaf=True,
        )

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_transpose_map_with_accessor(
        lambda a, u, v: {'d': len(a), 'u': u, 'v': v},
        x,
        y,
        none_is_leaf=True,
    )
    assert out == {
        'd': ((2, 2, 2), (2, 2, 2)),
        'u': ((1, 2, None), (3, 4, 5)),
        'v': ((6, [None], None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    with pytest.raises(ValueError, match=re.escape('Expected an instance of list, got None.')):
        optree.tree_transpose_map_with_accessor(
            lambda a, u, v: {'a': a, 'u': u, 'v': v},
            x,
            y,
            none_is_leaf=True,
        )
    out = optree.tree_transpose_map_with_accessor(
        lambda a, u, v: {'a': a, 'u': u, 'v': v},
        x,
        y,
        inner_treespec=optree.tree_structure({'a': 0, 'u': 1, 'v': 2}),
        none_is_leaf=True,
    )
    assert out == {
        'a': (
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
            ),
        ),
        'u': ((1, 2, None), (3, 4, 5)),
        'v': (([6], None, None), ({'foo': 'bar'}, 7, [8, 9])),
    }

    x = ((1, 2, None), (3, 4, 5))
    y = ((6, [None], 7), ({'foo': 'bar'}, 8, [9, 0]))
    out = optree.tree_transpose_map_with_accessor(
        lambda a, u, v: {'a': a, 'u': u, 'v': v},
        x,
        y,
        inner_treespec=optree.tree_structure({'a': 0, 'u': 1, 'v': 2}),
        none_is_leaf=True,
    )
    assert out == {
        'a': (
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
            ),
            (
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
                optree.PyTreeAccessor(
                    (
                        optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                        optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
                    ),
                ),
            ),
        ),
        'u': ((1, 2, None), (3, 4, 5)),
        'v': ((6, [None], 7), ({'foo': 'bar'}, 8, [9, 0])),
    }


def test_tree_map_key_order():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    leaves = []

    def add_leaves(x):
        leaves.append(x)
        return x

    for tree_map in (optree.tree_map, optree.tree_broadcast_map, optree.tree_transpose_map):
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


def test_tree_map_with_accessor_key_order():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    accessors = []
    leaves = []

    def add_leaves(a, x):
        accessors.append(a)
        leaves.append(x)
        return a, x

    for tree_map_with_accessor in (
        optree.tree_map_with_accessor,
        optree.tree_broadcast_map_with_accessor,
    ):
        accessors.clear()
        leaves.clear()
        mapped = tree_map_with_accessor(add_leaves, tree)
        expected = {
            'b': (
                optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
                2,
            ),
            'a': (
                optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
                1,
            ),
            'c': (
                optree.PyTreeAccessor((optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),)),
                3,
            ),
            'd': None,
            'e': (
                optree.PyTreeAccessor((optree.MappingEntry('e', dict, optree.PyTreeKind.DICT),)),
                4,
            ),
        }
        assert mapped == expected
        assert list(mapped.keys()) == list(expected.keys())
        assert list(mapped.values()) == list(expected.values())
        assert list(mapped.items()) == list(expected.items())
        assert accessors == [
            optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
            optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
            optree.PyTreeAccessor((optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),)),
            optree.PyTreeAccessor((optree.MappingEntry('e', dict, optree.PyTreeKind.DICT),)),
        ]
        assert leaves == [1, 2, 3, 4]


def test_tree_map_key_order_none_is_leaf():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    leaves = []

    def add_leaves(x):
        leaves.append(x)
        return x

    for tree_map in (optree.tree_map, optree.tree_broadcast_map, optree.tree_transpose_map):
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


def test_tree_map_with_accessor_key_order_none_is_leaf():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    accessors = []
    leaves = []

    def add_leaves(a, x):
        accessors.append(a)
        leaves.append(x)
        return a, x

    for tree_map_with_accessor in (
        optree.tree_map_with_accessor,
        optree.tree_broadcast_map_with_accessor,
    ):
        accessors.clear()
        leaves.clear()
        mapped = tree_map_with_accessor(add_leaves, tree, none_is_leaf=True)
        expected = {
            'b': (
                optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
                2,
            ),
            'a': (
                optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
                1,
            ),
            'c': (
                optree.PyTreeAccessor((optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),)),
                3,
            ),
            'd': (
                optree.PyTreeAccessor((optree.MappingEntry('d', dict, optree.PyTreeKind.DICT),)),
                None,
            ),
            'e': (
                optree.PyTreeAccessor((optree.MappingEntry('e', dict, optree.PyTreeKind.DICT),)),
                4,
            ),
        }
        assert mapped == expected
        assert list(mapped.keys()) == list(expected.keys())
        assert list(mapped.values()) == list(expected.values())
        assert list(mapped.items()) == list(expected.items())
        assert accessors == [
            optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
            optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
            optree.PyTreeAccessor((optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),)),
            optree.PyTreeAccessor((optree.MappingEntry('d', dict, optree.PyTreeKind.DICT),)),
            optree.PyTreeAccessor((optree.MappingEntry('e', dict, optree.PyTreeKind.DICT),)),
        ]
        assert leaves == [1, 2, 3, None, 4]


def test_tree_map_with_is_leaf_none():
    x = ((1, 2, None), (3, 4, 5))
    for tree_map in (optree.tree_map, optree.tree_broadcast_map):
        out = tree_map(lambda *xs: tuple(xs), x, none_is_leaf=False)
        assert out == (((1,), (2,), None), ((3,), (4,), (5,)))
        out = tree_map(lambda *xs: tuple(xs), x, none_is_leaf=True)
        assert out == (((1,), (2,), (None,)), ((3,), (4,), (5,)))
        out = tree_map(lambda *xs: tuple(xs), x, is_leaf=is_none)
        assert out == (((1,), (2,), (None,)), ((3,), (4,), (5,)))


def test_tree_map_with_is_leaf():
    x = ((1, 2, None), [3, 4, 5])
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map(lambda *xs: tuple(xs), x, y, is_leaf=is_list)
    assert out == (((1, [6]), (2, None), None), (([3, 4, 5], ({'foo': 'bar'}, 7, [8, 9]))))

    x = ((1, 2, None), [3, 4, 5])
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 0]))
    with pytest.raises(ValueError, match=re.escape('Expected None, got 7.')):
        optree.tree_map(lambda *xs: tuple(xs), x, y, is_leaf=is_list)


def test_tree_map_with_is_leaf_none_is_leaf():
    x = ((1, 2, None), [3, 4, 5])
    y = (([6], None, None), ({'foo': 'bar'}, 7, [8, 9]))
    out = optree.tree_map(
        lambda *xs: tuple(xs),
        x,
        y,
        is_leaf=is_list,
        none_is_leaf=True,
    )
    assert out == (((1, [6]), (2, None), (None, None)), (([3, 4, 5], ({'foo': 'bar'}, 7, [8, 9]))))

    x = ((1, 2, None), [3, 4, 5])
    y = (([6], None, 7), ({'foo': 'bar'}, 8, [9, 0]))
    out = optree.tree_map(
        lambda *xs: tuple(xs),
        x,
        y,
        is_leaf=is_list,
        none_is_leaf=True,
    )
    assert out == (((1, [6]), (2, None), (None, 7)), (([3, 4, 5], ({'foo': 'bar'}, 8, [9, 0]))))


def test_tree_broadcast_map_with_is_leaf():
    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, None), ({'foo': 'bar'}, 8, [9, 10]))
    out = optree.tree_broadcast_map(lambda *xs: tuple(xs), x, y, is_leaf=is_list)
    assert out == (
        ((1, [7]), None, None),
        ({'foo': (3, 'bar')}, ((4, 8), ([5], 8)), (6, [9, 10])),
    )

    x = ((1, 2, None), (3, (4, [5]), 6))
    y = (([7], None, 8), ({'foo': 'bar'}, 9, [10, 11]))
    out = optree.tree_broadcast_map(lambda *xs: tuple(xs), x, y, is_leaf=is_list)
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
        is_leaf=is_list,
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
        is_leaf=is_list,
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


def test_tree_map_with_accessor_ignore_return():
    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, None), ({'foo': 'bar'}, 7, [5, 6]))

    def fn1(a, *xs):
        if a[1].entry >= 1:
            leaves.append(xs)
        return 0

    leaves = []
    out = optree.tree_map_with_accessor_(fn1, x, y)
    assert out is x
    assert x == ((1, 2, None), (3, 4, 5))
    assert leaves == [(2, None), (4, 7), (5, [5, 6])]

    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))

    def fn2(a, *xs):
        if a[1].entry >= 1:
            leaves.append(xs)
        return 0

    leaves = []
    with pytest.raises(ValueError, match=re.escape('Expected None, got 4.')):
        optree.tree_map_with_accessor_(fn2, x, y)


def test_tree_map_ignore_return_none_is_leaf():
    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, None), ({'foo': 'bar'}, 7, [5, 6]))

    def fn1(*xs):
        leaves.append(xs)
        return 0

    leaves = []
    out = optree.tree_map_(fn1, x, y, none_is_leaf=True)
    assert out is x
    assert x == ((1, 2, None), (3, 4, 5))
    assert leaves == [(1, [3]), (2, None), (None, None), (3, {'foo': 'bar'}), (4, 7), (5, [5, 6])]

    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))

    def fn2(*xs):
        leaves.append(xs)
        return 0

    leaves = []
    out = optree.tree_map_(fn2, x, y, none_is_leaf=True)
    assert out is x
    assert x == ((1, 2, None), (3, 4, 5))
    assert leaves == [(1, [3]), (2, None), (None, 4), (3, {'foo': 'bar'}), (4, 7), (5, [5, 6])]


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


def test_tree_map_with_accessor_ignore_return_none_is_leaf():
    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, None), ({'foo': 'bar'}, 7, [5, 6]))

    def fn1(a, *xs):
        if a[1].entry >= 1:
            leaves.append(xs)
        return 0

    leaves = []
    out = optree.tree_map_with_accessor_(fn1, x, y, none_is_leaf=True)
    assert out is x
    assert x == ((1, 2, None), (3, 4, 5))
    assert leaves == [(2, None), (None, None), (4, 7), (5, [5, 6])]

    x = ((1, 2, None), (3, 4, 5))
    y = (([3], None, 4), ({'foo': 'bar'}, 7, [5, 6]))

    def fn2(a, *xs):
        if a[1].entry >= 1:
            leaves.append(xs)
        return 0

    leaves = []
    out = optree.tree_map_with_accessor_(fn2, x, y, none_is_leaf=True)
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


def test_tree_map_with_accessor_inplace():
    x = ((Counter(1), Counter(2), None), (Counter(3), Counter(4), Counter(5)))
    y = ((3, 0, None), (5, 7, 6))

    def fn1_(a, xi, yi):
        xi.increment(yi * (1 + sum(a.path)))

    out = optree.tree_map_with_accessor_(fn1_, x, y)
    assert out is x
    assert x == ((Counter(4), Counter(2), None), (Counter(13), Counter(25), Counter(29)))

    x = ((Counter(1), Counter(2), None), (Counter(3), Counter(4), Counter(5)))
    y = ((3, 0, 4), (5, 7, 6))

    def fn2_(a, xi, yi):
        xi.increment(yi * (1 + sum(a.path)))

    with pytest.raises(ValueError, match=re.escape('Expected None, got 4.')):
        optree.tree_map_with_accessor_(fn2_, x, y)


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


def test_tree_map_with_accessor_inplace_key_order():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    expected = tree.copy()
    accessors = []
    leaves = []

    def add_leaves(a, x):
        accessors.append(a)
        leaves.append(x)
        return a, x

    mapped = optree.tree_map_with_accessor_(add_leaves, tree)
    assert mapped is tree
    assert mapped == expected
    assert list(mapped.keys()) == list(expected.keys())
    assert list(mapped.values()) == list(expected.values())
    assert list(mapped.items()) == list(expected.items())
    assert accessors == [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('e', dict, optree.PyTreeKind.DICT),)),
    ]
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


def test_tree_map_with_accessor_inplace_key_order_none_is_leaf():
    tree = {'b': 2, 'a': 1, 'c': 3, 'd': None, 'e': 4}
    expected = tree.copy()
    accessors = []
    leaves = []

    def add_leaves(a, x):
        accessors.append(a)
        leaves.append(x)
        return a, x

    mapped = optree.tree_map_with_accessor_(add_leaves, tree, none_is_leaf=True)
    assert mapped is tree
    assert mapped == expected
    assert list(mapped.keys()) == list(expected.keys())
    assert list(mapped.values()) == list(expected.values())
    assert list(mapped.items()) == list(expected.items())
    assert accessors == [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('d', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('e', dict, optree.PyTreeKind.DICT),)),
    ]
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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_transpose(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
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
            with pytest.raises(
                ValueError,
                match=re.escape('Tree structures must have at least one leaf.'),
            ):
                optree.tree_transpose(outer_treespec, inner_treespec, nested)
            return
        with pytest.raises(
            ValueError,
            match=re.escape('Tree structures must have the same none_is_leaf value.'),
        ):
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
    with pytest.raises(ValueError, match='Tree structures must have the same namespace'):
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
    assert optree.tree_reduce(operator.add, {'x': 1, 'y': (2, 3)}) == 6
    assert optree.tree_reduce(operator.add, {'x': 1, 'y': (2, None), 'z': 3}) == 6
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
            initial=False,
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
        is_leaf=is_list,
    ) == [1, 2, None, 3]


def test_tree_max():
    with pytest.raises(ValueError, match='empty'):
        optree.tree_max({})
    assert optree.tree_max({}, default=0) == 0
    assert optree.tree_max({'x': 0, 'y': (2, 1)}) == 2
    assert optree.tree_max({'x': 0, 'y': (2, 1)}, key=operator.neg) == 0
    with pytest.raises(ValueError, match='empty'):
        optree.tree_max({'a': None})
    assert optree.tree_max({'a': None}, default=0) == 0
    assert optree.tree_max({'a': None}, none_is_leaf=True) is None
    with pytest.raises(ValueError, match='empty'):
        optree.tree_max(None)
    assert optree.tree_max(None, default=0) == 0
    assert optree.tree_max(None, none_is_leaf=True) is None
    assert optree.tree_max(None, default=0, key=operator.neg) == 0
    with pytest.raises(TypeError, match=r".*operand type for unary .+: 'NoneType'"):
        assert optree.tree_max(None, default=0, key=operator.neg, none_is_leaf=True) is None


def test_tree_min():
    with pytest.raises(ValueError, match='empty'):
        optree.tree_min({})
    assert optree.tree_min({}, default=0) == 0
    assert optree.tree_min({'x': 0, 'y': (2, 1)}) == 0
    assert optree.tree_min({'x': 0, 'y': (2, 1)}, key=operator.neg) == 2
    with pytest.raises(ValueError, match='empty'):
        optree.tree_min({'a': None})
    assert optree.tree_min({'a': None}, default=0) == 0
    assert optree.tree_min({'a': None}, none_is_leaf=True) is None
    with pytest.raises(ValueError, match='empty'):
        optree.tree_min(None)
    assert optree.tree_min(None, default=0) == 0
    assert optree.tree_min(None, none_is_leaf=True) is None
    assert optree.tree_min(None, default=0, key=operator.neg) == 0
    with pytest.raises(TypeError, match=r".*operand type for unary .+: 'NoneType'"):
        assert optree.tree_min(None, default=0, key=operator.neg, none_is_leaf=True) is None


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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_flatten_one_level(  # noqa: C901
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        actual_leaves = []
        actual_paths = []
        actual_accessors = []

        path_stack = []
        accessor_stack = []

        def flatten(node):  # noqa: C901
            counter = itertools.count()
            expected_children, one_level_treespec = optree.tree_flatten(
                node,
                is_leaf=lambda x: next(counter) > 0,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            node_type = type(node)
            node_kind = one_level_treespec.kind
            if one_level_treespec.is_leaf():
                assert expected_children == [node]
                assert node_kind == optree.PyTreeKind.LEAF
                with pytest.raises(
                    ValueError,
                    match=re.escape(f'Cannot flatten leaf-type: {node_type} (node: {node!r}).'),
                ):
                    optree.tree_flatten_one_level(
                        node,
                        none_is_leaf=none_is_leaf,
                        namespace=namespace,
                    )
                actual_leaves.append(node)
                actual_paths.append(tuple(path_stack))
                actual_accessors.append(optree.PyTreeAccessor(accessor_stack))
            else:
                output = (
                    children,
                    metadata,
                    entries,
                    unflatten_func,
                ) = optree.tree_flatten_one_level(
                    node,
                    none_is_leaf=none_is_leaf,
                    namespace=namespace,
                )
                assert children == expected_children
                assert output.type == one_level_treespec.type
                assert output.kind == one_level_treespec.kind
                if node_type in {type(None), tuple, list}:
                    assert metadata is None
                    if node_type is tuple:
                        assert node_kind == optree.PyTreeKind.TUPLE
                    elif node_type is list:
                        assert node_kind == optree.PyTreeKind.LIST
                    else:
                        assert node_kind == optree.PyTreeKind.NONE
                elif node_type is dict:
                    if dict_should_be_sorted or dict_session_namespace not in {'', namespace}:
                        assert metadata == sorted(node.keys())
                    else:
                        assert metadata == list(node.keys())
                    assert node_kind == optree.PyTreeKind.DICT
                elif node_type is OrderedDict:
                    assert metadata == list(node.keys())
                    assert node_kind == optree.PyTreeKind.ORDEREDDICT
                elif node_type is defaultdict:
                    if dict_should_be_sorted or dict_session_namespace not in {'', namespace}:
                        assert metadata == (node.default_factory, sorted(node.keys()))
                    else:
                        assert metadata == (node.default_factory, list(node.keys()))
                    assert node_kind == optree.PyTreeKind.DEFAULTDICT
                elif node_type is deque:
                    assert metadata == node.maxlen
                    assert node_kind == optree.PyTreeKind.DEQUE
                elif optree.is_structseq(node):
                    assert optree.is_structseq_class(node_type)
                    assert isinstance(node, optree.typing.structseq)
                    assert issubclass(node_type, optree.typing.structseq)
                    assert metadata is node_type
                    assert node_kind == optree.PyTreeKind.STRUCTSEQUENCE
                elif optree.is_namedtuple(node):
                    assert optree.is_namedtuple_class(node_type)
                    assert metadata is node_type
                    assert node_kind == optree.PyTreeKind.NAMEDTUPLE
                else:
                    assert node_kind == optree.PyTreeKind.CUSTOM
                assert len(entries) == len(children)
                if hasattr(node, '__getitem__'):
                    for child, entry in zip(children, entries):
                        assert node[entry] is child

                assert unflatten_func(metadata, children) == node
                if node_type is type(None):
                    assert unflatten_func(metadata, []) is None
                    with pytest.raises(ValueError, match=re.escape('Expected no children.')):
                        unflatten_func(metadata, range(1))

                for child, entry in zip(children, entries):
                    path_stack.append(entry)
                    accessor_stack.append(output.path_entry_type(entry, node_type, node_kind))
                    flatten(child)
                    path_stack.pop()
                    accessor_stack.pop()

        flatten(tree)
        assert len(path_stack) == 0
        assert len(accessor_stack) == 0
        assert actual_leaves == optree.tree_leaves(
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert actual_paths == optree.tree_paths(
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert_equal_type_and_value(
            actual_accessors,
            optree.tree_accessors(
                tree,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            ),
        )
