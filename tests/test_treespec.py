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

import itertools
import pickle

import optree

# pylint: disable-next=wrong-import-order
from helpers import NAMESPACED_TREE, TREE_STRINGS, TREES, parametrize


@parametrize(
    data=list(
        itertools.chain(
            zip(TREES, TREE_STRINGS[False], itertools.repeat(False)),
            zip(TREES, TREE_STRINGS[True], itertools.repeat(True)),
        )
    )
)
def test_treespec_string_representation(data):
    tree, correct_string, none_is_leaf = data
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf)
    assert str(treespec) == correct_string


def test_with_namespace():
    tree = NAMESPACED_TREE

    for namespace in ('', 'undefined'):
        leaves, treespec = optree.tree_flatten(tree, none_is_leaf=False, namespace=namespace)
        assert leaves == [tree]
        assert str(treespec) == ('PyTreeSpec(*)')
        paths, leaves, treespec = optree.tree_flatten_with_path(
            tree, none_is_leaf=False, namespace=namespace
        )
        assert paths == [()]
        assert leaves == [tree]
        assert str(treespec) == ('PyTreeSpec(*)')
    for namespace in ('', 'undefined'):
        leaves, treespec = optree.tree_flatten(tree, none_is_leaf=True, namespace=namespace)
        assert leaves == [tree]
        assert str(treespec) == ('PyTreeSpec(*, NoneIsLeaf)')
        paths, leaves, treespec = optree.tree_flatten_with_path(
            tree, none_is_leaf=True, namespace=namespace
        )
        assert paths == [()]
        assert leaves == [tree]
        assert str(treespec) == ('PyTreeSpec(*, NoneIsLeaf)')

    expected_string = "PyTreeSpec(CustomTreeNode(MyDictSubClass[['foo', 'baz']], [CustomTreeNode(MyDictSubClass[['c', 'b', 'a']], [None, *, *]), *]), namespace='namespace')"
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=False, namespace='namespace')
    assert leaves == [2, 1, 101]
    assert str(treespec) == expected_string
    paths, leaves, treespec = optree.tree_flatten_with_path(
        tree, none_is_leaf=False, namespace='namespace'
    )
    assert paths == [('foo', 'b'), ('foo', 'a'), ('baz',)]
    assert leaves == [2, 1, 101]
    assert str(treespec) == expected_string

    expected_string = "PyTreeSpec(CustomTreeNode(MyDictSubClass[['foo', 'baz']], [CustomTreeNode(MyDictSubClass[['c', 'b', 'a']], [*, *, *]), *]), NoneIsLeaf, namespace='namespace')"
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=True, namespace='namespace')
    assert leaves == [None, 2, 1, 101]
    assert str(treespec) == expected_string
    paths, leaves, treespec = optree.tree_flatten_with_path(
        tree, none_is_leaf=True, namespace='namespace'
    )
    assert paths == [('foo', 'c'), ('foo', 'b'), ('foo', 'a'), ('baz',)]
    assert leaves == [None, 2, 1, 101]
    assert str(treespec) == expected_string


@parametrize(tree=TREES, none_is_leaf=[False, True], namespace=['', 'undefined', 'namespace'])
def test_treespec_pickle_round_trip(tree, none_is_leaf, namespace):
    expected = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    actual = pickle.loads(pickle.dumps(expected))
    assert actual == expected


def test_treespec_with_empty_tuple_string_representation():
    assert str(optree.tree_structure(())) == r'PyTreeSpec(())'


def test_treespec_with_single_element_tuple_string_representation():
    assert str(optree.tree_structure((1,))) == r'PyTreeSpec((*,))'


def test_treespec_with_empty_list_string_representation():
    assert str(optree.tree_structure([])) == r'PyTreeSpec([])'


def test_treespec_with_empty_dict_string_representation():
    assert str(optree.tree_structure({})) == r'PyTreeSpec({})'


def test_treespec_children():
    _, treespec = optree.tree_flatten(((1, 2, 3), (4,)))
    _, c0 = optree.tree_flatten((0, 0, 0))
    _, c1 = optree.tree_flatten((7,))
    assert treespec.children() == [c0, c1]

    _, treespec = optree.tree_flatten(((1, 2, 3), (4,)))
    _, c0 = optree.tree_flatten((0, 0, 0))
    _, c1 = optree.tree_flatten((7,), none_is_leaf=True)
    assert treespec.children() != [c0, c1]

    _, treespec = optree.tree_flatten(((1, 2, None), (4,)), none_is_leaf=False)
    _, c0 = optree.tree_flatten((0, 0, None), none_is_leaf=False)
    _, c1 = optree.tree_flatten((7,), none_is_leaf=False)
    assert treespec.children() == [c0, c1]

    _, treespec = optree.tree_flatten(((1, 2, 3, None), (4,)), none_is_leaf=True)
    _, c0 = optree.tree_flatten((0, 0, 0, 0), none_is_leaf=True)
    _, c1 = optree.tree_flatten((7,), none_is_leaf=True)
    assert treespec.children() == [c0, c1]


@parametrize(
    none_is_leaf=[False, True],
)
def test_treespec_tuple_from_children(none_is_leaf):
    tree = ((1, 2, (3, 4)), (5,))
    leaves, treespec1 = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)
    treespec2 = optree.treespec_tuple(treespec1.children(), none_is_leaf=none_is_leaf)
    assert treespec1.num_leaves == len(leaves)
    assert treespec1.num_leaves == treespec2.num_leaves
    assert treespec1.num_nodes == treespec2.num_nodes

    tree = ((1, 2, None, (3, 4, None)), (5,), None)
    leaves, treespec1 = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)
    treespec2 = optree.treespec_tuple(treespec1.children(), none_is_leaf=none_is_leaf)
    assert treespec1.num_leaves == len(leaves)
    assert treespec1.num_leaves == treespec2.num_leaves
    assert treespec1.num_nodes == treespec2.num_nodes


@parametrize(
    none_is_leaf=[False, True],
)
def test_treespec_tuple_compares_equal(none_is_leaf):
    actual = optree.treespec_tuple(
        (optree.tree_structure(3, none_is_leaf=none_is_leaf),), none_is_leaf=none_is_leaf
    )
    expected = optree.tree_structure((3,), none_is_leaf=none_is_leaf)
    assert actual == expected

    actual = optree.treespec_tuple(
        (optree.tree_structure(None, none_is_leaf=none_is_leaf),), none_is_leaf=none_is_leaf
    )
    expected = optree.tree_structure((None,), none_is_leaf=none_is_leaf)
    assert actual == expected

    actual = optree.treespec_tuple(
        (
            optree.tree_structure(3, none_is_leaf=none_is_leaf),
            optree.tree_structure(None, none_is_leaf=none_is_leaf),
        ),
        none_is_leaf=none_is_leaf,
    )
    expected = optree.tree_structure((3, None), none_is_leaf=none_is_leaf)
    assert actual == expected


def test_treespec_is_leaf():
    assert optree.treespec_is_leaf(optree.tree_structure(1))
    assert not optree.treespec_is_leaf(optree.tree_structure((1, 2)))
    assert optree.treespec_is_leaf(optree.tree_structure(None))
    assert optree.treespec_is_leaf(optree.tree_structure(None, none_is_leaf=True))


def test_treespec_is_strict_leaf():
    assert optree.treespec_is_strict_leaf(optree.tree_structure(1))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure((1, 2)))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure(None))
    assert optree.treespec_is_strict_leaf(optree.tree_structure(None, none_is_leaf=True))


def test_treespec_leaf_none():
    assert optree.treespec_leaf(none_is_leaf=False) != optree.treespec_leaf(none_is_leaf=True)
    assert optree.treespec_leaf() == optree.tree_structure(1)
    assert optree.treespec_leaf(none_is_leaf=True) == optree.tree_structure(1, none_is_leaf=True)
    assert optree.treespec_leaf(none_is_leaf=True) == optree.tree_structure(None, none_is_leaf=True)
    assert optree.treespec_leaf(none_is_leaf=True) != optree.tree_structure(
        None, none_is_leaf=False
    )
    assert optree.treespec_leaf(none_is_leaf=True) == optree.treespec_none(none_is_leaf=True)
    assert optree.treespec_leaf(none_is_leaf=True) != optree.treespec_none(none_is_leaf=False)
    assert optree.treespec_leaf(none_is_leaf=False) != optree.treespec_none(none_is_leaf=True)
    assert optree.treespec_leaf(none_is_leaf=False) != optree.treespec_none(none_is_leaf=False)
    assert optree.treespec_none(none_is_leaf=False) != optree.treespec_none(none_is_leaf=True)
    assert optree.treespec_none() == optree.tree_structure(None)
    assert optree.treespec_none() != optree.tree_structure(1)
