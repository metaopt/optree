# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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
from collections import defaultdict

import pytest

import optree

# pylint: disable-next=wrong-import-order
from helpers import NAMESPACED_TREE, TREE_STRINGS, TREES, parametrize


def test_treespec_equal_hash():
    for i, tree1 in enumerate(TREES):
        treespec1 = optree.tree_structure(tree1)
        treespec1_none_is_leaf = optree.tree_structure(tree1, none_is_leaf=True)
        assert treespec1 != treespec1_none_is_leaf
        assert hash(treespec1) != hash(treespec1_none_is_leaf)
        for j, tree2 in enumerate(TREES):
            treespec2 = optree.tree_structure(tree2)
            treespec2_none_is_leaf = optree.tree_structure(tree2, none_is_leaf=True)
            if i == j:
                assert treespec1 == treespec2
                assert treespec1_none_is_leaf == treespec2_none_is_leaf
            if treespec1 == treespec2:
                assert hash(treespec1) == hash(treespec2)
            else:
                assert hash(treespec1) != hash(treespec2)
            if treespec1_none_is_leaf == treespec2_none_is_leaf:
                assert hash(treespec1_none_is_leaf) == hash(treespec2_none_is_leaf)
            else:
                assert hash(treespec1_none_is_leaf) != hash(treespec2_none_is_leaf)
            assert hash(treespec1) != hash(treespec2_none_is_leaf)
            assert hash(treespec1_none_is_leaf) != hash(treespec2)


@parametrize(tree=TREES, none_is_leaf=[False, True], namespace=['', 'undefined', 'namespace'])
def test_treespec_rich_compare(tree, none_is_leaf, namespace):
    count = itertools.count()

    def build_subtree(x):
        cnt = next(count)
        if cnt % 4 == 0:
            return (x,)
        if cnt % 4 == 1:
            return [x, x]
        if cnt % 4 == 2:
            return (x, [x])
        return {'a': x, 'b': [x]}

    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    suffix_treespec = optree.tree_structure(
        optree.tree_map(build_subtree, tree, none_is_leaf=none_is_leaf, namespace=namespace),
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    assert treespec == treespec
    assert not (treespec != treespec)
    assert not (treespec < treespec)
    assert not (treespec > treespec)
    assert treespec <= treespec
    assert treespec >= treespec
    assert optree.treespec_is_prefix(treespec, treespec, strict=False)
    assert not optree.treespec_is_prefix(treespec, treespec, strict=True)
    assert optree.treespec_is_suffix(treespec, treespec, strict=False)
    assert not optree.treespec_is_suffix(treespec, treespec, strict=True)

    if 'FlatCache' in str(treespec) or treespec == suffix_treespec:
        return

    assert treespec != suffix_treespec
    assert not (treespec == suffix_treespec)
    assert treespec != suffix_treespec
    assert treespec < suffix_treespec
    assert not (treespec > suffix_treespec)
    assert treespec <= suffix_treespec
    assert not (treespec >= suffix_treespec)
    assert suffix_treespec != treespec
    assert not (suffix_treespec == treespec)
    assert suffix_treespec > treespec
    assert not (suffix_treespec < treespec)
    assert suffix_treespec >= treespec
    assert not (suffix_treespec <= treespec)


@parametrize(
    data=list(
        itertools.chain(
            zip(TREES, TREE_STRINGS[False], itertools.repeat(False)),
            zip(TREES, TREE_STRINGS[True], itertools.repeat(True)),
        ),
    ),
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
        assert str(treespec) == 'PyTreeSpec(*)'
        paths, leaves, treespec = optree.tree_flatten_with_path(
            tree,
            none_is_leaf=False,
            namespace=namespace,
        )
        assert paths == [()]
        assert leaves == [tree]
        assert paths == treespec.paths()
        assert str(treespec) == 'PyTreeSpec(*)'
    for namespace in ('', 'undefined'):
        leaves, treespec = optree.tree_flatten(tree, none_is_leaf=True, namespace=namespace)
        assert leaves == [tree]
        assert str(treespec) == 'PyTreeSpec(*, NoneIsLeaf)'
        paths, leaves, treespec = optree.tree_flatten_with_path(
            tree,
            none_is_leaf=True,
            namespace=namespace,
        )
        assert paths == [()]
        assert leaves == [tree]
        assert paths == treespec.paths()
        assert str(treespec) == 'PyTreeSpec(*, NoneIsLeaf)'

    expected_string = "PyTreeSpec(CustomTreeNode(MyAnotherDict[['foo', 'baz']], [CustomTreeNode(MyAnotherDict[['c', 'b', 'a']], [None, *, *]), *]), namespace='namespace')"
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=False, namespace='namespace')
    assert leaves == [2, 1, 101]
    assert str(treespec) == expected_string
    paths, leaves, treespec = optree.tree_flatten_with_path(
        tree,
        none_is_leaf=False,
        namespace='namespace',
    )
    assert paths == [('foo', 'b'), ('foo', 'a'), ('baz',)]
    assert leaves == [2, 1, 101]
    assert paths == treespec.paths()
    assert str(treespec) == expected_string

    expected_string = "PyTreeSpec(CustomTreeNode(MyAnotherDict[['foo', 'baz']], [CustomTreeNode(MyAnotherDict[['c', 'b', 'a']], [*, *, *]), *]), NoneIsLeaf, namespace='namespace')"
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=True, namespace='namespace')
    assert leaves == [None, 2, 1, 101]
    assert str(treespec) == expected_string
    paths, leaves, treespec = optree.tree_flatten_with_path(
        tree,
        none_is_leaf=True,
        namespace='namespace',
    )
    assert paths == [('foo', 'c'), ('foo', 'b'), ('foo', 'a'), ('baz',)]
    assert leaves == [None, 2, 1, 101]
    assert paths == treespec.paths()
    assert str(treespec) == expected_string


@parametrize(tree=TREES, none_is_leaf=[False, True], namespace=['', 'undefined', 'namespace'])
def test_treespec_pickle_round_trip(tree, none_is_leaf, namespace):
    expected = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    try:
        pickle.loads(pickle.dumps(tree))
    except pickle.PicklingError:
        with pytest.raises(
            pickle.PicklingError,
            match="Can't pickle .*: it's not the same object as .*",
        ):
            pickle.loads(pickle.dumps(expected))
    else:
        actual = pickle.loads(pickle.dumps(expected))
        assert actual == expected
        if expected.type is dict or expected.type is defaultdict:
            assert list(optree.tree_unflatten(actual, range(len(actual)))) == list(
                optree.tree_unflatten(expected, range(len(expected))),
            )


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
)
def test_treespec_type(tree, none_is_leaf):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf)
    if treespec.is_leaf():
        assert treespec.type is None
    else:
        assert type(tree) is treespec.type


def test_treespec_with_empty_tuple_string_representation():
    assert str(optree.tree_structure(())) == r'PyTreeSpec(())'


def test_treespec_with_single_element_tuple_string_representation():
    assert str(optree.tree_structure((1,))) == r'PyTreeSpec((*,))'


def test_treespec_with_empty_list_string_representation():
    assert str(optree.tree_structure([])) == r'PyTreeSpec([])'


def test_treespec_with_empty_dict_string_representation():
    assert str(optree.tree_structure({})) == r'PyTreeSpec({})'


@parametrize(
    tree=TREES,
    inner_tree=[
        None,
        '*',
        (),
        (None,),
        ('*',),
        ['*', '*', '*'],
        ['*', '*', None],
        {'a': '*', 'b': None},
        {'a': '*', 'b': ('*', '*')},
    ],
    none_is_leaf=[False, True],
)
def test_treespec_compose_children(tree, inner_tree, none_is_leaf):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf)
    inner_treespec = optree.tree_structure(inner_tree, none_is_leaf=none_is_leaf)
    expected_treespec = optree.tree_structure(
        optree.tree_map(lambda _: inner_tree, tree, none_is_leaf=none_is_leaf),
        none_is_leaf=none_is_leaf,
    )
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

    if 'FlatCache' in str(treespec):
        return

    assert composed_treespec == expected_treespec

    stack = [(composed_treespec.children(), expected_treespec.children())]
    while stack:
        composed_children, expected_children = stack.pop()
        for composed_child, expected_child in zip(composed_children, expected_children):
            assert composed_child == expected_child
            stack.append((composed_child.children(), expected_child.children()))

    assert composed_treespec == optree.tree_structure(composed, none_is_leaf=none_is_leaf)

    if treespec == expected_treespec:
        assert not (treespec != expected_treespec)
        assert not (treespec < expected_treespec)
        assert treespec <= expected_treespec
        assert not (treespec > expected_treespec)
        assert treespec >= expected_treespec
        assert expected_treespec >= treespec
        assert not (expected_treespec > treespec)
        assert expected_treespec <= treespec
        assert not (expected_treespec < treespec)
        assert not optree.treespec_is_prefix(treespec, expected_treespec, strict=True)
        assert optree.treespec_is_prefix(treespec, expected_treespec, strict=False)
        assert not optree.treespec_is_suffix(treespec, expected_treespec, strict=True)
        assert optree.treespec_is_suffix(treespec, expected_treespec, strict=False)
        assert not optree.treespec_is_prefix(expected_treespec, treespec, strict=True)
        assert optree.treespec_is_prefix(expected_treespec, treespec, strict=False)
        assert not optree.treespec_is_suffix(expected_treespec, treespec, strict=True)
        assert optree.treespec_is_suffix(expected_treespec, treespec, strict=False)
    else:
        assert treespec != expected_treespec
        assert treespec < expected_treespec
        assert treespec <= expected_treespec
        assert not (treespec > expected_treespec)
        assert not (treespec >= expected_treespec)
        assert expected_treespec >= treespec
        assert expected_treespec > treespec
        assert not (expected_treespec <= treespec)
        assert not (expected_treespec < treespec)
        assert optree.treespec_is_prefix(treespec, expected_treespec, strict=True)
        assert optree.treespec_is_prefix(treespec, expected_treespec, strict=False)
        assert not optree.treespec_is_suffix(treespec, expected_treespec, strict=True)
        assert not optree.treespec_is_suffix(treespec, expected_treespec, strict=False)
        assert not optree.treespec_is_prefix(expected_treespec, treespec, strict=True)
        assert not optree.treespec_is_prefix(expected_treespec, treespec, strict=False)
        assert optree.treespec_is_suffix(expected_treespec, treespec, strict=True)
        assert optree.treespec_is_suffix(expected_treespec, treespec, strict=False)


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
)
def test_treespec_entries(tree, none_is_leaf):
    expected_paths, _, treespec = optree.tree_flatten_with_path(tree, none_is_leaf=none_is_leaf)
    assert optree.treespec_paths(treespec) == expected_paths

    def gen_path(spec, prefix):
        entries = optree.treespec_entries(spec)
        children = optree.treespec_children(spec)
        assert len(entries) == spec.num_children
        assert len(children) == spec.num_children
        if spec.is_leaf():
            yield prefix
        for entry, child in zip(entries, children):
            yield from gen_path(child, (*prefix, entry))

    paths = list(gen_path(treespec, ()))
    assert paths == expected_paths


def test_treespec_children():
    treespec = optree.tree_structure(((1, 2, 3), (4,)))
    c0 = optree.tree_structure((0, 0, 0))
    c1 = optree.tree_structure((7,))
    assert optree.treespec_children(treespec) == [c0, c1]

    treespec = optree.tree_structure(((1, 2, 3), (4,)))
    c0 = optree.tree_structure((0, 0, 0))
    c1 = optree.tree_structure((7,), none_is_leaf=True)
    assert optree.treespec_children(treespec) != [c0, c1]

    treespec = optree.tree_structure(((1, 2, None), (4,)), none_is_leaf=False)
    c0 = optree.tree_structure((0, 0, None), none_is_leaf=False)
    c1 = optree.tree_structure((7,), none_is_leaf=False)
    assert optree.treespec_children(treespec) == [c0, c1]

    treespec = optree.tree_structure(((1, 2, 3, None), (4,)), none_is_leaf=True)
    c0 = optree.tree_structure((0, 0, 0, 0), none_is_leaf=True)
    c1 = optree.tree_structure((7,), none_is_leaf=True)
    assert optree.treespec_children(treespec) == [c0, c1]


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
)
def test_treespec_num_children(tree, none_is_leaf):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf)
    assert treespec.num_children == len(treespec.entries())
    assert treespec.num_children == len(treespec.children())


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
)
def test_treespec_num_leaves(tree, none_is_leaf):
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf)
    assert treespec.num_leaves == len(leaves)
    assert treespec.num_leaves == len(treespec)
    assert treespec.num_leaves == len(treespec.paths())


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
)
def test_treespec_num_nodes(tree, none_is_leaf):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf)
    nodes = []
    stack = [treespec]
    while stack:
        spec = stack.pop()
        nodes.append(spec)
        stack.extend(reversed(spec.children()))
    assert treespec.num_nodes == len(nodes)


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
        (optree.tree_structure(3, none_is_leaf=none_is_leaf),),
        none_is_leaf=none_is_leaf,
    )
    expected = optree.tree_structure((3,), none_is_leaf=none_is_leaf)
    assert actual == expected

    actual = optree.treespec_tuple(
        (optree.tree_structure(None, none_is_leaf=none_is_leaf),),
        none_is_leaf=none_is_leaf,
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
    assert optree.treespec_is_leaf(optree.tree_structure(()))
    assert optree.treespec_is_leaf(optree.tree_structure([]))
    assert optree.tree_structure(1).is_leaf(strict=False)
    assert not optree.tree_structure((1, 2)).is_leaf(strict=False)
    assert optree.tree_structure(None).is_leaf(strict=False)
    assert optree.tree_structure(None, none_is_leaf=True).is_leaf(strict=False)
    assert optree.tree_structure(()).is_leaf(strict=False)
    assert optree.tree_structure([]).is_leaf(strict=False)


def test_treespec_is_strict_leaf():
    assert optree.treespec_is_strict_leaf(optree.tree_structure(1))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure((1, 2)))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure(None))
    assert optree.treespec_is_strict_leaf(optree.tree_structure(None, none_is_leaf=True))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure(()))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure([]))
    assert optree.tree_structure(1).is_leaf(strict=True)
    assert not optree.tree_structure((1, 2)).is_leaf(strict=True)
    assert not optree.tree_structure(None).is_leaf(strict=True)
    assert optree.tree_structure(None, none_is_leaf=True).is_leaf(strict=True)
    assert not optree.tree_structure(()).is_leaf(strict=True)
    assert not optree.tree_structure([]).is_leaf(strict=True)


def test_treespec_leaf_none():
    assert optree.treespec_leaf(none_is_leaf=False) != optree.treespec_leaf(none_is_leaf=True)
    assert optree.treespec_leaf() == optree.tree_structure(1)
    assert optree.treespec_leaf(none_is_leaf=True) == optree.tree_structure(1, none_is_leaf=True)
    assert optree.treespec_leaf(none_is_leaf=True) == optree.tree_structure(None, none_is_leaf=True)
    assert optree.treespec_leaf(none_is_leaf=True) != optree.tree_structure(
        None,
        none_is_leaf=False,
    )
    assert optree.treespec_leaf(none_is_leaf=True) == optree.treespec_none(none_is_leaf=True)
    assert optree.treespec_leaf(none_is_leaf=True) != optree.treespec_none(none_is_leaf=False)
    assert optree.treespec_leaf(none_is_leaf=False) != optree.treespec_none(none_is_leaf=True)
    assert optree.treespec_leaf(none_is_leaf=False) != optree.treespec_none(none_is_leaf=False)
    assert optree.treespec_none(none_is_leaf=False) != optree.treespec_none(none_is_leaf=True)
    assert optree.treespec_none() == optree.tree_structure(None)
    assert optree.treespec_none() != optree.tree_structure(1)
