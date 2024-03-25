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

import itertools
import pickle
import re
import subprocess
import sys
import textwrap
from collections import OrderedDict, UserList, defaultdict, deque

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


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
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


def test_treespec_self_referential():
    class Holder:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Holder) and self.value == other.value

        def __hash__(self):
            return hash(self.value)

        def __repr__(self):
            return f'Holder({self.value!r})'

    key = Holder('a')

    hashes = set()
    treespec = optree.tree_structure({key: 0})
    assert str(treespec) == "PyTreeSpec({Holder('a'): *})"
    assert hash(treespec) == hash(treespec)
    hashes.add(hash(treespec))

    key.value = 'b'
    assert str(treespec) == "PyTreeSpec({Holder('b'): *})"
    assert hash(treespec) == hash(treespec)
    assert hash(treespec) not in hashes
    hashes.add(hash(treespec))

    key.value = treespec
    assert str(treespec) == 'PyTreeSpec({Holder(...): *})'
    assert hash(treespec) == hash(treespec)
    assert hash(treespec) not in hashes
    hashes.add(hash(treespec))

    key.value = ('a', treespec, treespec)
    assert str(treespec) == "PyTreeSpec({Holder(('a', ..., ...)): *})"
    assert hash(treespec) == hash(treespec)
    assert hash(treespec) not in hashes
    hashes.add(hash(treespec))

    other = optree.tree_structure({Holder(treespec): 1})
    assert str(other) == "PyTreeSpec({Holder(PyTreeSpec({Holder(('a', ..., ...)): *})): *})"
    assert hash(other) == hash(other)
    assert hash(other) not in hashes
    hashes.add(hash(other))

    key.value = other
    assert str(treespec) == 'PyTreeSpec({Holder(PyTreeSpec({Holder(...): *})): *})'
    assert str(other) == 'PyTreeSpec({Holder(PyTreeSpec({Holder(...): *})): *})'
    assert hash(treespec) == hash(treespec)
    assert hash(treespec) not in hashes
    hashes.add(hash(treespec))
    assert hash(other) == hash(other)
    assert hash(treespec) == hash(other)
    with pytest.raises(RecursionError):
        assert treespec != other


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


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
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
        if expected.type in {dict, OrderedDict, defaultdict}:
            assert list(optree.tree_unflatten(actual, range(len(actual)))) == list(
                optree.tree_unflatten(expected, range(len(expected))),
            )


class Foo:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_treespec_pickle_missing_registration():
    optree.register_pytree_node(
        Foo,
        lambda foo: ((foo.x, foo.y), None),
        lambda _, children: Foo(children[0], children[1]),
        namespace='foo',
    )

    treespec = optree.tree_structure(Foo(0, 1), namespace='foo')
    serialized = pickle.dumps(treespec)

    error = subprocess.check_output(
        [
            sys.executable,
            '-c',
            textwrap.dedent(
                f"""
                import pickle
                import sys

                try:
                    treespec = pickle.loads({serialized!r})
                except Exception as ex:
                    print(ex)
                else:
                    sys.exit(1)
                """,
            ),
        ],
    ).decode('utf-8')
    assert re.match(
        r"Unknown custom type in pickled PyTreeSpec: <class '.*'> in namespace 'foo'\.",
        string=error,
    )

    optree.unregister_pytree_node(Foo, namespace='foo')
    with pytest.raises(
        RuntimeError,
        match=r"Unknown custom type in pickled PyTreeSpec: <class '.*'> in namespace 'foo'\.",
    ):
        treespec = pickle.loads(serialized)


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_type(tree, none_is_leaf, namespace):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
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
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_compose_children(tree, inner_tree, none_is_leaf, namespace):
    treespec = optree.tree_structure(
        tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    inner_treespec = optree.tree_structure(
        inner_tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    expected_treespec = optree.tree_structure(
        optree.tree_map(
            lambda _: inner_tree,
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    composed_treespec = treespec.compose(inner_treespec)
    expected_leaves = treespec.num_leaves * inner_treespec.num_leaves
    assert composed_treespec.num_leaves == treespec.num_leaves * inner_treespec.num_leaves
    expected_nodes = (treespec.num_nodes - treespec.num_leaves) + (
        inner_treespec.num_nodes * treespec.num_leaves
    )
    assert composed_treespec.num_nodes == expected_nodes
    leaves = list(range(expected_leaves))
    composed = optree.tree_unflatten(composed_treespec, leaves)
    assert leaves == optree.tree_leaves(
        composed,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )

    if 'FlatCache' in str(treespec):
        return

    assert composed_treespec == expected_treespec

    stack = [(composed_treespec.children(), expected_treespec.children())]
    while stack:
        composed_children, expected_children = stack.pop()
        for composed_child, expected_child in zip(composed_children, expected_children):
            assert composed_child == expected_child
            stack.append((composed_child.children(), expected_child.children()))

    assert composed_treespec == optree.tree_structure(
        composed,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )

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
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_entries(tree, none_is_leaf, namespace):
    expected_paths, _, treespec = optree.tree_flatten_with_path(
        tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    assert optree.treespec_paths(treespec) == expected_paths

    def gen_path(spec, prefix):
        entries = optree.treespec_entries(spec)
        children = optree.treespec_children(spec)
        assert len(entries) == spec.num_children
        assert len(children) == spec.num_children
        assert entries is not optree.treespec_entries(spec)
        assert children is not optree.treespec_children(spec)
        optree.treespec_entries(spec).clear()
        optree.treespec_children(spec).clear()
        if spec.is_leaf():
            yield prefix
        for entry, child in zip(entries, children):
            yield from gen_path(child, (*prefix, entry))

    paths = list(gen_path(treespec, ()))
    assert paths == expected_paths


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_entry(tree, none_is_leaf, namespace):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    if treespec.type is None or treespec.type is type(None):
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
            optree.treespec_entry(treespec, 0)
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
            optree.treespec_entry(treespec, -1)
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
            optree.treespec_entry(treespec, 1)
    if treespec.is_leaf(strict=False):
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
            optree.treespec_entry(treespec, 0)
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
            optree.treespec_entry(treespec, -1)
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
            optree.treespec_entry(treespec, 1)
    expected_entries = optree.treespec_entries(treespec)
    for i, entry in enumerate(expected_entries):
        assert entry == optree.treespec_entry(treespec, i)
        assert entry == optree.treespec_entry(treespec, i - len(expected_entries))
        assert optree.treespec_entry(treespec, i) == optree.treespec_entry(treespec, i)
        assert optree.treespec_entry(treespec, i - len(expected_entries)) == optree.treespec_entry(
            treespec,
            i - len(expected_entries),
        )
        assert optree.treespec_entry(treespec, i) == optree.treespec_entry(
            treespec,
            i - len(expected_entries),
        )
    with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
        optree.treespec_entry(treespec, len(expected_entries))
    with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
        optree.treespec_entry(treespec, -len(expected_entries) - 1)

    assert expected_entries == [
        optree.treespec_entry(treespec, i) for i in range(len(expected_entries))
    ]


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
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_child(tree, none_is_leaf, namespace):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    if treespec.type is None or treespec.type is type(None):
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
            optree.treespec_child(treespec, 0)
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
            optree.treespec_child(treespec, -1)
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
            optree.treespec_child(treespec, 1)
    if treespec.is_leaf(strict=False):
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
            optree.treespec_child(treespec, 0)
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
            optree.treespec_child(treespec, -1)
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
            optree.treespec_child(treespec, 1)
    expected_children = optree.treespec_children(treespec)
    for i, child in enumerate(expected_children):
        assert child == optree.treespec_child(treespec, i)
        assert child == optree.treespec_child(treespec, i - len(expected_children))
        assert optree.treespec_child(treespec, i) == optree.treespec_child(treespec, i)
        assert optree.treespec_child(treespec, i - len(expected_children)) == optree.treespec_child(
            treespec,
            i - len(expected_children),
        )
        assert optree.treespec_child(treespec, i) == optree.treespec_child(
            treespec,
            i - len(expected_children),
        )
    with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
        optree.treespec_child(treespec, len(expected_children))
    with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
        optree.treespec_child(treespec, -len(expected_children) - 1)

    assert expected_children == [
        optree.treespec_child(treespec, i) for i in range(len(expected_children))
    ]


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_num_nodes(tree, none_is_leaf, namespace):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    nodes = []
    stack = [treespec]
    while stack:
        spec = stack.pop()
        nodes.append(spec)
        children = spec.children()
        stack.extend(reversed(children))
        assert spec.num_nodes == sum(child.num_nodes for child in children) + 1
    assert treespec.num_nodes == len(nodes)


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_num_leaves(tree, none_is_leaf, namespace):
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    assert treespec.num_leaves == len(leaves)
    assert treespec.num_leaves == len(treespec)
    assert treespec.num_leaves == len(treespec.paths())


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_num_children(tree, none_is_leaf, namespace):
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    assert treespec.num_children == len(treespec.entries())
    assert treespec.num_children == len(treespec.children())


def test_treespec_is_leaf():
    assert optree.treespec_is_strict_leaf(optree.tree_structure(1))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure((1, 2)))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure(None))
    assert optree.treespec_is_strict_leaf(optree.tree_structure(None, none_is_leaf=True))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure(()))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure([]))
    assert optree.treespec_is_leaf(optree.tree_structure(1))
    assert not optree.treespec_is_leaf(optree.tree_structure((1, 2)))
    assert not optree.treespec_is_leaf(optree.tree_structure(None))
    assert optree.treespec_is_leaf(optree.tree_structure(None, none_is_leaf=True))
    assert not optree.treespec_is_leaf(optree.tree_structure(()))
    assert not optree.treespec_is_leaf(optree.tree_structure([]))
    assert optree.tree_structure(1).is_leaf(strict=True)
    assert not optree.tree_structure((1, 2)).is_leaf(strict=True)
    assert not optree.tree_structure(None).is_leaf(strict=True)
    assert optree.tree_structure(None, none_is_leaf=True).is_leaf(strict=True)
    assert not optree.tree_structure(()).is_leaf(strict=True)
    assert not optree.tree_structure([]).is_leaf(strict=True)

    assert optree.treespec_is_leaf(optree.tree_structure(1), strict=False)
    assert not optree.treespec_is_leaf(optree.tree_structure((1, 2)), strict=False)
    assert optree.treespec_is_leaf(optree.tree_structure(None), strict=False)
    assert optree.treespec_is_leaf(optree.tree_structure(None, none_is_leaf=True), strict=False)
    assert optree.treespec_is_leaf(optree.tree_structure(()), strict=False)
    assert optree.treespec_is_leaf(optree.tree_structure([]), strict=False)
    assert optree.tree_structure(1).is_leaf(strict=False)
    assert not optree.tree_structure((1, 2)).is_leaf(strict=False)
    assert optree.tree_structure(None).is_leaf(strict=False)
    assert optree.tree_structure(None, none_is_leaf=True).is_leaf(strict=False)
    assert optree.tree_structure(()).is_leaf(strict=False)
    assert optree.tree_structure([]).is_leaf(strict=False)


@parametrize(
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_leaf_none(namespace):
    assert optree.treespec_leaf(none_is_leaf=False, namespace=namespace) != optree.treespec_leaf(
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(namespace=namespace) == optree.tree_structure(
        1,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) == optree.tree_structure(
        1,
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) == optree.tree_structure(
        None,
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) != optree.tree_structure(
        None,
        none_is_leaf=False,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) == optree.treespec_none(
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) != optree.treespec_none(
        none_is_leaf=False,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=False, namespace=namespace) != optree.treespec_none(
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=False, namespace=namespace) != optree.treespec_none(
        none_is_leaf=False,
        namespace=namespace,
    )

    assert optree.treespec_none(none_is_leaf=False, namespace=namespace) != optree.treespec_none(
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_none(namespace=namespace) == optree.tree_structure(
        None,
        namespace=namespace,
    )
    assert optree.treespec_none(namespace=namespace) != optree.tree_structure(
        1,
        namespace=namespace,
    )
    assert optree.treespec_none(none_is_leaf=True, namespace=namespace) == optree.tree_structure(
        1,
        none_is_leaf=True,
        namespace=namespace,
    )

    with pytest.warns(
        UserWarning,
        match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
    ):
        assert optree.treespec_from_collection(
            1,
            namespace=namespace,
        ) == optree.treespec_leaf(
            namespace=namespace,
        )
    with pytest.warns(
        UserWarning,
        match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
    ):
        assert optree.treespec_from_collection(
            1,
            none_is_leaf=True,
            namespace=namespace,
        ) == optree.treespec_leaf(
            none_is_leaf=True,
            namespace=namespace,
        )
    assert optree.treespec_from_collection(
        None,
        namespace=namespace,
    ) == optree.treespec_none(
        namespace=namespace,
    )
    with pytest.warns(
        UserWarning,
        match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
    ):
        assert optree.treespec_from_collection(
            None,
            none_is_leaf=True,
            namespace=namespace,
        ) == optree.treespec_none(
            none_is_leaf=True,
            namespace=namespace,
        )


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_constructor(tree, none_is_leaf, namespace):  # noqa: C901
    for passed_namespace in sorted({'', namespace}):
        stack = [tree]
        while stack:
            node = stack.pop()
            counter = itertools.count()
            expected_treespec = optree.tree_structure(
                node,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            children, one_level_treespec = optree.tree_flatten(
                node,
                is_leaf=lambda x: next(counter) > 0,  # noqa: B023
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            node_type = type(node)
            if one_level_treespec.is_leaf():
                assert len(children) == 1
                with pytest.warns(
                    UserWarning,
                    match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
                ):
                    assert (
                        optree.treespec_from_collection(
                            node,
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                assert (
                    optree.treespec_leaf(
                        none_is_leaf=none_is_leaf,
                        namespace=passed_namespace,
                    )
                    == expected_treespec
                )
            else:
                children_treespecs = [
                    optree.tree_structure(
                        child,
                        none_is_leaf=none_is_leaf,
                        namespace=namespace,
                    )
                    for child in children
                ]
                collection_of_treespecs = optree.tree_unflatten(
                    one_level_treespec,
                    children_treespecs,
                )
                assert (
                    optree.treespec_from_collection(
                        collection_of_treespecs,
                        none_is_leaf=none_is_leaf,
                        namespace=namespace,
                    )
                    == expected_treespec
                )

                if node_type in (type(None), tuple, list):
                    if node_type is tuple:
                        assert (
                            optree.treespec_tuple(
                                children_treespecs,
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        assert (
                            optree.treespec_from_collection(
                                tuple(children_treespecs),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                    elif node_type is list:
                        assert (
                            optree.treespec_list(
                                children_treespecs,
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        assert (
                            optree.treespec_from_collection(
                                list(children_treespecs),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                    else:
                        assert len(children_treespecs) == 0
                        assert (
                            optree.treespec_none(
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        assert (
                            optree.treespec_from_collection(
                                None,
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                elif node_type is dict:
                    assert (
                        optree.treespec_dict(
                            zip(sorted(node), children_treespecs),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                    assert (
                        optree.treespec_from_collection(
                            dict(zip(sorted(node), children_treespecs)),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                elif node_type is OrderedDict:
                    assert (
                        optree.treespec_ordereddict(
                            zip(node, children_treespecs),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                    assert (
                        optree.treespec_from_collection(
                            OrderedDict(zip(node, children_treespecs)),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                elif node_type is defaultdict:
                    assert (
                        optree.treespec_defaultdict(
                            node.default_factory,
                            zip(sorted(node), children_treespecs),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                    assert (
                        optree.treespec_from_collection(
                            defaultdict(
                                node.default_factory,
                                zip(sorted(node), children_treespecs),
                            ),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                elif node_type is deque:
                    assert (
                        optree.treespec_deque(
                            children_treespecs,
                            maxlen=node.maxlen,
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                    assert (
                        optree.treespec_from_collection(
                            deque(children_treespecs, maxlen=node.maxlen),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                elif optree.is_structseq(node):
                    assert (
                        optree.treespec_structseq(
                            node_type(children_treespecs),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                    assert (
                        optree.treespec_from_collection(
                            node_type(children_treespecs),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                    with pytest.raises(
                        ValueError,
                        match=r'Expected a namedtuple of PyTreeSpec\(s\), got .*\.',
                    ):
                        optree.treespec_namedtuple(
                            node_type(children_treespecs),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                elif optree.is_namedtuple(node):
                    assert (
                        optree.treespec_namedtuple(
                            node_type(*children_treespecs),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                    assert (
                        optree.treespec_from_collection(
                            node_type(*children_treespecs),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                    with pytest.raises(
                        ValueError,
                        match=r'Expected a PyStructSequence of PyTreeSpec\(s\), got .*\.',
                    ):
                        optree.treespec_structseq(
                            node_type(*children_treespecs),
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )

                stack.extend(reversed(children))


def test_treespec_constructor_namespace():
    @optree.register_pytree_node_class(namespace='mylist')
    class MyList(UserList):
        def tree_flatten(self):
            return self.data, None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    with pytest.warns(
        UserWarning,
        match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
    ):
        assert (
            optree.treespec_from_collection(
                MyList([optree.treespec_leaf(), optree.treespec_leaf(), optree.treespec_leaf()]),
            )
            == optree.treespec_leaf()
        )

    expected_treespec = optree.tree_structure(MyList([1, 2, 3]), namespace='mylist')
    actual_treespec = optree.treespec_from_collection(
        MyList([optree.treespec_leaf(), optree.treespec_leaf(), optree.treespec_leaf()]),
        namespace='mylist',
    )
    assert actual_treespec == expected_treespec
    assert actual_treespec.type is MyList
    assert actual_treespec.namespace == 'mylist'

    children_treespecs = actual_treespec.children()
    assert all(child.namespace == 'mylist' for child in children_treespecs)
    treespec1 = optree.treespec_from_collection(list(children_treespecs), namespace='')
    assert treespec1.type is list
    assert treespec1.namespace == 'mylist'

    treespec2 = optree.treespec_from_collection(
        [optree.treespec_leaf(), optree.treespec_leaf(), optree.treespec_leaf()],
        namespace='mylist',
    )
    assert treespec2.type is list
    assert treespec2.namespace == ''

    assert treespec1 == treespec2


def test_treespec_constructor_none_treespec_inputs():
    with pytest.raises(ValueError, match=r'Expected a\(n\) list of PyTreeSpec\(s\), got .*\.'):
        optree.treespec_list([optree.treespec_leaf(), 1])

    with pytest.raises(ValueError, match=r'Expected a\(n\) list of PyTreeSpec\(s\), got .*\.'):
        optree.treespec_from_collection([optree.treespec_leaf(), 1])

    with pytest.raises(ValueError, match=r'Expected a\(n\) list of PyTreeSpec\(s\), got .*\.'):
        optree.treespec_from_collection(
            [
                optree.treespec_leaf(),
                (optree.treespec_leaf(), optree.treespec_leaf()),
            ],
        )

    assert optree.treespec_from_collection(
        [
            optree.treespec_leaf(),
            optree.treespec_tuple((optree.treespec_leaf(), optree.treespec_leaf())),
        ],
    ) == optree.tree_structure([0, (1, 2)])
