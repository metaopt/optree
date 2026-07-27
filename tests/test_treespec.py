# Copyright 2022-2026 MetaOPT Team. All Rights Reserved.
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

import contextlib
import itertools
import os
import pickle
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import weakref
from collections import OrderedDict, UserList, defaultdict, deque, namedtuple

import pytest

import helpers
import optree
from helpers import (
    GLOBAL_NAMESPACE,
    NAMESPACED_TREE,
    PYPY,
    STANDARD_DICT_TYPES,
    TEST_ROOT,
    TREE_STRINGS,
    TREES,
    MyAnotherDict,
    MyDict,
    Py_DEBUG,
    check_script_in_subprocess,
    disable_systrace,
    gc_collect,
    parametrize,
    recursionlimit,
    skipif_android,
    skipif_ios,
    skipif_pypy,
    skipif_wasm,
)


@pytest.mark.skipif(
    platform.machine().lower() not in ('x86_64', 'amd64'),
    reason='Only run on x86_64 and AMD64 architectures',
)
@skipif_wasm
@skipif_android
@skipif_ios
@skipif_pypy
@disable_systrace
def test_treespec_construct():
    with pytest.raises(TypeError, match=re.escape('No constructor defined!')):
        optree.PyTreeSpec()
    treespec = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
    with pytest.raises(TypeError, match=re.escape('No constructor defined!')):
        treespec.__init__()
    del treespec

    gc_collect()

    returncode = 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            check_script_in_subprocess(
                r"""
                import signal
                import sys

                import optree
                import optree._C

                for _ in range(32):
                    treespec = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
                    try:
                        repr(treespec)
                    except optree._C.InternalError as ex:
                        assert 'src/treespec/serialization.cpp' in str(ex).replace('\\', '/')
                        sys.exit(0)
                """,
                cwd=tmpdir,
                output=None,
            )
    except subprocess.CalledProcessError as ex:
        returncode = abs(ex.returncode)
        if 128 < returncode < 256:
            returncode -= 128
    assert returncode in (
        0,
        signal.SIGSEGV,
        signal.SIGABRT,
        0xC0000005,  # STATUS_ACCESS_VIOLATION on Windows
    )


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


def test_treespec_equal_hash_with_namespace():
    # `optree.functools.partial` is registered in the global namespace, so it is recognized under
    # any namespace. Flattening the same object with and without an explicit namespace yields
    # structurally identical treespecs that compare equal, because an empty namespace is treated as
    # a wildcard compatible with any namespace (see `PyTreeSpec::EqualTo`). Equal treespecs MUST
    # hash equally, otherwise hash-based containers (`dict` / `set`) break.
    obj = optree.functools.partial(int, base=2)

    treespec_no_namespace = optree.tree_structure(obj)
    treespec_namespace = optree.tree_structure(obj, namespace='namespace')

    assert treespec_no_namespace.namespace == ''
    assert treespec_namespace.namespace == 'namespace'

    # The empty namespace is a wildcard compatible with any namespace: these compare equal.
    assert treespec_no_namespace == treespec_namespace

    # Hash/equality contract: equal objects must have equal hashes.
    assert hash(treespec_no_namespace) == hash(treespec_namespace)

    # Consequences for hash-based containers when the contract is honored.
    assert treespec_namespace in {treespec_no_namespace: 'value'}
    assert len({treespec_no_namespace, treespec_namespace}) == 1


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_rich_compare(
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
    tree, expected_string, none_is_leaf = data
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf)
    assert str(treespec) == expected_string
    assert repr(treespec) == expected_string

    assert expected_string.startswith('PyTreeSpec(')
    assert expected_string.endswith(')')
    if none_is_leaf:
        assert expected_string.endswith(', NoneIsLeaf)')
        representation = expected_string[len('PyTreeSpec(') : -len(', NoneIsLeaf)')]
    else:
        representation = expected_string[len('PyTreeSpec(') : -len(')')]

    if (
        'CustomTreeNode' not in representation
        and 'sys.float_info' not in representation
        and 'time.struct_time' not in representation
    ):
        representation = re.sub(
            r"<class '([\w\.]+)'>",
            lambda match: match.group(1),
            representation,
        )
        counter = itertools.count()
        representation = re.sub(r'\*', lambda _: str(next(counter)), representation)
        new_tree = optree.tree_unflatten(treespec, range(treespec.num_leaves))
        reconstructed_tree = eval(representation, helpers.__dict__.copy())
        assert new_tree == reconstructed_tree


@skipif_pypy  # CPython-only: `os.stat_result` slots 7, 8, 9 are unnamed; PyPy names them
def test_treespec_structseq_unnamed_field_string_representation():
    # `os.stat_result` renders its UNNAMED sequence slots (7, 8, 9) with the synthetic `<unnamed@N>`
    # placeholder, following CPython's `<lambda>` convention for names that are not identifiers,
    # rather than the bare `unnamed field` marker which reads as an invalid keyword. CPython's
    # `stat_result_desc` has pinned the 7 named + 3 unnamed sequence fields for 16 years, so the
    # repr is asserted exactly; the hidden float `st_atime` fields (indices >= 10) are not part of
    # the sequence and must not leak into it.
    assert os.stat_result.n_sequence_fields == 10
    assert os.stat_result.n_unnamed_fields == 3
    st = os.stat_result(range(os.stat_result.n_fields))
    representation = str(optree.tree_structure(st))
    assert representation == (
        'PyTreeSpec(os.stat_result('
        'st_mode=*, st_ino=*, st_dev=*, st_nlink=*, st_uid=*, st_gid=*, st_size=*, '
        '<unnamed@7>=*, <unnamed@8>=*, <unnamed@9>=*))'
    )


def test_treespec_with_empty_tuple_string_representation():
    assert str(optree.tree_structure(())) == r'PyTreeSpec(())'


def test_treespec_with_single_element_tuple_string_representation():
    assert str(optree.tree_structure((1,))) == r'PyTreeSpec((*,))'


def test_treespec_with_empty_list_string_representation():
    assert str(optree.tree_structure([])) == r'PyTreeSpec([])'


def test_treespec_with_empty_dict_string_representation():
    assert str(optree.tree_structure({})) == r'PyTreeSpec({})'


def test_treespec_namedtuple_repr_with_divergent_fields_raises_value_error():
    # If a namedtuple's `_fields` is mutated after the treespec is built, the recorded arity and the
    # now-divergent field count disagree. The repr must raise a clear `ValueError` attributing the
    # cause, not an `InternalError` telling the user to file a bug report.
    Point = namedtuple('Point', ('x', 'y'))  # noqa: PYI024
    treespec = optree.tree_structure(Point(1, 2))
    assert str(treespec) == 'PyTreeSpec(Point(x=*, y=*))'

    Point._fields = ('x', 'y', 'z')  # diverge: 3 fields vs the treespec's arity of 2
    with pytest.raises(ValueError, match=r'does not match the arity'):
        repr(treespec)


def test_treespec_setstate_rejects_structseq_field_arity_mismatch():
    # A PyStructSequence type's sequence-field count is fixed in C, so a node's arity must equal it
    # (unlike a namedtuple, whose `_fields` can be mutated after the fact). `FromPicklable` (via
    # `__setstate__`/`pickle`) must reject a crafted state pairing a PyStructSequence type with a
    # mismatched arity at load time, rather than build a corrupt treespec that later aborts (e.g. in
    # repr with an `InternalError`).
    spec = optree.tree_structure(time.gmtime())  # struct_time: 9 sequence fields
    node_states, none_is_leaf, namespace = spec.__getstate__()
    # Swap the type to os.stat_result (10 sequence fields) while keeping the arity of 9.
    crafted = tuple(
        (kind, arity, os.stat_result if data is time.struct_time else data, *remaining)
        for (kind, arity, data, *remaining) in node_states
    )
    obj = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
    with pytest.raises(RuntimeError, match=r'does not match the arity'):
        obj.__setstate__((crafted, none_is_leaf, namespace))


def test_treespec_setstate_rejects_namedtuple_field_arity_mismatch():
    # A namedtuple's `_fields` can be mutated, so a crafted state can pair the type with an arity
    # that no longer matches its field count. `FromPicklable` must reject it at load, rather than
    # build a corrupt spec (the repr guards the post-load mutation case separately).
    Point = namedtuple('Point', ('x', 'y'))  # noqa: PYI024
    state = optree.tree_structure(Point(1, 2)).__getstate__()  # arity 2
    Point._fields = ('x', 'y', 'z')  # diverge: 3 fields vs the pickled arity of 2

    obj = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
    with pytest.raises(RuntimeError, match=r'does not match the arity'):
        obj.__setstate__(state)


@disable_systrace
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

    gc_collect()
    if not PYPY:
        with recursionlimit(64):
            with pytest.raises(RecursionError):
                assert treespec != other

        wr = weakref.ref(treespec)
        del treespec, key, other
        gc_collect()
        assert wr() is None


@disable_systrace
def test_treeiter_self_referential():
    sentinel = object()

    d = {'a': 1}
    it = optree.tree_iter(d)
    assert next(it) == 1
    d['b'] = 2
    assert next(it, sentinel) is sentinel

    d = {'a': 1, 'b': {'c': 2}}
    it = optree.tree_iter(d)
    assert next(it) == 1
    d['b']['d'] = it
    assert next(it) == 2
    assert next(it) is it
    assert next(it, sentinel) is sentinel

    d = {'a': 1, 'b': {'c': 2}}
    it = optree.tree_iter(d)
    wr = weakref.ref(it)
    assert next(it) == 1
    d['b']['d'] = it
    assert next(it) == 2

    del it, d
    gc_collect()
    if not PYPY:
        assert wr() is None


def test_treespec_with_namespace():
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
        accessors, leaves, treespec = optree.tree_flatten_with_accessor(
            tree,
            none_is_leaf=False,
            namespace=namespace,
        )
        assert accessors == [optree.PyTreeAccessor()]
        assert leaves == [tree]
        assert accessors == treespec.accessors()
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
        accessors, leaves, treespec = optree.tree_flatten_with_accessor(
            tree,
            none_is_leaf=True,
            namespace=namespace,
        )
        assert accessors == [optree.PyTreeAccessor()]
        assert leaves == [tree]
        assert accessors == treespec.accessors()
        assert str(treespec) == 'PyTreeSpec(*, NoneIsLeaf)'

    expected_string = "PyTreeSpec(CustomTreeNode(MyAnotherDict[['foo', 'baz']], [CustomTreeNode(MyDict[['c', 'b', 'a']], [None, *, *]), *]), namespace='namespace')"
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
    accessors, leaves, treespec = optree.tree_flatten_with_accessor(
        tree,
        none_is_leaf=False,
        namespace='namespace',
    )
    assert accessors == [
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('b', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', MyAnotherDict, optree.PyTreeKind.CUSTOM),),
        ),
    ]
    assert leaves == [2, 1, 101]
    assert accessors == treespec.accessors()
    assert str(treespec) == expected_string

    expected_string = "PyTreeSpec(CustomTreeNode(MyAnotherDict[['foo', 'baz']], [CustomTreeNode(MyDict[['c', 'b', 'a']], [*, *, *]), *]), NoneIsLeaf, namespace='namespace')"
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
    accessors, leaves, treespec = optree.tree_flatten_with_accessor(
        tree,
        none_is_leaf=True,
        namespace='namespace',
    )
    assert accessors == [
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('c', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('b', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', MyAnotherDict, optree.PyTreeKind.CUSTOM),),
        ),
    ]
    assert leaves == [None, 2, 1, 101]
    assert accessors == treespec.accessors()
    assert str(treespec) == expected_string


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_pickle_roundtrip(
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
        expected = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        try:
            pickle.loads(pickle.dumps(tree))
        except pickle.PicklingError:
            with pytest.raises(pickle.PicklingError, match=r"Can't pickle .*:"):
                pickle.loads(pickle.dumps(expected))
        else:
            actual = pickle.loads(pickle.dumps(expected))
            assert actual == expected
            if expected.type in STANDARD_DICT_TYPES:
                assert list(optree.tree_unflatten(actual, range(len(actual)))) == list(
                    optree.tree_unflatten(expected, range(len(expected))),
                )


@skipif_wasm
@skipif_android
@skipif_ios
def test_treespec_pickle_all_protocols_roundtrip():
    # pybind11's pickle support reconstructs cleanly only at protocol >= 2. Protocols 0 and 1 used
    # to reconstruct via `object.__new__`, which pybind11 rejects with an untranslated C++ exception
    # that aborts the interpreter (SIGABRT). Run in a subprocess so a regression fails this test
    # rather than killing the whole suite.
    check_script_in_subprocess(
        r"""
        import pickle

        import optree

        spec = optree.tree_structure({'a': [1, 2], 'b': (3, 4)})
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(spec, protocol=protocol))
            assert restored == spec, (protocol, restored, spec)
        """,
        output=None,
    )


class Foo:
    def __init__(self, x, y):
        self.x = x
        self.y = y


@skipif_wasm
@skipif_android
@skipif_ios
def test_treespec_pickle_missing_registration():
    if sys.version_info[:2] == (3, 11) and platform.system() == 'Windows' and Py_DEBUG:
        pytest.skip('Python 3.11 on Windows has a bug during PyStructSequence type deallocation.')

    optree.register_pytree_node(
        Foo,
        lambda foo: ((foo.x, foo.y), None),
        lambda _, children: Foo(*children),
        namespace='foo',
    )

    treespec = optree.tree_structure(Foo(0, 1), namespace='foo')
    serialized = pickle.dumps(treespec)

    check_script_in_subprocess(
        f"""
        import pickle
        import sys

        sys.path.insert(0, {str(TEST_ROOT)!r})

        try:
            treespec = pickle.loads({serialized!r})
        except Exception as ex:
            print(ex)
        else:
            print('No exception was raised.', file=sys.stderr)
            sys.exit(1)
        """,
        output=re.compile(
            r"Unknown custom type in pickled PyTreeSpec: <class '.*'> in namespace 'foo'\.",
        ),
    )

    optree.unregister_pytree_node(Foo, namespace='foo')
    with pytest.raises(
        RuntimeError,
        match=r"^Unknown custom type in pickled PyTreeSpec: <class '.*'> in namespace 'foo'\.$",
    ):
        treespec = pickle.loads(serialized)


def test_treespec_getstate_does_not_alias_internal_node_data():
    # `__getstate__` (used by `pickle`) must return a snapshot, not aliases of the immutable spec's
    # internal mutable containers: the keys of a dict/OrderedDict/defaultdict node and its
    # insertion-order keys dict. Mutating the returned state otherwise reaches back into the spec,
    # desyncing the keys from the arity (repr raises an InternalError) or adding a spurious
    # original key (unflatten returns an extra entry). A custom node's entries are immutable.
    class Custom:
        def __init__(self, *values):
            self.values = values

    optree.register_pytree_node(
        Custom,
        lambda custom: (custom.values, None, tuple(range(len(custom.values)))),
        lambda metadata, children: Custom(*children),
        namespace='getstate_snapshot',
    )
    try:
        tree = {
            'b': Custom(1, 2),
            'a': 3,
            'od': OrderedDict([('y', 4), ('x', 5)]),
            'dd': defaultdict(int, {'q': 6, 'p': 7}),
        }
        spec = optree.tree_structure(tree, namespace='getstate_snapshot')
        node_states, _, _ = state = spec.__getstate__()
        before = repr(state)

        for node in node_states:
            kind, node_data, node_entries, original_keys = node[0], node[2], node[3], node[7]
            if kind in {optree.PyTreeKind.DICT, optree.PyTreeKind.ORDEREDDICT}:
                node_data.append('injected')  # a dict/OrderedDict node's keys list
            elif kind == optree.PyTreeKind.DEFAULTDICT:
                node_data[1].append('injected')  # a defaultdict's (default_factory, keys) tuple
            if isinstance(original_keys, dict):
                original_keys['injected'] = None
            assert node_entries is None or isinstance(node_entries, tuple)  # entries are immutable

        assert repr(spec.__getstate__()) == before, 'mutating the pickled state corrupted the spec'
    finally:
        optree.unregister_pytree_node(Custom, namespace='getstate_snapshot')


def test_treespec_getstate_aliases_custom_node_data():
    # Limitation (characterization test): a custom node's `node_data` is the user-provided metadata,
    # which `__getstate__` passes through by reference. optree copies its own dict/defaultdict keys
    # (see `test_treespec_getstate_does_not_alias_internal_node_data`) but cannot generically
    # deep-copy arbitrary metadata, so mutating it via the pickled state reaches back into the spec.
    # Protecting custom metadata is the caller's responsibility; this pins the behavior.
    class Custom:
        def __init__(self, *children, alpha, beta=None):
            self.children = children
            self.metadata = {'alpha': alpha, 'beta': beta}

        def __eq__(self, other):
            return (
                isinstance(other, Custom)
                and self.children == other.children
                and self.metadata == other.metadata
            )

        __hash__ = None

    optree.register_pytree_node(
        Custom,
        lambda custom: (custom.children, custom.metadata),  # mutable dict metadata
        lambda metadata, children: Custom(*children, **metadata),
        namespace='getstate_alias_custom',
    )
    try:
        leaves, treespec = optree.tree_flatten(
            Custom(1, 2, alpha=3, beta=4),
            namespace='getstate_alias_custom',
        )
        before = repr(treespec)
        custom_state = next(
            node for node in treespec.__getstate__()[0] if node[0] == optree.PyTreeKind.CUSTOM
        )
        assert custom_state[2] == {'alpha': 3, 'beta': 4}

        # Mutate the aliased metadata in place via the pickled state.
        custom_state[2]['gamma'] = 5  # mutate the aliased metadata in place

        aliased = next(
            node for node in treespec.__getstate__()[0] if node[0] == optree.PyTreeKind.CUSTOM
        )
        assert aliased[2] is custom_state[2]
        assert aliased[2] == {'alpha': 3, 'beta': 4, 'gamma': 5}  # the mutation reached the spec
        assert repr(treespec) == before.replace(
            repr({'alpha': 3, 'beta': 4}),
            repr({'alpha': 3, 'beta': 4, 'gamma': 5}),
        )
        # The corruption even reaches what `tree_unflatten` rebuilds, not just repr/getstate.
        with pytest.raises(TypeError, match=r'unexpected keyword argument'):
            optree.tree_unflatten(treespec, leaves)

        # Replacing the metadata wholesale reaches the spec the same way, but here unflatten
        # succeeds and rebuilds a different object: the corruption is silent, not an error.
        custom_state[2].clear()
        custom_state[2]['alpha'] = 42
        aliased = next(
            node for node in treespec.__getstate__()[0] if node[0] == optree.PyTreeKind.CUSTOM
        )
        assert aliased[2] is custom_state[2]
        assert aliased[2] == {'alpha': 42}
        assert repr(treespec) == before.replace(
            repr({'alpha': 3, 'beta': 4}),
            repr({'alpha': 42}),
        )
        reconstructed = optree.tree_unflatten(treespec, leaves)
        reconstructed_treespec = optree.tree_structure(
            reconstructed,
            namespace='getstate_alias_custom',
        )
        assert reconstructed == Custom(1, 2, alpha=42)
        assert reconstructed.metadata == {'alpha': 42, 'beta': None}
        assert reconstructed_treespec != treespec
        assert repr(reconstructed_treespec) == before.replace(
            repr({'alpha': 3, 'beta': 4}),
            repr({'alpha': 42, 'beta': None}),
        )
    finally:
        optree.unregister_pytree_node(Custom, namespace='getstate_alias_custom')


def test_treespec_setstate_does_not_alias_supplied_node_data():
    # The symmetric half of `test_treespec_getstate_does_not_alias_internal_node_data`:
    # `__setstate__` validated the supplied key list and then BORROWED it into the node, so
    # mutating the state afterwards silently corrupted an already-restored treespec (the keys
    # desync from the children, and `unflatten` pairs them up wrongly). It must copy instead.
    # `original_keys` is already rebuilt via `dict.fromkeys`, so it is covered too.
    def setstate(state):
        obj = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
        obj.__setstate__(state)
        return obj

    trees = [
        {'a': 0, 'b': 0},
        OrderedDict([('a', 0), ('b', 0)]),
        defaultdict(int, {'a': 0, 'b': 0}),
        {'b': 0, 'a': 0},  # sorted keys differ from the insertion order recorded in original_keys
    ]
    for tree in trees:
        spec = optree.tree_structure(tree)
        state = spec.__getstate__()
        restored = setstate(state)
        before = repr(restored)
        expected_entries = restored.entries()
        expected_tree = restored.unflatten([10, 20])

        for node in state[0]:
            kind, node_data, original_keys = node[0], node[2], node[7]
            if kind in {optree.PyTreeKind.DICT, optree.PyTreeKind.ORDEREDDICT}:
                node_data.reverse()  # a dict/OrderedDict node's keys list
                node_data.append('injected')
            elif kind == optree.PyTreeKind.DEFAULTDICT:
                node_data[1].reverse()  # a defaultdict's (default_factory, keys) tuple
                node_data[1].append('injected')
            if isinstance(original_keys, dict):
                original_keys['injected'] = None

        assert repr(restored) == before, tree
        assert restored.entries() == expected_entries, tree
        assert restored.unflatten([10, 20]) == expected_tree, tree
        assert restored == spec, tree


def test_treespec_setstate_rejects_malformed_state():
    # `PyTreeSpec.__setstate__` (used by `pickle`) must reject structurally malformed state rather
    # than build a corrupt spec that triggers out-of-bounds reads / crashes when later used. The
    # per-node tuple layout is (kind, arity, node_data, node_entries, custom, num_leaves, num_nodes,
    # original_keys); see `PyTreeSpec::FromPicklable`.
    def setstate(state):
        obj = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
        obj.__setstate__(state)
        return obj

    CUSTOM = int(optree.PyTreeKind.CUSTOM)  # noqa: N806
    LEAF = int(optree.PyTreeKind.LEAF)  # noqa: N806
    NONE = int(optree.PyTreeKind.NONE)  # noqa: N806
    TUPLE = int(optree.PyTreeKind.TUPLE)  # noqa: N806
    DICT = int(optree.PyTreeKind.DICT)  # noqa: N806
    NAMEDTUPLE = int(optree.PyTreeKind.NAMEDTUPLE)  # noqa: N806
    DEFAULTDICT = int(optree.PyTreeKind.DEFAULTDICT)  # noqa: N806
    DEQUE = int(optree.PyTreeKind.DEQUE)  # noqa: N806
    STRUCTSEQUENCE = int(optree.PyTreeKind.STRUCTSEQUENCE)  # noqa: N806
    NUM_KINDS = int(optree.PyTreeKind.NUM_KINDS)  # noqa: N806
    leaf_node = (LEAF, 0, None, None, None, 1, 1, None)  # arity 0, 1 leaf, 1 node
    keys_ab = {'a': None, 'b': None}  # original_keys for a 2-key ('a', 'b') dict node

    # Sanity: well-formed states still round-trip.
    for spec in [
        optree.tree_structure((0, 0)),
        optree.tree_structure({'a': 0, 'b': 0}),
        optree.tree_structure(defaultdict(int, {'a': 0, 'b': 0})),
    ]:
        assert setstate(spec.__getstate__()) == spec

    malformed_exceptions = (RuntimeError, ValueError, TypeError)

    # The rejection cases below follow the order of the checks in `PyTreeSpec::FromPicklable`.

    # A state that is not a 3-tuple.
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node,), False))

    # A node state that is not a 7- or 8-tuple.
    with pytest.raises(malformed_exceptions):
        setstate((((LEAF, 0, None, None, None, 1),), False, ''))

    # Kind out of range: the raw integer is validated before the narrowing `uint8_t` enum cast,
    # which would otherwise wrap a bogus value to a valid-looking kind.
    with pytest.raises(malformed_exceptions):
        setstate((((NUM_KINDS, 0, None, None, None, 0, 1, None),), False, ''))

    # Negative arity.
    with pytest.raises(malformed_exceptions):
        setstate((((TUPLE, -1, None, None, None, 0, 1, None),), False, ''))

    # A dict node missing its original keys, and a non-dict node carrying them.
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, (DICT, 2, ['a', 'b'], None, None, 2, 3, None)), False, ''))
    with pytest.raises(malformed_exceptions):
        setstate((((LEAF, 0, None, None, None, 1, 1, keys_ab),), False, ''))

    # A negative leaf count, or a non-positive node count.
    with pytest.raises(malformed_exceptions):
        setstate((((LEAF, 0, None, None, None, -1, 1, None),), False, ''))
    with pytest.raises(malformed_exceptions):
        setstate((((LEAF, 0, None, None, None, 1, 0, None),), False, ''))

    # Node data on a leaf or none node (childless kinds that must not carry any).
    with pytest.raises(malformed_exceptions):
        setstate((((LEAF, 0, 'data', None, None, 1, 1, None),), False, ''))

    # Leaf or none nodes are childless; a nonzero arity absorbs the preceding subtrees while still
    # folding consistently, so the reconstructed spec reports a leaf/None while its num_leaves counts
    # the absorbed children and unflatten silently drops them.
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, (NONE, 1, None, None, None, 1, 2, None)), False, ''))
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, (LEAF, 1, None, None, None, 1, 2, None)), False, ''))

    # A None-kind node cannot appear when none_is_leaf is set (None is flattened as a leaf then, so
    # a flattened tree never contains a None node); accepting one later raises an InternalError.
    with pytest.raises(malformed_exceptions):
        setstate((((NONE, 0, None, None, None, 0, 1, None),), True, ''))

    # Node data on a tuple or list node.
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, (TUPLE, 2, 'data', None, None, 2, 3, None)), False, ''))

    # Dict key list shorter than arity (MakeNode would index past the list end).
    short_keys = (DICT, 2, ['a'], None, None, 2, 3, keys_ab)
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, short_keys), False, ''))

    # Dict with duplicate keys (would collapse the rebuilt dict), and with an unhashable key.
    dup_keys = (DICT, 2, ['a', 'a'], None, None, 2, 3, keys_ab)
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, dup_keys), False, ''))
    unhashable_key = (DICT, 2, [[], []], None, None, 2, 3, keys_ab)
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, unhashable_key), False, ''))

    # NamedTuple / StructSequence node_data that is not the expected kind of type.
    with pytest.raises(malformed_exceptions):
        setstate((((NAMEDTUPLE, 0, int, None, None, 0, 1, None),), False, ''))
    with pytest.raises(malformed_exceptions):
        setstate((((STRUCTSEQUENCE, 0, int, None, None, 0, 1, None),), False, ''))

    # DefaultDict metadata as a list where a 2-tuple is expected previously caused a raw tuple-item
    # read to segfault; it is now coerced to a tuple and used safely.
    restored = setstate(
        (
            (
                leaf_node,
                leaf_node,
                (DEFAULTDICT, 2, [int, ['a', 'b']], None, None, 2, 3, keys_ab),
            ),
            False,
            '',
        ),
    )
    assert optree.tree_unflatten(restored, [10, 20]) == defaultdict(int, {'a': 10, 'b': 20})

    # DefaultDict metadata with the wrong tuple size is rejected.
    wrong_metadata = (DEFAULTDICT, 2, (int, ['a', 'b'], 'extra'), None, None, 2, 3, keys_ab)
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, wrong_metadata), False, ''))

    # DefaultDict default_factory that is neither None nor callable.
    bad_factory = (DEFAULTDICT, 2, (42, ['a', 'b']), None, None, 2, 3, keys_ab)
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, bad_factory), False, ''))

    # DefaultDict keys too few, and DefaultDict keys not distinct (the Dict variants are above).
    defaultdict_short = (DEFAULTDICT, 2, (int, ['a']), None, None, 2, 3, keys_ab)
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, defaultdict_short), False, ''))
    defaultdict_dup = (DEFAULTDICT, 2, (int, ['a', 'a']), None, None, 2, 3, keys_ab)
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, defaultdict_dup), False, ''))

    # Deque maxlen that is neither None nor an int, and maxlen smaller than the arity (a deque holds
    # at most maxlen items, so arity <= maxlen).
    with pytest.raises(malformed_exceptions):
        setstate((((DEQUE, 0, 'x', None, None, 0, 1, None),), False, ''))
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, (DEQUE, 2, 1, None, None, 2, 3, None)), False, ''))

    # A non-custom node carrying node entries or a custom type.
    with pytest.raises(malformed_exceptions):
        setstate(
            ((leaf_node, leaf_node, (TUPLE, 2, None, ('a', 'b'), None, 2, 3, None)), False, ''),
        )

    # Original keys whose count (not just key set) disagrees with the arity.
    short_original = (DICT, 2, ['a', 'b'], None, None, 2, 3, {'a': None})
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, short_original), False, ''))

    # Dict original_keys whose key set differs from the sorted key list.
    mismatched_original = (DICT, 2, ['a', 'b'], None, None, 2, 3, {'a': None, 'c': None})
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node, mismatched_original), False, ''))

    # A custom node whose node-entries count disagrees with the arity (needs a registered type).
    class MalformedCustomNode:
        pass

    optree.register_pytree_node(
        MalformedCustomNode,
        lambda obj: ((), None),
        lambda metadata, children: MalformedCustomNode(),
        namespace='malformed',
    )
    try:
        custom_node = (CUSTOM, 2, None, ('one-entry',), MalformedCustomNode, 2, 3, None)
        with pytest.raises(malformed_exceptions):
            setstate(((leaf_node, leaf_node, custom_node), False, 'malformed'))
    finally:
        optree.unregister_pytree_node(MalformedCustomNode, namespace='malformed')

    # A node claiming more children than the traversal provides.
    with pytest.raises(malformed_exceptions):
        setstate((((TUPLE, 2, None, None, None, 2, 3, None),), False, ''))

    # Inconsistent intermediate num_nodes (previously only the last node was checked).
    with pytest.raises(malformed_exceptions):
        setstate(
            (
                (
                    (LEAF, 0, None, None, None, 1, 5, None),  # leaf claims num_nodes == 5
                    leaf_node,
                    (TUPLE, 2, None, None, None, 2, 3, None),
                ),
                False,
                '',
            ),
        )

    # A traversal that yields more than one tree.
    with pytest.raises(malformed_exceptions):
        setstate(((leaf_node, leaf_node), False, ''))


def test_treespec_setstate_rejects_builtin_custom_type():
    # Regression: `FromPicklable` accepted any registration found for a CUSTOM node's custom type.
    # The built-in registrations (NoneType/tuple/list/dict/...) live in the same map but carry empty
    # flatten/unflatten callables, so the reconstructed node later called a null function pointer
    # and crashed the interpreter. Both `__setstate__` and `pickle.loads` must reject it.
    CUSTOM = int(optree.PyTreeKind.CUSTOM)  # noqa: N806
    LEAF = int(optree.PyTreeKind.LEAF)  # noqa: N806
    leaf_node = (LEAF, 0, None, None, None, 1, 1, None)

    for builtin_type in (list, dict, tuple, deque, OrderedDict, defaultdict, type(None)):
        state = ((leaf_node, (CUSTOM, 1, None, None, builtin_type, 1, 2, None)), False, '')
        obj = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
        with pytest.raises(RuntimeError, match=r'the custom type is a built-in type'):
            obj.__setstate__(state)

        # The same state shipped as a pickle payload, hand-assembled the way an attacker would:
        # `NEWOBJ` an empty spec, then `BUILD` it from the crafted state.
        blob = b''.join(
            [
                pickle.PROTO + bytes([2]),
                pickle.GLOBAL + b'optree\nPyTreeSpec\n',
                pickle.EMPTY_TUPLE + pickle.NEWOBJ,
                pickle.dumps(state, protocol=2)[2:-1],  # strip the PROTO / STOP framing
                pickle.BUILD + pickle.STOP,
            ],
        )
        with pytest.raises(RuntimeError, match=r'the custom type is a built-in type'):
            pickle.loads(blob)  # crafted payload, the point of the test


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_type(
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
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.is_leaf():
            assert treespec.type is None
        else:
            assert type(tree) is treespec.type


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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_compose_children(
    tree,
    inner_tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
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
        transformed_treespec = treespec.transform(None, lambda _: inner_treespec)
        expected_leaves = treespec.num_leaves * inner_treespec.num_leaves
        assert composed_treespec.num_leaves == treespec.num_leaves * inner_treespec.num_leaves
        assert transformed_treespec.num_leaves == expected_leaves
        expected_nodes = (treespec.num_nodes - treespec.num_leaves) + (
            inner_treespec.num_nodes * treespec.num_leaves
        )
        assert composed_treespec.num_nodes == expected_nodes
        assert transformed_treespec.num_nodes == expected_nodes
        leaves = list(range(expected_leaves))
        composed = optree.tree_unflatten(composed_treespec, leaves)
        transformed = optree.tree_unflatten(transformed_treespec, leaves)
        assert composed == transformed

        if 'FlatCache' in str(treespec):
            return

        assert (leaves, composed_treespec) == optree.tree_flatten(
            composed,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert (leaves, transformed_treespec) == optree.tree_flatten(
            transformed,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )

        assert composed_treespec == expected_treespec
        assert transformed_treespec == expected_treespec

        stack = [(composed_treespec.children(), expected_treespec.children())]
        while stack:
            composed_children, expected_children = stack.pop()
            for composed_child, expected_child in zip(composed_children, expected_children):
                assert composed_child == expected_child
                stack.append((composed_child.children(), expected_child.children()))

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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_entries(
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
        expected_paths, _, treespec = optree.tree_flatten_with_path(
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert optree.treespec_paths(treespec) == expected_paths

        def gen_path(spec):
            entries = optree.treespec_entries(spec)
            children = optree.treespec_children(spec)
            assert len(entries) == spec.num_children
            assert len(children) == spec.num_children
            assert entries is not optree.treespec_entries(spec)
            assert children is not optree.treespec_children(spec)
            optree.treespec_entries(spec).clear()
            optree.treespec_children(spec).clear()

            if spec.is_leaf():
                assert spec.num_children == 0
                yield ()
                return

            for entry, child in zip(entries, children):
                for suffix in gen_path(child):
                    yield (entry, *suffix)

        paths = list(gen_path(treespec))
        assert paths == expected_paths

        expected_accessors, _, other_treespec = optree.tree_flatten_with_accessor(
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert optree.treespec_accessors(treespec) == expected_accessors
        assert optree.treespec_accessors(other_treespec) == expected_accessors
        assert treespec == other_treespec

        def gen_typed_path(spec):
            entries = optree.treespec_entries(spec)
            children = optree.treespec_children(spec)
            assert len(entries) == spec.num_children
            assert len(children) == spec.num_children

            if spec.is_leaf():
                assert spec.num_children == 0
                yield ()
                return

            node_type = spec.type
            node_kind = spec.kind
            for entry, child in zip(entries, children):
                for suffix in gen_typed_path(child):
                    yield ((entry, node_type, node_kind), *suffix)

        typed_paths = list(gen_typed_path(treespec))
        expected_typed_paths = [
            tuple((e.entry, e.type, e.kind) for e in accessor) for accessor in expected_accessors
        ]
        assert typed_paths == expected_typed_paths


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_entry(
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
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.type is None or treespec.type is type(None):
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, 0)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, -1)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, 1)
        if treespec.is_leaf(strict=False):
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, 0)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, -1)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, 1)
        expected_entries = optree.treespec_entries(treespec)
        for i, entry in enumerate(expected_entries):
            assert entry == optree.treespec_entry(treespec, i)
            assert entry == optree.treespec_entry(treespec, i - len(expected_entries))
            assert optree.treespec_entry(treespec, i) == optree.treespec_entry(treespec, i)
            assert optree.treespec_entry(
                treespec,
                i - len(expected_entries),
            ) == optree.treespec_entry(
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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_child(
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
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.type is None or treespec.type is type(None):
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, 0)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, -1)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, 1)
        if treespec.is_leaf(strict=False):
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, 0)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, -1)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, 1)
        expected_children = optree.treespec_children(treespec)
        for i, child in enumerate(expected_children):
            assert child == optree.treespec_child(treespec, i)
            assert child == optree.treespec_child(treespec, i - len(expected_children))
            assert optree.treespec_child(treespec, i) == optree.treespec_child(treespec, i)
            assert optree.treespec_child(
                treespec,
                i - len(expected_children),
            ) == optree.treespec_child(
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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_one_level(
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
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.type is None:
            assert treespec.is_leaf()
            assert optree.treespec_one_level(treespec) is None
            assert optree.treespec_children(treespec) == []
            assert treespec.num_children == 0
        else:
            one_level = optree.treespec_one_level(treespec)
            counter = itertools.count()
            expected_treespec = optree.tree_structure(
                tree,
                is_leaf=lambda x: next(counter) > 0,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            num_children = treespec.num_children
            assert not treespec.is_leaf()
            assert not one_level.is_leaf()
            assert not expected_treespec.is_leaf()
            assert one_level == expected_treespec
            assert optree.treespec_one_level(one_level) == one_level
            assert optree.treespec_one_level(expected_treespec) == expected_treespec
            assert one_level.num_nodes == num_children + 1
            assert one_level.num_leaves == num_children
            assert one_level.num_children == num_children
            assert len(one_level) == num_children
            assert optree.treespec_entries(one_level) == optree.treespec_entries(treespec)
            assert all(optree.treespec_child(one_level, i).is_leaf() for i in range(num_children))
            assert all(child.is_leaf() for child in optree.treespec_children(one_level))
            assert optree.treespec_is_prefix(one_level, treespec)
            assert optree.treespec_is_suffix(treespec, one_level)
            assert (
                optree.treespec_from_collection(
                    optree.tree_unflatten(one_level, treespec.children()),
                    none_is_leaf=none_is_leaf,
                    namespace=namespace,
                )
                == treespec
            )
            it = iter(treespec.children())
            assert optree.treespec_transform(one_level, None, lambda _: next(it)) == treespec


def test_treespec_transform():
    treespec = optree.tree_structure(((1, 2, 3), (4,)))
    assert optree.treespec_transform(treespec) == treespec
    assert optree.treespec_transform(treespec) is not treespec
    assert optree.treespec_transform(
        treespec,
        None,
        lambda _: optree.tree_structure((1, [2])),
    ) == optree.tree_structure((((0, [1]), (2, [3]), (4, [5])), ((6, [7]),)))
    assert optree.treespec_transform(
        treespec,
        lambda spec: optree.treespec_list(spec.children()),
    ) == optree.tree_structure([[1, 2, 3], [4]])
    assert optree.treespec_transform(
        treespec,
        lambda spec: optree.treespec_dict(zip('abcd', spec.children())),
    ) == optree.tree_structure({'a': {'a': 0, 'b': 1, 'c': 2}, 'b': {'a': 3}})
    assert optree.treespec_transform(
        treespec,
        lambda spec: optree.treespec_dict(zip('abcd', spec.children())),
        lambda spec: optree.tree_structure([0, None, 1]),
    ) == optree.tree_structure(
        {'a': {'a': [0, None, 1], 'b': [2, None, 3], 'c': [4, None, 5]}, 'b': {'a': [6, None, 7]}},
    )
    namespaced_treespec = optree.tree_structure(
        MyAnotherDict({1: MyAnotherDict({2: 1, 1: 2, 0: 3}), 0: MyAnotherDict({0: 4})}),
        namespace='namespace',
    )
    assert (
        optree.treespec_transform(
            treespec,
            lambda spec: optree.tree_structure(
                MyAnotherDict(zip(spec.entries(), spec.children())),
                namespace='namespace',
            ),
        )
        == namespaced_treespec
    )
    assert optree.treespec_transform(
        namespaced_treespec,
        lambda spec: optree.treespec_list(spec.children()),
    ) == optree.tree_structure([[1, 2, 3], [4]])

    with pytest.raises(
        TypeError,
        match=re.escape('Expected the PyTreeSpec transform function returns a PyTreeSpec'),
    ):
        optree.treespec_transform(treespec, lambda _: None)

    with pytest.raises(
        TypeError,
        match=re.escape('Expected the PyTreeSpec transform function returns a PyTreeSpec'),
    ):
        optree.treespec_transform(treespec, None, lambda _: None)

    with pytest.raises(
        ValueError,
        match=(
            r'Expected the PyTreeSpec transform function returns '
            r'a PyTreeSpec with the same value of `none_is_leaf=\w+` as the input'
        ),
    ):
        optree.treespec_transform(
            treespec,
            lambda spec: optree.treespec_list(
                [optree.treespec_leaf(none_is_leaf=True)] * spec.num_children,
                none_is_leaf=True,
            ),
        )

    def fn(spec):
        with optree.dict_insertion_ordered(True, namespace='undefined'):
            return optree.treespec_dict(zip('abcd', spec.children()), namespace='undefined')

    with pytest.raises(ValueError, match=r'Expected treespec\(s\) with namespace .*, got .*\.'):
        optree.treespec_transform(namespaced_treespec, fn)

    with pytest.raises(
        ValueError,
        match=re.escape(
            'Expected the PyTreeSpec transform function returns '
            'a PyTreeSpec with the same number of arity as the input',
        ),
    ):
        optree.treespec_transform(treespec, lambda _: optree.tree_structure([0, 1]))

    with pytest.raises(
        ValueError,
        match=re.escape(
            'Expected the PyTreeSpec transform function returns '
            'a one-level PyTreeSpec as the input',
        ),
    ):
        optree.treespec_transform(
            treespec,
            lambda spec: optree.tree_structure([None] + [0] * spec.num_children),
        )


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_num_nodes(
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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_num_leaves(
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
        assert treespec.num_leaves == len(leaves)
        assert treespec.num_leaves == len(treespec)
        assert treespec.num_leaves == len(treespec.paths())
        assert treespec.num_leaves == len(treespec.accessors())


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_num_children(
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
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_is_one_level(
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
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.type is None:
            assert treespec.is_leaf()
            assert optree.treespec_one_level(treespec) is None
            assert not optree.treespec_is_one_level(treespec)
        else:
            one_level = optree.treespec_one_level(treespec)
            counter = itertools.count()
            expected_treespec = optree.tree_structure(
                tree,
                is_leaf=lambda x: next(counter) > 0,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            assert not treespec.is_leaf()
            assert not one_level.is_leaf()
            assert not expected_treespec.is_leaf()
            assert one_level == expected_treespec
            assert optree.treespec_one_level(one_level) == one_level
            assert optree.treespec_one_level(expected_treespec) == expected_treespec
            assert optree.treespec_is_one_level(one_level)
            assert optree.treespec_is_one_level(expected_treespec)
            assert optree.treespec_is_one_level(treespec) == (treespec == one_level)
            assert optree.treespec_is_one_level(treespec) == (treespec == expected_treespec)


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
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_constructor(  # noqa: C901
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    use_sorted_keys = dict_should_be_sorted or dict_session_namespace not in {'', namespace}
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
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

                    if node_type in {type(None), tuple, list}:
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
                        if use_sorted_keys:
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
                        else:
                            context = (
                                optree.dict_insertion_ordered(
                                    True,
                                    namespace=passed_namespace or GLOBAL_NAMESPACE,
                                )
                                if dict_session_namespace != passed_namespace
                                else contextlib.nullcontext()
                            )
                            with context:
                                assert (
                                    optree.treespec_dict(
                                        zip(node, children_treespecs),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                                assert (
                                    optree.treespec_from_collection(
                                        dict(zip(node, children_treespecs)),
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
                        if use_sorted_keys:
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
                        else:
                            context = (
                                optree.dict_insertion_ordered(
                                    True,
                                    namespace=passed_namespace or GLOBAL_NAMESPACE,
                                )
                                if dict_session_namespace != passed_namespace
                                else contextlib.nullcontext()
                            )
                            with context:
                                assert (
                                    optree.treespec_defaultdict(
                                        node.default_factory,
                                        zip(node, children_treespecs),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                                assert (
                                    optree.treespec_from_collection(
                                        defaultdict(
                                            node.default_factory,
                                            zip(node, children_treespecs),
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
        def __tree_flatten__(self):
            return self.data, None, None

        @classmethod
        def __tree_unflatten__(cls, metadata, children):
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
