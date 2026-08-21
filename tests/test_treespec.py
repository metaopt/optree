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

import builtins
import contextlib
import gc
import itertools
import math
import os
import pickle
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import warnings
import weakref
from collections import OrderedDict, UserList, defaultdict, deque, namedtuple

import pytest

import helpers
import optree
from helpers import (
    GLOBAL_NAMESPACE,
    HAS_DEFERRED_TYPE_REFS,
    NAMESPACED_TREE,
    NODETYPE_REGISTRY,
    OPTREE_HAS_FROZENDICT,
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
    skipif_deferred_type_refs,
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


@skipif_pypy  # relies on CPython's reference-cycle collector
@skipif_deferred_type_refs
def test_treespec_custom_node_reference_cycle_is_collectable():
    # A treespec reaches a registered custom type through the shared registration held by its custom
    # node. Once the registry no longer pins the registration, the node is its sole owner, so
    # `PyTreeSpec::PyTpTraverse` reports those objects and the cyclic GC can collect a cycle through
    # them. The cycle keeps the heap type alive, and free-threaded builds before 3.14 hold deferred
    # references to type objects in per-thread caches, so `gc_collect()` cannot reclaim it there.
    # 3.14t does, so this keeps free-threaded coverage.
    class Cyclic:
        pass

    optree.register_pytree_node(
        Cyclic,
        lambda cyclic: ((), None),
        lambda metadata, children: None,  # never called; the test only needs the registration
        namespace='cycle_gc',
    )
    try:
        treespec = optree.tree_structure(Cyclic(), namespace='cycle_gc')
        # Cycle: Cyclic -> __dict__ -> treespec -> registration -> Cyclic
        Cyclic.self_spec = treespec
    finally:
        optree.unregister_pytree_node(Cyclic, namespace='cycle_gc')

    wr = weakref.ref(Cyclic)
    del Cyclic, treespec
    gc_collect()
    assert wr() is None


@skipif_pypy  # relies on CPython's reference-cycle collector
@skipif_deferred_type_refs
def test_treespec_custom_node_reference_cycle_is_collectable_with_repeated_nodes():
    # Regression: the traverse reported a registration's members only when a single node held it,
    # so a treespec containing the same registered type more than once never reported them and the
    # cycle leaked. The treespec collectively owns the registration in that case too: what matters
    # is that no one outside it holds a reference.
    for num_nodes in (1, 2, 5):

        class Cyclic:
            pass

        optree.register_pytree_node(
            Cyclic,
            lambda cyclic: ((), None),
            lambda metadata, children: None,
            namespace='cycle_gc_repeated',
        )
        try:
            tree = [Cyclic() for _ in range(num_nodes)]
            treespec = optree.tree_structure(tree, namespace='cycle_gc_repeated')
            Cyclic.self_spec = treespec
        finally:
            optree.unregister_pytree_node(Cyclic, namespace='cycle_gc_repeated')

        wr = weakref.ref(Cyclic)
        del Cyclic, treespec, tree
        gc_collect()
        assert wr() is None, num_nodes


@skipif_pypy  # relies on CPython's reference-cycle collector
def test_treespec_shared_registration_refs_are_not_reported():
    # `PyTreeSpec::PyTpTraverse` must report only references the treespec owns.
    # A shared registration holds one reference to each member however many nodes point at it, so
    # reporting per node would underflow the object's shadow refcount and abort on debug builds.
    # `gc.get_referrers()` walks `tp_traverse`, so it shows what the traversal reports.
    class Shared:
        pass

    optree.register_pytree_node(
        Shared,
        lambda shared: ((), None),
        lambda metadata, children: None,
        namespace='shared_gc',
    )
    try:
        # The registry holds the registration, so no treespec is its sole owner.
        treespecs = [optree.tree_structure(Shared(), namespace='shared_gc') for _ in range(4)]
        gc_collect()
        assert not any(treespec in gc.get_referrers(Shared) for treespec in treespecs)
        # The registration is also shared between treespecs, so dropping the registry's hold while
        # more than one treespec remains must not make them report it either.
        optree.unregister_pytree_node(Shared, namespace='shared_gc')
        gc_collect()
        assert not any(treespec in gc.get_referrers(Shared) for treespec in treespecs)
    finally:
        del treespecs

    wr = weakref.ref(Shared)
    del Shared
    gc_collect()
    if not HAS_DEFERRED_TYPE_REFS:
        assert wr() is None


@skipif_pypy  # relies on CPython's reference-cycle collector
def test_treespec_shared_registration_is_still_not_reported_with_repeated_nodes():
    # The counterpart: while anything outside the treespec holds the registration, its members must
    # not be reported however many nodes reference it, or the collector's shadow refcount underflows.
    class Shared:
        pass

    optree.register_pytree_node(
        Shared,
        lambda shared: ((), None),
        lambda metadata, children: None,
        namespace='shared_gc_repeated',
    )
    treespec = optree.tree_structure([Shared(), Shared()], namespace='shared_gc_repeated')
    other = optree.tree_structure(Shared(), namespace='shared_gc_repeated')
    gc_collect()
    # The registry still holds it.
    assert treespec not in gc.get_referrers(Shared)
    optree.unregister_pytree_node(Shared, namespace='shared_gc_repeated')
    gc_collect()
    # `other` still holds it.
    assert treespec not in gc.get_referrers(Shared)
    del other
    gc_collect()
    # Now the treespec's two nodes are the only holders, so it reports the members once.
    assert treespec in gc.get_referrers(Shared)
    del treespec

    wr = weakref.ref(Shared)
    del Shared
    gc_collect()
    if not HAS_DEFERRED_TYPE_REFS:
        assert wr() is None


@skipif_pypy  # relies on CPython's reference-cycle collector
@pytest.mark.xfail(
    strict=True,
    reason='known limitation: a treespec cannot see registration references held by another treespec',
    raises=AssertionError,
)
def test_treespec_reference_cycle_across_treespecs_is_collectable():
    # Known limitation. A treespec reports a registration's members only when its own nodes hold
    # every reference to it, because that is all it can count. When two treespecs each hold some of
    # the references, neither sees the other's, so neither reports the members and a cycle through
    # them survives even though the two treespecs jointly own the registration.
    #
    # Resolving this needs the registration itself to be a garbage-collected object with its own
    # `tp_traverse`, so each edge is reported by whoever owns it and no counting is required.
    # Until then this is strictly better than reporting nothing at all, which never collects any of
    # these cycles.
    class Cyclic:
        pass

    optree.register_pytree_node(
        Cyclic,
        lambda cyclic: ((), None),
        lambda metadata, children: None,
        namespace='cycle_gc_across',
    )
    try:
        treespecs = [optree.tree_structure(Cyclic(), namespace='cycle_gc_across') for _ in range(2)]
        Cyclic.self_specs = treespecs
    finally:
        optree.unregister_pytree_node(Cyclic, namespace='cycle_gc_across')

    wr = weakref.ref(Cyclic)
    del Cyclic, treespecs
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


def test_treeiter_leaf_predicate_no_reference_leak():
    # A reference cycle that runs through the `leaf_predicate` callback must be collectable.
    # Regression: `PyTreeIter` tp_traverse / tp_clear previously ignored `m_leaf_predicate`, so a
    # cycle through the predicate was invisible to the cyclic garbage collector and leaked.
    def is_leaf(x):
        return False

    it = optree.tree_iter({'a': 1, 'b': {'c': 2}}, is_leaf)
    wr = weakref.ref(it)
    assert next(it) == 1
    is_leaf.self_ref = it  # cycle: it -> m_leaf_predicate (is_leaf) -> is_leaf.self_ref -> it

    del it, is_leaf
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

        import sys
        from collections import OrderedDict, defaultdict

        trees = [
            {'a': [1, 2], 'b': (3, 4)},
            OrderedDict([('b', 1), ('a', 2)]),
            defaultdict(int, {'b': 1, 'a': 2}),
        ]
        if sys.version_info >= (3, 15) and optree._C.OPTREE_HAS_FROZENDICT:
            # `frozendict` node data is the same mutable key list as `dict`, and the protocol 0/1
            # path reduces through `copyreg.__newobj__` rather than pybind11's own reduction.
            trees.append(frozendict({'b': 1, 'a': 2, 'c': (3, 4)}))

        for tree in trees:
            spec = optree.tree_structure(tree)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                restored = pickle.loads(pickle.dumps(spec, protocol=protocol))
                assert restored == spec, (protocol, restored, spec)
                assert restored.unflatten(range(spec.num_leaves)) == spec.unflatten(
                    range(spec.num_leaves),
                ), (protocol, tree)
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

    # The same negatives for FROZENDICT. `FrozenDict` rides on shared `||` conditions with `Dict`
    # at ~10 sites in `PyTreeSpec::FromPicklable`; dropping one would let a crafted pickle build a
    # corrupt spec with no other test noticing.
    if sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT:  # pragma: >=3.15 cover
        FROZENDICT = int(optree.PyTreeKind.FROZENDICT)  # noqa: N806

        # Missing its original keys, and short/duplicate/unhashable key lists.
        with pytest.raises(malformed_exceptions):
            setstate(
                (
                    (leaf_node, leaf_node, (FROZENDICT, 2, ['a', 'b'], None, None, 2, 3, None)),
                    False,
                    '',
                ),
            )
        for bad_keys in (['a'], ['a', 'a'], [[], []]):
            with pytest.raises(malformed_exceptions):
                setstate(
                    (
                        (
                            leaf_node,
                            leaf_node,
                            (FROZENDICT, 2, bad_keys, None, None, 2, 3, keys_ab),
                        ),
                        False,
                        '',
                    ),
                )
        # Original keys that do not match the node's own keys.
        with pytest.raises(malformed_exceptions):
            setstate(
                (
                    (
                        leaf_node,
                        leaf_node,
                        (FROZENDICT, 2, ['a', 'b'], None, None, 2, 3, {'a': None, 'c': None}),
                    ),
                    False,
                    '',
                ),
            )
        # A well-formed frozendict node still round-trips.
        restored = setstate(
            (
                (leaf_node, leaf_node, (FROZENDICT, 2, ['a', 'b'], None, None, 2, 3, keys_ab)),
                False,
                '',
            ),
        )
        assert restored == optree.tree_structure(builtins.frozendict({'a': 1, 'b': 2}))

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

    builtin_types = [list, dict, tuple, deque, OrderedDict, defaultdict, type(None)]
    if sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT:  # pragma: >=3.15 cover
        builtin_types.append(builtins.frozendict)  # type: ignore[attr-defined]

    for builtin_type in builtin_types:
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
            for composed_child, expected_child in zip(
                composed_children,
                expected_children,
                strict=True,
            ):
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


def test_treespec_compose_rejects_incompatible_namespace_merge():
    # Regression: composing an empty-namespace spec (whose custom nodes are resolved globally) with
    # a spec in another namespace adopted that namespace but kept the global registrations. When the
    # same type is registered differently in the two namespaces, the composed spec silently used the
    # wrong flatten/unflatten (spurious flatten_up_to errors; corrupt pickle). Reject the merge.
    class Pair:
        def __init__(self, a, b):
            self.a, self.b = a, b

    class Single:
        def __init__(self, x):
            self.x = x

    optree.register_pytree_node(
        Pair,
        lambda t: ((t.a, t.b), None, None),
        lambda m, c: Pair(c[0], c[1]),
        namespace=GLOBAL_NAMESPACE,
    )
    optree.register_pytree_node(  # behavior differs from the global registration
        Pair,
        lambda t: ((t.b, t.a), None, None),
        lambda m, c: Pair(c[1], c[0]),
        namespace='behavior_change',
    )
    optree.register_pytree_node(
        Single,
        lambda t: ((t.x,), None, None),
        lambda m, c: Single(c[0]),
        namespace='behavior_change',
    )
    try:
        outer = optree.tree_structure(Pair(0, 0))
        inner = optree.tree_structure(Single(0), namespace='behavior_change')
        assert outer.namespace == ''
        assert inner.namespace == 'behavior_change'
        with pytest.raises(ValueError, match='original registration'):
            outer.compose(inner)

        # `tree_transpose` builds its expected structure with `compose`, so the rejection surfaces
        # through the public API too (here via its structure-mismatch diagnostic path).
        with pytest.raises(ValueError, match='original registration'):
            optree.tree_transpose(outer, inner, [1, 2, 3])

        # `broadcast_to_common_suffix` adopts the namespace the same way.
        self_spec = optree.tree_structure({'k': Pair(0, 0)})
        other_spec = optree.tree_structure({'k': Single(0)}, namespace='behavior_change')
        with pytest.raises(ValueError, match='original registration'):
            self_spec.broadcast_to_common_suffix(other_spec)
    finally:
        optree.unregister_pytree_node(Pair, namespace=GLOBAL_NAMESPACE)
        optree.unregister_pytree_node(Pair, namespace='behavior_change')
        optree.unregister_pytree_node(Single, namespace='behavior_change')


def test_treespec_compose_allows_compatible_namespace_merge():
    # The namespace-merge rejection must not over-reject: a custom type registered only globally
    # resolves identically under any namespace (via global fallback), so merging an empty-namespace
    # spec that uses it into another namespace is allowed and the result stays consistent.
    class GlobalOnly:
        def __init__(self, a, b):
            self.a, self.b = a, b

    optree.register_pytree_node(
        GlobalOnly,
        lambda t: ((t.a, t.b), None, None),
        lambda m, c: GlobalOnly(c[0], c[1]),
        namespace=GLOBAL_NAMESPACE,
    )
    try:
        outer = optree.tree_structure(GlobalOnly(0, 0))
        assert outer.namespace == ''

        # Both empty -> the merge stays in the global namespace.
        assert outer.compose(optree.tree_structure(0)).namespace == ''

        # Empty side (global-only custom) merged into a namespace: allowed, adopts the namespace,
        # and unflattens consistently (the global registration is used throughout).
        with optree.dict_insertion_ordered(True, namespace='no_override'):
            inner = optree.tree_structure({'x': 0}, namespace='no_override')
        assert inner.namespace == 'no_override'
        composed = outer.compose(inner)
        assert composed.namespace == 'no_override'
        result = optree.tree_unflatten(composed, [1, 2])
        assert isinstance(result, GlobalOnly)
        assert result.a == {'x': 1}
        assert result.b == {'x': 2}

        # The cross-namespace merge equals building the composed structure directly with `tree_map`
        # in the adopted namespace, compose's defining identity.
        expected = optree.tree_structure(
            optree.tree_map(lambda _: {'x': 0}, GlobalOnly(0, 0), namespace='no_override'),
            namespace='no_override',
        )
        assert composed == expected

        # broadcast_to_common_suffix likewise allows the compatible merge.
        broadcasted = outer.broadcast_to_common_suffix(
            optree.tree_structure(GlobalOnly(0, 0), namespace='no_override'),
        )
        assert broadcasted.namespace == 'no_override'
    finally:
        optree.unregister_pytree_node(GlobalOnly, namespace=GLOBAL_NAMESPACE)


def test_treespec_broadcast_to_common_suffix_does_not_mutate_argument_on_key_mismatch():
    # Regression: BroadcastToCommonSuffixImpl built the "got key(s)" part of its key-mismatch error
    # message by sorting the ARGUMENT spec's live dict-node key list IN PLACE: `other_keys` was a
    # borrow of `node_data`, not a copy. For an OrderedDict the child subtrees stay in insertion
    # order while the keys get permuted, silently corrupting a spec the caller still holds: repr,
    # equality, hash, and unflatten all go wrong. The message must be built from a sorted COPY.
    other = optree.tree_structure(OrderedDict([('c', 1), ('b', 2)]))
    before_repr = str(other)
    before_hash = hash(other)
    this = optree.tree_structure({'a': 1})
    with pytest.raises(ValueError, match='dictionary key mismatch'):
        this.broadcast_to_common_suffix(other)
    # The argument spec must be byte-for-byte unchanged by the failed call.
    assert str(other) == before_repr
    assert hash(other) == before_hash
    # And it must still unflatten in its ORIGINAL insertion order (c, b), not a sorted (b, c) order.
    assert other.unflatten([10, 20]) == OrderedDict([('c', 10), ('b', 20)])


def test_treespec_broadcast_to_common_suffix_preserves_custom_node_entries():
    # Broadcasting rebuilds each non-leaf node, so `.node_entries` must be carried across: losing
    # it breaks accessors, which fall back to `range(arity)` (`GetAttrEntry(entry=0)`).
    class Vector:
        def __init__(self, a, c):
            self.a, self.c = a, c

    optree.register_pytree_node(
        Vector,
        lambda o: ((o.a, o.c), None, ('a', 'c')),  # 3-tuple flatten -> node_entries = ('a', 'c')
        lambda metadata, children: Vector(*children),
        path_entry_type=optree.GetAttrEntry,
        namespace=GLOBAL_NAMESPACE,
    )
    try:
        spec = optree.tree_structure(Vector(1, 2))
        other = optree.tree_structure(Vector(3, 4))
        assert spec.entries() == ['a', 'c']

        # Both specs share the same custom structure, so the common suffix is that structure and the
        # explicit string entries must survive unchanged, not degrade to the fallback [0, 1].
        broadcasted = spec.broadcast_to_common_suffix(other)
        assert broadcasted.entries() == ['a', 'c']
        assert broadcasted.paths() == spec.paths()
        assert broadcasted.accessors() == spec.accessors()
    finally:
        optree.unregister_pytree_node(Vector, namespace=GLOBAL_NAMESPACE)


def test_treespec_deep_walk_raises_recursion_error_not_segfault():
    # Regression: `PathsImpl`, `AccessorsImpl`, and `BroadcastToCommonSuffixImpl` recurse once per
    # tree level. Without a depth guard, a deeply-nested spec (trivially built via doubling
    # `compose`) overflowed the native C++ stack and crashed the interpreter with a SIGSEGV instead
    # of raising a catchable `RecursionError`.
    # Each `compose` doubles the depth, so ceil(log2(limit)) + 1 composes push it above the limit.
    num_composes = math.ceil(math.log2(optree.MAX_RECURSION_DEPTH)) + 1
    deep = optree.tree_structure([0])
    for _ in range(num_composes):
        deep = deep.compose(deep)
    assert 2**num_composes > optree.MAX_RECURSION_DEPTH
    with pytest.raises(RecursionError):
        deep.paths()
    with pytest.raises(RecursionError):
        deep.accessors()
    with pytest.raises(RecursionError):
        deep.broadcast_to_common_suffix(deep)

    # Broadcasting the deep spec against a shallower spec whose depth is still below the limit
    # recurses only as far as the common suffix, so it must succeed (not raise RecursionError or
    # crash) in either direction, returning the deeper spec.
    shallower = optree.tree_structure([0])
    for _ in range(num_composes - 2):  # depth 2 ** (num_composes - 2), safely below the limit
        shallower = shallower.compose(shallower)
    assert 2 ** (num_composes - 2) < optree.MAX_RECURSION_DEPTH
    assert deep.broadcast_to_common_suffix(shallower) == deep
    assert shallower.broadcast_to_common_suffix(deep) == deep


def test_treespec_compose_rejects_namespace_override_with_different_arity():
    # A type registered globally flattens both members as children (arity 2); a namespace override
    # flattens one member as a child and stores the other as node metadata (arity 1). Both
    # registrations round-trip, but merging an empty-namespace spec (global, arity 2) into that
    # namespace must be rejected: the composed spec would claim the namespace while carrying an
    # arity-2 node that the namespace's registration cannot unflatten.
    class TwoMember:
        def __init__(self, a, b):
            self.a, self.b = a, b

        def __eq__(self, other):
            return isinstance(other, TwoMember) and (self.a, self.b) == (other.a, other.b)

        __hash__ = None

    optree.register_pytree_node(
        TwoMember,
        lambda t: ((t.a, t.b), None, None),  # global: both members are children
        lambda metadata, children: TwoMember(children[0], children[1]),
        namespace=GLOBAL_NAMESPACE,
    )
    optree.register_pytree_node(
        TwoMember,
        lambda t: ((t.a,), t.b, None),  # override: one child, the other is metadata
        lambda metadata, children: TwoMember(children[0], metadata),
        namespace='arity_change',
    )
    try:
        obj = TwoMember(1, 2)

        # Both registrations round-trip on their own.
        global_leaves, global_spec = optree.tree_flatten(obj)
        assert global_leaves == [1, 2]
        assert optree.tree_unflatten(global_spec, global_leaves) == obj
        custom_leaves, custom_spec = optree.tree_flatten(obj, namespace='arity_change')
        assert custom_leaves == [1]
        assert optree.tree_unflatten(custom_spec, custom_leaves) == obj

        assert global_spec.namespace == ''
        assert global_spec.num_leaves == 2
        assert custom_spec.namespace == 'arity_change'
        assert custom_spec.num_leaves == 1
        with pytest.raises(ValueError, match='original registration'):
            global_spec.compose(custom_spec)
    finally:
        optree.unregister_pytree_node(TwoMember, namespace=GLOBAL_NAMESPACE)
        optree.unregister_pytree_node(TwoMember, namespace='arity_change')


def test_treespec_transform_rejects_incompatible_namespace_merge():
    # `transform` unifies the namespace across the input spec and the transform outputs. If that
    # unified (non-empty) namespace rebinds a custom node (e.g. the input's globally-resolved
    # custom node) to a different registration, the transform must be rejected (same class as the
    # compose / broadcast merge rejection). A globally-only-registered type is still allowed via
    # fallback.
    class Diverge:  # variable arity; registered differently in the global and named namespaces
        def __init__(self, *children):
            self.children = children

    class GlobalOnly:  # variable arity; registered only globally -> resolves via fallback anywhere
        def __init__(self, *children):
            self.children = children

    optree.register_pytree_node(
        Diverge,
        lambda d: (d.children, None, None),
        lambda metadata, children: Diverge(*children),
        namespace=GLOBAL_NAMESPACE,
    )
    optree.register_pytree_node(
        Diverge,
        lambda d: (tuple(reversed(d.children)), None, None),  # divergent from the global reg
        lambda metadata, children: Diverge(*reversed(children)),
        namespace='transform_change',
    )
    optree.register_pytree_node(
        GlobalOnly,
        lambda g: (g.children, None, None),
        lambda metadata, children: GlobalOnly(*children),
        namespace=GLOBAL_NAMESPACE,
    )

    def to_namespaced_leaf(_):
        # Replace a leaf with a namespaced Diverge to inject the namespace (leaves have no arity).
        return optree.tree_structure(Diverge(0), namespace='transform_change')

    def to_global_node(spec):
        # Replace a node with a same-arity globally-resolved Diverge; it rebinds under the promoted
        # namespace (Diverge is registered differently there). Generic over the node's arity.
        return optree.tree_structure(Diverge(*range(spec.num_children)))

    def to_namespaced_node(spec):
        # Outer node -> global Diverge (rebinds); inner nodes -> namespaced Diverge (injects the
        # namespace). Both same-arity, so `f_node` alone drives the (f_node, None) rejection.
        if spec.type is tuple:
            return to_global_node(spec)
        return optree.tree_structure(
            Diverge(*range(spec.num_children)),
            namespace='transform_change',
        )

    try:
        # The rejection must fire for every `(f_node, f_leaf)` combination that puts a
        # globally-resolved custom node under the non-empty unified namespace.

        # (None, f_leaf): the input's global Diverge is kept, f_leaf injects the namespace.
        outer = optree.tree_structure(Diverge(0, 0))
        assert outer.namespace == ''
        with pytest.raises(ValueError, match='original registration'):
            outer.transform(None, to_namespaced_leaf)

        # (f_node, None): f_node alone yields a global Diverge above namespaced children.
        with pytest.raises(ValueError, match='original registration'):
            optree.tree_structure(([0], [0])).transform(to_namespaced_node, None)

        # (f_node, f_leaf): f_node injects the global Diverge, f_leaf injects the namespace.
        with pytest.raises(ValueError, match='original registration'):
            optree.tree_structure([0, 0]).transform(to_global_node, to_namespaced_leaf)

        # Compatible: GlobalOnly resolves identically under any namespace via fallback.
        global_outer = optree.tree_structure(GlobalOnly(0, 0))
        transformed = global_outer.transform(None, to_namespaced_leaf)
        assert transformed.namespace == 'transform_change'
    finally:
        optree.unregister_pytree_node(Diverge, namespace=GLOBAL_NAMESPACE)
        optree.unregister_pytree_node(Diverge, namespace='transform_change')
        optree.unregister_pytree_node(GlobalOnly, namespace=GLOBAL_NAMESPACE)


def test_treespec_from_collection_rejects_incompatible_namespace_promotion():
    # `treespec_from_collection` promotes an empty caller namespace to a child spec's namespace. If
    # that promoted namespace rebinds a custom node the collection resolved globally (the root node,
    # or a globally-resolved child) to a different registration, the result would claim the
    # namespace while carrying the wrong registration: it must be rejected, exactly like compose /
    # transform / broadcast. A globally-only-registered type is still allowed via fallback.
    class Diverge:  # variable arity; registered differently in the global and named namespaces
        def __init__(self, *children):
            self.children = children

    class GlobalOnly:  # variable arity; registered only globally -> resolves via fallback anywhere
        def __init__(self, *children):
            self.children = children

    optree.register_pytree_node(
        Diverge,
        lambda d: (d.children, None, None),
        lambda metadata, children: Diverge(*children),
        namespace=GLOBAL_NAMESPACE,
    )
    optree.register_pytree_node(
        Diverge,
        lambda d: (tuple(reversed(d.children)), None, None),  # divergent from the global reg
        lambda metadata, children: Diverge(*reversed(children)),
        namespace='from_coll_change',
    )
    optree.register_pytree_node(
        GlobalOnly,
        lambda g: (g.children, None, None),
        lambda metadata, children: GlobalOnly(*children),
        namespace=GLOBAL_NAMESPACE,
    )
    try:
        # Incompatible: the globally-resolved Diverge rebinds under the promoted namespace.
        foo = optree.tree_structure(Diverge(0, 0))
        child = optree.tree_structure(Diverge(0), namespace='from_coll_change')
        assert foo.namespace == ''
        with pytest.raises(ValueError, match='original registration'):
            optree.treespec_from_collection([foo, child], namespace='')

        # Compatible: GlobalOnly resolves identically under any namespace via fallback.
        global_spec = optree.tree_structure(GlobalOnly(0, 0))
        promoted = optree.treespec_from_collection([global_spec, child], namespace='')
        assert promoted.namespace == 'from_coll_change'
    finally:
        optree.unregister_pytree_node(Diverge, namespace=GLOBAL_NAMESPACE)
        optree.unregister_pytree_node(Diverge, namespace='from_coll_change')
        optree.unregister_pytree_node(GlobalOnly, namespace=GLOBAL_NAMESPACE)


def test_treespec_dict_key_order_survives_namespace_promotion():
    # A dict node's key order is fixed at BUILD time by the namespace passed then. Operations that
    # merge/promote a spec's namespace (`treespec_from_collection`, `compose`, `transform`) only
    # re-tag it for custom-node resolution; like `compose` they NEVER reorder an already-built dict.
    # So a dict built under the global ('') namespace (sorted keys) keeps that order even after
    # promotion to an insertion-ordered namespace, intentionally differing from the same dict built
    # directly under that namespace, while a dict built directly under the namespace keeps its
    # insertion order (matching). This test locks that behavior across all three operations.
    class Wrap:  # a variable-arity custom node, so `f_node` can build same-arity replacements
        def __init__(self, *children):
            self.children = children

    optree.register_pytree_node(
        Wrap,
        lambda w: (w.children, None, None),
        lambda metadata, children: Wrap(*children),
        namespace='promote_order',
    )
    try:
        with optree.dict_insertion_ordered(True, namespace='promote_order'):
            child = optree.tree_structure(Wrap(0), namespace='promote_order')
            # Built directly under the insertion-ordered namespace: keys in insertion order (b, a).
            genuine = optree.tree_structure({'b': Wrap(0), 'a': Wrap(0)}, namespace='promote_order')
            assert genuine.entries() == ['b', 'a']

            def to_namespaced_node(spec):
                # Rewrite every non-dict node into a same-arity `Wrap` in the namespace (generic
                # over the node's arity rather than tied to this test's shapes) so `f_node` alone
                # can promote the spec. `transform` promotes only when some output carries a
                # namespace, and only a custom node can. The outer dict node is kept so its key
                # order stays observable.
                if spec.type is dict:
                    return spec
                return optree.tree_structure(
                    Wrap(*range(spec.num_children)),
                    namespace='promote_order',
                )

            def transform_combos(outer):
                return {
                    'transform(None, f_leaf)': outer.transform(None, lambda _: child),
                    'transform(f_node, None)': outer.transform(to_namespaced_node, None),
                    'transform(f_node, f_leaf)': outer.transform(
                        to_namespaced_node,
                        lambda _: child,
                    ),
                }

            # Dicts built under the GLOBAL ('') namespace, sorted keys (a, b), then promoted.
            from_global = {
                'from_collection': optree.treespec_from_collection(
                    {'b': child, 'a': child},
                    namespace='',
                ),
                'compose': optree.tree_structure({'b': 0, 'a': 0}).compose(child),
                **transform_combos(optree.tree_structure({'b': [0], 'a': [0]})),
            }
            # Dicts built directly under the namespace, insertion-order keys (b, a).
            from_namespace = {
                'from_collection': optree.treespec_from_collection(
                    {'b': child, 'a': child},
                    namespace='promote_order',
                ),
                'compose': optree.tree_structure(
                    {'b': 0, 'a': 0},
                    namespace='promote_order',
                ).compose(child),
                **transform_combos(
                    optree.tree_structure({'b': [0], 'a': [0]}, namespace='promote_order'),
                ),
            }

        for name, spec in from_global.items():
            assert spec.namespace == 'promote_order', name  # promoted for custom resolution ...
            assert spec.entries() == ['a', 'b'], name  # ... but the dict keeps '' (sorted) order

        for name, spec in from_namespace.items():
            assert spec.namespace == 'promote_order', name
            assert spec.entries() == ['b', 'a'], name  # insertion order kept, matches direct build

        # from_collection / compose reproduce genuine's flat structure exactly, so the only
        # difference is the dict key order: global-built differs, namespace-built matches.
        assert from_global['from_collection'] != genuine
        assert from_global['compose'] != genuine
        assert from_namespace['from_collection'] == genuine
        assert from_namespace['compose'] == genuine
    finally:
        optree.unregister_pytree_node(Wrap, namespace='promote_order')


def test_treespec_is_prefix_nested_dict_key_reorder():
    # Regression: `IsPrefix` reorders a dict node's children in a working copy of the traversal to
    # make key order irrelevant. When a NESTED dict also needed reordering, it indexed the pristine
    # traversal by an offset into the already-mutated working copy, corrupting it -> a spurious
    # `optree._C.InternalError` or a wrong boolean. Two treespecs that describe the SAME tree
    # (differing only in dict key insertion order, at nested levels) must be mutual non-strict
    # prefixes / suffixes.

    # Top-level AND nested dict keys reordered; the top-level reorder relocates the nested dict.
    tree_a = OrderedDict([('a', 0), ('b', OrderedDict([('e', 0), ('g', 0)])), ('d', 0)])
    tree_b = OrderedDict([('b', OrderedDict([('g', 0), ('e', 0)])), ('a', 0), ('d', 0)])
    a = optree.tree_structure(tree_a)
    b = optree.tree_structure(tree_b)
    assert optree.treespec_is_prefix(a, b, strict=False)
    assert optree.treespec_is_prefix(b, a, strict=False)
    assert optree.treespec_is_suffix(a, b, strict=False)
    assert optree.treespec_is_suffix(b, a, strict=False)
    assert a <= b
    assert b <= a
    assert a >= b
    assert b >= a

    # A nested dict whose reorder relocates a subtree containing another out-of-order dict.
    tree_a2 = OrderedDict([('a', 0), ('d', 0), ('b', OrderedDict([('e', 0), ('f', 0)]))])
    tree_b2 = OrderedDict([('b', OrderedDict([('f', 0), ('e', 0)])), ('d', 0), ('a', 0)])
    a2 = optree.tree_structure(tree_a2)
    b2 = optree.tree_structure(tree_b2)
    assert optree.treespec_is_prefix(a2, b2, strict=False)
    assert optree.treespec_is_prefix(b2, a2, strict=False)
    assert a2 <= b2
    assert b2 <= a2


def test_treespec_is_prefix_deque_maxlen_agnostic():
    # A deque's treespec stores both its arity and its `maxlen`, but `is_prefix` is arity-based and
    # deliberately `maxlen`-AGNOSTIC. A deque holds at most `maxlen` items, so `arity <= maxlen`
    # always holds and two flatten-compatible deques necessarily share the same arity while carrying
    # any `maxlen1`/`maxlen2`; `maxlen` does not affect how children are partitioned, so gating the
    # prefix relation on it would wrongly reject valid `flatten_up_to`/`broadcast_prefix` operations.
    # `EqualTo`, by contrast, IS `maxlen`-sensitive (`unflatten` restores the exact `maxlen`), so
    # `a <= b and b <= a` does NOT imply `a == b`: `is_prefix` is a preorder, not a partial order.
    a = optree.tree_structure(deque([1, 2, 3], maxlen=3))
    b = optree.tree_structure(deque([1, 2, 3], maxlen=5))
    unbounded = optree.tree_structure(deque([1, 2, 3]))  # maxlen=None
    # Equality distinguishes maxlen (bounded vs bounded, and bounded vs unbounded).
    assert a != b
    assert a != unbounded
    assert b != unbounded

    # Same arity, any maxlen (bounded or unbounded): mutual non-strict prefixes and suffixes,
    # even though the specs are unequal, so mutual prefixes do NOT imply equality (a preorder).
    for x, y in itertools.permutations([a, b, unbounded], 2):
        assert x != y
        assert optree.treespec_is_prefix(x, y, strict=False)
        assert optree.treespec_is_suffix(x, y, strict=False)
        assert x <= y
        assert x >= y

    # Practical consequence: a prefix deque flattens / broadcasts a full deque of a different maxlen.
    prefix_spec = optree.tree_structure(deque([1, 2, 3], maxlen=None))
    assert prefix_spec.flatten_up_to(deque([[10], [20, 21], [30]], maxlen=5)) == [
        [10],
        [20, 21],
        [30],
    ]
    assert optree.broadcast_prefix(
        deque([1, 2, 3], maxlen=3),
        deque([[0], [0, 0], [0]], maxlen=7),
    ) == [1, 2, 2, 3]

    # Arity still gates the relation: a different-arity deque is not a prefix.
    assert not optree.treespec_is_prefix(
        optree.tree_structure(deque([1, 2], maxlen=9)),
        a,
        strict=False,
    )


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

            for entry, child in zip(entries, children, strict=True):
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
            for entry, child in zip(entries, children, strict=True):
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


def test_treespec_entry_and_child_accept_int_like_indices():
    # The compiled signatures advertise `SupportsInt | SupportsIndex`, and the runtime honors both,
    # so the stubs must not narrow them to `int`.
    class OnlyIndex:
        def __index__(self):
            return 1

    class OnlyInt:
        def __int__(self):
            return 1

    treespec = optree.tree_structure({'a': 1, 'b': 2})
    for index in (1, OnlyIndex(), OnlyInt()):
        assert treespec.entry(index) == treespec.entry(1), index
        assert treespec.child(index) == treespec.child(1), index
        # The public wrappers must not narrow what the methods they forward to accept.
        assert optree.treespec_entry(treespec, index) == treespec.entry(1), index
        assert optree.treespec_child(treespec, index) == treespec.child(1), index


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
        lambda spec: optree.treespec_dict(zip('abcd', spec.children(), strict=False)),
    ) == optree.tree_structure({'a': {'a': 0, 'b': 1, 'c': 2}, 'b': {'a': 3}})
    assert optree.treespec_transform(
        treespec,
        lambda spec: optree.treespec_dict(zip('abcd', spec.children(), strict=False)),
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
                MyAnotherDict(zip(spec.entries(), spec.children(), strict=True)),
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
            return optree.treespec_dict(
                zip('abcd', spec.children(), strict=False),
                namespace='undefined',
            )

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


def test_treespec_from_collection_on_leaf_propagates_escalated_warning():
    # `treespec_from_collection()` on a leaf issues a UserWarning via `PyErr_WarnEx()`. When
    # warnings are escalated to errors (e.g. `-W error`), the escalation must propagate cleanly as
    # that UserWarning: the C++ code must check `PyErr_WarnEx()`'s return value and raise, not
    # ignore it and return a result with the exception left set (which pybind11 surfaces as a
    # confusing `SystemError: ... returned a result with an exception set`).
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(
            UserWarning,
            match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
        ):
            optree.treespec_from_collection(1)


def test_treespec_from_collection_drops_namespace_for_childless_roots():
    # Regression: a leaf or `None` root skipped the namespace-dropping step, so the caller's
    # namespace stuck to a treespec that resolves no custom type. Two otherwise-identical treespecs
    # built under different namespaces then compared unequal.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for collection in (None, 1, 'leaf'):
            first = optree.treespec_from_collection(collection, namespace='first')
            second = optree.treespec_from_collection(collection, namespace='second')
            assert first.namespace == '', (collection, first.namespace)
            assert second.namespace == '', (collection, second.namespace)
            assert first == second, collection
            assert hash(first) == hash(second), collection


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
                                    zip(sorted(node), children_treespecs, strict=True),
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                            assert (
                                optree.treespec_from_collection(
                                    dict(zip(sorted(node), children_treespecs, strict=True)),
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
                                        zip(node, children_treespecs, strict=True),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                                assert (
                                    optree.treespec_from_collection(
                                        dict(zip(node, children_treespecs, strict=True)),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                    elif node_type is OrderedDict:
                        assert (
                            optree.treespec_ordereddict(
                                zip(node, children_treespecs, strict=True),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        assert (
                            optree.treespec_from_collection(
                                OrderedDict(zip(node, children_treespecs, strict=True)),
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
                                    zip(sorted(node), children_treespecs, strict=True),
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                            assert (
                                optree.treespec_from_collection(
                                    defaultdict(
                                        node.default_factory,
                                        zip(sorted(node), children_treespecs, strict=True),
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
                                        zip(node, children_treespecs, strict=True),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                                assert (
                                    optree.treespec_from_collection(
                                        defaultdict(
                                            node.default_factory,
                                            zip(node, children_treespecs, strict=True),
                                        ),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                    elif (
                        sys.version_info >= (3, 15)
                        and OPTREE_HAS_FROZENDICT
                        and node_type is builtins.frozendict  # type: ignore[attr-defined]
                    ):
                        if use_sorted_keys:
                            assert (
                                optree.treespec_frozendict(
                                    zip(sorted(node), children_treespecs, strict=True),
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                            assert (
                                optree.treespec_from_collection(
                                    builtins.frozendict(  # type: ignore[attr-defined]
                                        zip(sorted(node), children_treespecs, strict=True),
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
                                    optree.treespec_frozendict(
                                        zip(node, children_treespecs, strict=True),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                                assert (
                                    optree.treespec_from_collection(
                                        builtins.frozendict(  # type: ignore[attr-defined]
                                            zip(node, children_treespecs, strict=True),
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


def test_treespec_dict_constructor_preserves_insertion_ordered_namespace():
    # Regression: under `dict_insertion_ordered` mode the key order of a dict spec depends on the
    # namespace, so `treespec_dict(..., namespace=...)` must keep that namespace (like
    # `tree_flatten`) instead of resetting it to '': an empty-namespace spec with unsorted keys is
    # otherwise unreachable via `tree_flatten` and breaks equality/consistency.
    leaf = optree.tree_structure(0)

    with optree.dict_insertion_ordered(True, namespace='namespace'):
        constructed = optree.treespec_dict({'b': leaf, 'a': leaf}, namespace='namespace')
        _, flattened = optree.tree_flatten({'b': 1, 'a': 2}, namespace='namespace')

    assert constructed.entries() == ['b', 'a']  # insertion order preserved
    assert flattened.namespace == 'namespace'
    assert constructed.namespace == 'namespace'  # was '' before the fix
    assert constructed == flattened

    # Without the mode, keys are sorted and the namespace is dropped, same as `tree_flatten`.
    outside = optree.treespec_dict({'b': leaf, 'a': leaf}, namespace='namespace')
    _, flattened_outside = optree.tree_flatten({'b': 1, 'a': 2}, namespace='namespace')
    assert outside.entries() == ['a', 'b']
    assert outside.namespace == ''
    assert outside == flattened_outside


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


@pytest.mark.skipif(
    not (sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT),
    reason='`frozendict` requires Python 3.15+',
)
def test_treespec_frozendict_distinct_from_dict():
    # Even with identical keys and structure, a `frozendict` treespec must be distinct from a
    # `dict` treespec (and from `OrderedDict`/`defaultdict`) so users can rely on container
    # identity surviving flatten/unflatten and pickle round-trips. A regression collapsing
    # `PyTreeKind::FrozenDict` into `PyTreeKind::Dict` would only be caught indirectly by the
    # parametrized `TREES` suite; this test makes the invariant explicit.
    frozendict = builtins.frozendict  # type: ignore[attr-defined] # pylint: disable=no-member

    frozendict_treespec = optree.tree_structure(frozendict({'a': 1, 'b': 2}))
    dict_treespec = optree.tree_structure({'a': 1, 'b': 2})
    ordereddict_treespec = optree.tree_structure(OrderedDict([('a', 1), ('b', 2)]))
    defaultdict_treespec = optree.tree_structure(defaultdict(int, {'a': 1, 'b': 2}))

    assert frozendict_treespec != dict_treespec
    assert frozendict_treespec != ordereddict_treespec
    assert frozendict_treespec != defaultdict_treespec
    assert hash(frozendict_treespec) != hash(dict_treespec)
    assert hash(frozendict_treespec) != hash(ordereddict_treespec)
    assert hash(frozendict_treespec) != hash(defaultdict_treespec)

    # Empty frozendict spec is also distinct from empty dict spec.
    assert optree.treespec_frozendict() != optree.treespec_dict()
    assert hash(optree.treespec_frozendict()) != hash(optree.treespec_dict())

    # Round-trip via `pickle` (already exercised elsewhere in this file) must preserve the
    # `FrozenDict` kind. Asserting equality is necessary but not sufficient, so also assert the
    # unpickled spec still differs from the dict spec, which would fail if the kind were silently
    # demoted during serialization.
    nested = optree.tree_structure(frozendict({'a': frozendict({'b': 1, 'c': 2})}))
    restored = pickle.loads(pickle.dumps(nested))
    assert restored == nested
    assert restored != optree.tree_structure({'a': {'b': 1, 'c': 2}})


@pytest.mark.skipif(
    not (sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT),
    reason='`frozendict` requires Python 3.15+',
)
def test_treespec_frozendict_dict_insertion_ordered():
    # Under the default (sorted-keys) regime, two `frozendict`s built with reversed key order
    # produce the same treespec. Under `dict_insertion_ordered(True)`, they produce different
    # treespecs, mirroring the behavior of `dict`. This guards the per-kind branch added in
    # `optree/registry.py` for `_FROZENDICT_INSERTION_ORDERED_REGISTRY_ENTRY`.
    frozendict = builtins.frozendict  # type: ignore[attr-defined] # pylint: disable=no-member

    forward = frozendict({'a': 1, 'b': 2})
    reverse = frozendict({'b': 2, 'a': 1})

    assert optree.tree_structure(forward) == optree.tree_structure(reverse)

    with optree.dict_insertion_ordered(True, namespace=GLOBAL_NAMESPACE):
        forward_treespec = optree.tree_structure(forward)
        reverse_treespec = optree.tree_structure(reverse)
        assert forward_treespec != reverse_treespec
        assert hash(forward_treespec) != hash(reverse_treespec)


@pytest.mark.skipif(
    not (sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT),
    reason='`frozendict` requires Python 3.15+',
)
def test_treespec_frozendict_from_collection_retains_insertion_ordered_namespace():
    # `MakeFromCollection` resets the namespace for nodes that do not depend on it. An
    # insertion-ordered dict node DOES depend on it, since the same collection sorts under any
    # other namespace, so dropping `frozendict` yields a spec no `tree_flatten()` can produce.
    frozendict = builtins.frozendict  # type: ignore[attr-defined] # pylint: disable=no-member

    leaf = optree.treespec_leaf()
    namespace = 'frozendict-insertion-ordered'

    with optree.dict_insertion_ordered(True, namespace=namespace):
        treespec = optree.treespec_frozendict({'b': leaf, 'a': leaf}, namespace=namespace)
        assert treespec.namespace == namespace
        assert (
            str(treespec)
            == "PyTreeSpec(frozendict({'b': *, 'a': *}), namespace=" + repr(namespace) + ')'
        )
        # Round-trips: the spec is reachable by flattening under the same namespace.
        assert treespec == optree.tree_structure(
            frozendict({'b': 1, 'a': 2}),
            namespace=namespace,
        )

    # Outside the insertion-ordered namespace the keys are sorted, so the node no longer depends on
    # the namespace and it is reset, matching `dict` / `defaultdict`.
    treespec = optree.treespec_frozendict({'b': leaf, 'a': leaf}, namespace=namespace)
    assert treespec.namespace == ''
    assert str(treespec) == "PyTreeSpec(frozendict({'a': *, 'b': *}))"


@pytest.mark.skipif(
    not (sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT),
    reason='`frozendict` requires Python 3.15+',
)
def test_treespec_frozendict_pickled_state_does_not_alias_keys():
    # A `frozendict` node keeps its keys in the same internal `py::list` as `dict`, so
    # `__getstate__` must hand out a copy and `__setstate__` must take one, or mutating the state
    # corrupts an immutable treespec. Companion to the `*_alias_*_node_data` tests above.
    frozendict = builtins.frozendict  # type: ignore[attr-defined] # pylint: disable=no-member

    def setstate(state):
        obj = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
        obj.__setstate__(state)
        return obj

    # Sorted keys differ from the insertion order recorded in `original_keys`.
    treespec = optree.tree_structure(frozendict({'b': 0, 'a': 0}))

    # `__getstate__` must not expose the treespec's own containers. Compare the repr of the STATE,
    # not of the treespec: `repr(treespec)` never reads `original_keys`, so it would not notice a
    # borrowed `original_keys` dict being mutated.
    state = treespec.__getstate__()
    before = repr(treespec.__getstate__())
    for node in state[0]:
        kind, node_data, original_keys = node[0], node[2], node[7]
        if kind == optree.PyTreeKind.FROZENDICT:
            node_data.append('injected')
        if isinstance(original_keys, dict):
            original_keys['injected'] = None
    assert repr(treespec.__getstate__()) == before, 'mutating the pickled state corrupted the spec'
    assert treespec.unflatten([10, 20]) == frozendict({'b': 20, 'a': 10})

    # `__setstate__` must copy the supplied containers rather than borrow them.
    state = treespec.__getstate__()
    restored = setstate(state)
    before = repr(restored)
    expected_entries = restored.entries()
    expected_tree = restored.unflatten([10, 20])
    for node in state[0]:
        kind, node_data, original_keys = node[0], node[2], node[7]
        if kind == optree.PyTreeKind.FROZENDICT:
            node_data.reverse()
            node_data.append('injected')
        if isinstance(original_keys, dict):
            original_keys['injected'] = None
    assert repr(restored) == before
    assert restored.entries() == expected_entries
    assert restored.unflatten([10, 20]) == expected_tree
    assert restored == treespec


def test_treespec_frozendict_pickle_cross_version():
    # A treespec pickled on a Python 3.15+ build names `PyTreeKind::FrozenDict`, whose enum value
    # exists on every build. Runs on every interpreter and asserts whichever half applies: without
    # support the restore must fail loudly (see the `#if !defined(OPTREE_HAS_FROZENDICT)` guard in
    # `PyTreeSpec::FromPicklable`), with support it must rebuild a real `frozendict` node.
    DICT = int(optree.PyTreeKind.DICT)  # noqa: N806
    FROZENDICT = int(optree.PyTreeKind.FROZENDICT)  # noqa: N806

    # Retag a plain `dict` spec instead of checking in an opaque blob, so the payload tracks the
    # current state layout. On supporting builds it is asserted equal to the native one below.
    node_states, none_is_leaf, namespace = optree.tree_structure(
        {'x': {'b': 1, 'a': 2}, 'y': [3]},
    ).__getstate__()
    retagged = []
    for node in node_states:
        # The inner (first, in post-order) dict node; leave the outer one alone.
        if node[0] == DICT and not any(n[0] == FROZENDICT for n in retagged):
            node = (FROZENDICT, *node[1:])
        retagged.append(node)
    state = (tuple(retagged), none_is_leaf, namespace)
    assert any(node[0] == FROZENDICT for node in state[0]), 'retagging produced no frozendict node'

    # Ship it the way a real pickle would arrive: `NEWOBJ` an empty spec, then `BUILD` from state.
    blob = b''.join(
        [
            pickle.PROTO + bytes([2]),
            pickle.GLOBAL + b'optree\nPyTreeSpec\n',
            pickle.EMPTY_TUPLE + pickle.NEWOBJ,
            pickle.dumps(state, protocol=2)[2:-1],  # strip the PROTO / STOP framing
            pickle.BUILD + pickle.STOP,
        ],
    )

    if not OPTREE_HAS_FROZENDICT:
        obj = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
        with pytest.raises(
            ValueError,
            match=re.escape(
                'Cannot restore a PyTreeSpec containing a `frozendict` node: this build of optree '
                'was compiled without `frozendict` support (requires Python 3.15+).',
            ),
        ):
            obj.__setstate__(state)
        with pytest.raises(ValueError, match=r'compiled without `frozendict` support'):
            pickle.loads(blob)  # crafted payload, the point of the test
        return

    frozendict = builtins.frozendict  # type: ignore[attr-defined] # pylint: disable=no-member

    restored = pickle.loads(blob)  # crafted payload, the point of the test
    expected = optree.tree_structure({'x': frozendict({'b': 1, 'a': 2}), 'y': [3]})
    assert restored == expected
    assert restored.__getstate__() == expected.__getstate__()
    assert str(restored) == "PyTreeSpec({'x': frozendict({'a': *, 'b': *}), 'y': [*]})"

    tree = restored.unflatten([10, 20, 30])
    assert type(tree['x']) is frozendict
    assert list(tree['x']) == ['b', 'a']
    assert tree == {'x': frozendict({'b': 20, 'a': 10}), 'y': [30]}


@pytest.mark.skipif(
    sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT,
    reason='`frozendict` IS supported on this interpreter; behavior tested elsewhere',
)
def test_treespec_frozendict_runtime_error_on_unsupported_interpreter():
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            '`optree.treespec_frozendict` requires Python 3.15+ with `frozendict` support.',
        ),
    ):
        optree.treespec_frozendict()

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            '`optree.treespec_frozendict` requires Python 3.15+ with `frozendict` support.',
        ),
    ):
        optree.treespec.frozendict({'a': optree.treespec_leaf()})

    # Keyword and mapping-plus-keyword forms take the same path.
    with pytest.raises(RuntimeError, match=r'requires Python 3\.15\+'):
        optree.treespec_frozendict(a=optree.treespec_leaf())
    with pytest.raises(RuntimeError, match=r'requires Python 3\.15\+'):
        optree.treespec_frozendict({'a': optree.treespec_leaf()}, b=optree.treespec_leaf())


def test_frozendict_kind_is_defined_on_every_build():
    # `PyTreeKind.FROZENDICT` is registered unconditionally, even where `frozendict` is unsupported,
    # so that a kind value never means two different things across builds. Its numeric value is part
    # of the pickle format (`PyTreeSpec.__getstate__` stores `int(kind)`), so it must not shift.
    assert optree.PyTreeKind.FROZENDICT.name == 'FROZENDICT'
    assert int(optree.PyTreeKind.FROZENDICT) == 11
    assert int(optree.PyTreeKind.NUM_KINDS) == 12
    assert int(optree.PyTreeKind.FROZENDICT) + 1 == int(optree.PyTreeKind.NUM_KINDS)


@pytest.mark.skipif(
    sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT,
    reason='`frozendict` IS supported on this interpreter; behavior tested elsewhere',
)
def test_frozendict_unsupported_build_surface():
    # The negative half of the feature contract, exercised by every CI job that is not 3.15+ with
    # `frozendict` support, i.e. most of the matrix. Pins that an unsupported build advertises
    # nothing it cannot deliver, and that no `FROZENDICT` node can come into existence.
    assert optree._C.OPTREE_HAS_FROZENDICT is False  # pylint: disable=protected-access

    # The constructors stay importable, but must not be silently usable.
    assert callable(optree.treespec_frozendict)
    assert optree.treespec.frozendict is optree.treespec_frozendict
    assert 'treespec_frozendict' in optree.__all__

    # No dict-family surface claims `frozendict`.
    assert STANDARD_DICT_TYPES == frozenset({dict, OrderedDict, defaultdict})
    assert not any(
        getattr(entry.type, '__name__', None) == 'frozendict'
        for entry in NODETYPE_REGISTRY.values()
    )

    # No tree flattens to a FROZENDICT node, so the kind is unreachable by construction.
    assert all(
        node[0] != int(optree.PyTreeKind.FROZENDICT)
        for tree in TREES
        for node in optree.tree_structure(tree).__getstate__()[0]
    )
