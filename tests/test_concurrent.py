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

import atexit
import itertools
import pickle
import weakref
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

import optree
from helpers import GLOBAL_NAMESPACE, PYPY, TREES, Py_GIL_DISABLED, gc_collect, parametrize


if PYPY:
    pytest.skip('Test for CPython only', allow_module_level=True)


if Py_GIL_DISABLED:
    NUM_WORKERS = 32
    NUM_FUTURES = 128
else:
    NUM_WORKERS = 4
    NUM_FUTURES = 16


EXECUTOR = ThreadPoolExecutor(max_workers=NUM_WORKERS)
atexit.register(EXECUTOR.shutdown)


def concurrent_run(func):
    futures = [EXECUTOR.submit(func) for _ in range(NUM_FUTURES)]
    future2index = {future: i for i, future in enumerate(futures)}
    completed_futures = sorted(as_completed(futures), key=future2index.get)
    first_exception = next(filter(None, (future.exception() for future in completed_futures)), None)
    if first_exception is not None:
        raise first_exception
    return [future.result() for future in completed_futures]


concurrent_run(object)  # warm-up


@parametrize(
    tree=TREES,
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_flatten_unflatten_thread_safe(
    tree,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    def test_fn():
        return optree.tree_flatten(tree, namespace=namespace)

    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        leaves, treespec = expected = test_fn()
        for result in concurrent_run(test_fn):
            assert result == expected

    for result in concurrent_run(lambda: optree.tree_unflatten(treespec, leaves)):
        assert result == tree


@parametrize(
    tree=TREES,
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_flatten_with_path_thread_safe(
    tree,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    def test_fn():
        return optree.tree_flatten_with_path(tree, namespace=namespace)

    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        expected = test_fn()
        for result in concurrent_run(test_fn):
            assert result == expected


@parametrize(
    tree=TREES,
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_flatten_with_accessor_thread_safe(
    tree,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    def test_fn():
        return optree.tree_flatten_with_accessor(tree, namespace=namespace)

    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        expected = test_fn()
        for result in concurrent_run(test_fn):
            assert result == expected


@parametrize(tree=TREES)
def test_treespec_string_representation(tree):
    expected_string = repr(optree.tree_structure(tree))

    def check1():
        treespec = optree.tree_structure(tree)
        assert str(treespec) == expected_string
        assert repr(treespec) == expected_string

    concurrent_run(check1)

    treespec = optree.tree_structure(tree)

    def check2():
        assert str(treespec) == expected_string
        assert repr(treespec) == expected_string

    concurrent_run(check2)


def test_treespec_self_referential():  # noqa: C901
    class Holder:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Holder) and self.value == other.value

        def __hash__(self):
            return hash(self.value)

        def __repr__(self):
            return f'Holder({self.value!r})'

    hashes = set()
    key = Holder('a')

    treespec = optree.tree_structure({key: 0})

    def check1():
        assert str(treespec) == "PyTreeSpec({Holder('a'): *})"  # noqa: F821
        assert hash(treespec) == hash(treespec)  # noqa: F821

    concurrent_run(check1)

    hashes.add(hash(treespec))

    key.value = 'b'

    def check2():
        assert str(treespec) == "PyTreeSpec({Holder('b'): *})"  # noqa: F821
        assert hash(treespec) == hash(treespec)  # noqa: F821
        assert hash(treespec) not in hashes  # noqa: F821

    concurrent_run(check2)

    hashes.add(hash(treespec))

    key.value = treespec

    def check3():
        assert str(treespec) == 'PyTreeSpec({Holder(...): *})'  # noqa: F821
        assert hash(treespec) == hash(treespec)  # noqa: F821
        assert hash(treespec) not in hashes  # noqa: F821

    concurrent_run(check3)

    hashes.add(hash(treespec))

    key.value = ('a', treespec, treespec)

    def check4():
        assert str(treespec) == "PyTreeSpec({Holder(('a', ..., ...)): *})"  # noqa: F821
        assert hash(treespec) == hash(treespec)  # noqa: F821
        assert hash(treespec) not in hashes  # noqa: F821

    concurrent_run(check4)

    hashes.add(hash(treespec))

    other = optree.tree_structure({Holder(treespec): 1})

    def check5():
        assert (
            str(other)  # noqa: F821
            == "PyTreeSpec({Holder(PyTreeSpec({Holder(('a', ..., ...)): *})): *})"
        )
        assert hash(other) == hash(other)  # noqa: F821
        assert hash(other) not in hashes  # noqa: F821

    concurrent_run(check5)

    hashes.add(hash(other))

    key.value = other

    def check6():
        assert (
            str(treespec) == 'PyTreeSpec({Holder(PyTreeSpec({Holder(...): *})): *})'  # noqa: F821
        )
        assert str(other) == 'PyTreeSpec({Holder(PyTreeSpec({Holder(...): *})): *})'  # noqa: F821
        assert hash(treespec) == hash(treespec)  # noqa: F821
        assert hash(treespec) not in hashes  # noqa: F821

    concurrent_run(check6)

    hashes.add(hash(treespec))

    def check7():
        assert hash(other) == hash(other)  # noqa: F821
        assert hash(treespec) == hash(other)  # noqa: F821

    concurrent_run(check7)

    with pytest.raises(RecursionError):
        assert treespec != other

    wr = weakref.ref(treespec)
    del treespec, key, other
    gc_collect()
    assert wr() is None


@parametrize(
    tree=TREES,
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_pickle_round_trip(
    tree,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    def check1():
        assert pickle.loads(pickle.dumps(expected)) == expected

    def check2():
        assert pickle.dumps(expected) == expected_serialized
        assert pickle.loads(expected_serialized) == expected

    def check3():
        assert list(optree.tree_unflatten(actual, range(len(actual)))) == list(
            optree.tree_unflatten(expected, range(len(expected))),
        )

    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        expected = optree.tree_structure(tree, namespace=namespace)
        expected_serialized = b''
        try:
            pickle.loads(pickle.dumps(tree))
        except pickle.PicklingError:
            with pytest.raises(pickle.PicklingError, match=r"Can't pickle .*:"):
                pickle.loads(pickle.dumps(expected))
        else:
            expected_serialized = pickle.dumps(expected)
            actual = pickle.loads(expected_serialized)
            concurrent_run(check1)
            concurrent_run(check2)
            if expected.type in {dict, OrderedDict, defaultdict}:
                concurrent_run(check3)


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_tree_iter_thread_safe(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    counter = itertools.count()
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        new_tree = optree.tree_map(
            lambda x: next(counter),
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        num_leaves = next(counter)
        it = optree.tree_iter(
            new_tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )

    results = concurrent_run(lambda: list(it))
    for seq in results:
        assert sorted(seq) == seq
    assert sorted(itertools.chain.from_iterable(results)) == list(range(num_leaves))
