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

import atexit
import contextlib
import random
import sys

import pytest

from helpers import (
    ANDROID,
    IOS,
    OPTREE_HAS_SUBINTERPRETER_SUPPORT,
    PYPY,
    WASM,
    Py_DEBUG,
    check_script_in_subprocess,
)


if (
    PYPY
    or WASM
    or IOS
    or ANDROID
    or sys.version_info < (3, 14)
    or not getattr(sys.implementation, 'supports_isolated_interpreters', False)
    or not OPTREE_HAS_SUBINTERPRETER_SUPPORT
):
    pytest.skip('Test for CPython 3.14+ only', allow_module_level=True)


from concurrent import interpreters
from concurrent.futures import InterpreterPoolExecutor, as_completed


if not Py_DEBUG:
    NUM_WORKERS = 8
    NUM_FUTURES = 32
    NUM_FLAKY_RERUNS = 16
else:
    NUM_WORKERS = 4
    NUM_FUTURES = 16
    NUM_FLAKY_RERUNS = 8


EXECUTOR = InterpreterPoolExecutor(max_workers=NUM_WORKERS)
atexit.register(EXECUTOR.shutdown)


def run(func, /, *args, **kwargs):
    future = EXECUTOR.submit(func, *args, **kwargs)
    exception = future.exception()
    if exception is not None:
        raise exception
    return future.result()


def concurrent_run(func, /, *args, **kwargs):
    futures = [EXECUTOR.submit(func, *args, **kwargs) for _ in range(NUM_FUTURES)]
    future2index = {future: i for i, future in enumerate(futures)}
    completed_futures = sorted(as_completed(futures), key=future2index.get)
    first_exception = next(filter(None, (future.exception() for future in completed_futures)), None)
    if first_exception is not None:
        raise first_exception
    return [future.result() for future in completed_futures]


def check_module_importable():
    import collections
    import sys
    import time

    import optree
    import optree._C

    is_current_interpreter_main = optree._C.is_current_interpreter_main()
    main_interpreter_id = optree._C.get_main_interpreter_id()
    current_interpreter_id = optree._C.get_current_interpreter_id()

    if is_current_interpreter_main != (main_interpreter_id == current_interpreter_id):
        raise RuntimeError('interpreter identity mismatch')

    if not is_current_interpreter_main and optree._C.get_registry_size() != (
        9 if sys.version_info >= (3, 15) and optree._C.OPTREE_HAS_FROZENDICT else 8
    ):
        raise RuntimeError('registry size mismatch')

    tree = {
        'b': [2, (3, 4)],
        'a': 1,
        'c': collections.OrderedDict(
            f=None,
            d=5,
            e=time.struct_time([6] + [None] * (time.struct_time.n_sequence_fields - 1)),
        ),
        'g': collections.defaultdict(list, h=collections.deque([7, 8, 9], maxlen=10)),
    }
    expected_leaves1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_leaves2 = [
        1,
        2,
        3,
        4,
        None,
        5,
        6,
        *([None] * (time.struct_time.n_sequence_fields - 1)),
        7,
        8,
        9,
    ]
    if sys.version_info >= (3, 15) and optree._C.OPTREE_HAS_FROZENDICT:
        from builtins import frozendict  # pylint: disable=no-name-in-module

        tree['i'] = frozendict({'k': 11, 'j': 10})
        expected_leaves1.extend([10, 11])
        expected_leaves2.extend([10, 11])

    leaves1, treespec1 = optree.tree_flatten(tree, none_is_leaf=False)
    reconstructed1 = optree.tree_unflatten(treespec1, leaves1)
    if reconstructed1 != tree:
        raise RuntimeError('unflatten/flatten mismatch')
    if treespec1.num_leaves != len(leaves1):
        raise RuntimeError(f'num_leaves mismatch: ({leaves1}, {treespec1})')
    if leaves1 != expected_leaves1:
        raise RuntimeError(f'flattened leaves mismatch: ({leaves1}, {treespec1})')

    leaves2, treespec2 = optree.tree_flatten(tree, none_is_leaf=True)
    reconstructed2 = optree.tree_unflatten(treespec2, leaves2)
    if reconstructed2 != tree:
        raise RuntimeError('unflatten/flatten mismatch')
    if treespec2.num_leaves != len(leaves2):
        raise RuntimeError(f'num_leaves mismatch: ({leaves2}, {treespec2})')
    if leaves2 != expected_leaves2:
        raise RuntimeError(f'flattened leaves mismatch: ({leaves2}, {treespec2})')

    _ = optree.tree_flatten_with_path(tree, none_is_leaf=False)
    _ = optree.tree_flatten_with_path(tree, none_is_leaf=True)
    _ = optree.tree_flatten_with_accessor(tree, none_is_leaf=False)
    _ = optree.tree_flatten_with_accessor(tree, none_is_leaf=True)

    return (
        is_current_interpreter_main,
        main_interpreter_id,
        id(type(None)),
        id(tuple),
        id(list),
        id(dict),
        id(collections.OrderedDict),
    )


def test_import():
    import collections

    expected = (
        False,
        0,
        id(type(None)),
        id(tuple),
        id(list),
        id(dict),
        id(collections.OrderedDict),
    )

    assert check_module_importable() == (True, *expected[1:])
    assert run(check_module_importable) == expected

    for _ in range(random.randint(5, 10)):
        with contextlib.closing(interpreters.create()) as subinterpreter:
            subinterpreter.exec('import optree')
        with contextlib.closing(interpreters.create()) as subinterpreter:
            assert subinterpreter.call(check_module_importable) == expected

    for actual in concurrent_run(check_module_importable):
        assert actual == expected

    with contextlib.ExitStack() as stack:
        subinterpreters = [
            stack.enter_context(contextlib.closing(interpreters.create()))
            for _ in range(random.randint(5, 10))
        ]
        random.shuffle(subinterpreters)
        for subinterpreter in subinterpreters:
            subinterpreter.exec('import optree')

    with contextlib.ExitStack() as stack:
        subinterpreters = [
            stack.enter_context(contextlib.closing(interpreters.create()))
            for _ in range(random.randint(5, 10))
        ]
        random.shuffle(subinterpreters)
        for subinterpreter in subinterpreters:
            assert subinterpreter.call(check_module_importable) == expected


def test_import_in_subinterpreter_after_main():
    check_script_in_subprocess(
        """
        import contextlib
        import gc
        from concurrent import interpreters

        import optree

        subinterpreter = None
        with contextlib.closing(interpreters.create()) as subinterpreter:
            subinterpreter.exec('import optree')

        del optree, subinterpreter
        for _ in range(10):
            gc.collect()
        """,
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )

    check_script_in_subprocess(
        f"""
        import contextlib
        import gc
        import random
        from concurrent import interpreters

        import optree

        subinterpreter = subinterpreters = stack = None
        with contextlib.ExitStack() as stack:
            subinterpreters = [
                stack.enter_context(contextlib.closing(interpreters.create()))
                for _ in range({NUM_FUTURES})
            ]
            random.shuffle(subinterpreters)
            for subinterpreter in subinterpreters:
                subinterpreter.exec('import optree')

        del optree, subinterpreter, subinterpreters, stack
        for _ in range(10):
            gc.collect()
        """,
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )


def test_import_in_subinterpreter_before_main():
    check_script_in_subprocess(
        """
        import contextlib
        import gc
        from concurrent import interpreters

        subinterpreter = None
        with contextlib.closing(interpreters.create()) as subinterpreter:
            subinterpreter.exec('import optree')

        import optree

        del optree, subinterpreter
        for _ in range(10):
            gc.collect()
        """,
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )

    check_script_in_subprocess(
        f"""
        import contextlib
        import gc
        import random
        from concurrent import interpreters

        subinterpreter = subinterpreters = stack = None
        with contextlib.ExitStack() as stack:
            subinterpreters = [
                stack.enter_context(contextlib.closing(interpreters.create()))
                for _ in range({NUM_FUTURES})
            ]
            random.shuffle(subinterpreters)
            for subinterpreter in subinterpreters:
                subinterpreter.exec('import optree')

        import optree

        del optree, subinterpreter, subinterpreters, stack
        for _ in range(10):
            gc.collect()
        """,
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )

    check_script_in_subprocess(
        f"""
        import contextlib
        import gc
        import random
        from concurrent import interpreters

        subinterpreter = subinterpreters = stack = None
        with contextlib.ExitStack() as stack:
            subinterpreters = [
                stack.enter_context(contextlib.closing(interpreters.create()))
                for _ in range({NUM_FUTURES})
            ]
            random.shuffle(subinterpreters)
            for subinterpreter in subinterpreters:
                subinterpreter.exec('import optree')

            import optree

        del optree, subinterpreter, subinterpreters, stack
        for _ in range(10):
            gc.collect()
        """,
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )


def test_import_in_subinterpreters_concurrently():
    check_script_in_subprocess(
        f"""
        from concurrent.futures import InterpreterPoolExecutor, as_completed

        def check_import():
            import sys
            import optree
            import optree._C

            if optree._C.get_registry_size() != (
                9 if sys.version_info >= (3, 15) and optree._C.OPTREE_HAS_FROZENDICT else 8
            ):
                raise RuntimeError('registry size mismatch')
            if optree._C.is_current_interpreter_main():
                raise RuntimeError('expected subinterpreter')

        with InterpreterPoolExecutor(max_workers={NUM_WORKERS}) as executor:
            futures = [executor.submit(check_import) for _ in range({NUM_FUTURES})]
            for future in as_completed(futures):
                future.result()
        """,
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )


def test_type_cache_cleanup_across_subinterpreters():
    # Exercise the process-global type caches across subinterpreter churn. The caches key on the
    # type's memory ADDRESS, which the allocator may recycle after a type is freed. Each interpreter
    # flattens a RANDOMLY chosen type and reads its repr, firing both the classification and the
    # field-name cache; distinct field names make a stale entry on a recycled address observable.
    # The subprocess makes finalization real.
    check_script_in_subprocess(
        f"""
        import contextlib
        import textwrap
        from concurrent import interpreters

        resolve = textwrap.dedent(
            '''
            import keyword
            import os
            import random
            import string
            import time
            from collections import namedtuple

            import optree

            # Distinct PyStructSequence types, each with a distinct first sequence field.
            cases = [
                (os.stat_result, 'st_mode'),
                (time.struct_time, 'tm_year'),
                (os.times_result, 'user'),
            ]
            case = random.choice([*cases, None])
            if case is None:
                # Exercise the classification cache with a namedtuple type.
                field_names = []
                while not field_names:
                    field_names = [
                        ''.join(random.choices(string.ascii_lowercase, k=random.randint(1, 8)))
                        for _ in range(random.randint(2, 5))
                    ]
                    field_names = [n for n in field_names if not keyword.iskeyword(n)]  # avoid reserved names
                    field_names = list(dict.fromkeys(field_names))  # deduplicate
                NamedTupleType = namedtuple('NamedTupleType', field_names)
                leaves, treespec = optree.tree_flatten(NamedTupleType(*range(len(field_names))))
                assert len(leaves) == len(field_names), (leaves, treespec)
                assert (field_names[0] + '=*') in repr(treespec), repr(treespec)
            else:
                # Exercise the classification cache and the field-name cache with a PyStructSequence type.
                PyStructSequenceType, first_field = case
                obj = PyStructSequenceType(range(PyStructSequenceType.n_sequence_fields))
                assert (first_field + '=*') in repr(optree.tree_structure(obj)), first_field
            '''
        ).strip()

        exec(resolve)  # the main interpreter resolves a random type
        for _ in range({NUM_FUTURES}):
            with contextlib.closing(interpreters.create()) as subinterpreter:
                subinterpreter.exec(resolve)
        exec(resolve)  # the main interpreter must still resolve correctly after the churn
        """,
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )


def test_registry_init_failure_does_not_leak_the_interpreter_id():
    # Regression: `PyTreeTypeRegistry::Init` inserted the interpreter into the alive set and bumped
    # the seen counter BEFORE `atexit.register(&Clear)` succeeded, with no rollback. A failed import
    # then left behind an ID that no callback can ever remove, so the registry keeps believing a
    # dead interpreter is alive and its shutdown invariants never hold again. `Init` must mark the
    # interpreter only once the cleanup is registered, the way `WeakKeyCache::LookupOrInsert` does.
    check_script_in_subprocess(
        """
        import contextlib
        import faulthandler
        import textwrap
        from concurrent import interpreters

        import optree

        faulthandler.dump_traceback_later(60, exit=True)  # watchdog: abort the process on a hang

        before_ids = optree._C.get_alive_interpreter_ids()
        before_seen = optree._C.get_num_interpreters_seen()
        assert before_ids == {optree._C.get_current_interpreter_id()}, before_ids

        with contextlib.closing(interpreters.create()) as subinterpreter:
            subinterpreter.exec(
                textwrap.dedent(
                    '''
                    import atexit
                    import typing

                    def failing_register(*args, **kwargs):
                        raise RuntimeError('injected atexit failure')

                    atexit.register = failing_register
                    try:
                        import optree
                    except ImportError:
                        pass
                    else:
                        raise AssertionError('the injected failure did not propagate')
                    ''',
                ).strip(),
            )

        after_ids = optree._C.get_alive_interpreter_ids()
        after_seen = optree._C.get_num_interpreters_seen()
        assert after_ids == before_ids, (before_ids, after_ids)
        assert after_seen == before_seen, (before_seen, after_seen)
        """,
        output='',
    )


def test_registry_lookup_no_gil_lock_order_deadlock():
    # Regression: `Lookup`/`Register`/`Unregister` called `GetSingleton()` while holding `sm_mutex`.
    # Under `per_interpreter_gil` that call releases the GIL once any subinterpreter has imported
    # `optree._C`, so a flatten thread drops the GIL holding the read lock while a registration
    # thread holds the GIL waiting on the write lock: a lock-order inversion that hangs the process.
    # The watchdog turns the hang into a non-zero exit; the fix acquires the singleton before
    # locking.
    check_script_in_subprocess(
        f"""
        import contextlib
        import faulthandler
        import threading
        import time
        from concurrent import interpreters

        import optree

        # Latch pybind11's `has_seen_non_main_interpreter` so `GetSingleton()` releases the GIL on
        # every subsequent call (the single-interpreter fast path is disabled process-wide).
        with contextlib.closing(interpreters.create()) as subinterpreter:
            subinterpreter.exec('import optree._C')

        faulthandler.dump_traceback_later(20, exit=True)  # a deadlock becomes a non-zero exit

        stop = threading.Event()
        tree = {{'a': 1, 'b': [2, 3], 'c': (4, 5)}}

        def flatten_worker():
            while not stop.is_set():
                optree.tree_flatten(tree)

        def register_worker(index):
            cls = type(f'DeadlockNode_{{index}}', (), {{}})
            while not stop.is_set():
                optree.register_pytree_node(
                    cls,
                    lambda x: ((), None),
                    lambda metadata, children: cls(),
                    namespace='deadlock'
                )
                optree.unregister_pytree_node(cls, namespace='deadlock')

        workers = [threading.Thread(target=flatten_worker) for _ in range({NUM_WORKERS})]
        workers += [threading.Thread(target=register_worker, args=(index,)) for index in range(2)]
        for worker in workers:
            worker.start()
        time.sleep(0.5)  # let readers and writers contend on `sm_mutex`
        stop.set()
        for worker in workers:
            worker.join(timeout=5)
            assert not worker.is_alive(), 'worker thread did not terminate (deadlock)'

        faulthandler.cancel_dump_traceback_later()
        """,
        output=None,
        rerun=NUM_FLAKY_RERUNS,
    )
