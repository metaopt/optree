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

import atexit
import contextlib
import random
import sys
import textwrap

import pytest

from helpers import (
    ANDROID,
    IOS,
    OPTREE_HAS_SUBINTERPRETER_SUPPORT,
    PYPY,
    WASM,
    Py_DEBUG,
    Py_GIL_DISABLED,
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


if Py_GIL_DISABLED and not Py_DEBUG:
    NUM_WORKERS = 32
    NUM_FUTURES = 128
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


run(object)  # warm-up


def check_module_importable():
    import collections
    import time

    import optree._C

    if optree._C.get_registry_size() != 8:
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

    leaves1, treespec1 = optree.tree_flatten(tree, none_is_leaf=False)
    reconstructed1 = optree.tree_unflatten(treespec1, leaves1)
    if reconstructed1 != tree:
        raise RuntimeError('unflatten/flatten mismatch')
    if treespec1.num_leaves != len(leaves1):
        raise RuntimeError(f'num_leaves mismatch: ({leaves1}, {treespec1})')
    if leaves1 != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise RuntimeError(f'flattened leaves mismatch: ({leaves1}, {treespec1})')

    leaves2, treespec2 = optree.tree_flatten(tree, none_is_leaf=True)
    reconstructed2 = optree.tree_unflatten(treespec2, leaves2)
    if reconstructed2 != tree:
        raise RuntimeError('unflatten/flatten mismatch')
    if treespec2.num_leaves != len(leaves2):
        raise RuntimeError(f'num_leaves mismatch: ({leaves2}, {treespec2})')
    if leaves2 != [
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
    ]:
        raise RuntimeError(f'flattened leaves mismatch: ({leaves2}, {treespec2})')

    _ = optree.tree_flatten_with_path(tree, none_is_leaf=False)
    _ = optree.tree_flatten_with_path(tree, none_is_leaf=True)
    _ = optree.tree_flatten_with_accessor(tree, none_is_leaf=False)
    _ = optree.tree_flatten_with_accessor(tree, none_is_leaf=True)

    return (
        optree._C.get_main_interpreter_id(),
        id(type(None)),
        id(tuple),
        id(list),
        id(dict),
        id(collections.OrderedDict),
    )


def test_import():
    import collections

    expected = (
        0,
        id(type(None)),
        id(tuple),
        id(list),
        id(dict),
        id(collections.OrderedDict),
    )

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
        textwrap.dedent(
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
        ).strip(),
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )

    check_script_in_subprocess(
        textwrap.dedent(
            f"""
            import contextlib
            import gc
            from concurrent import interpreters

            import optree

            subinterpreter = subinterpreters = stack = None
            with contextlib.ExitStack() as stack:
                subinterpreters = [
                    stack.enter_context(contextlib.closing(interpreters.create()))
                    for _ in range({NUM_FUTURES})
                ]
                for subinterpreter in subinterpreters:
                    subinterpreter.exec('import optree')

            del optree, subinterpreter, subinterpreters, stack
            for _ in range(10):
                gc.collect()
            """,
        ).strip(),
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )


def test_import_in_subinterpreter_before_main():
    check_script_in_subprocess(
        textwrap.dedent(
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
        ).strip(),
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )

    check_script_in_subprocess(
        textwrap.dedent(
            f"""
            import contextlib
            import gc
            from concurrent import interpreters

            subinterpreter = subinterpreters = stack = None
            with contextlib.ExitStack() as stack:
                subinterpreters = [
                    stack.enter_context(contextlib.closing(interpreters.create()))
                    for _ in range({NUM_FUTURES})
                ]
                for subinterpreter in subinterpreters:
                    subinterpreter.exec('import optree')

            import optree

            del optree, subinterpreter, subinterpreters, stack
            for _ in range(10):
                gc.collect()
            """,
        ).strip(),
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )

    check_script_in_subprocess(
        textwrap.dedent(
            f"""
            import contextlib
            import gc
            from concurrent import interpreters

            subinterpreter = subinterpreters = stack = None
            with contextlib.ExitStack() as stack:
                subinterpreters = [
                    stack.enter_context(contextlib.closing(interpreters.create()))
                    for _ in range({NUM_FUTURES})
                ]
                for subinterpreter in subinterpreters:
                    subinterpreter.exec('import optree')

                import optree

            del optree, subinterpreter, subinterpreters, stack
            for _ in range(10):
                gc.collect()
            """,
        ).strip(),
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )


def test_import_in_subinterpreters_concurrently():
    check_script_in_subprocess(
        textwrap.dedent(
            f"""
            from concurrent.futures import InterpreterPoolExecutor, as_completed

            def check_import():
                import optree

                if optree._C.get_registry_size() != 8:
                    raise RuntimeError('registry size mismatch')

            with InterpreterPoolExecutor(max_workers={NUM_WORKERS}) as executor:
                futures = [executor.submit(check_import) for _ in range({NUM_FUTURES})]
                for future in as_completed(futures):
                    future.result()
            """,
        ).strip(),
        output='',
        rerun=NUM_FLAKY_RERUNS,
    )
