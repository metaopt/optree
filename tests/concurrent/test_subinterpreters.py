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
import sys

import pytest

from helpers import (
    PYPY,
    WASM,
    Py_DEBUG,
    Py_GIL_DISABLED,
)


if PYPY or WASM or sys.version_info < (3, 14):
    pytest.skip('Test for CPython 3.14+ only', allow_module_level=True)


from concurrent.futures import InterpreterPoolExecutor, as_completed


if Py_GIL_DISABLED and not Py_DEBUG:
    NUM_WORKERS = 32
    NUM_FUTURES = 128
else:
    NUM_WORKERS = 4
    NUM_FUTURES = 16


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
