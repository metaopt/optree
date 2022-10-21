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
"""Utility functions for OpTree."""

from typing import Any, Iterable, List, Sequence, Tuple


def safe_zip(*args: Sequence[Any]) -> List[Tuple[Any, ...]]:
    """Strict zip that requires all arguments to be the same length."""
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(zip(*args))


def unzip2(xys: Iterable[Tuple[Any, Any]]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Unzip sequence of length-2 tuples into two tuples."""
    # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
    # is too permissive about inputs, and does not guarantee a length-2 output.
    # For example, for empty dict: tuple(zip(*{}.items())) -> ()
    xs = []
    ys = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)
