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

import pytest

from optree.utils import safe_zip, total_order_sorted, unzip2


def test_total_order_sorted():
    assert total_order_sorted([]) == []
    assert total_order_sorted([1, 5, 4, 2, 3]) == [1, 2, 3, 4, 5]
    assert total_order_sorted([1, 5, 4.1, 2, 3]) == [1, 2, 3, 4.1, 5]
    assert total_order_sorted([1, 5, 4, 2, 3], reverse=True) == [5, 4, 3, 2, 1]
    assert total_order_sorted([1, 5, 4.1, 2, 3], reverse=True) == [5, 4.1, 3, 2, 1]
    assert total_order_sorted([1, 5, 4, '20', '3']) == [1, 4, 5, '20', '3']
    assert total_order_sorted([1, 5, 4.5, '20', '3']) == [4.5, 1, 5, '20', '3']
    assert total_order_sorted(
        {1: 1, 5: 2, 4.5: 3, '20': 4, '3': 5}.items(),
        key=lambda kv: kv[0],
    ) == [(4.5, 3), (1, 1), (5, 2), ('20', 4), ('3', 5)]

    class NonSortable:
        def __init__(self, x):
            self.x = x

        def __repr__(self) -> str:
            return f'{self.__class__.__name__}({self.x})'

        def __eq__(self, other):
            return isinstance(other, self.__class__) and self.x == other.x

        def __hash__(self):
            return hash(self.x)

    assert total_order_sorted([1, 5, 4, NonSortable(2), NonSortable(3)]) == [
        1,
        5,
        4,
        NonSortable(2),
        NonSortable(3),
    ]


def test_safe_zip():
    assert safe_zip([]) == []
    assert safe_zip([1]) == [(1,)]
    assert safe_zip([1, 2]) == [(1,), (2,)]
    assert safe_zip([1, 2], [3, 4]) == [(1, 3), (2, 4)]
    assert safe_zip([1, 2], [3, 4], [5, 6]) == [(1, 3, 5), (2, 4, 6)]
    with pytest.raises(ValueError, match='length mismatch'):
        safe_zip([1, 2], [3, 4, 5])
    with pytest.raises(ValueError, match='length mismatch'):
        safe_zip([1, 2], [3, 4], [5, 6, 7])


def test_unzip2():
    assert unzip2([]) == ((), ())
    assert unzip2([(1, 2)]) == ((1,), (2,))
    assert unzip2([(1, 2), (3, 4)]) == ((1, 3), (2, 4))
    assert unzip2({}.items()) == ((), ())
    assert unzip2({1: 2}.items()) == ((1,), (2,))
    assert unzip2({1: 2, 3: 4}.items()) == ((1, 3), (2, 4))
