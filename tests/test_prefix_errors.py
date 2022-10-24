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

# pylint: disable=missing-function-docstring,invalid-name,implicit-str-concat

import pytest

import optree

# pylint: disable-next=wrong-import-order
from helpers import Vector2D


def test_different_types():
    (e,) = optree.prefix_errors((1, 2), [1, 2])
    expected = 'pytree structure error: different types at key path\n' '    in_axes tree root'
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_types_nested():
    (e,) = optree.prefix_errors(((1,), (2,)), ([3], (4,)))
    expected = 'pytree structure error: different types at key path\n' r'    in_axes\[0\]'
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_types_multiple():
    e1, e2 = optree.prefix_errors(((1,), (2,)), ([3], [4]))
    expected = 'pytree structure error: different types at key path\n' r'    in_axes\[0\]'
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = 'pytree structure error: different types at key path\n' r'    in_axes\[1\]'
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_different_num_children():
    (e,) = optree.prefix_errors((1,), (2, 3))
    expected = (
        'pytree structure error: different numbers of pytree children '
        'at key path\n'
        '    in_axes tree root'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_num_children_nested():
    (e,) = optree.prefix_errors([[1]], [[2, 3]])
    expected = (
        'pytree structure error: different numbers of pytree children '
        'at key path\n'
        r'    in_axes\[0\]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_num_children_multiple():
    e1, e2 = optree.prefix_errors([[1], [2]], [[3, 4], [5, 6]])
    expected = (
        'pytree structure error: different numbers of pytree children '
        'at key path\n'
        r'    in_axes\[0\]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = (
        'pytree structure error: different numbers of pytree children '
        'at key path\n'
        r'    in_axes\[1\]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_different_metadata():
    (e,) = optree.prefix_errors({1: 2}, {3: 4})
    expected = (
        'pytree structure error: different pytree metadata ' 'at key path\n' '    in_axes tree root'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_metadata_nested():
    (e,) = optree.prefix_errors([{1: 2}], [{3: 4}])
    expected = (
        'pytree structure error: different pytree metadata ' 'at key path\n' r'    in_axes\[0\]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_metadata_multiple():
    e1, e2 = optree.prefix_errors([{1: 2}, {3: 4}], [{3: 4}, {5: 6}])
    expected = (
        'pytree structure error: different pytree metadata ' 'at key path\n' r'    in_axes\[0\]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = (
        'pytree structure error: different pytree metadata ' 'at key path\n' r'    in_axes\[1\]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_fallback_keypath():
    (e,) = optree.prefix_errors(Vector2D(1, [2]), Vector2D(3, 4))
    expected = (
        'pytree structure error: different types at key path\n' r'    in_axes\[<flat index 1>\]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_no_errors():
    () = optree.prefix_errors((1, 2), ((11, 12, 13), 2))


def test_different_structure_no_children():
    (e,) = optree.prefix_errors({}, {'a': []})
    expected = (
        'pytree structure error: different numbers of pytree children '
        'at key path\n'
        '    in_axes tree root'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')
