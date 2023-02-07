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

# pylint: disable=missing-function-docstring,invalid-name,implicit-str-concat

import re
from collections import OrderedDict, defaultdict, deque

import pytest

import optree

# pylint: disable-next=wrong-import-order
from helpers import CustomTuple, TimeStructTime, Vector2D
from optree.registry import (
    AttributeKeyPathEntry,
    FlattenedKeyPathEntry,
    GetitemKeyPathEntry,
    KeyPath,
    KeyPathEntry,
)


def test_different_types():
    (e,) = optree.prefix_errors((1, 2), [1, 2])
    expected = re.escape(
        'pytree structure error: different types at key path\n' '    in_axes tree root'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_types_nested():
    (e,) = optree.prefix_errors(((1,), (2,)), ([3], (4,)))
    expected = re.escape('pytree structure error: different types at key path\n' '    in_axes[0]')
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_types_multiple():
    e1, e2 = optree.prefix_errors(((1,), (2,)), ([3], [4]))
    expected = re.escape('pytree structure error: different types at key path\n' '    in_axes[0]')
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = re.escape('pytree structure error: different types at key path\n' '    in_axes[1]')
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_different_num_children():
    (e,) = optree.prefix_errors((1,), (2, 3))
    expected = re.escape(
        'pytree structure error: different numbers of pytree children at key path\n'
        '    in_axes tree root'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_num_children_nested():
    (e,) = optree.prefix_errors([[1]], [[2, 3]])
    expected = re.escape(
        'pytree structure error: different numbers of pytree children at key path\n'
        '    in_axes[0]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_num_children_multiple():
    e1, e2 = optree.prefix_errors([[1], [2]], [[3, 4], [5, 6]])
    expected = re.escape(
        'pytree structure error: different numbers of pytree children at key path\n'
        '    in_axes[0]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = re.escape(
        'pytree structure error: different numbers of pytree children at key path\n'
        '    in_axes[1]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_different_metadata():
    (e,) = optree.prefix_errors({1: 2}, {3: 4})
    expected = re.escape(
        'pytree structure error: different pytree metadata at key path\n' '    in_axes tree root'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_metadata_nested():
    (e,) = optree.prefix_errors([{1: 2}], [{3: 4}])
    expected = re.escape(
        'pytree structure error: different pytree metadata at key path\n' '    in_axes[0]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_metadata_multiple():
    e1, e2 = optree.prefix_errors([{1: 2}, {3: 4}], [{3: 4}, {5: 6}])
    expected = re.escape(
        'pytree structure error: different pytree metadata at key path\n' '    in_axes[0]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = re.escape(
        'pytree structure error: different pytree metadata at key path\n' '    in_axes[1]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_namedtuple():
    (e,) = optree.prefix_errors(CustomTuple(1, [2, [3]]), CustomTuple(4, [5, 6]))
    expected = re.escape(
        'pytree structure error: different types at key path\n' '    in_axes.bar[1]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_structseq():
    (e,) = optree.prefix_errors(
        TimeStructTime((1, [2, [3]], *range(7))), TimeStructTime((4, [5, 6], *range(7)))
    )
    expected = re.escape(
        'pytree structure error: different types at key path\n' '    in_axes.tm_mon[1]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_fallback_keypath():
    (e,) = optree.prefix_errors(Vector2D(1, [2]), Vector2D(3, 4))
    expected = re.escape(
        'pytree structure error: different types at key path\n' '    in_axes[<flat index 1>]'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_no_errors():
    () = optree.prefix_errors((1, 2), ((11, 12, 13), 2))


def test_different_structure_no_children():
    (e,) = optree.prefix_errors({}, {'a': []})
    expected = re.escape(
        'pytree structure error: different numbers of pytree children at key path\n'
        '    in_axes tree root'
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_key_path():
    with pytest.raises(NotImplementedError):
        KeyPathEntry('a').pprint()

    root = KeyPath()
    sequence_key_path = GetitemKeyPathEntry(0)
    dict_key_path = GetitemKeyPathEntry('a')
    namedtuple_key_path = AttributeKeyPathEntry('attr')
    fallback_key_path = FlattenedKeyPathEntry(1)

    with pytest.raises(
        TypeError,
        match=re.escape("unsupported operand type(s) for +: 'GetitemKeyPathEntry' and 'int'"),
    ):
        sequence_key_path + 1
    with pytest.raises(
        TypeError,
        match=re.escape("unsupported operand type(s) for +: 'int' and 'GetitemKeyPathEntry'"),
    ):
        1 + sequence_key_path

    with pytest.raises(
        TypeError, match=re.escape("unsupported operand type(s) for +: 'KeyPath' and 'int'")
    ):
        root + 1
    with pytest.raises(
        TypeError, match=re.escape("unsupported operand type(s) for +: 'int' and 'KeyPath'")
    ):
        1 + root

    assert root.pprint() == ' tree root'
    assert root + root == root
    assert root + sequence_key_path == KeyPath((sequence_key_path,))
    assert (
        root + sequence_key_path + dict_key_path + namedtuple_key_path + fallback_key_path
        == KeyPath((sequence_key_path, dict_key_path, namedtuple_key_path, fallback_key_path))
    )
    assert (root + sequence_key_path).pprint() == '[0]'
    assert (sequence_key_path + root).pprint() == '[0]'
    assert (root + dict_key_path).pprint() == "['a']"
    assert (root + namedtuple_key_path).pprint() == '.attr'
    assert (root + fallback_key_path).pprint() == '[<flat index 1>]'
    assert sequence_key_path + dict_key_path == KeyPath((sequence_key_path, dict_key_path))
    assert (sequence_key_path + dict_key_path).pprint() == "[0]['a']"
    assert (dict_key_path + sequence_key_path).pprint() == "['a'][0]"
    assert (sequence_key_path + namedtuple_key_path).pprint() == '[0].attr'
    assert (namedtuple_key_path + sequence_key_path).pprint() == '.attr[0]'
    assert (dict_key_path + namedtuple_key_path).pprint() == "['a'].attr"
    assert (namedtuple_key_path + dict_key_path).pprint() == ".attr['a']"
    assert (sequence_key_path + fallback_key_path).pprint() == '[0][<flat index 1>]'
    assert (fallback_key_path + sequence_key_path).pprint() == '[<flat index 1>][0]'
    assert (dict_key_path + fallback_key_path).pprint() == "['a'][<flat index 1>]"
    assert (fallback_key_path + dict_key_path).pprint() == "[<flat index 1>]['a']"
    assert (namedtuple_key_path + fallback_key_path).pprint() == '.attr[<flat index 1>]'
    assert (fallback_key_path + namedtuple_key_path).pprint() == '[<flat index 1>].attr'
    assert sequence_key_path + dict_key_path + namedtuple_key_path + fallback_key_path == KeyPath(
        (sequence_key_path, dict_key_path, namedtuple_key_path, fallback_key_path)
    )
    assert (
        sequence_key_path + dict_key_path + namedtuple_key_path + fallback_key_path
    ).pprint() == "[0]['a'].attr[<flat index 1>]"

    for node, key_paths in (
        ([0, 1], [GetitemKeyPathEntry(0), GetitemKeyPathEntry(1)]),
        ((0, 1), [GetitemKeyPathEntry(0), GetitemKeyPathEntry(1)]),
        ({'b': 1, 'a': 2}, [GetitemKeyPathEntry('a'), GetitemKeyPathEntry('b')]),
        (OrderedDict([('b', 1), ('a', 2)]), [GetitemKeyPathEntry('b'), GetitemKeyPathEntry('a')]),
        (defaultdict(int, {'b': 1, 'a': 2}), [GetitemKeyPathEntry('a'), GetitemKeyPathEntry('b')]),
        (deque([0, 1]), [GetitemKeyPathEntry(0), GetitemKeyPathEntry(1)]),
        (CustomTuple(0, 1), [AttributeKeyPathEntry('foo'), AttributeKeyPathEntry('bar')]),
        (Vector2D(1, 2), [FlattenedKeyPathEntry(0), FlattenedKeyPathEntry(1)]),
    ):
        assert optree.ops._child_keys(node) == key_paths
