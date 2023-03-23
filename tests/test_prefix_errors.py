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
import textwrap
from collections import OrderedDict, defaultdict, deque

import pytest

import optree

# pylint: disable-next=wrong-import-order
from helpers import CustomTuple, FlatCache, TimeStructTime, Vector2D
from optree.registry import (
    AttributeKeyPathEntry,
    FlattenedKeyPathEntry,
    GetitemKeyPathEntry,
    KeyPath,
    KeyPathEntry,
)


def test_different_types():
    lhs, rhs = (1, 2), [1, 2]
    with pytest.raises(ValueError, match=r'Expected an instance of tuple, got .*\.'):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = {'a': 1, 'b': 2}, [1, 2]
    with pytest.raises(
        ValueError,
        match=(
            r'Expected an instance of dict, collections.OrderedDict, or collections.defaultdict, '
            r'got .*\.'
        ),
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = {'a': 1, 'b': 2}, OrderedDict({'a': 1, 'b': 2})
    optree.tree_map_(lambda x, y: None, lhs, rhs)
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = {'a': 1, 'b': 2}, defaultdict(int, {'a': 1, 'b': 2})
    optree.tree_map_(lambda x, y: None, lhs, rhs)
    () = optree.prefix_errors(lhs, rhs)


def test_different_types_nested():
    lhs, rhs = ((1,), (2,)), ([3], (4,))
    with pytest.raises(ValueError, match=r'Expected an instance of .*, got .*\.'):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_types_multiple():
    lhs, rhs = ((1,), (2,)), ([3], [4])
    with pytest.raises(ValueError, match=r'Expected an instance of .*, got .*\.'):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    e1, e2 = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_different_num_children():
    lhs, rhs = (1,), (2, 3)
    with pytest.raises(
        ValueError,
        match=r'tuple arity mismatch; expected: \d+, got: \d+; tuple: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_num_children_nested():
    lhs, rhs = [[1]], [[2, 3]]
    with pytest.raises(
        ValueError,
        match=r'list arity mismatch; expected: \d+, got: \d+; list: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_num_children_multiple():
    lhs, rhs = [[1], [2]], [[3, 4], [5, 6]]
    with pytest.raises(
        ValueError,
        match=r'list arity mismatch; expected: \d+, got: \d+; list: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    e1, e2 = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_different_metadata():
    lhs, rhs = {1: 2}, {3: 4}
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; dict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = OrderedDict({'a': 1, 'b': 2}), OrderedDict({'a': 3, 'c': 4})
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; OrderedDict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = OrderedDict({'a': 1, 'b': 2}), OrderedDict({'b': 4, 'a': 3})
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore key ordering
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = defaultdict(list, {'a': 1, 'b': 2}), defaultdict(set, {'b': 4, 'a': 3})
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore default factory
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = {'a': 1, 'b': 2}, defaultdict(list, {'b': 4, 'a': 3})
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore dictionary types
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = OrderedDict({'b': 4, 'a': 3}), {'a': 1, 'b': 2}
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore dictionary types
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = deque([1, 2], maxlen=None), deque([3, 4], maxlen=3)
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore maxlen
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = FlatCache([None, 1]), FlatCache(1)
    with pytest.raises(
        ValueError,
        match=r'Mismatch custom node data; expected: .*, got: .*; value: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree metadata at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_metadata_nested():
    lhs, rhs = [{1: 2}], [{3: 4}]
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; dict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_metadata_multiple():
    lhs, rhs = [{1: 2}, {3: 4}], [{3: 4}, {5: 6}]
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; dict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    e1, e2 = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_namedtuple():
    lhs, rhs = CustomTuple(1, [2, [3]]), CustomTuple(4, [5, 6])
    with pytest.raises(ValueError):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes.bar[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_structseq():
    lhs, rhs = TimeStructTime((1, [2, [3]], *range(7))), TimeStructTime((4, [5, 6], *range(7)))
    with pytest.raises(ValueError):
        optree.tree_map_(lambda x, y: None, lhs, rhs)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes.tm_mon[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_fallback_keypath():
    (e,) = optree.prefix_errors(Vector2D(1, [2]), Vector2D(3, 4))
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes[<flat index 1>]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_no_errors():
    for lhs, rhs in (
        ((1, 2), ((11, 12, 13), 2)),
        ({'a': 1, 'b': 2}, {'b': (11, 12, 13), 'a': 2}),
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
        () = optree.prefix_errors(lhs, rhs)


def test_different_structure_no_children():
    (e,) = optree.prefix_errors((), ([],))
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    (e,) = optree.prefix_errors({}, {'a': []})
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes tree root
            """,
        ).strip(),
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
        TypeError,
        match=re.escape("unsupported operand type(s) for +: 'KeyPath' and 'int'"),
    ):
        root + 1
    with pytest.raises(
        TypeError,
        match=re.escape("unsupported operand type(s) for +: 'int' and 'KeyPath'"),
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
        (
            sequence_key_path,
            dict_key_path,
            namedtuple_key_path,
            fallback_key_path,
        ),
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
