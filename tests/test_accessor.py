# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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

import dataclasses
import itertools
import re
from collections import UserDict, UserList
from typing import Any, NamedTuple

import pytest

import optree
from helpers import TREE_ACCESSORS, SysFloatInfoType, parametrize


def assert_equal_type_and_value(a, b):
    assert type(a) == type(b)
    assert a == b


def test_pytree_accessor_new():
    assert_equal_type_and_value(optree.PyTreeAccessor(), optree.PyTreeAccessor(()))
    assert_equal_type_and_value(
        optree.PyTreeAccessor(
            [
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ],
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        optree.PyTreeAccessor(
            [
                optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),
                optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ],
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry(key, dict, optree.PyTreeKind.DICT) for key in ('a', 'b', 'c')),
        ),
    )

    with pytest.raises(TypeError, match=r'Expected a path of PyTreeEntry, got .*\.'):
        optree.PyTreeAccessor([optree.MappingEntry('a', dict, optree.PyTreeKind.DICT), 'b'])


def test_pytree_accessor_add():
    assert_equal_type_and_value(
        optree.PyTreeAccessor() + optree.PyTreeAccessor(),
        optree.PyTreeAccessor(),
    )
    assert_equal_type_and_value(
        optree.PyTreeAccessor() + optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
    )
    assert_equal_type_and_value(
        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE) + optree.PyTreeAccessor(),
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
    )
    assert_equal_type_and_value(
        (
            optree.PyTreeAccessor()
            + optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE)
            + optree.SequenceEntry(1, list, optree.PyTreeKind.LIST)
            + optree.MappingEntry('c', dict, optree.PyTreeKind.DICT)
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        (
            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE)
            + optree.SequenceEntry(1, list, optree.PyTreeKind.LIST)
            + optree.MappingEntry('c', dict, optree.PyTreeKind.DICT)
            + optree.PyTreeAccessor()
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        (
            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE)
            + optree.SequenceEntry(1, list, optree.PyTreeKind.LIST)
            + optree.MappingEntry('c', dict, optree.PyTreeKind.DICT)
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        (
            optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE)
            + optree.PyTreeAccessor(
                (
                    optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                    optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
                ),
            )
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ),
        ),
    )

    with pytest.raises(TypeError, match=re.escape(r'unsupported operand type(s) for +')):
        optree.PyTreeAccessor() + 'a'
    with pytest.raises(TypeError, match=re.escape(r'unsupported operand type(s) for +')):
        1 + optree.PyTreeAccessor()
    with pytest.raises(TypeError, match=re.escape(r'unsupported operand type(s) for +')):
        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE) + 'a'
    with pytest.raises(TypeError, match=re.escape(r'unsupported operand type(s) for +')):
        1 + optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE)


def test_pytree_accessor_mul():
    assert_equal_type_and_value(optree.PyTreeAccessor() * 3, optree.PyTreeAccessor())
    assert 3 * optree.PyTreeAccessor() == optree.PyTreeAccessor()
    assert_equal_type_and_value(
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ),
        )
        * 2,
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        2
        * optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
            ),
        ),
    )


def test_pytree_accessor_getitem():
    entries = (
        optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
        optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
        optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),
    )
    accessor = optree.PyTreeAccessor(entries)

    for i in range(-len(entries) - 2, len(entries) + 1):
        if -len(entries) <= i < len(entries):
            assert_equal_type_and_value(accessor[i], entries[i])
        else:
            with pytest.raises(IndexError, match=r'index out of range'):
                accessor[i]

        for j in range(-len(entries) - 2, len(entries) + 1):
            assert len(accessor[i:j]) == len(entries[i:j])
            assert_equal_type_and_value(accessor[i:j], optree.PyTreeAccessor(entries[i:j]))


@parametrize(
    none_is_leaf=[False, True],
)
def test_pytree_accessor_equal_hash(none_is_leaf):
    for i, accessor1 in enumerate(itertools.chain.from_iterable(TREE_ACCESSORS[none_is_leaf])):
        for j, accessor2 in enumerate(itertools.chain.from_iterable(TREE_ACCESSORS[none_is_leaf])):
            if i == j:
                assert accessor1 == accessor2
                assert hash(accessor1) == hash(accessor2)
            if accessor1 == accessor2:
                assert hash(accessor1) == hash(accessor2)
            else:
                assert hash(accessor1) != hash(accessor2)


def test_pytree_entry_init():
    for path_entry_type in (
        optree.PyTreeEntry,
        optree.GetAttrEntry,
        optree.GetItemEntry,
        optree.FlattenedEntry,
        optree.AutoEntry,
        optree.SequenceEntry,
        optree.MappingEntry,
        optree.NamedTupleEntry,
        optree.StructSequenceEntry,
        optree.DataclassEntry,
    ):
        entry = path_entry_type(0, int, optree.PyTreeKind.CUSTOM)
        assert entry.entry == 0
        assert entry.type is int
        assert entry.kind == optree.PyTreeKind.CUSTOM

        with pytest.raises(
            ValueError,
            match=(
                re.escape('Cannot create a leaf path entry.')
                if path_entry_type is not optree.AutoEntry
                else r'Cannot create an automatic path entry for PyTreeKind .*\.'
            ),
        ):
            path_entry_type(0, int, optree.PyTreeKind.LEAF)
        with pytest.raises(
            ValueError,
            match=(
                re.escape('Cannot create a path entry for None.')
                if path_entry_type is not optree.AutoEntry
                else r'Cannot create an automatic path entry for PyTreeKind .*\.'
            ),
        ):
            path_entry_type(None, type(None), optree.PyTreeKind.NONE)


def test_pytree_auto_entry_new_invalid_kind():
    with pytest.raises(
        ValueError,
        match=r'Cannot create an automatic path entry for PyTreeKind .*\.',
    ):
        optree.AutoEntry(0, int, optree.PyTreeKind.LEAF)
    with pytest.raises(
        ValueError,
        match=r'Cannot create an automatic path entry for PyTreeKind .*\.',
    ):
        optree.AutoEntry(None, type(None), optree.PyTreeKind.NONE)
    with pytest.raises(
        ValueError,
        match=r'Cannot create an automatic path entry for PyTreeKind .*\.',
    ):
        optree.AutoEntry(0, tuple, optree.PyTreeKind.TUPLE)


def test_pytree_auto_entry_new_dispatch():
    class CustomTuple(NamedTuple):
        x: Any
        y: Any
        z: Any

    @dataclasses.dataclass
    class CustomDataclass:
        foo: Any
        bar: Any

    class MyMapping(UserDict):
        pass

    class MySequence(UserList):
        pass

    class MyObject:
        pass

    entry = optree.AutoEntry(0, SysFloatInfoType, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 0
    assert entry.type is SysFloatInfoType
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.StructSequenceEntry

    entry = optree.AutoEntry(0, CustomTuple, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 0
    assert entry.type is CustomTuple
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.NamedTupleEntry

    entry = optree.AutoEntry('foo', CustomDataclass, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 'foo'
    assert entry.type is CustomDataclass
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.DataclassEntry

    entry = optree.AutoEntry('foo', dict, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 'foo'
    assert entry.type is dict
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.MappingEntry

    entry = optree.AutoEntry('foo', MyMapping, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 'foo'
    assert entry.type is MyMapping
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.MappingEntry

    entry = optree.AutoEntry(0, tuple, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 0
    assert entry.type is tuple
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.SequenceEntry

    entry = optree.AutoEntry(0, list, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 0
    assert entry.type is list
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.SequenceEntry

    entry = optree.AutoEntry(0, str, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 0
    assert entry.type is str
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.SequenceEntry

    entry = optree.AutoEntry(0, bytes, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 0
    assert entry.type is bytes
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.SequenceEntry

    entry = optree.AutoEntry(0, MySequence, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 0
    assert entry.type is MySequence
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.SequenceEntry

    entry = optree.AutoEntry(0, MyObject, optree.PyTreeKind.CUSTOM)
    assert entry.entry == 0
    assert entry.type is MyObject
    assert entry.kind == optree.PyTreeKind.CUSTOM
    assert type(entry) is optree.FlattenedEntry
