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

import itertools

import pytest

import optree
from helpers import TREE_ACCESSORS, parametrize


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
