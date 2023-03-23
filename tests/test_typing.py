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

# pylint: disable=missing-function-docstring

import re
import sys
import time

import pytest

import optree
from helpers import CustomNamedTupleSubclass, CustomTuple, Vector2D


class FakeNamedTuple(tuple):
    _fields = ('a', 2, 'b')


class FakeStructSequence(tuple):
    n_sequence_fields = 9
    n_fields = 11
    n_unnamed_fields = 0


def test_is_namedtuple():
    assert not optree.is_namedtuple((1, 2))
    assert not optree.is_namedtuple([1, 2])
    assert not optree.is_namedtuple(sys.float_info)
    assert not optree.is_namedtuple(time.gmtime())
    assert optree.is_namedtuple(CustomTuple(1, 2))
    assert optree.is_namedtuple(CustomNamedTupleSubclass(1, 2))
    assert not optree.is_namedtuple(FakeNamedTuple((1, 2, 3)))
    assert not optree.is_namedtuple(Vector2D(1, 2))
    assert not optree.is_namedtuple(FakeStructSequence((1, 2)))
    assert not optree.is_namedtuple_class(CustomTuple(1, 2))
    assert not optree.is_namedtuple_class(CustomNamedTupleSubclass(1, 2))
    assert not optree.is_namedtuple_class(FakeNamedTuple((1, 2, 3)))

    assert not optree.is_namedtuple(type(sys.float_info))
    assert not optree.is_namedtuple(time.struct_time)
    assert optree.is_namedtuple(CustomTuple)
    assert optree.is_namedtuple(CustomNamedTupleSubclass)
    assert not optree.is_namedtuple(FakeNamedTuple)
    assert not optree.is_namedtuple(Vector2D)
    assert not optree.is_namedtuple_class(type(sys.float_info))
    assert not optree.is_namedtuple_class(time.struct_time)
    assert optree.is_namedtuple_class(CustomTuple)
    assert optree.is_namedtuple_class(CustomNamedTupleSubclass)
    assert not optree.is_namedtuple_class(FakeNamedTuple)
    assert not optree.is_namedtuple_class(Vector2D)
    assert not optree.is_namedtuple_class(FakeStructSequence)


def test_is_structseq():
    assert not optree.is_structseq((1, 2))
    assert not optree.is_structseq([1, 2])
    assert optree.is_structseq(sys.float_info)
    assert optree.is_structseq(time.gmtime())
    assert not optree.is_structseq(CustomTuple(1, 2))
    assert not optree.is_structseq(CustomNamedTupleSubclass(1, 2))
    assert not optree.is_structseq(FakeNamedTuple((1, 2, 3)))
    assert not optree.is_structseq(Vector2D(1, 2))
    assert not optree.is_structseq(FakeStructSequence((1, 2)))
    assert not optree.is_structseq_class(sys.float_info)
    assert not optree.is_structseq_class(time.gmtime())

    assert optree.is_structseq(type(sys.float_info))
    assert optree.is_structseq(time.struct_time)
    assert not optree.is_structseq(CustomTuple)
    assert not optree.is_structseq(CustomNamedTupleSubclass)
    assert not optree.is_structseq(FakeNamedTuple)
    assert not optree.is_structseq(Vector2D)
    assert optree.is_structseq_class(type(sys.float_info))
    assert optree.is_structseq_class(time.struct_time)
    assert not optree.is_structseq_class(CustomTuple)
    assert not optree.is_structseq_class(CustomNamedTupleSubclass)
    assert not optree.is_structseq_class(FakeNamedTuple)
    assert not optree.is_structseq_class(Vector2D)
    assert not optree.is_structseq_class(FakeStructSequence)


def test_namedtuple_fields():
    assert optree.namedtuple_fields(CustomTuple) == ('foo', 'bar')
    assert optree.namedtuple_fields(CustomTuple(1, 2)) == ('foo', 'bar')
    assert optree.namedtuple_fields(CustomNamedTupleSubclass) == ('foo', 'bar')
    assert optree.namedtuple_fields(CustomNamedTupleSubclass(1, 2)) == ('foo', 'bar')

    with pytest.raises(
        TypeError,
        match=re.escape(r'Expected an instance of collections.namedtuple type, got [1, 2].'),
    ):
        optree.namedtuple_fields([1, 2])
    with pytest.raises(
        TypeError,
        match=re.escape(r"Expected a collections.namedtuple type, got <class 'list'>."),
    ):
        optree.namedtuple_fields(list)

    with pytest.raises(
        TypeError,
        match=re.escape(r'Expected an instance of collections.namedtuple type, got (1, 2).'),
    ):
        optree.namedtuple_fields((1, 2))
    with pytest.raises(
        TypeError,
        match=re.escape(r"Expected a collections.namedtuple type, got <class 'tuple'>."),
    ):
        optree.namedtuple_fields(tuple)

    with pytest.raises(
        TypeError,
        match=re.escape(
            r'Expected an instance of collections.namedtuple type, '
            r'got time.struct_time(tm_year=0, tm_mon=1, tm_mday=2, tm_hour=3, tm_min=4, tm_sec=5, tm_wday=6, tm_yday=7, tm_isdst=8).',
        ),
    ):
        optree.namedtuple_fields(time.struct_time(range(9)))
    with pytest.raises(
        TypeError,
        match=re.escape(r"Expected a collections.namedtuple type, got <class 'time.struct_time'>."),
    ):
        optree.namedtuple_fields(time.struct_time)

    with pytest.raises(
        TypeError,
        match=re.escape(r'Expected an instance of collections.namedtuple type, got (1, 2, 3).'),
    ):
        optree.namedtuple_fields(FakeNamedTuple((1, 2, 3)))
    with pytest.raises(
        TypeError,
        match=r"Expected a collections.namedtuple type, got <class '.*\.FakeNamedTuple'>\.",
    ):
        optree.namedtuple_fields(FakeNamedTuple)


def test_structseq_fields():
    assert optree.structseq_fields(sys.float_info) == (
        'max',
        'max_exp',
        'max_10_exp',
        'min',
        'min_exp',
        'min_10_exp',
        'dig',
        'mant_dig',
        'epsilon',
        'radix',
        'rounds',
    )
    assert optree.structseq_fields(type(sys.float_info)) == (
        'max',
        'max_exp',
        'max_10_exp',
        'min',
        'min_exp',
        'min_10_exp',
        'dig',
        'mant_dig',
        'epsilon',
        'radix',
        'rounds',
    )
    assert optree.structseq_fields(time.gmtime()) == (
        'tm_year',
        'tm_mon',
        'tm_mday',
        'tm_hour',
        'tm_min',
        'tm_sec',
        'tm_wday',
        'tm_yday',
        'tm_isdst',
    )
    assert optree.structseq_fields(time.struct_time) == (
        'tm_year',
        'tm_mon',
        'tm_mday',
        'tm_hour',
        'tm_min',
        'tm_sec',
        'tm_wday',
        'tm_yday',
        'tm_isdst',
    )

    with pytest.raises(
        TypeError,
        match=re.escape(r'Expected an instance of PyStructSequence type, got [1, 2].'),
    ):
        optree.structseq_fields([1, 2])
    with pytest.raises(
        TypeError,
        match=re.escape(r"Expected a PyStructSequence type, got <class 'list'>."),
    ):
        optree.structseq_fields(list)

    with pytest.raises(
        TypeError,
        match=re.escape(r'Expected an instance of PyStructSequence type, got (1, 2).'),
    ):
        optree.structseq_fields((1, 2))
    with pytest.raises(
        TypeError,
        match=re.escape(r"Expected a PyStructSequence type, got <class 'tuple'>."),
    ):
        optree.structseq_fields(tuple)

    with pytest.raises(
        TypeError,
        match=re.escape(
            r'Expected an instance of PyStructSequence type, got CustomTuple(foo=1, bar=2).',
        ),
    ):
        optree.structseq_fields(CustomTuple(1, 2))
    with pytest.raises(
        TypeError,
        match=re.escape(r"Expected a PyStructSequence type, got <class 'helpers.CustomTuple'>."),
    ):
        optree.structseq_fields(CustomTuple)

    with pytest.raises(
        TypeError,
        match=re.escape(r'Expected an instance of PyStructSequence type, got (1, 2).'),
    ):
        optree.structseq_fields(FakeStructSequence((1, 2)))
    with pytest.raises(
        TypeError,
        match=r"Expected a PyStructSequence type, got <class '.*\.FakeStructSequence'>\.",
    ):
        optree.structseq_fields(FakeStructSequence)
