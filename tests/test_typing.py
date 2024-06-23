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

# pylint: disable=missing-function-docstring

import re
import sys
import time
import weakref
from collections import namedtuple

import pytest

import optree
from helpers import (
    CustomNamedTupleSubclass,
    CustomTuple,
    Vector2D,
    gc_collect,
    getrefcount,
    skipif_pypy,
)


class FakeNamedTuple(tuple):
    _fields = ('a', 'b', 'c')

    def __new__(cls, a, b, c):
        return super().__new__(cls, (a, b, c))

    @property
    def a(self):
        return self[0]

    @property
    def b(self):
        return self[1]

    @property
    def c(self):
        return self[2]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(a={self.a}, b={self.b}, c={self.c})'


class FakeStructSequence(tuple):
    n_fields = 11
    n_sequence_fields = 9
    n_unnamed_fields = 0


def test_is_namedtuple():
    assert not optree.is_namedtuple((1, 2))
    assert not optree.is_namedtuple([1, 2])
    assert not optree.is_namedtuple(sys.float_info)
    assert not optree.is_namedtuple(time.gmtime())
    assert optree.is_namedtuple(CustomTuple(1, 2))
    assert optree.is_namedtuple(CustomNamedTupleSubclass(1, 2))
    assert not optree.is_namedtuple(FakeNamedTuple(1, 2, 3))
    assert not optree.is_namedtuple(Vector2D(1, 2))
    assert not optree.is_namedtuple(FakeStructSequence((1, 2)))
    assert not optree.is_namedtuple_class(CustomTuple(1, 2))
    assert not optree.is_namedtuple_class(CustomNamedTupleSubclass(1, 2))
    assert not optree.is_namedtuple_class(FakeNamedTuple(1, 2, 3))

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


@skipif_pypy
def test_is_namedtuple_cache():
    Point = namedtuple('Point', ('x', 'y'))  # noqa: PYI024

    refcount = getrefcount(Point)
    weakrefcount = weakref.getweakrefcount(Point)
    assert optree.is_namedtuple(Point)
    new_refcount = getrefcount(Point)
    new_weakrefcount = weakref.getweakrefcount(Point)
    assert new_refcount == refcount
    assert new_weakrefcount == weakrefcount + 1
    assert optree.is_namedtuple_class(Point)
    assert weakref.getweakrefcount(Point) == new_weakrefcount
    wr = weakref.getweakrefs(Point)[0]
    assert wr() is Point
    del Point
    gc_collect()
    assert wr() is None

    refcount = getrefcount(time.struct_time)
    weakrefcount = weakref.getweakrefcount(time.struct_time)
    assert not optree.is_namedtuple(time.struct_time)
    new_refcount = getrefcount(time.struct_time)
    new_weakrefcount = weakref.getweakrefcount(time.struct_time)
    assert new_refcount == refcount
    assert new_weakrefcount <= weakrefcount + 1
    assert not optree.is_namedtuple_class(time.struct_time)
    assert weakref.getweakrefcount(time.struct_time) == new_weakrefcount

    called_with = ''

    class FooMeta(type):
        def __del__(cls):
            nonlocal called_with
            called_with = cls.__name__

    class Foo(metaclass=FooMeta):
        pass

    refcount = getrefcount(Foo)
    weakrefcount = weakref.getweakrefcount(Foo)
    assert not optree.is_namedtuple(Foo)
    new_refcount = getrefcount(Foo)
    new_weakrefcount = weakref.getweakrefcount(Foo)
    assert new_refcount == refcount
    assert new_weakrefcount == weakrefcount + 1
    assert not optree.is_namedtuple_class(Foo)
    assert weakref.getweakrefcount(Foo) == new_weakrefcount

    assert called_with == ''
    wr = weakref.getweakrefs(Foo)[0]
    assert wr() is Foo
    del Foo
    gc_collect()
    assert called_with == 'Foo'
    assert wr() is None


@skipif_pypy
def test_namedtuple_fields_cache():
    Point = namedtuple('Point', ('x', 'y'))  # noqa: PYI024

    refcount = getrefcount(Point)
    weakrefcount = weakref.getweakrefcount(Point)
    assert optree.namedtuple_fields(Point) == ('x', 'y')
    new_refcount = getrefcount(Point)
    new_weakrefcount = weakref.getweakrefcount(Point)
    assert new_refcount == refcount
    assert new_weakrefcount == weakrefcount + 1
    assert optree.namedtuple_fields(Point(0, 1)) == ('x', 'y')
    assert weakref.getweakrefcount(Point) == new_weakrefcount
    wr = weakref.getweakrefs(Point)[0]
    assert wr() is Point

    fields = optree.namedtuple_fields(Point)
    assert optree.namedtuple_fields(Point) is fields
    assert optree.namedtuple_fields(Point(0, 1)) is fields
    new_fields = ('a', 'b')
    Point._fields = new_fields
    assert optree.namedtuple_fields(Point) is new_fields
    assert optree.namedtuple_fields(Point(0, 1)) is new_fields

    del Point
    gc_collect()
    assert wr() is None

    with pytest.raises(
        TypeError,
        match=r"Expected a collections.namedtuple type, got <class '.*'>\.",
    ):
        assert optree.namedtuple_fields(time.struct_time)

    called_with = ''

    class FooMeta(type):
        def __del__(cls):
            nonlocal called_with
            called_with = cls.__name__

    class Foo(metaclass=FooMeta):
        pass

    refcount = getrefcount(Foo)
    weakrefcount = weakref.getweakrefcount(Foo)
    with pytest.raises(
        TypeError,
        match=r"Expected a collections.namedtuple type, got <class '.*'>\.",
    ):
        optree.namedtuple_fields(Foo)
    new_refcount = getrefcount(Foo)
    new_weakrefcount = weakref.getweakrefcount(Foo)
    assert new_refcount == refcount
    assert new_weakrefcount == weakrefcount + 1

    assert called_with == ''
    wr = weakref.getweakrefs(Foo)[0]
    assert wr() is Foo
    del Foo
    gc_collect()
    assert called_with == 'Foo'
    assert wr() is None


def test_is_structseq():
    with pytest.raises(TypeError, match="type 'structseq' is not an acceptable base type"):

        class MyTuple(optree.typing.structseq):
            pass

    with pytest.raises(NotImplementedError):
        optree.typing.structseq(range(1))

    assert not optree.is_structseq((1, 2))
    assert not optree.is_structseq([1, 2])
    assert optree.is_structseq(sys.float_info)
    assert optree.is_structseq(time.gmtime())
    assert not optree.is_structseq(CustomTuple(1, 2))
    assert not optree.is_structseq(CustomNamedTupleSubclass(1, 2))
    assert not optree.is_structseq(FakeNamedTuple(1, 2, 3))
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


@skipif_pypy
def test_is_structseq_cache():
    Point = namedtuple('Point', ('x', 'y'))  # noqa: PYI024

    refcount = getrefcount(Point)
    weakrefcount = weakref.getweakrefcount(Point)
    assert not optree.is_structseq(Point)
    new_refcount = getrefcount(Point)
    new_weakrefcount = weakref.getweakrefcount(Point)
    assert new_refcount == refcount
    assert new_weakrefcount == weakrefcount + 1
    assert not optree.is_structseq_class(Point)
    assert weakref.getweakrefcount(Point) == new_weakrefcount
    wr = weakref.getweakrefs(Point)[0]
    assert wr() is Point
    del Point
    gc_collect()
    assert wr() is None

    refcount = getrefcount(time.struct_time)
    weakrefcount = weakref.getweakrefcount(time.struct_time)
    assert optree.is_structseq(time.struct_time)
    new_refcount = getrefcount(time.struct_time)
    new_weakrefcount = weakref.getweakrefcount(time.struct_time)
    assert new_refcount == refcount
    assert new_weakrefcount <= weakrefcount + 1
    assert optree.is_structseq_class(time.struct_time)
    assert weakref.getweakrefcount(time.struct_time) == new_weakrefcount

    called_with = ''

    class FooMeta(type):
        def __del__(cls):
            nonlocal called_with
            called_with = cls.__name__

    class Foo(metaclass=FooMeta):
        pass

    refcount = getrefcount(Foo)
    weakrefcount = weakref.getweakrefcount(Foo)
    assert not optree.is_structseq(Foo)
    new_refcount = getrefcount(Foo)
    new_weakrefcount = weakref.getweakrefcount(Foo)
    assert new_refcount == refcount
    assert new_weakrefcount == weakrefcount + 1
    assert not optree.is_structseq_class(Foo)
    assert weakref.getweakrefcount(Foo) == new_weakrefcount

    assert called_with == ''
    wr = weakref.getweakrefs(Foo)[0]
    assert wr() is Foo
    del Foo
    gc_collect()
    assert called_with == 'Foo'
    assert wr() is None


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
        match=re.escape(
            r'Expected an instance of collections.namedtuple type, '
            r'got FakeNamedTuple(a=1, b=2, c=3).',
        ),
    ):
        optree.namedtuple_fields(FakeNamedTuple(1, 2, 3))
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


@skipif_pypy
def test_structseq_fields_cache():
    Point = namedtuple('Point', ('x', 'y'))  # noqa: PYI024

    refcount = getrefcount(Point)
    weakrefcount = weakref.getweakrefcount(Point)
    with pytest.raises(TypeError, match=r"Expected a PyStructSequence type, got <class '.*'>\."):
        optree.structseq_fields(Point)
    new_refcount = getrefcount(Point)
    new_weakrefcount = weakref.getweakrefcount(Point)
    assert new_refcount == refcount
    assert new_weakrefcount == weakrefcount + 1
    with pytest.raises(
        TypeError,
        match=re.escape('Expected an instance of PyStructSequence type, got Point(x=0, y=1).'),
    ):
        optree.structseq_fields(Point(0, 1))
    assert weakref.getweakrefcount(Point) == new_weakrefcount
    wr = weakref.getweakrefs(Point)[0]
    assert wr() is Point
    del Point
    gc_collect()
    assert wr() is None

    refcount = getrefcount(time.struct_time)
    weakrefcount = weakref.getweakrefcount(time.struct_time)
    assert optree.structseq_fields(time.struct_time) is optree.structseq_fields(time.struct_time)
    new_refcount = getrefcount(time.struct_time)
    new_weakrefcount = weakref.getweakrefcount(time.struct_time)
    assert new_refcount == refcount
    assert new_weakrefcount <= weakrefcount + 2
    assert optree.structseq_fields(time.gmtime()) is optree.structseq_fields(time.struct_time)
    assert weakref.getweakrefcount(time.struct_time) == new_weakrefcount

    called_with = ''

    class FooMeta(type):
        def __del__(cls):
            nonlocal called_with
            called_with = cls.__name__

    class Foo(metaclass=FooMeta):
        pass

    refcount = getrefcount(Foo)
    weakrefcount = weakref.getweakrefcount(Foo)
    with pytest.raises(TypeError, match=r"Expected a PyStructSequence type, got <class '.*'>\."):
        optree.structseq_fields(Foo)
    new_refcount = getrefcount(Foo)
    new_weakrefcount = weakref.getweakrefcount(Foo)
    assert new_refcount == refcount
    assert new_weakrefcount == weakrefcount + 1

    assert called_with == ''
    wr = weakref.getweakrefs(Foo)[0]
    assert wr() is Foo
    del Foo
    gc_collect()
    assert called_with == 'Foo'
    assert wr() is None
