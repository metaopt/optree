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
    Py_GIL_DISABLED,
    Vector2D,
    gc_collect,
    getrefcount,
    skipif_pypy,
)


class FakeNamedTuple(tuple):
    __slots__ = ()

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
    __slots__ = ()

    n_fields = 11
    n_sequence_fields = 9
    n_unnamed_fields = 0


def test_is_namedtuple():
    def is_namedtuple_(obj):
        nonlocal is_namedtuple, is_namedtuple_class, is_namedtuple_instance
        assert is_namedtuple(obj) == (is_namedtuple_class(obj) or is_namedtuple_instance(obj))
        assert is_namedtuple_class(obj) == (isinstance(obj, type) and is_namedtuple(obj))
        assert is_namedtuple_instance(obj) == (not isinstance(obj, type) and is_namedtuple(obj))
        return is_namedtuple(obj)

    for is_namedtuple, is_namedtuple_class, is_namedtuple_instance in (  # noqa: B007
        (
            optree.is_namedtuple,
            optree.is_namedtuple_class,
            optree.is_namedtuple_instance,
        ),
        (
            optree.is_namedtuple.__cxx_implementation__,
            optree.is_namedtuple_class.__cxx_implementation__,
            optree.is_namedtuple_instance.__cxx_implementation__,
        ),
        (
            optree.is_namedtuple.__python_implementation__,
            optree.is_namedtuple_class.__python_implementation__,
            optree.is_namedtuple_instance.__python_implementation__,
        ),
    ):
        assert not is_namedtuple_((1, 2))
        assert not is_namedtuple_([1, 2])
        assert not is_namedtuple_(sys.float_info)
        assert not is_namedtuple_(time.gmtime())
        assert is_namedtuple_(CustomTuple(1, 2))
        assert is_namedtuple_(CustomNamedTupleSubclass(1, 2))
        assert not is_namedtuple_(FakeNamedTuple(1, 2, 3))
        assert not is_namedtuple_(Vector2D(1, 2))
        assert not is_namedtuple_(FakeStructSequence((1, 2)))
        assert not is_namedtuple_class(CustomTuple(1, 2))
        assert not is_namedtuple_class(CustomNamedTupleSubclass(1, 2))
        assert not is_namedtuple_class(FakeNamedTuple(1, 2, 3))

        assert not is_namedtuple_(type(sys.float_info))
        assert not is_namedtuple_(time.struct_time)
        assert is_namedtuple_(CustomTuple)
        assert is_namedtuple_(CustomNamedTupleSubclass)
        assert not is_namedtuple_(FakeNamedTuple)
        assert not is_namedtuple_(Vector2D)
        assert not is_namedtuple_class(type(sys.float_info))
        assert not is_namedtuple_class(time.struct_time)
        assert is_namedtuple_class(CustomTuple)
        assert is_namedtuple_class(CustomNamedTupleSubclass)
        assert not is_namedtuple_class(FakeNamedTuple)
        assert not is_namedtuple_class(Vector2D)
        assert not is_namedtuple_class(FakeStructSequence)


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
    if not Py_GIL_DISABLED:
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
    if not Py_GIL_DISABLED:
        assert called_with == 'Foo'
        assert wr() is None


def test_namedtuple_fields():
    for namedtuple_fields in (
        optree.namedtuple_fields,
        optree.namedtuple_fields.__cxx_implementation__,
        optree.namedtuple_fields.__python_implementation__,
    ):
        assert namedtuple_fields(CustomTuple) == ('foo', 'bar')
        assert namedtuple_fields(CustomTuple(1, 2)) == ('foo', 'bar')
        assert namedtuple_fields(CustomNamedTupleSubclass) == ('foo', 'bar')
        assert namedtuple_fields(CustomNamedTupleSubclass(1, 2)) == ('foo', 'bar')

        with pytest.raises(
            TypeError,
            match=re.escape(r'Expected an instance of collections.namedtuple type, got [1, 2].'),
        ):
            namedtuple_fields([1, 2])
        with pytest.raises(
            TypeError,
            match=re.escape(r"Expected a collections.namedtuple type, got <class 'list'>."),
        ):
            namedtuple_fields(list)

        with pytest.raises(
            TypeError,
            match=re.escape(r'Expected an instance of collections.namedtuple type, got (1, 2).'),
        ):
            namedtuple_fields((1, 2))
        with pytest.raises(
            TypeError,
            match=re.escape(r"Expected a collections.namedtuple type, got <class 'tuple'>."),
        ):
            namedtuple_fields(tuple)

        with pytest.raises(
            TypeError,
            match=re.escape(
                r'Expected an instance of collections.namedtuple type, '
                r'got time.struct_time(tm_year=0, tm_mon=1, tm_mday=2, tm_hour=3, tm_min=4, tm_sec=5, tm_wday=6, tm_yday=7, tm_isdst=8).',
            ),
        ):
            namedtuple_fields(time.struct_time(range(9)))
        with pytest.raises(
            TypeError,
            match=re.escape(
                r"Expected a collections.namedtuple type, got <class 'time.struct_time'>.",
            ),
        ):
            namedtuple_fields(time.struct_time)

        with pytest.raises(
            TypeError,
            match=re.escape(
                r'Expected an instance of collections.namedtuple type, '
                r'got FakeNamedTuple(a=1, b=2, c=3).',
            ),
        ):
            namedtuple_fields(FakeNamedTuple(1, 2, 3))
        with pytest.raises(
            TypeError,
            match=r"Expected a collections.namedtuple type, got <class '.*\.FakeNamedTuple'>\.",
        ):
            namedtuple_fields(FakeNamedTuple)


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
    if not Py_GIL_DISABLED:
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
    if not Py_GIL_DISABLED:
        assert called_with == 'Foo'
        assert wr() is None


def test_is_structseq():
    def is_structseq_(obj):
        nonlocal is_structseq, is_structseq_class, is_structseq_instance
        assert is_structseq(obj) == (is_structseq_class(obj) or is_structseq_instance(obj))
        assert is_structseq_class(obj) == (isinstance(obj, type) and is_structseq(obj))
        assert is_structseq_instance(obj) == (not isinstance(obj, type) and is_structseq(obj))
        return is_structseq(obj)

    with pytest.raises(TypeError, match="type 'structseq' is not an acceptable base type"):

        class MyTuple(optree.typing.structseq):
            pass

    with pytest.raises(NotImplementedError):
        optree.typing.structseq(range(1))

    for is_structseq, is_structseq_class, is_structseq_instance in (  # noqa: B007
        (
            optree.is_structseq,
            optree.is_structseq_class,
            optree.is_structseq_instance,
        ),
        (
            optree.is_structseq.__cxx_implementation__,
            optree.is_structseq_class.__cxx_implementation__,
            optree.is_structseq_instance.__cxx_implementation__,
        ),
        (
            optree.is_structseq.__python_implementation__,
            optree.is_structseq_class.__python_implementation__,
            optree.is_structseq_instance.__python_implementation__,
        ),
    ):
        assert not is_structseq_((1, 2))
        assert not is_structseq_([1, 2])
        assert is_structseq_(sys.float_info)
        assert is_structseq_(time.gmtime())
        assert not is_structseq_(CustomTuple(1, 2))
        assert not is_structseq_(CustomNamedTupleSubclass(1, 2))
        assert not is_structseq_(FakeNamedTuple(1, 2, 3))
        assert not is_structseq_(Vector2D(1, 2))
        assert not is_structseq_(FakeStructSequence((1, 2)))
        assert not is_structseq_class(sys.float_info)
        assert not is_structseq_class(time.gmtime())

        assert is_structseq_(type(sys.float_info))
        assert is_structseq_(time.struct_time)
        assert not is_structseq_(CustomTuple)
        assert not is_structseq_(CustomNamedTupleSubclass)
        assert not is_structseq_(FakeNamedTuple)
        assert not is_structseq_(Vector2D)
        assert is_structseq_class(type(sys.float_info))
        assert is_structseq_class(time.struct_time)
        assert not is_structseq_class(CustomTuple)
        assert not is_structseq_class(CustomNamedTupleSubclass)
        assert not is_structseq_class(FakeNamedTuple)
        assert not is_structseq_class(Vector2D)
        assert not is_structseq_class(FakeStructSequence)


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
    if not Py_GIL_DISABLED:
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
    if not Py_GIL_DISABLED:
        assert called_with == 'Foo'
        assert wr() is None


def test_structseq_fields():
    for structseq_fields in (
        optree.structseq_fields,
        optree.structseq_fields.__cxx_implementation__,
        optree.structseq_fields.__python_implementation__,
    ):
        assert structseq_fields(sys.float_info) == (
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
        assert structseq_fields(type(sys.float_info)) == (
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
        assert structseq_fields(time.gmtime()) == (
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
        assert structseq_fields(time.struct_time) == (
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
            structseq_fields([1, 2])
        with pytest.raises(
            TypeError,
            match=re.escape(r"Expected a PyStructSequence type, got <class 'list'>."),
        ):
            structseq_fields(list)

        with pytest.raises(
            TypeError,
            match=re.escape(r'Expected an instance of PyStructSequence type, got (1, 2).'),
        ):
            structseq_fields((1, 2))
        with pytest.raises(
            TypeError,
            match=re.escape(r"Expected a PyStructSequence type, got <class 'tuple'>."),
        ):
            structseq_fields(tuple)

        with pytest.raises(
            TypeError,
            match=re.escape(
                r'Expected an instance of PyStructSequence type, got CustomTuple(foo=1, bar=2).',
            ),
        ):
            structseq_fields(CustomTuple(1, 2))
        with pytest.raises(
            TypeError,
            match=re.escape(
                r"Expected a PyStructSequence type, got <class 'helpers.CustomTuple'>.",
            ),
        ):
            structseq_fields(CustomTuple)

        with pytest.raises(
            TypeError,
            match=re.escape(r'Expected an instance of PyStructSequence type, got (1, 2).'),
        ):
            structseq_fields(FakeStructSequence((1, 2)))
        with pytest.raises(
            TypeError,
            match=r"Expected a PyStructSequence type, got <class '.*\.FakeStructSequence'>\.",
        ):
            structseq_fields(FakeStructSequence)


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
    if not Py_GIL_DISABLED:
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
    if not Py_GIL_DISABLED:
        assert called_with == 'Foo'
        assert wr() is None
