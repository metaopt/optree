# Copyright 2022-2026 MetaOPT Team. All Rights Reserved.
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

import builtins
import enum
import os
import re
import sys
import time
import weakref
from collections import namedtuple
from typing import TypeVar, Union, get_args, get_origin

import pytest

import optree
import optree.typing
from helpers import (
    OPTREE_HAS_FROZENDICT,
    PYBIND11_HAS_NATIVE_ENUM,
    PYPY,
    CustomNamedTupleSubclass,
    CustomTuple,
    Py_GIL_DISABLED,
    Vector2D,
    check_script_in_subprocess,
    disable_systrace,
    gc_collect,
    getrefcount,
    skipif_android,
    skipif_ios,
    skipif_pypy,
    skipif_wasm,
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


def test_pytreekind_enum():
    if PYBIND11_HAS_NATIVE_ENUM:
        all_kinds = list(optree.PyTreeKind)
        assert len(all_kinds) == optree.PyTreeKind.NUM_KINDS
        assert issubclass(optree.PyTreeKind, enum.IntEnum)
        assert issubclass(optree.PyTreeKind, int)

        assert optree.PyTreeKind.CUSTOM == 0
        assert optree.PyTreeKind.LEAF == 1
        assert optree.PyTreeKind.NONE == 2
        assert optree.PyTreeKind.CUSTOM.name == 'CUSTOM'
        assert optree.PyTreeKind.LEAF.name == 'LEAF'
        assert optree.PyTreeKind.NONE.name == 'NONE'
        for i, kind in enumerate(all_kinds):
            assert isinstance(kind, int)
            assert kind == i
            assert kind is optree.PyTreeKind(i)
            assert kind is getattr(optree.PyTreeKind, kind.name)

        with pytest.raises(ValueError, match=r'.* is not a valid .*\bPyTreeKind\b.*'):
            optree.PyTreeKind(optree.PyTreeKind.NUM_KINDS)
    else:
        all_kinds = [optree.PyTreeKind(i) for i in range(optree.PyTreeKind.NUM_KINDS)]

    assert optree.PyTreeKind.CUSTOM.value == 0
    assert optree.PyTreeKind.LEAF.value == 1
    assert optree.PyTreeKind.NONE.value == 2
    assert optree.PyTreeKind.CUSTOM.name == 'CUSTOM'
    assert optree.PyTreeKind.LEAF.name == 'LEAF'
    assert optree.PyTreeKind.NONE.name == 'NONE'
    for i, kind in enumerate(all_kinds):
        assert isinstance(kind, optree.PyTreeKind)
        assert int(kind) == i
        assert kind.value == i
        assert kind == optree.PyTreeKind(i)
        assert kind == getattr(optree.PyTreeKind, kind.name)


def test_pytree_typing():
    T = TypeVar('T')

    optree.PyTree[int]
    optree.PyTree[Union[int, str]]  # noqa: UP007
    optree.PyTree[T]
    assert optree.PyTree[optree.PyTree[int]] == optree.PyTree[int]
    assert optree.PyTree[optree.PyTree[Union[int, str]]] == optree.PyTree[Union[int, str]]  # noqa: UP007
    assert optree.PyTree[optree.PyTree[T]] == optree.PyTree[T]
    optree.PyTree[float | bytes]
    assert optree.PyTree[optree.PyTree[float | bytes]] == optree.PyTree[float | bytes]

    IntTree = optree.PyTreeTypeVar('IntTree', int)  # noqa: N806
    assert IntTree == optree.PyTree[IntTree]

    if sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT:
        frozendict = builtins.frozendict  # type: ignore[attr-defined] # pylint: disable=no-member

        assert optree.typing.FrozenDict is frozendict  # type: ignore[attr-defined]
        assert 'FrozenDict' in optree.typing.__all__
        # `FrozenDict` is inserted immediately after `Dict` in `__all__`.
        assert optree.typing.__all__[optree.typing.__all__.index('Dict') + 1] == 'FrozenDict'
        assert optree.PyTreeKind.FROZENDICT.name == 'FROZENDICT'

        # The `PyTree` generic alias includes a `frozendict[Any, ...]` member on Python 3.15+.
        assert any(get_origin(arg) is frozendict for arg in get_args(optree.PyTree[int]))
    else:
        # Without `frozendict` support the typing surface must not advertise it at all. This runs
        # on the majority of the CI matrix, so it is the negative half of the contract above.
        assert not hasattr(optree.typing, 'FrozenDict')
        assert 'FrozenDict' not in optree.typing.__all__
        assert not any(
            getattr(get_origin(arg), '__name__', None) == 'frozendict'
            for arg in get_args(optree.PyTree[int])
        )


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
@disable_systrace
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
@disable_systrace
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

    with pytest.raises(TypeError, match="type 'StructSequence' is not an acceptable base type"):

        class MyTuple(optree.typing.StructSequence):
            pass

    with pytest.raises(NotImplementedError):
        optree.typing.StructSequence(range(1))

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
@disable_systrace
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
        # On CPython, `os.stat_result` has UNNAMED sequence slots 7, 8, 9 (the integer
        # atime/mtime/ctime); the st_atime/st_mtime/st_ctime attributes are hidden FLOAT fields at
        # higher field indices. `tp_members` must be mapped by offset, not by position, or those
        # slots get mislabeled with the trailing hidden float names.
        stat_fields = structseq_fields(os.stat_result)
        assert len(stat_fields) == os.stat_result.n_sequence_fields
        assert stat_fields[:7] == (
            'st_mode',
            'st_ino',
            'st_dev',
            'st_nlink',
            'st_uid',
            'st_gid',
            'st_size',
        )
        if not PYPY:
            # PyPy has no unnamed fields: it names slots 7-9 `_integer_atime`/etc. and puts the
            # hidden float `st_atime` at a later index, so this CPython-only check does not apply.
            for name in stat_fields[7:10]:
                assert name not in {'st_atime', 'st_mtime', 'st_ctime'}
                assert not name.isidentifier()  # the PyStructSequence unnamed-field marker

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


@skipif_pypy  # PyPy reports `n_unnamed_fields == 0` and takes the index-based branch instead
def test_structseq_fields_python_implementation_falls_back_when_the_probe_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    # A type with unnamed slots that rejects the sentinel probe leaves nothing to match positions
    # against, so the implementation falls back to a trailing layout: the named fields keep the
    # leading positions and the rest get the unnamed marker. Pinning the lenient pairing that makes
    # that possible, since a strict one would raise instead of degrading.
    #
    # No stdlib type reaches this path: `os.stat_result`, the only CPython type with unnamed fields,
    # accepts arbitrary objects. The stand-in below is not a real PyStructSequence, hence the
    # `is_structseq_class` swap; `__slots__` supplies the `member_descriptor` fields it looks for.
    class RejectsProbe:
        __slots__ = ('st_alpha', 'st_beta', 'st_gamma')

        n_fields = 4
        n_sequence_fields = 4
        n_unnamed_fields = 1

        def __new__(cls, sequence, /):
            raise TypeError(f'cannot build {cls.__name__} from placeholder values: {sequence!r}')

    python_implementation = optree.structseq_fields.__python_implementation__
    original_is_structseq_class = optree.typing.is_structseq_class
    monkeypatch.setattr(
        optree.typing,
        'is_structseq_class',
        lambda cls, /: cls is RejectsProbe or original_is_structseq_class(cls),
    )
    fields = python_implementation(RejectsProbe)

    assert fields == (
        'st_alpha',
        'st_beta',
        'st_gamma',
        optree.typing.PyStructSequence_UnnamedField,
    )


def test_structseq_accessor_unnamed_fields_codify_by_index():
    # The accessor round-trip (the generated code evaluates to the accessed value) must hold for
    # every slot on every implementation. It exercises both codify styles: CPython leaves
    # `os.stat_result` slots 7, 8, 9 UNNAMED, so their accessors codify to index access (matching the
    # index-based `__call__`); PyPy names those slots (`_integer_atime` etc.) and codifies them by
    # attribute. Either way `accessor.codify(...)` and `accessor(...)` resolve to the same `st[i]`.
    st = os.stat(os.curdir)  # a real stat_result, valid on both CPython and PyPy
    accessors = optree.tree_accessors(st)
    assert len(accessors) == os.stat_result.n_sequence_fields
    for i, accessor in enumerate(accessors):
        assert eval(accessor.codify('__st'), {'__st': st}, {}) == accessor(st) == st[i]
    assert accessors[6].codify('__st') == '__st.st_size'  # a named slot -> attribute access

    # Repeat with DISTINCT per-field values (`st[i] == i`) so the round-trip reliably catches the
    # unnamed-slot mislabel: a real stat's whole-second atime could coincide with integer slot 7. On
    # CPython slots 7, 8, 9 are UNNAMED, so their accessors must codify to index access; codifying slot
    # 7 as `.st_atime` (the hidden FLOAT field CPython's own repr mislabels it with) would eval to
    # that wrong value. PyPy names those slots (`_integer_atime` etc.) and aliases `st_atime` back to
    # `self[7]`, so the unnamed-slot specifics below are asserted CPython-only.
    st = os.stat_result(range(os.stat_result.n_fields))
    accessors = optree.tree_accessors(st)
    assert len(accessors) == os.stat_result.n_sequence_fields
    for i, accessor in enumerate(accessors):
        assert eval(accessor.codify('__st'), {'__st': st}, {}) == accessor(st) == i
    if not PYPY:
        assert st.st_atime != st[7]  # a different (hidden) field, not sequence slot 7
        for i in (7, 8, 9):
            assert accessors[i].codify('__st') == f'__st[{i}]'


@skipif_pypy
@disable_systrace
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


@skipif_wasm
@skipif_android
@skipif_ios
@skipif_pypy  # CPython-only: uses `atexit._ncallbacks()` and CPython type caches
def test_type_caches_register_interpreter_cleanup():
    # optree keeps three process-global type caches: namedtuple classification, PyStructSequence
    # classification, and PyStructSequence field names. Each registers one per-interpreter `atexit`
    # cleanup on its first insert (the classification caches at import via the registry, the
    # field-name cache on first use). Measuring in a clean subprocess before importing optree pins
    # optree's whole footprint: one callback for the registry plus one per cache.
    check_script_in_subprocess(
        r"""
        import atexit
        import time
        import typing

        n0 = atexit._ncallbacks()
        import optree
        n1 = atexit._ncallbacks()
        optree.is_namedtuple(int)
        n2 = atexit._ncallbacks()
        optree.is_structseq(int)
        n3 = atexit._ncallbacks()
        optree.structseq_fields(time.struct_time)
        n4 = atexit._ncallbacks()

        assert n0 < n1, (n0, n1)
        assert n1 <= n2, (n1, n2)
        assert n2 <= n3, (n2, n3)
        assert n3 <= n4, (n3, n4)
        assert n4 - n0 == 4, (n0, n1, n2, n3, n4)
        """,
        output=None,
    )


@skipif_wasm
@skipif_android
@skipif_ios
@skipif_pypy  # CPython-only: uses the CPython type caches
def test_type_cache_insert_failure_does_not_leave_a_dangling_entry():
    # Regression: the caches published an entry before taking a reference to the value and before
    # creating the weakref that evicts it. If registering the per-interpreter `atexit` cleanup
    # raised in between, the entry survived owning nothing and with no eviction hook, so the next
    # lookup read the freed value and segfaulted. Run in a subprocess so a crash is a non-zero exit
    # rather than a lost test session.
    check_script_in_subprocess(
        r"""
        import atexit
        import time

        import optree

        real_register = atexit.register

        def failing_register(*args, **kwargs):
            raise RuntimeError('injected atexit failure')

        atexit.register = failing_register
        try:
            optree.structseq_fields(time.struct_time)
        except RuntimeError:
            pass
        else:
            raise AssertionError('the injected failure did not propagate')
        finally:
            atexit.register = real_register

        # The interpreter must not be marked as cleaned-up-registered by the failed attempt, or the
        # cleanup would never be retried. Sample before the retry, which is the call that registers.
        before = atexit._ncallbacks()

        # The failed insert must not be observable: the value is recomputed, not read back from a
        # dangling entry.
        fields = optree.structseq_fields(time.struct_time)
        after = atexit._ncallbacks()
        assert fields[:2] == ('tm_year', 'tm_mon'), fields
        assert after == before + 1, (before, after)
        """,
        output=None,
    )


@skipif_wasm
@skipif_android
@skipif_ios
@skipif_pypy  # CPython-only: uses the CPython type caches
def test_type_cache_insert_failure_before_import_does_not_crash():
    # The same hazard on the import-time path: the registry and the classification caches take their
    # first entries while `optree` is being imported, so break `atexit.register` before the import
    # rather than after. The initialization must fail as a normal `ImportError` and leave nothing
    # half-registered behind, rather than caching an entry it does not own and crashing later.
    # Re-importing in the same process is not possible once initialization has failed part way
    # through, which is a pybind11 module-init limitation rather than something optree controls.
    check_script_in_subprocess(
        r"""
        import atexit
        import sys
        import typing

        real_register = atexit.register

        def failing_register(*args, **kwargs):
            raise RuntimeError('injected atexit failure')

        atexit.register = failing_register
        try:
            import optree
        except ImportError:
            pass
        else:
            raise AssertionError('the injected failure did not propagate')
        finally:
            atexit.register = real_register

        assert 'optree' not in sys.modules
        assert 'optree._C' not in sys.modules
        """,
        output=None,
    )
