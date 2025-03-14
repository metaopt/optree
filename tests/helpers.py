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

# pylint: disable=missing-class-docstring,missing-function-docstring,invalid-name

import dataclasses
import gc
import itertools
import platform
import sys
import sysconfig
import time
from collections import OrderedDict, UserDict, defaultdict, deque, namedtuple
from pathlib import Path
from typing import Any, NamedTuple

import pytest

import optree
from optree.registry import __GLOBAL_NAMESPACE as GLOBAL_NAMESPACE


TEST_ROOT = Path(__file__).absolute().parent


PYPY = platform.python_implementation() == 'PyPy'
skipif_pypy = pytest.mark.skipif(
    PYPY,
    reason='PyPy does not support weakref and refcount correctly',
)

Py_GIL_DISABLED = sysconfig.get_config_var('Py_GIL_DISABLED') is not None
NUM_GC_REPEAT = 10 if Py_GIL_DISABLED else 5


def gc_collect():
    for _ in range(NUM_GC_REPEAT):
        gc.collect()


def getrefcount(obj=None):
    gc_collect()
    return sys.getrefcount(obj)


def parametrize(**argvalues):
    arguments = list(argvalues)
    argvalues = list(itertools.product(*tuple(map(argvalues.get, arguments))))

    ids = tuple(
        '-'.join(f'{arg}({value!r})' for arg, value in zip(arguments, values))
        for values in argvalues
    )

    return pytest.mark.parametrize(arguments, argvalues, ids=ids)


MISSING = object()


def assert_equal_type_and_value(actual, expected=MISSING, *, expected_type=None):
    if expected_type is None:
        assert expected is not MISSING
        expected_type = type(expected)
    assert type(actual) is expected_type

    if expected is MISSING:
        return

    assert actual == expected
    if isinstance(expected, optree.PyTreeAccessor):
        assert hash(actual) == hash(expected)
        for i, j in zip(actual, expected):
            assert_equal_type_and_value(i, j)


def is_tuple(tup):
    return isinstance(tup, tuple)


def is_list(lst):
    return isinstance(lst, list)


def is_dict(dct):
    return isinstance(dct, dict)


def is_primitive_collection(obj):
    if type(obj) in {tuple, list, deque}:
        return all(isinstance(item, (int, float, str, bool, type(None))) for item in obj)
    if type(obj) in {dict, OrderedDict, defaultdict}:
        return all(isinstance(value, (int, float, str, bool, type(None))) for value in obj.values())
    return False


def is_none(none):
    return none is None


def is_not_none(none):
    return none is not None


def always(obj):  # pylint: disable=unused-argument
    return True


def never(obj):  # pylint: disable=unused-argument
    return False


IS_LEAF_FUNCTIONS = (
    is_tuple,
    is_list,
    is_dict,
    is_primitive_collection,
    is_none,
    is_not_none,
    always,
    never,
)


CustomTuple = namedtuple('CustomTuple', ('foo', 'bar'))  # noqa: PYI024


class CustomNamedTupleSubclass(CustomTuple):
    pass


class EmptyTuple(NamedTuple):
    pass


# sys.float_info(max=*, max_exp=*, max_10_exp=*, min=*, min_exp=*, min_10_exp=*, dig=*, mant_dig=*, epsilon=*, radix=*, rounds=*)
SysFloatInfoType = type(sys.float_info)
# time.struct_time(tm_year=*, tm_mon=*, tm_mday=*, tm_hour=*, tm_min=*, tm_sec=*, tm_wday=*, tm_yday=*, tm_isdst=*)
TimeStructTimeType = time.struct_time

if PYPY:
    SysFloatInfoType.__module__ = 'sys'
    TimeStructTimeType.__module__ = 'time'


class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})'


class Vector3DEntry(optree.PyTreeEntry):
    def __call__(self, obj):
        assert self.entry in {0, 1}
        return obj.x if self.entry == 0 else obj.y


optree.register_pytree_node(
    Vector3D,
    lambda o: ((o.x, o.y), o.z),
    lambda z, xy: Vector3D(xy[0], xy[1], z),
    path_entry_type=Vector3DEntry,
    namespace=GLOBAL_NAMESPACE,
)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        assert index in {0, 1}
        return self.x if index == 0 else self.y

    def __eq__(self, other):
        return isinstance(other, Vector2D) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y})'

    def tree_flatten(self):
        return (self.x, self.y), None

    @classmethod
    def tree_unflatten(cls, metadata, children):  # pylint: disable=unused-argument
        return cls(*children)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
@dataclasses.dataclass
class MyDataclass:
    alpha: Any
    beta: Any
    gamma: Any
    delta: Any

    def tree_flatten(self):
        return (
            (self.alpha, self.beta, self.gamma, self.delta),
            None,
            ('alpha', 'beta', 'gamma', 'delta'),
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children)


@optree.register_pytree_node_class(path_entry_type=optree.GetAttrEntry, namespace=GLOBAL_NAMESPACE)
@dataclasses.dataclass
class MyOtherDataclass:
    a: Any
    b: Any
    c: Any
    d: Any

    def tree_flatten(self):
        return (
            (self.a, self.c),
            (self.b, self.d),
            ('a', 'c'),
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        a, c = children
        b, d = metadata
        return cls(a, b, c, d)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
@dataclasses.dataclass
class MyAnotherDataclass:
    x: Any
    y: Any
    z: Any

    def tree_flatten(self):
        return (self.x, self.y, self.z), None

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
class FlatCache:
    TREE_PATH_ENTRY_TYPE = optree.GetItemEntry

    def __init__(self, structured, *, leaves=None, treespec=None):
        if treespec is None:
            leaves, treespec = optree.tree_flatten(structured)
        self._structured = structured
        self.treespec = treespec
        self.leaves = leaves

    def __getitem__(self, index):
        return self.leaves[index]

    def __eq__(self, other):
        return isinstance(other, FlatCache) and self.structured == other.structured

    def __hash__(self):
        return hash(self.structured)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.structured!r})'

    @property
    def structured(self):
        if self._structured is None:
            self._structured = optree.tree_unflatten(self.treespec, self.leaves)
        return self._structured

    def tree_flatten(self):
        return self.leaves, self.treespec

    @classmethod
    def tree_unflatten(cls, metadata, children):
        if not optree.all_leaves(children):
            children, metadata = optree.tree_flatten(optree.tree_unflatten(metadata, children))
        return cls(structured=None, leaves=children, treespec=metadata)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
class MyDict(UserDict):
    TREE_PATH_ENTRY_TYPE = optree.MappingEntry

    def tree_flatten(self):
        reversed_keys = sorted(self.keys(), reverse=True)
        return [self[key] for key in reversed_keys], reversed_keys, reversed_keys

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(zip(metadata, children))

    def __repr__(self):
        return f'{self.__class__.__name__}({super().__repr__()})'


@optree.register_pytree_node_class('namespace')
class MyAnotherDict(MyDict):
    pass


class Counter:
    def __init__(self, start=0):
        self.count = start

    def increment(self, n=1):
        self.count += n
        return self.count

    def __int__(self):
        return self.count

    def __eq__(self, other):
        return isinstance(other, Counter) and self.count == other.count

    def __hash__(self):
        return hash(self.count)

    def __repr__(self):
        return f'Counter({self.count})'

    def __next__(self):
        return self.increment()


NAMESPACED_TREE = MyAnotherDict([('baz', 101), ('foo', MyDict(a=1, b=2, c=None))])


# pylint: disable=line-too-long
TREES = (
    1,
    None,
    (None,),
    (1, None),
    (),
    [],
    ([()]),
    (1, 2),
    ((1, 'foo'), ['bar', (3, None, 7)]),
    [3],
    EmptyTuple(),
    [3, CustomTuple(foo=(3, CustomTuple(foo=3, bar=None)), bar={'baz': 34})],
    TimeStructTimeType((*range(1, 3), None, *range(3, 9))),
    SysFloatInfoType(
        (*range(1, 10), None, TimeStructTimeType((*range(10, 15), None, *range(15, 20)))),
    ),
    [Vector3D(3, None, [4, 'foo'])],
    Vector2D(2, 3.0),
    {},
    {'a': 1, 'b': 2},
    {'b': (2, 3), 'a': 1, 'c': None, 'd': {'f': 4, 'e': None}},
    OrderedDict(),
    OrderedDict([('foo', 34), ('baz', 101), ('something', -42)]),
    OrderedDict([('foo', 34), ('baz', 101), ('something', deque([None, 2, 3]))]),
    OrderedDict([('foo', 34), ('baz', 101), ('something', deque([None, None, 3], maxlen=2))]),
    OrderedDict([('foo', 34), ('baz', 101), ('something', deque([None, 2, 3], maxlen=2))]),
    defaultdict(),
    defaultdict(int),
    defaultdict(dict, [('foo', 34), ('baz', 101), ('something', -42)]),
    deque(),
    deque(maxlen=0),
    deque([None, 2, 3]),
    deque([None, None, 3], maxlen=2),
    MyDict([('baz', 101), ('foo', MyDict(a=1, b=2, c=None))]),
    NAMESPACED_TREE,
    CustomNamedTupleSubclass(foo='hello', bar=3.5),
    MyDataclass(2, None, 3, 5),
    MyOtherDataclass(7, 11, None, 13),
    MyAnotherDataclass(MyDataclass(2, 3, None, 5), 7, MyOtherDataclass(11, None, 13, 19)),
    FlatCache(None),
    FlatCache(1),
    FlatCache({'a': [1, 2]}),
)


TREE_PATHS_NONE_IS_NODE = [
    [()],
    [],
    [],
    [(0,)],
    [],
    [],
    [],
    [(0,), (1,)],
    [(0, 0), (0, 1), (1, 0), (1, 1, 0), (1, 1, 2)],
    [(0,)],
    [],
    [(0,), (1, 0, 0), (1, 0, 1, 0), (1, 1, 'baz')],
    [(0,), (1,), (3,), (4,), (5,), (6,), (7,), (8,)],
    [
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (8,),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 6),
        (10, 7),
        (10, 8),
    ],
    [(0, 0)],
    [(0,), (1,)],
    [],
    [('a',), ('b',)],
    [('a',), ('b', 0), ('b', 1), ('d', 'f')],
    [],
    [('foo',), ('baz',), ('something',)],
    [('foo',), ('baz',), ('something', 1), ('something', 2)],
    [('foo',), ('baz',), ('something', 1)],
    [('foo',), ('baz',), ('something', 0), ('something', 1)],
    [],
    [],
    [('baz',), ('foo',), ('something',)],
    [],
    [],
    [(1,), (2,)],
    [(1,)],
    [('foo', 'b'), ('foo', 'a'), ('baz',)],
    [()],
    [(0,), (1,)],
    [('alpha',), ('gamma',), ('delta',)],
    [('a',)],
    [(0, 'alpha'), (0, 'beta'), (0, 'delta'), (1,), (2, 'a'), (2, 'c')],
    [],
    [(0,)],
    [(0,), (1,)],
]

TREE_PATHS_NONE_IS_LEAF = [
    [()],
    [()],
    [(0,)],
    [(0,), (1,)],
    [],
    [],
    [],
    [(0,), (1,)],
    [(0, 0), (0, 1), (1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2)],
    [(0,)],
    [],
    [(0,), (1, 0, 0), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 'baz')],
    [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)],
    [
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (8,),
        (9,),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 5),
        (10, 6),
        (10, 7),
        (10, 8),
    ],
    [(0, 0), (0, 1)],
    [(0,), (1,)],
    [],
    [('a',), ('b',)],
    [('a',), ('b', 0), ('b', 1), ('c',), ('d', 'e'), ('d', 'f')],
    [],
    [('foo',), ('baz',), ('something',)],
    [('foo',), ('baz',), ('something', 0), ('something', 1), ('something', 2)],
    [('foo',), ('baz',), ('something', 0), ('something', 1)],
    [('foo',), ('baz',), ('something', 0), ('something', 1)],
    [],
    [],
    [('baz',), ('foo',), ('something',)],
    [],
    [],
    [(0,), (1,), (2,)],
    [(0,), (1,)],
    [('foo', 'c'), ('foo', 'b'), ('foo', 'a'), ('baz',)],
    [()],
    [(0,), (1,)],
    [('alpha',), ('beta',), ('gamma',), ('delta',)],
    [('a',), ('c',)],
    [(0, 'alpha'), (0, 'beta'), (0, 'gamma'), (0, 'delta'), (1,), (2, 'a'), (2, 'c')],
    [],
    [(0,)],
    [(0,), (1,)],
]

TREE_PATHS = {
    optree.NONE_IS_NODE: TREE_PATHS_NONE_IS_NODE,
    optree.NONE_IS_LEAF: TREE_PATHS_NONE_IS_LEAF,
}


TREE_ACCESSORS_NONE_IS_NODE = [
    [optree.PyTreeAccessor()],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
    ],
    [],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),)),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
    ],
    [optree.PyTreeAccessor((optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),))],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),)),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(1, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.MappingEntry('baz', dict, optree.PyTreeKind.DICT),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(0, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(1, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(3, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(4, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(5, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(6, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(7, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(8, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(0, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(1, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(2, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(3, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(4, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(5, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(6, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(7, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(8, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(0, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(1, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(2, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(3, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(4, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(6, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(7, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(8, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                Vector3DEntry(0, Vector3D, optree.PyTreeKind.CUSTOM),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor((optree.FlattenedEntry(0, Vector2D, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.FlattenedEntry(1, Vector2D, optree.PyTreeKind.CUSTOM),)),
    ],
    [],
    [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
    ],
    [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('d', dict, optree.PyTreeKind.DICT),
                optree.MappingEntry('f', dict, optree.PyTreeKind.DICT),
            ),
        ),
    ],
    [],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(2, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [],
    [],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('something', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
    ],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(2, deque, optree.PyTreeKind.DEQUE),)),
    ],
    [optree.PyTreeAccessor((optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),))],
    [
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('b', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor((optree.MappingEntry('baz', MyDict, optree.PyTreeKind.CUSTOM),)),
    ],
    [optree.PyTreeAccessor()],
    [
        optree.PyTreeAccessor(
            (optree.NamedTupleEntry(0, CustomNamedTupleSubclass, optree.PyTreeKind.NAMEDTUPLE),),
        ),
        optree.PyTreeAccessor(
            (optree.NamedTupleEntry(1, CustomNamedTupleSubclass, optree.PyTreeKind.NAMEDTUPLE),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.DataclassEntry('alpha', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('gamma', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('delta', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.GetAttrEntry('a', MyOtherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('alpha', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('beta', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('delta', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry(1, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(2, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.GetAttrEntry('a', MyOtherDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(2, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.GetAttrEntry('c', MyOtherDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
    ],
    [],
    [optree.PyTreeAccessor((optree.GetItemEntry(0, FlatCache, optree.PyTreeKind.CUSTOM),))],
    [
        optree.PyTreeAccessor((optree.GetItemEntry(0, FlatCache, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.GetItemEntry(1, FlatCache, optree.PyTreeKind.CUSTOM),)),
    ],
]

TREE_ACCESSORS_NONE_IS_LEAF = [
    [optree.PyTreeAccessor()],
    [optree.PyTreeAccessor()],
    [optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),))],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),)),
    ],
    [],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),)),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
    ],
    [optree.PyTreeAccessor((optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),))],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),)),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.NamedTupleEntry(1, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(1, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.MappingEntry('baz', dict, optree.PyTreeKind.DICT),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(0, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(1, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(2, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(3, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(4, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(5, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(6, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(7, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(8, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(0, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(1, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(2, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(3, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(4, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(5, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(6, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(7, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(8, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(9, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(0, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(1, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(2, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(3, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(4, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(5, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(6, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(7, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(8, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                Vector3DEntry(0, Vector3D, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                Vector3DEntry(1, Vector3D, optree.PyTreeKind.CUSTOM),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor((optree.FlattenedEntry(0, Vector2D, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.FlattenedEntry(1, Vector2D, optree.PyTreeKind.CUSTOM),)),
    ],
    [],
    [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
    ],
    [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor((optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('d', dict, optree.PyTreeKind.DICT),
                optree.MappingEntry('e', dict, optree.PyTreeKind.DICT),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('d', dict, optree.PyTreeKind.DICT),
                optree.MappingEntry('f', dict, optree.PyTreeKind.DICT),
            ),
        ),
    ],
    [],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(2, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [],
    [],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('something', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
    ],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(2, deque, optree.PyTreeKind.DEQUE),)),
    ],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),)),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('c', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('b', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor((optree.MappingEntry('baz', MyDict, optree.PyTreeKind.CUSTOM),)),
    ],
    [optree.PyTreeAccessor()],
    [
        optree.PyTreeAccessor(
            (optree.NamedTupleEntry(0, CustomNamedTupleSubclass, optree.PyTreeKind.NAMEDTUPLE),),
        ),
        optree.PyTreeAccessor(
            (optree.NamedTupleEntry(1, CustomNamedTupleSubclass, optree.PyTreeKind.NAMEDTUPLE),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.DataclassEntry('alpha', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('beta', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('gamma', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('delta', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.GetAttrEntry('a', MyOtherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.GetAttrEntry('c', MyOtherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('alpha', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('beta', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('gamma', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('delta', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry(1, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(2, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.GetAttrEntry('a', MyOtherDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(2, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.GetAttrEntry('c', MyOtherDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
    ],
    [],
    [optree.PyTreeAccessor((optree.GetItemEntry(0, FlatCache, optree.PyTreeKind.CUSTOM),))],
    [
        optree.PyTreeAccessor((optree.GetItemEntry(0, FlatCache, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.GetItemEntry(1, FlatCache, optree.PyTreeKind.CUSTOM),)),
    ],
]
TREE_ACCESSORS = {
    optree.NONE_IS_NODE: TREE_ACCESSORS_NONE_IS_NODE,
    optree.NONE_IS_LEAF: TREE_ACCESSORS_NONE_IS_LEAF,
}


TREE_STRINGS_NONE_IS_NODE = (
    'PyTreeSpec(*)',
    'PyTreeSpec(None)',
    'PyTreeSpec((None,))',
    'PyTreeSpec((*, None))',
    'PyTreeSpec(())',
    'PyTreeSpec([])',
    'PyTreeSpec([()])',
    'PyTreeSpec((*, *))',
    'PyTreeSpec(((*, *), [*, (*, None, *)]))',
    'PyTreeSpec([*])',
    'PyTreeSpec(EmptyTuple())',
    "PyTreeSpec([*, CustomTuple(foo=(*, CustomTuple(foo=*, bar=None)), bar={'baz': *})])",
    'PyTreeSpec(time.struct_time(tm_year=*, tm_mon=*, tm_mday=None, tm_hour=*, tm_min=*, tm_sec=*, tm_wday=*, tm_yday=*, tm_isdst=*))',
    'PyTreeSpec(sys.float_info(max=*, max_exp=*, max_10_exp=*, min=*, min_exp=*, min_10_exp=*, dig=*, mant_dig=*, epsilon=*, radix=None, rounds=time.struct_time(tm_year=*, tm_mon=*, tm_mday=*, tm_hour=*, tm_min=*, tm_sec=None, tm_wday=*, tm_yday=*, tm_isdst=*)))',
    "PyTreeSpec([CustomTreeNode(Vector3D[[4, 'foo']], [*, None])])",
    'PyTreeSpec(CustomTreeNode(Vector2D[None], [*, *]))',
    'PyTreeSpec({})',
    "PyTreeSpec({'a': *, 'b': *})",
    "PyTreeSpec({'a': *, 'b': (*, *), 'c': None, 'd': {'e': None, 'f': *}})",
    'PyTreeSpec(OrderedDict())',
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': *}))",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([None, *, *])}))",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([None, *], maxlen=2)}))",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([*, *], maxlen=2)}))",
    'PyTreeSpec(defaultdict(None, {}))',
    "PyTreeSpec(defaultdict(<class 'int'>, {}))",
    "PyTreeSpec(defaultdict(<class 'dict'>, {'baz': *, 'foo': *, 'something': *}))",
    'PyTreeSpec(deque([]))',
    'PyTreeSpec(deque([], maxlen=0))',
    'PyTreeSpec(deque([None, *, *]))',
    'PyTreeSpec(deque([None, *], maxlen=2))',
    "PyTreeSpec(CustomTreeNode(MyDict[['foo', 'baz']], [CustomTreeNode(MyDict[['c', 'b', 'a']], [None, *, *]), *]))",
    'PyTreeSpec(*)',
    'PyTreeSpec(CustomNamedTupleSubclass(foo=*, bar=*))',
    'PyTreeSpec(CustomTreeNode(MyDataclass[None], [*, None, *, *]))',
    'PyTreeSpec(CustomTreeNode(MyOtherDataclass[(11, 13)], [*, None]))',
    'PyTreeSpec(CustomTreeNode(MyAnotherDataclass[None], [CustomTreeNode(MyDataclass[None], [*, *, None, *]), *, CustomTreeNode(MyOtherDataclass[(None, 19)], [*, *])]))',
    'PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec(None)], []))',
    'PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec(*)], [*]))',
    "PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec({'a': [*, *]})], [*, *]))",
)

TREE_STRINGS_NONE_IS_LEAF = (
    'PyTreeSpec(*, NoneIsLeaf)',
    'PyTreeSpec(*, NoneIsLeaf)',
    'PyTreeSpec((*,), NoneIsLeaf)',
    'PyTreeSpec((*, *), NoneIsLeaf)',
    'PyTreeSpec((), NoneIsLeaf)',
    'PyTreeSpec([], NoneIsLeaf)',
    'PyTreeSpec([()], NoneIsLeaf)',
    'PyTreeSpec((*, *), NoneIsLeaf)',
    'PyTreeSpec(((*, *), [*, (*, *, *)]), NoneIsLeaf)',
    'PyTreeSpec([*], NoneIsLeaf)',
    'PyTreeSpec(EmptyTuple(), NoneIsLeaf)',
    "PyTreeSpec([*, CustomTuple(foo=(*, CustomTuple(foo=*, bar=*)), bar={'baz': *})], NoneIsLeaf)",
    'PyTreeSpec(time.struct_time(tm_year=*, tm_mon=*, tm_mday=*, tm_hour=*, tm_min=*, tm_sec=*, tm_wday=*, tm_yday=*, tm_isdst=*), NoneIsLeaf)',
    'PyTreeSpec(sys.float_info(max=*, max_exp=*, max_10_exp=*, min=*, min_exp=*, min_10_exp=*, dig=*, mant_dig=*, epsilon=*, radix=*, rounds=time.struct_time(tm_year=*, tm_mon=*, tm_mday=*, tm_hour=*, tm_min=*, tm_sec=*, tm_wday=*, tm_yday=*, tm_isdst=*)), NoneIsLeaf)',
    "PyTreeSpec([CustomTreeNode(Vector3D[[4, 'foo']], [*, *])], NoneIsLeaf)",
    'PyTreeSpec(CustomTreeNode(Vector2D[None], [*, *]), NoneIsLeaf)',
    'PyTreeSpec({}, NoneIsLeaf)',
    "PyTreeSpec({'a': *, 'b': *}, NoneIsLeaf)",
    "PyTreeSpec({'a': *, 'b': (*, *), 'c': *, 'd': {'e': *, 'f': *}}, NoneIsLeaf)",
    'PyTreeSpec(OrderedDict(), NoneIsLeaf)',
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': *}), NoneIsLeaf)",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([*, *, *])}), NoneIsLeaf)",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([*, *], maxlen=2)}), NoneIsLeaf)",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([*, *], maxlen=2)}), NoneIsLeaf)",
    'PyTreeSpec(defaultdict(None, {}), NoneIsLeaf)',
    "PyTreeSpec(defaultdict(<class 'int'>, {}), NoneIsLeaf)",
    "PyTreeSpec(defaultdict(<class 'dict'>, {'baz': *, 'foo': *, 'something': *}), NoneIsLeaf)",
    'PyTreeSpec(deque([]), NoneIsLeaf)',
    'PyTreeSpec(deque([], maxlen=0), NoneIsLeaf)',
    'PyTreeSpec(deque([*, *, *]), NoneIsLeaf)',
    'PyTreeSpec(deque([*, *], maxlen=2), NoneIsLeaf)',
    "PyTreeSpec(CustomTreeNode(MyDict[['foo', 'baz']], [CustomTreeNode(MyDict[['c', 'b', 'a']], [*, *, *]), *]), NoneIsLeaf)",
    'PyTreeSpec(*, NoneIsLeaf)',
    'PyTreeSpec(CustomNamedTupleSubclass(foo=*, bar=*), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(MyDataclass[None], [*, *, *, *]), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(MyOtherDataclass[(11, 13)], [*, *]), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(MyAnotherDataclass[None], [CustomTreeNode(MyDataclass[None], [*, *, *, *]), *, CustomTreeNode(MyOtherDataclass[(None, 19)], [*, *])]), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec(None)], []), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec(*)], [*]), NoneIsLeaf)',
    "PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec({'a': [*, *]})], [*, *]), NoneIsLeaf)",
)

TREE_STRINGS = {
    optree.NONE_IS_NODE: TREE_STRINGS_NONE_IS_NODE,
    optree.NONE_IS_LEAF: TREE_STRINGS_NONE_IS_LEAF,
}


LEAVES = (
    'foo',
    0.1,
    1,
    object(),
)
