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

# pylint: disable=missing-class-docstring,missing-function-docstring,invalid-name

import gc
import itertools
import sys
import time
from collections import OrderedDict, UserDict, defaultdict, deque, namedtuple
from typing import NamedTuple

import pytest

import optree


def getrefcount(obj=None):
    for _ in range(10):
        gc.collect()
    return sys.getrefcount(obj)


def parametrize(**argvalues) -> pytest.mark.parametrize:
    arguments = list(argvalues)
    argvalues = list(itertools.product(*tuple(map(argvalues.get, arguments))))

    ids = tuple(
        '-'.join(f'{arg}({val})' for arg, val in zip(arguments, values)) for values in argvalues
    )

    return pytest.mark.parametrize(arguments, argvalues, ids=ids)


CustomTuple = namedtuple('CustomTuple', ('foo', 'bar'))  # noqa: PYI024


class CustomNamedTupleSubclass(CustomTuple):
    pass


class EmptyTuple(NamedTuple):
    pass


# sys.float_info(max=*, max_exp=*, max_10_exp=*, min=*, min_exp=*, min_10_exp=*, dig=*, mant_dig=*, epsilon=*, radix=*, rounds=*)
SysFloatInfoType = type(sys.float_info)
# time.struct_time(tm_year=*, tm_mon=*, tm_mday=*, tm_hour=*, tm_min=*, tm_sec=*, tm_wday=*, tm_yday=*, tm_isdst=*)
TimeStructTimeType = time.struct_time


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


optree.register_pytree_node(
    Vector3D,
    lambda o: ((o.x, o.y), o.z),
    lambda z, xy: Vector3D(xy[0], xy[1], z),
    namespace=optree.registry.__GLOBAL_NAMESPACE,  # pylint: disable=protected-access
)


@optree.register_pytree_node_class(
    namespace=optree.registry.__GLOBAL_NAMESPACE,  # pylint: disable=protected-access
)
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Vector2D) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y})'

    def tree_flatten(self):
        return ((self.x, self.y), None)

    @classmethod
    def tree_unflatten(cls, metadata, children):  # pylint: disable=unused-argument
        return cls(*children)


# pylint: disable-next=protected-access
@optree.register_pytree_node_class(namespace=optree.registry.__GLOBAL_NAMESPACE)
class FlatCache:
    def __init__(self, structured, *, leaves=None, treespec=None):
        if treespec is None:
            leaves, treespec = optree.tree_flatten(structured)
        self._structured = structured
        self.treespec = treespec
        self.leaves = leaves

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


@optree.register_pytree_node_class(
    namespace=optree.registry.__GLOBAL_NAMESPACE,  # pylint: disable=protected-access
)
class MyDict(UserDict):
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

    def __repr__(self):
        return f'Counter({self.count})'

    def __next__(self):
        return self.increment()


NAMESPACED_TREE = MyAnotherDict([('baz', 101), ('foo', MyAnotherDict(a=1, b=2, c=None))])


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
    [],
    [(0,)],
    [(0,), (1,)],
]

TREE_PATHS = {
    optree.NONE_IS_NODE: TREE_PATHS_NONE_IS_NODE,
    optree.NONE_IS_LEAF: TREE_PATHS_NONE_IS_LEAF,
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
