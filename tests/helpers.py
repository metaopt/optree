# Copyright 2022 MetaOPT Team. All Rights Reserved.
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

import itertools
from collections import OrderedDict, UserDict, defaultdict, deque, namedtuple

import pytest

import optree


def parametrize(**argvalues) -> pytest.mark.parametrize:
    arguments = list(argvalues)
    argvalues = list(itertools.product(*tuple(map(argvalues.get, arguments))))

    ids = tuple(
        '-'.join(f'{arg}({val})' for arg, val in zip(arguments, values)) for values in argvalues
    )

    return pytest.mark.parametrize(arguments, argvalues, ids=ids)


CustomTuple = namedtuple('CustomTuple', ('foo', 'bar'))


class CustomNamedTupleSubclass(CustomTuple):
    pass


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
    namespace=optree.registry.__GLOBAL_NAMESPACE  # pylint: disable=protected-access
)
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y})'

    def tree_flatten(self):
        return ((self.x, self.y), None)

    @classmethod
    def tree_unflatten(cls, metadata, children):  # pylint: disable=unused-argument
        return cls(*children)

    def __eq__(self, other):
        return isinstance(other, Vector2D) and (self.x, self.y) == (other.x, other.y)


@optree.register_pytree_node_class(
    namespace=optree.registry.__GLOBAL_NAMESPACE,  # pylint: disable=protected-access
)
class FlatCache:
    def __init__(self, structured, *, leaves=None, treespec=None):
        if treespec is None:
            leaves, treespec = optree.tree_flatten(structured)
        self._structured = structured
        self.treespec = treespec
        self.leaves = leaves

    def __hash__(self):
        return hash(self.structured)

    def __eq__(self, other):
        return isinstance(other, FlatCache) and self.structured == other.structured

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
class MyDictSubClass(MyDict):
    pass


NAMESPACED_TREE = MyDictSubClass([('baz', 101), ('foo', MyDictSubClass(a=1, b=2, c=None))])


# pylint: disable=line-too-long
TREES = (
    1,
    None,
    (None,),
    (1, None),
    (),
    ([()]),
    (1, 2),
    ((1, 'foo'), ['bar', (3, None, 7)]),
    [3],
    [3, CustomTuple(foo=(3, CustomTuple(foo=3, bar=None)), bar={'baz': 34})],
    [Vector3D(3, None, [4, 'foo'])],
    Vector2D(2, 3.0),
    {'a': 1, 'b': 2},
    OrderedDict([('foo', 34), ('baz', 101), ('something', -42)]),
    OrderedDict([('foo', 34), ('baz', 101), ('something', deque([None, 2, 3]))]),
    OrderedDict([('foo', 34), ('baz', 101), ('something', deque([None, None, 3], maxlen=2))]),
    OrderedDict([('foo', 34), ('baz', 101), ('something', deque([None, 2, 3], maxlen=2))]),
    defaultdict(dict, [('foo', 34), ('baz', 101), ('something', -42)]),
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
    [(0,), (1,)],
    [(0, 0), (0, 1), (1, 0), (1, 1, 0), (1, 1, 2)],
    [(0,)],
    [(0,), (1, 0, 0), (1, 0, 1, 0), (1, 1, 'baz')],
    [(0, 0)],
    [(0,), (1,)],
    [('a',), ('b',)],
    [('foo',), ('baz',), ('something',)],
    [('foo',), ('baz',), ('something', 1), ('something', 2)],
    [('foo',), ('baz',), ('something', 1)],
    [('foo',), ('baz',), ('something', 0), ('something', 1)],
    [('baz',), ('foo',), ('something',)],
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
    [(0,), (1,)],
    [(0, 0), (0, 1), (1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2)],
    [(0,)],
    [(0,), (1, 0, 0), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 'baz')],
    [(0, 0), (0, 1)],
    [(0,), (1,)],
    [('a',), ('b',)],
    [('foo',), ('baz',), ('something',)],
    [('foo',), ('baz',), ('something', 0), ('something', 1), ('something', 2)],
    [('foo',), ('baz',), ('something', 0), ('something', 1)],
    [('foo',), ('baz',), ('something', 0), ('something', 1)],
    [('baz',), ('foo',), ('something',)],
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
    'PyTreeSpec([()])',
    'PyTreeSpec((*, *))',
    'PyTreeSpec(((*, *), [*, (*, None, *)]))',
    'PyTreeSpec([*])',
    "PyTreeSpec([*, CustomTuple(foo=(*, CustomTuple(foo=*, bar=None)), bar={'baz': *})])",
    "PyTreeSpec([CustomTreeNode(Vector3D[[4, 'foo']], [*, None])])",
    'PyTreeSpec(CustomTreeNode(Vector2D[None], [*, *]))',
    "PyTreeSpec({'a': *, 'b': *})",
    "PyTreeSpec(OrderedDict([('foo', *), ('baz', *), ('something', *)]))",
    "PyTreeSpec(OrderedDict([('foo', *), ('baz', *), ('something', deque([None, *, *]))]))",
    "PyTreeSpec(OrderedDict([('foo', *), ('baz', *), ('something', deque([None, *], maxlen=2))]))",
    "PyTreeSpec(OrderedDict([('foo', *), ('baz', *), ('something', deque([*, *], maxlen=2))]))",
    "PyTreeSpec(defaultdict(<class 'dict'>, {'baz': *, 'foo': *, 'something': *}))",
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
    'PyTreeSpec([()], NoneIsLeaf)',
    'PyTreeSpec((*, *), NoneIsLeaf)',
    'PyTreeSpec(((*, *), [*, (*, *, *)]), NoneIsLeaf)',
    'PyTreeSpec([*], NoneIsLeaf)',
    "PyTreeSpec([*, CustomTuple(foo=(*, CustomTuple(foo=*, bar=*)), bar={'baz': *})], NoneIsLeaf)",
    "PyTreeSpec([CustomTreeNode(Vector3D[[4, 'foo']], [*, *])], NoneIsLeaf)",
    'PyTreeSpec(CustomTreeNode(Vector2D[None], [*, *]), NoneIsLeaf)',
    "PyTreeSpec({'a': *, 'b': *}, NoneIsLeaf)",
    "PyTreeSpec(OrderedDict([('foo', *), ('baz', *), ('something', *)]), NoneIsLeaf)",
    "PyTreeSpec(OrderedDict([('foo', *), ('baz', *), ('something', deque([*, *, *]))]), NoneIsLeaf)",
    "PyTreeSpec(OrderedDict([('foo', *), ('baz', *), ('something', deque([*, *], maxlen=2))]), NoneIsLeaf)",
    "PyTreeSpec(OrderedDict([('foo', *), ('baz', *), ('something', deque([*, *], maxlen=2))]), NoneIsLeaf)",
    "PyTreeSpec(defaultdict(<class 'dict'>, {'baz': *, 'foo': *, 'something': *}), NoneIsLeaf)",
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
