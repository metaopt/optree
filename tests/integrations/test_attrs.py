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

# pylint: disable=missing-function-docstring,wrong-import-position

import re

import pytest


pytest.importorskip('attrs')

import attrs

import optree
from helpers import GLOBAL_NAMESPACE


def test_field_pytree_node():
    f1 = optree.integrations.attrs.field()
    assert f1.metadata['pytree_node'] is True
    f2 = optree.integrations.attrs.field(pytree_node=False)
    assert f2.metadata['pytree_node'] is False
    f3 = optree.integrations.attrs.field(pytree_node=True)
    assert f3.metadata['pytree_node'] is True


def test_field_pytree_node_from_metadata():
    f1 = optree.integrations.attrs.field(metadata={'pytree_node': False})
    assert f1.metadata['pytree_node'] is False
    f2 = optree.integrations.attrs.field(metadata={'pytree_node': True})
    assert f2.metadata['pytree_node'] is True
    # Explicit pytree_node overrides metadata
    f3 = optree.integrations.attrs.field(metadata={'pytree_node': True}, pytree_node=False)
    assert f3.metadata['pytree_node'] is False


def test_field_init_false():
    with pytest.raises(
        TypeError,
        match=re.escape('`pytree_node=True` is not allowed for non-init fields.'),
    ):
        optree.integrations.attrs.field(init=False)
    with pytest.raises(
        TypeError,
        match=re.escape('`pytree_node=True` is not allowed for non-init fields.'),
    ):
        optree.integrations.attrs.field(init=False, pytree_node=True)
    f = optree.integrations.attrs.field(init=False, pytree_node=False)
    assert f.metadata['pytree_node'] is False


def test_define():
    @optree.integrations.attrs.define(namespace='test-attrs')
    class Point:
        x: float
        y: float
        z: float = 0.0

    point = Point(1.0, 2.0, 3.0)
    leaves, treespec = optree.tree_flatten(point)
    assert leaves == [point]
    assert treespec.is_leaf()

    leaves, treespec = optree.tree_flatten(point, namespace='test-attrs')
    assert leaves == [1.0, 2.0, 3.0]
    assert point == optree.tree_unflatten(treespec, leaves)


def test_define_with_mixed_fields():
    @optree.integrations.attrs.define(namespace='test-attrs-mixed')
    class Foo:
        a: int
        b: int = 2
        c: int = optree.integrations.attrs.field(default=0, init=False, pytree_node=False)
        d: float = optree.integrations.attrs.field(default=42.0)
        e: int = optree.integrations.attrs.field(default=7, pytree_node=False)

        def __attrs_post_init__(self):
            self.c = self.a + self.b

    foo = Foo(1, d=4.5, e=8)
    leaves, treespec = optree.tree_flatten(foo, namespace='test-attrs-mixed')
    assert leaves == [1, 2, 4.5]
    assert foo == optree.tree_unflatten(treespec, leaves)


def test_define_frozen():
    @optree.integrations.attrs.frozen(namespace='test-attrs-frozen')
    class FrozenPoint:
        x: float
        y: float

    point = FrozenPoint(1.0, 2.0)
    leaves, treespec = optree.tree_flatten(point, namespace='test-attrs-frozen')
    assert leaves == [1.0, 2.0]
    assert point == optree.tree_unflatten(treespec, leaves)

    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        point.x = 3.0


def test_define_with_alias():
    @optree.integrations.attrs.define(namespace='test-attrs-alias')
    class Aliased:
        _x: float
        _y: float

    obj = Aliased(x=1.0, y=2.0)
    assert obj._x == 1.0
    assert obj._y == 2.0

    leaves, treespec = optree.tree_flatten(obj, namespace='test-attrs-alias')
    assert leaves == [1.0, 2.0]
    reconstructed = optree.tree_unflatten(treespec, [3.0, 4.0])
    assert reconstructed._x == 3.0
    assert reconstructed._y == 4.0


def test_define_with_non_class():
    with pytest.raises(
        TypeError,
        match=r'@optree\.integrations\.attrs\.define\(\) can only be used with classes, not .*',
    ):

        @optree.integrations.attrs.define(namespace='error')
        def foo():
            pass


def test_define_with_duplicate_registrations():
    with pytest.raises(
        TypeError,
        match=r'Cannot register .* as a pytree node more than once\.',
    ):

        @optree.integrations.attrs.define(namespace='error')
        @optree.integrations.attrs.define(namespace='error')
        class Foo1:
            x: int
            y: float


def test_define_with_init_false_field():
    # Using attrs.field (not our field) to bypass the early check in field(),
    # so the error comes from define()'s field inspection.
    with pytest.raises(
        TypeError,
        match=r'PyTree node field .* must be included in `__init__\(\)`\.',
    ):

        @optree.integrations.attrs.define(namespace='error')
        class Foo:
            x: int = attrs.field(init=False, metadata={'pytree_node': True})
            y: int = 123


def test_define_with_invalid_namespace():
    with pytest.raises(TypeError, match='The namespace must be a string'):

        @optree.integrations.attrs.define(namespace=1)
        class Foo1:
            x: int
            y: float

    @optree.integrations.attrs.define(namespace='')
    class Foo2:
        x: int
        y: float

    foo = Foo2(1, 2.0)
    leaves, treespec = optree.tree_flatten(foo)
    assert leaves == [1, 2.0]
    assert foo == optree.tree_unflatten(treespec, leaves)

    @optree.integrations.attrs.define(namespace=GLOBAL_NAMESPACE)
    class Foo3:
        x: int
        y: float

    foo = Foo3(1, 2.0)
    leaves, treespec = optree.tree_flatten(foo)
    assert leaves == [1, 2.0]
    assert foo == optree.tree_unflatten(treespec, leaves)


def test_mutable():
    @optree.integrations.attrs.mutable(namespace='test-attrs-mutable')
    class MutablePoint:
        x: float
        y: float

    point = MutablePoint(1.0, 2.0)
    leaves, treespec = optree.tree_flatten(point, namespace='test-attrs-mutable')
    assert leaves == [1.0, 2.0]
    assert point == optree.tree_unflatten(treespec, leaves)

    # Verify it's mutable
    point.x = 3.0
    assert point.x == 3.0


def test_mutable_is_define():
    assert optree.integrations.attrs.mutable is optree.integrations.attrs.define


def test_make_class():
    cls = optree.integrations.attrs.make_class(
        'Point',
        ['x', 'y'],
        namespace='test-attrs-make-class',
    )

    point = cls(1.0, 2.0)
    leaves, treespec = optree.tree_flatten(point, namespace='test-attrs-make-class')
    assert leaves == [1.0, 2.0]
    assert point == optree.tree_unflatten(treespec, leaves)


def test_make_class_with_dict_attrs():
    cls = optree.integrations.attrs.make_class(
        'Point',
        {'x': attrs.field(), 'y': attrs.field(metadata={'pytree_node': False})},
        namespace='test-attrs-make-class-dict',
    )

    point = cls(1.0, 2.0)
    leaves, treespec = optree.tree_flatten(point, namespace='test-attrs-make-class-dict')
    assert leaves == [1.0]
    assert point == optree.tree_unflatten(treespec, leaves)


def test_register_existing_class():
    @attrs.define
    class Existing:
        x: float
        y: float

    optree.integrations.attrs.register_node(Existing, namespace='test-attrs-register')

    obj = Existing(1.0, 2.0)
    leaves, treespec = optree.tree_flatten(obj, namespace='test-attrs-register')
    assert leaves == [1.0, 2.0]
    assert obj == optree.tree_unflatten(treespec, leaves)


def test_register_existing_class_with_metadata():
    @attrs.define
    class ExistingMixed:
        a: int
        b: int = attrs.field(metadata={'pytree_node': False})

    optree.integrations.attrs.register_node(ExistingMixed, namespace='test-attrs-register-meta')

    obj = ExistingMixed(1, 2)
    leaves, treespec = optree.tree_flatten(obj, namespace='test-attrs-register-meta')
    assert leaves == [1]
    assert obj == optree.tree_unflatten(treespec, leaves)


def test_register_non_attrs_class():
    class NotAttrs:
        pass

    with pytest.raises(TypeError, match='is not an attrs-decorated class'):
        optree.integrations.attrs.register_node(NotAttrs, namespace='error')


def test_register_non_class():
    with pytest.raises(TypeError, match='Expected a class'):
        optree.integrations.attrs.register_node(42, namespace='error')


def test_register_double_registration():
    @attrs.define
    class Double:
        x: int

    optree.integrations.attrs.register_node(Double, namespace='test-attrs-double')

    with pytest.raises(
        TypeError,
        match=r'Cannot register .* as a pytree node more than once\.',
    ):
        optree.integrations.attrs.register_node(Double, namespace='test-attrs-double-2')


def test_accessor_support():
    @optree.integrations.attrs.define(namespace='test-attrs-accessor')
    class AccessorTest:
        x: int
        y: float

    obj = AccessorTest(1, 2.0)
    accessors, leaves, treespec = optree.tree_flatten_with_accessor(
        obj,
        namespace='test-attrs-accessor',
    )
    assert leaves == [1, 2.0]
    assert len(accessors) == 2
    assert accessors == [
        optree.PyTreeAccessor(
            (optree.integrations.attrs.AttrsEntry('x', AccessorTest, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.integrations.attrs.AttrsEntry('y', AccessorTest, optree.PyTreeKind.CUSTOM),),
        ),
    ]
    assert [a(obj) for a in accessors] == [1, 2.0]
    assert treespec.kind == optree.PyTreeKind.CUSTOM
    assert treespec.type is AccessorTest


def test_attrs_entry():
    @attrs.define
    class EntryTest:
        x: int
        y: float
        z: int = attrs.field(init=False, default=0)

    entry_str = optree.integrations.attrs.AttrsEntry('x', EntryTest, optree.PyTreeKind.CUSTOM)
    assert entry_str.field == 'x'
    assert entry_str.name == 'x'
    assert entry_str.fields == ('x', 'y', 'z')
    assert entry_str.init_fields == ('x', 'y')
    assert 'AttrsEntry' in repr(entry_str)
    assert "'x'" in repr(entry_str)

    entry_int = optree.integrations.attrs.AttrsEntry(1, EntryTest, optree.PyTreeKind.CUSTOM)
    assert entry_int.field == 'y'
    assert entry_int.name == 'y'


def test_accessor_codify():
    @optree.integrations.attrs.define(namespace='test-attrs-codify')
    class CodifyTest:
        x: int
        y: float

    obj = CodifyTest(1, 2.0)
    accessors, _, _ = optree.tree_flatten_with_accessor(
        obj,
        namespace='test-attrs-codify',
    )
    assert accessors[0].codify() == '*.x'
    assert accessors[1].codify() == '*.y'


def test_define_as_decorator_factory():
    decorator = optree.integrations.attrs.define(namespace='test-attrs-factory')

    @decorator
    class Point:
        x: float
        y: float

    point = Point(1.0, 2.0)
    leaves, treespec = optree.tree_flatten(point, namespace='test-attrs-factory')
    assert leaves == [1.0, 2.0]
    assert point == optree.tree_unflatten(treespec, leaves)


def test_register_as_decorator():
    @optree.integrations.attrs.register_node(namespace='test-attrs-register-dec')
    @attrs.define
    class DecoratorTest:
        x: float
        y: float

    obj = DecoratorTest(3.0, 4.0)
    leaves, treespec = optree.tree_flatten(obj, namespace='test-attrs-register-dec')
    assert leaves == [3.0, 4.0]
    assert obj == optree.tree_unflatten(treespec, leaves)


def test_register_as_decorator_with_metadata():
    @optree.integrations.attrs.register_node(namespace='test-attrs-register-dec-meta')
    @attrs.define
    class DecoratorMixed:
        a: int
        b: int = attrs.field(metadata={'pytree_node': False})

    obj = DecoratorMixed(10, 20)
    leaves, treespec = optree.tree_flatten(obj, namespace='test-attrs-register-dec-meta')
    assert leaves == [10]
    assert obj == optree.tree_unflatten(treespec, leaves)


def test_register_namespace_as_positional():
    @optree.integrations.attrs.register_node('test-attrs-register-pos')
    @attrs.define
    class PosNsTest:
        x: int

    obj = PosNsTest(42)
    leaves, treespec = optree.tree_flatten(obj, namespace='test-attrs-register-pos')
    assert leaves == [42]
    assert obj == optree.tree_unflatten(treespec, leaves)


def test_register_namespace_as_positional_with_kwarg_error():
    with pytest.raises(
        ValueError,
        match=r'Cannot specify `namespace` when the first argument is a string\.',
    ):
        optree.integrations.attrs.register_node('ns1', namespace='ns2')


def test_register_namespace_empty_string_as_positional():
    with pytest.raises(
        ValueError,
        match=r'The namespace cannot be an empty string\.',
    ):
        optree.integrations.attrs.register_node('')


def test_register_missing_namespace():
    @attrs.define
    class MissingNs:
        x: int

    with pytest.raises(
        ValueError,
        match=r'Must specify `namespace` when the first argument is a class\.',
    ):
        optree.integrations.attrs.register_node(MissingNs)


def test_register_invalid_namespace():
    @attrs.define
    class Foo1:
        x: int

    with pytest.raises(TypeError, match='The namespace must be a string'):
        optree.integrations.attrs.register_node(Foo1, namespace=1)

    @attrs.define
    class Foo2:
        x: int
        y: float

    optree.integrations.attrs.register_node(Foo2, namespace='')

    foo = Foo2(1, 2.0)
    leaves, treespec = optree.tree_flatten(foo)
    assert leaves == [1, 2.0]
    assert foo == optree.tree_unflatten(treespec, leaves)

    @attrs.define
    class Foo3:
        x: int
        y: float

    optree.integrations.attrs.register_node(Foo3, namespace=GLOBAL_NAMESPACE)

    foo = Foo3(1, 2.0)
    leaves, treespec = optree.tree_flatten(foo)
    assert leaves == [1, 2.0]
    assert foo == optree.tree_unflatten(treespec, leaves)
