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


def test_define_reconstruction_reapplies_field_converters():
    # Characterization: the generated unflatten rebuilds via `cls(**fields)`, which re-runs attrs
    # field converters. Since `flatten` reads the already-converted value, an idempotent converter
    # round-trips exactly while a non-idempotent one does not. A class needing exact reconstruction
    # should register explicitly with custom flatten/unflatten (see `register_node`).

    # Idempotent converter: `int(int(x)) == int(x)`, so the round-trip is exact.
    @optree.integrations.attrs.define(namespace='test-attrs-converter-idempotent')
    class Cast:
        x: int = optree.integrations.attrs.field(converter=int)

    cast = Cast(5.0)  # __init__ converter: x = int(5.0) = 5
    leaves, treespec = optree.tree_flatten(cast, namespace='test-attrs-converter-idempotent')
    assert leaves == [5]
    assert optree.tree_unflatten(treespec, leaves).x == 5

    # Non-idempotent converter: unflatten re-applies it, so the stored `6` becomes `7`.
    @optree.integrations.attrs.define(namespace='test-attrs-converter-shift')
    class Shift:
        x: int = optree.integrations.attrs.field(converter=lambda v: v + 1)

    shift = Shift(5)  # __init__ converter: x = 5 + 1 = 6
    leaves, treespec = optree.tree_flatten(shift, namespace='test-attrs-converter-shift')
    assert leaves == [6]
    assert optree.tree_unflatten(treespec, leaves).x == 7  # re-applied: 6 + 1 = 7, not 6


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
        match=(
            r'Cannot register .* as a pytree node more than once with '
            r'`optree\.integrations\.attrs\.register_node\(\)`\. '
            r'Use `optree\.register_pytree_node\(\)` or `optree\.register_pytree_node_class\(\)` '
            r'with explicit flatten/unflatten functions'
        ),
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
        match=(
            r'Cannot register .* as a pytree node more than once with '
            r'`optree\.integrations\.attrs\.register_node\(\)`\. '
            r'Use `optree\.register_pytree_node\(\)` or `optree\.register_pytree_node_class\(\)` '
            r'with explicit flatten/unflatten functions'
        ),
    ):
        optree.integrations.attrs.register_node(Double, namespace='test-attrs-double-2')

    # As the error suggests, the class can still be registered in another namespace via the generic
    # API with explicit flatten/unflatten functions.
    optree.register_pytree_node(
        Double,
        lambda d: ((d.x,), None, None),
        lambda _, children: Double(*children),
        namespace='test-attrs-double-2',
    )
    optree.unregister_pytree_node(Double, namespace='test-attrs-double-2')


def test_register_node_failure_does_not_leak_fields_guard():
    @attrs.define
    class Leak:
        x: int

    namespace = 'test-attrs-register-leak'
    # Occupy the (class, namespace) slot directly so the attrs `register_node()`'s internal
    # `register_pytree_node()` call fails, after `register_node()` would set its `_FIELDS` guard.
    optree.register_pytree_node(
        Leak,
        lambda leak: ((leak.x,), None, None),
        lambda _, children: Leak(*children),
        namespace=namespace,
    )
    try:
        with pytest.raises(ValueError, match='already registered'):
            optree.integrations.attrs.register_node(Leak, namespace=namespace)
    finally:
        optree.unregister_pytree_node(Leak, namespace=namespace)

    # A failed registration must not leave the `_FIELDS` guard behind, or the class becomes
    # impossible to register ever again: every retry would raise "... more than once".
    # Re-registration after clearing the conflicting entry must succeed.
    optree.integrations.attrs.register_node(Leak, namespace=namespace)
    optree.unregister_pytree_node(Leak, namespace=namespace)


def test_register_node_rolls_back_when_the_guard_cannot_be_set():
    # `register_node()` commits the registry entry first and sets the `__optree_attrs_fields__`
    # guard afterwards. If the class refuses the guard, both must be rolled back: a class left
    # registered but unguarded is just as unrecoverable, because the retry then fails with
    # "already registered" instead.
    class RejectGuardOnce(type):
        rejected = False

        def __setattr__(cls, name, value, /):
            if name == '__optree_attrs_fields__' and not RejectGuardOnce.rejected:
                RejectGuardOnce.rejected = True  # a transient failure: reject only the first time
                raise RuntimeError('cannot set the guard attribute')
            super().__setattr__(name, value)

    @attrs.define
    class Rejecting(metaclass=RejectGuardOnce):
        x: int

    namespace = 'test-attrs-register-rollback'
    with pytest.raises(RuntimeError, match=r'cannot set the guard attribute'):
        optree.integrations.attrs.register_node(Rejecting, namespace=namespace)

    assert '__optree_attrs_fields__' not in Rejecting.__dict__
    with pytest.raises(ValueError, match=r'is not registered'):
        optree.unregister_pytree_node(Rejecting, namespace=namespace)

    # The class is left exactly as it was, so the retry succeeds.
    optree.integrations.attrs.register_node(Rejecting, namespace=namespace)
    optree.unregister_pytree_node(Rejecting, namespace=namespace)


def test_register_init_false_class_warns():
    @attrs.define(init=False)
    class InitFalse:
        x: int
        y: int

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Attrs class 'InitFalse' does not use an attrs-generated `__init__` "
            '(for example, `init=False`). '
            '`tree_unflatten()` may fail because '
            '`optree.integrations.attrs.register_node()` reconstructs instances with `cls(**kwargs)`.',
        ),
    ):
        optree.integrations.attrs.register_node(InitFalse, namespace='test-attrs-init-false')


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


def test_attrs_entry_integer_indexes_children():
    # For a class registered with `optree.integrations.attrs.register_node()`, an integer entry
    # indexes the children that registration emits (the fields that are BOTH `pytree_node=True` and
    # `init`). A non-child field interleaved between children must not shift the mapping.
    @optree.integrations.attrs.register_node(namespace='test-attrs-entry-integer')
    @attrs.define
    class Foo:
        a: int
        b: int = optree.integrations.attrs.field(default=0, pytree_node=False)  # not a child
        # non-init -> not a tree child
        d: int = optree.integrations.attrs.field(init=False, default=0, pytree_node=False)
        c: int = 0

    foo = Foo(1, 2, 3)  # a=1, b=2, c=3 (d defaults to 0, not an init parameter)
    assert tuple(a.name for a in attrs.fields(Foo)) == ('a', 'b', 'd', 'c')

    entry_int = optree.integrations.attrs.AttrsEntry(1, Foo, optree.PyTreeKind.CUSTOM)
    assert entry_int.init_fields == ('a', 'b', 'c')  # `d` is not an init field
    assert entry_int.children_fields == ('a', 'c')  # `b` is metadata and `d` is non-init
    # The 2nd child is `c` (not the metadata field `b` nor the non-init field `d`).
    assert entry_int.field == 'c'
    assert entry_int.name == 'c'
    assert entry_int.codify('x') == 'x.c'
    assert entry_int(foo) == foo.c == 3


def test_attrs_entry_integer_follows_generic_flatten_func():
    # Regression: an integer entry was resolved with the `optree.integrations.attrs` field
    # predicate, which does not apply to a class registered with the generic
    # `register_pytree_node()`. There the flatten function alone decides the children, so
    # `pytree_node=False` must not drop a field and leave the entry indexing past the end of an
    # internal list.
    @attrs.define
    class Foo:
        a: int
        b: int = optree.integrations.attrs.field(default=0, pytree_node=False)

    optree.register_pytree_node(
        Foo,
        lambda foo: ((foo.a, foo.b), None, None),  # both fields are children, entries default
        lambda _, children: Foo(*children),
        path_entry_type=optree.integrations.attrs.AttrsEntry,
        namespace='test-attrs-entry-generic',
    )

    foo = Foo(1, 2)
    paths, leaves, treespec = optree.tree_flatten_with_path(
        foo,
        namespace='test-attrs-entry-generic',
    )
    assert paths == [(0,), (1,)]
    assert leaves == [1, 2]

    accessors = treespec.accessors()
    assert [accessor(foo) for accessor in accessors] == [1, 2]
    assert [accessor[0].field for accessor in accessors] == ['a', 'b']
    assert [accessor.codify('x') for accessor in accessors] == ['x.a', 'x.b']


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
