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

# pylint: disable=missing-function-docstring,invalid-name,wrong-import-order

import dataclasses
import inspect
import math
import re
import sys
from collections import OrderedDict

import pytest

import optree
from helpers import GLOBAL_NAMESPACE


def test_public_api():
    assert optree.dataclasses.__all__ == dataclasses.__all__
    for name in dataclasses.__all__:
        if name in {'field', 'dataclass', 'make_dataclass'}:
            assert getattr(optree.dataclasses, name) != getattr(dataclasses, name)
        else:
            assert getattr(optree.dataclasses, name) is getattr(dataclasses, name)


def test_same_signature():
    field_parameters = inspect.signature(optree.dataclasses.field).parameters.copy()
    field_original_parameters = inspect.signature(dataclasses.field).parameters.copy()
    assert len(field_parameters) >= len(field_original_parameters) + 1
    assert next(reversed(field_parameters)) == 'pytree_node'
    assert field_parameters['pytree_node'].kind == inspect.Parameter.KEYWORD_ONLY
    assert field_parameters['pytree_node'].default is None
    field_parameters.pop('pytree_node')
    assert OrderedDict(
        [
            (name, (param.name, param.kind, param.default))
            for name, param in field_parameters.items()
        ][: len(field_original_parameters)],
    ) == OrderedDict(
        (name, (param.name, param.kind, param.default))
        for name, param in field_original_parameters.items()
    )

    dataclass_parameters = inspect.signature(optree.dataclasses.dataclass).parameters.copy()
    dataclass_original_parameters = inspect.signature(dataclasses.dataclass).parameters.copy()
    assert len(dataclass_parameters) >= len(dataclass_original_parameters) + 1
    assert next(reversed(dataclass_parameters)) == 'namespace'
    assert dataclass_parameters['namespace'].kind == inspect.Parameter.KEYWORD_ONLY
    assert dataclass_parameters['namespace'].default is inspect.Parameter.empty
    dataclass_parameters.pop('namespace')
    assert OrderedDict(
        [
            (name, (param.name, param.kind, param.default))
            for name, param in dataclass_parameters.items()
        ][: len(dataclass_original_parameters)],
    ) == OrderedDict(
        (name, (param.name, param.kind, param.default))
        for name, param in dataclass_original_parameters.items()
    )

    make_dataclass_parameters = inspect.signature(
        optree.dataclasses.make_dataclass,
    ).parameters.copy()
    make_dataclass_original_parameters = inspect.signature(
        dataclasses.make_dataclass,
    ).parameters.copy()
    assert len(make_dataclass_parameters) >= len(make_dataclass_original_parameters) + 1
    assert next(reversed(make_dataclass_parameters)) == 'namespace'
    assert make_dataclass_parameters['namespace'].kind == inspect.Parameter.KEYWORD_ONLY
    assert make_dataclass_parameters['namespace'].default is inspect.Parameter.empty
    make_dataclass_parameters.pop('namespace')
    assert 'ns' in make_dataclass_parameters
    assert 'ns' not in make_dataclass_original_parameters
    assert OrderedDict(
        [
            (
                {'ns': 'namespace'}.get(name, name),
                ({'ns': 'namespace'}.get(param.name, param.name), param.kind, param.default),
            )
            for name, param in make_dataclass_parameters.items()
        ][: len(make_dataclass_original_parameters)],
    ) == OrderedDict(
        (name, (param.name, param.kind, param.default))
        for name, param in make_dataclass_original_parameters.items()
    )


def test_field_future_parameters():
    # pylint: disable=invalid-field-call
    optree.dataclasses.field()
    dataclasses.field()
    if sys.version_info >= (3, 10):
        optree.dataclasses.field(kw_only=True)
        dataclasses.field(kw_only=True)
        optree.dataclasses.field(kw_only=False)
        dataclasses.field(kw_only=False)
    else:
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.field(kw_only=True)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.field(kw_only=True)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.field(kw_only=False)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.field(kw_only=False)


def test_field_with_init():
    with pytest.raises(
        TypeError,
        match=re.escape("field() got an unexpected keyword argument 'pytree_node'"),
    ):
        dataclasses.field(pytree_node=True)

    f1 = optree.dataclasses.field()
    assert f1.metadata['pytree_node'] is True
    f2 = optree.dataclasses.field(pytree_node=False)
    assert f2.metadata['pytree_node'] is False
    f3 = optree.dataclasses.field(pytree_node=True)
    assert f3.metadata['pytree_node'] is True
    with pytest.raises(
        TypeError,
        match=re.escape('`pytree_node=True` is not allowed for non-init fields.'),
    ):
        optree.dataclasses.field(init=False)
    f4 = optree.dataclasses.field(init=False, metadata={'pytree_node': False})
    assert f4.metadata['pytree_node'] is False
    with pytest.raises(
        TypeError,
        match=re.escape('`pytree_node=True` is not allowed for non-init fields.'),
    ):
        optree.dataclasses.field(init=False, metadata={'pytree_node': True})
    f5 = optree.dataclasses.field(init=False, pytree_node=False)
    assert f5.metadata['pytree_node'] is False
    with pytest.raises(
        TypeError,
        match=re.escape('`pytree_node=True` is not allowed for non-init fields.'),
    ):
        optree.dataclasses.field(init=False, pytree_node=True)


def test_dataclass_future_parameters():
    with pytest.raises(
        TypeError,
        match=re.escape("dataclass() missing 1 required keyword-only argument: 'namespace'"),
    ):
        optree.dataclasses.dataclass()
    optree.dataclasses.dataclass(namespace='namespace')
    dataclasses.dataclass()
    if sys.version_info >= (3, 11):
        optree.dataclasses.dataclass(weakref_slot=True, namespace='namespace')
        dataclasses.dataclass(weakref_slot=True)
        optree.dataclasses.dataclass(weakref_slot=False, namespace='namespace')
        dataclasses.dataclass(weakref_slot=False)
    else:
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.dataclass(weakref_slot=True, namespace='error')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(weakref_slot=True)
        optree.dataclasses.dataclass(weakref_slot=False, namespace='namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(weakref_slot=False)
    if sys.version_info >= (3, 10):
        optree.dataclasses.dataclass(match_args=True, namespace='namespace')
        dataclasses.dataclass(match_args=True)
        optree.dataclasses.dataclass(match_args=False, namespace='namespace')
        dataclasses.dataclass(match_args=False)
        optree.dataclasses.dataclass(kw_only=True, namespace='namespace')
        dataclasses.dataclass(kw_only=True)
        optree.dataclasses.dataclass(kw_only=False, namespace='namespace')
        dataclasses.dataclass(kw_only=False)
        optree.dataclasses.dataclass(slots=True, namespace='namespace')
        dataclasses.dataclass(slots=True)
        optree.dataclasses.dataclass(slots=False, namespace='namespace')
        dataclasses.dataclass(slots=False)
    else:
        optree.dataclasses.dataclass(match_args=True, namespace='namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(match_args=True)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.dataclass(match_args=False, namespace='error')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(match_args=False)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.dataclass(kw_only=True, namespace='error')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(kw_only=True)
        optree.dataclasses.dataclass(kw_only=False, namespace='namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(kw_only=False)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.dataclass(slots=True, namespace='error')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(slots=True)
        optree.dataclasses.dataclass(slots=False, namespace='namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(slots=False)


def test_dataclass_with_init():
    @optree.dataclasses.dataclass(namespace='namespace')
    class Foo:
        a: int
        b: int = 2
        c: int = optree.dataclasses.field(init=False, pytree_node=False)
        d: float = dataclasses.field(init=True, default=42.0)
        e: int = dataclasses.field(init=False, metadata={'pytree_node': False})
        f: float = optree.dataclasses.field(init=True, default=6.0, metadata={'pytree_node': True})
        g: int = dataclasses.field(init=True, default=7, metadata={'pytree_node': False})

        def __post_init__(self):
            self.c = self.a + self.b
            self.e = self.d * self.f

    foo = Foo(1, d=4.5, f=3.0, g=8)
    leaves, treespec = optree.tree_flatten(foo)
    assert leaves == [foo]
    assert treespec.is_leaf()
    leaves, treespec = optree.tree_flatten(foo, namespace='namespace')
    assert leaves == [1, 2, 4.5, 3.0]
    assert foo == optree.tree_unflatten(treespec, leaves)

    with pytest.raises(
        TypeError,
        match=r'PyTree node field .* must be included in `__init__\(\)`\.',
    ):

        @optree.dataclasses.dataclass(namespace='error')
        class Foo1:
            x: int = dataclasses.field(init=False)
            y: int = 123

    with pytest.raises(
        TypeError,
        match=r'PyTree node field .* must be included in `__init__\(\)`\.',
    ):

        @optree.dataclasses.dataclass(namespace='error')
        class Foo2:
            x: int = dataclasses.field(init=False, metadata={'pytree_node': True})
            y: int = 123

    with pytest.raises(
        TypeError,
        match=re.escape('`pytree_node=True` is not allowed for non-init fields.'),
    ):

        @optree.dataclasses.dataclass(namespace='error')
        class Foo3:
            x: int = optree.dataclasses.field(init=False, pytree_node=True)
            y: int = 123


def test_dataclass_with_non_class():
    with pytest.raises(
        TypeError,
        match=r'@optree\.dataclasses\.dataclass\(\) can only be used with classes, not .*',
    ):

        @optree.dataclasses.dataclass(namespace='error')
        def foo():
            pass

    with pytest.raises(
        TypeError,
        match=r'@optree\.dataclasses\.dataclass\(\) can only be used with classes, not .*',
    ):
        optree.dataclasses.dataclass(lambda x: x, namespace='error')

    class Foo:
        x: int
        y: float

    optree.dataclasses.dataclass(Foo, namespace='namespace')


def test_dataclass_with_duplicate_registrations():
    with pytest.raises(
        TypeError,
        match=r'@optree\.dataclasses\.dataclass\(\) cannot be applied to .* more than once\.',
    ):

        @optree.dataclasses.dataclass(namespace='error')
        @optree.dataclasses.dataclass(namespace='error')
        class Foo1:
            x: int
            y: float

    with pytest.raises(
        TypeError,
        match=r'@optree\.dataclasses\.dataclass\(\) cannot be applied to .* more than once\.',
    ):

        @optree.dataclasses.dataclass(namespace='other-error')
        @optree.dataclasses.dataclass(namespace='error')
        class Foo2:
            x: int
            y: float

    @optree.register_pytree_node_class(namespace='other-namespace')
    @optree.dataclasses.dataclass(namespace='namespace')
    class Foo:
        x: int
        y: float

        def tree_flatten(self):
            return [self.y], self.x, ['y']

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(metadata, children[0])

    foo = Foo(1, 2.0)
    accessors1, leaves1, treespec1 = optree.tree_flatten_with_accessor(foo)
    assert optree.tree_unflatten(treespec1, leaves1) == foo
    assert accessors1 == [optree.PyTreeAccessor()]
    assert leaves1 == [foo]
    assert treespec1.namespace == ''
    assert treespec1.is_leaf()
    assert treespec1.kind == optree.PyTreeKind.LEAF
    assert treespec1.type is None
    accessors2, leaves2, treespec2 = optree.tree_flatten_with_accessor(foo, namespace='namespace')
    assert optree.tree_unflatten(treespec2, leaves2) == foo
    assert accessors2 == [
        optree.PyTreeAccessor((optree.DataclassEntry('x', Foo, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.DataclassEntry('y', Foo, optree.PyTreeKind.CUSTOM),)),
    ]
    assert [a(foo) for a in accessors2] == [1, 2.0]
    assert leaves2 == [1, 2.0]
    assert treespec2.namespace == 'namespace'
    assert treespec2.kind == optree.PyTreeKind.CUSTOM
    assert treespec2.type is Foo
    (
        accessors3,
        leaves3,
        treespec3,
    ) = optree.tree_flatten_with_accessor(foo, namespace='other-namespace')
    assert optree.tree_unflatten(treespec3, leaves3) == foo
    assert accessors3 == [
        optree.PyTreeAccessor((optree.DataclassEntry('y', Foo, optree.PyTreeKind.CUSTOM),)),
    ]
    assert [a(foo) for a in accessors3] == [2.0]
    assert leaves3 == [2.0]
    assert treespec3.namespace == 'other-namespace'
    assert treespec3.kind == optree.PyTreeKind.CUSTOM
    assert treespec3.type is Foo


def test_dataclass_with_invalid_namespace():
    with pytest.raises(TypeError, match='The namespace must be a string'):

        @optree.dataclasses.dataclass(namespace=1)
        class Foo1:
            x: int
            y: float

    with pytest.raises(ValueError, match=re.escape('The namespace cannot be an empty string.')):

        @optree.dataclasses.dataclass(namespace='')
        class Foo2:
            x: int
            y: float

    @optree.dataclasses.dataclass(namespace=GLOBAL_NAMESPACE)
    class Foo:
        x: int
        y: float

    foo = Foo(1, 2.0)
    accessors, leaves, treespec = optree.tree_flatten_with_accessor(foo)
    assert optree.tree_unflatten(treespec, leaves) == foo
    assert accessors == [
        optree.PyTreeAccessor((optree.DataclassEntry('x', Foo, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.DataclassEntry('y', Foo, optree.PyTreeKind.CUSTOM),)),
    ]
    assert [a(foo) for a in accessors] == [1, 2.0]
    assert leaves == [1, 2.0]
    assert treespec.namespace == ''
    assert treespec.kind == optree.PyTreeKind.CUSTOM
    assert treespec.type is Foo


def test_make_dataclass_future_parameters():
    with pytest.raises(
        TypeError,
        match=re.escape("make_dataclass() missing 1 required keyword-only argument: 'namespace'"),
    ):
        optree.dataclasses.make_dataclass('Foo1', ['x', ('y', int), ('z', float, 0.0)])
    with pytest.raises(
        TypeError,
        match=re.escape("make_dataclass() missing 1 required keyword-only argument: 'ns'"),
    ):
        optree.dataclasses.make_dataclass(
            'Foo1',
            ['x', ('y', int), ('z', float, 0.0)],
            namespace={
                'norm': lambda self: math.hypot(self.x, self.y, self.z),
            },
        )
    if sys.version_info >= (3, 12):
        assert (
            optree.dataclasses.make_dataclass(
                'Foo2',
                ['x', ('y', int), ('z', float, 0.0)],
                namespace='namespace',
            ).__module__
            == __name__
        )
        assert (
            dataclasses.make_dataclass('Foo3', ['x', ('y', int), ('z', float, 0.0)]).__module__
            == __name__
        )
        assert (
            optree.dataclasses.make_dataclass(
                'Foo4',
                ['x', ('y', int), ('z', float, 0.0)],
                module='some.module',
                namespace='namespace',
            ).__module__
            == 'some.module'
        )
        assert (
            dataclasses.make_dataclass(
                'Foo5',
                ['x', ('y', int), ('z', float, 0.0)],
                module='some.module',
            ).__module__
            == 'some.module'
        )
    else:
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.make_dataclass(
                'Foo4',
                ['x', ('y', int), ('z', float, 0.0)],
                module='some.module',
                namespace='error',
            )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.make_dataclass(
                'Foo5',
                ['x', ('y', int), ('z', float, 0.0)],
                module='some.module',
            )
    if sys.version_info >= (3, 11):
        optree.dataclasses.make_dataclass(
            'Foo6',
            ['x', ('y', int), ('z', float, 0.0)],
            slots=True,
            weakref_slot=True,
            namespace='namespace',
        )
        dataclasses.make_dataclass(
            'Foo7',
            ['x', ('y', int), ('z', float, 0.0)],
            slots=True,
            weakref_slot=True,
        )
        optree.dataclasses.make_dataclass(
            'Foo8',
            ['x', ('y', int), ('z', float, 0.0)],
            weakref_slot=False,
            namespace='namespace',
        )
        dataclasses.make_dataclass('Foo9', ['x', ('y', int), ('z', float, 0.0)], weakref_slot=False)
    else:
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.make_dataclass(
                'Foo6',
                ['x', ('y', int), ('z', float, 0.0)],
                weakref_slot=True,
                namespace='error',
            )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.make_dataclass(
                'Foo7',
                ['x', ('y', int), ('z', float, 0.0)],
                weakref_slot=True,
            )
        optree.dataclasses.make_dataclass(
            'Foo8',
            ['x', ('y', int), ('z', float, 0.0)],
            weakref_slot=False,
            namespace='namespace',
        )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.make_dataclass(
                'Foo9',
                ['x', ('y', int), ('z', float, 0.0)],
                weakref_slot=False,
            )
    if sys.version_info >= (3, 10):
        optree.dataclasses.make_dataclass(
            'Foo10',
            ['x', ('y', int), ('z', float, 0.0)],
            match_args=True,
            namespace='namespace',
        )
        dataclasses.make_dataclass('Foo11', ['x', ('y', int), ('z', float, 0.0)], match_args=True)
        optree.dataclasses.make_dataclass(
            'Foo12',
            ['x', ('y', int), ('z', float, 0.0)],
            match_args=False,
            namespace='namespace',
        )
        dataclasses.make_dataclass('Foo13', ['x', ('y', int), ('z', float, 0.0)], match_args=False)
        optree.dataclasses.make_dataclass(
            'Foo14',
            ['x', ('y', int), ('z', float, 0.0)],
            kw_only=True,
            namespace='namespace',
        )
        dataclasses.make_dataclass('Foo15', ['x', ('y', int), ('z', float, 0.0)], kw_only=True)
        optree.dataclasses.make_dataclass(
            'Foo16',
            ['x', ('y', int), ('z', float, 0.0)],
            kw_only=False,
            namespace='namespace',
        )
        dataclasses.make_dataclass('Foo17', ['x', ('y', int), ('z', float, 0.0)], kw_only=False)
        optree.dataclasses.make_dataclass(
            'Foo18',
            ['x', ('y', int), ('z', float, 0.0)],
            slots=True,
            namespace='namespace',
        )
        dataclasses.make_dataclass('Foo19', ['x', ('y', int), ('z', float, 0.0)], slots=True)
        optree.dataclasses.make_dataclass(
            'Foo20',
            ['x', ('y', int), ('z', float, 0.0)],
            slots=False,
            namespace='namespace',
        )
        dataclasses.make_dataclass('Foo21', ['x', ('y', int), ('z', float, 0.0)], slots=False)
    else:
        optree.dataclasses.make_dataclass(
            'Foo10',
            ['x', ('y', int), ('z', float, 0.0)],
            match_args=True,
            namespace='namespace',
        )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.make_dataclass(
                'Foo11',
                ['x', ('y', int), ('z', float, 0.0)],
                match_args=True,
            )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.make_dataclass(
                'Foo12',
                ['x', ('y', int), ('z', float, 0.0)],
                match_args=False,
                namespace='error',
            )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.make_dataclass(
                'Foo13',
                ['x', ('y', int), ('z', float, 0.0)],
                match_args=False,
            )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.make_dataclass(
                'Foo14',
                ['x', ('y', int), ('z', float, 0.0)],
                kw_only=True,
                namespace='error',
            )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.make_dataclass('Foo15', ['x', ('y', int), ('z', float, 0.0)], kw_only=True)
        optree.dataclasses.make_dataclass(
            'Foo16',
            ['x', ('y', int), ('z', float, 0.0)],
            kw_only=False,
            namespace='namespace',
        )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.make_dataclass('Foo17', ['x', ('y', int), ('z', float, 0.0)], kw_only=False)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.make_dataclass(
                'Foo18',
                ['x', ('y', int), ('z', float, 0.0)],
                slots=True,
                namespace='error',
            )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.make_dataclass('Foo19', ['x', ('y', int), ('z', float, 0.0)], slots=True)
        optree.dataclasses.make_dataclass(
            'Foo20',
            ['x', ('y', int), ('z', float, 0.0)],
            slots=False,
            namespace='namespace',
        )
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.make_dataclass('Foo21', ['x', ('y', int), ('z', float, 0.0)], slots=False)


def test_make_dataclass_with_duplicate_registrations():
    Foo1 = optree.dataclasses.make_dataclass(  # noqa: N806
        'Foo1',
        [('x', int), ('y', float)],
        namespace='error',
    )
    with pytest.raises(
        TypeError,
        match=r'@optree\.dataclasses\.dataclass\(\) cannot be applied to .* more than once\.',
    ):
        optree.dataclasses.dataclass(Foo1, namespace='error')

    Foo2 = optree.dataclasses.make_dataclass(  # noqa: N806
        'Foo2',
        [('x', int), ('y', float)],
        namespace='error',
    )
    with pytest.raises(
        TypeError,
        match=r'@optree\.dataclasses\.dataclass\(\) cannot be applied to .* more than once\.',
    ):
        optree.dataclasses.dataclass(Foo2, namespace='other-error')

    Foo = optree.register_pytree_node_class(namespace='other-namespace')(  # noqa: N806
        optree.dataclasses.make_dataclass(
            'Foo',
            [('x', int), ('y', float)],
            ns={
                'tree_flatten': lambda self: ([self.y], self.x, ['y']),
                'tree_unflatten': classmethod(
                    lambda cls, metadata, children: cls(metadata, children[0]),
                ),
            },
            namespace='namespace',
        ),
    )

    foo = Foo(1, 2.0)
    accessors1, leaves1, treespec1 = optree.tree_flatten_with_accessor(foo)
    assert optree.tree_unflatten(treespec1, leaves1) == foo
    assert accessors1 == [optree.PyTreeAccessor()]
    assert leaves1 == [foo]
    assert treespec1.namespace == ''
    assert treespec1.is_leaf()
    assert treespec1.kind == optree.PyTreeKind.LEAF
    assert treespec1.type is None
    accessors2, leaves2, treespec2 = optree.tree_flatten_with_accessor(foo, namespace='namespace')
    assert optree.tree_unflatten(treespec2, leaves2) == foo
    assert accessors2 == [
        optree.PyTreeAccessor((optree.DataclassEntry('x', Foo, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.DataclassEntry('y', Foo, optree.PyTreeKind.CUSTOM),)),
    ]
    assert [a(foo) for a in accessors2] == [1, 2.0]
    assert leaves2 == [1, 2.0]
    assert treespec2.namespace == 'namespace'
    assert treespec2.kind == optree.PyTreeKind.CUSTOM
    assert treespec2.type is Foo
    (
        accessors3,
        leaves3,
        treespec3,
    ) = optree.tree_flatten_with_accessor(foo, namespace='other-namespace')
    assert optree.tree_unflatten(treespec3, leaves3) == foo
    assert accessors3 == [
        optree.PyTreeAccessor((optree.DataclassEntry('y', Foo, optree.PyTreeKind.CUSTOM),)),
    ]
    assert [a(foo) for a in accessors3] == [2.0]
    assert leaves3 == [2.0]
    assert treespec3.namespace == 'other-namespace'
    assert treespec3.kind == optree.PyTreeKind.CUSTOM
    assert treespec3.type is Foo


def test_make_dataclass_with_invalid_namespace():
    with pytest.raises(TypeError, match='The namespace must be a string'):
        optree.dataclasses.make_dataclass('Foo1', ['x', ('y', int), ('z', float, 0.0)], namespace=1)

    with pytest.raises(ValueError, match=re.escape('The namespace cannot be an empty string.')):
        optree.dataclasses.make_dataclass(
            'Foo2',
            ['x', ('y', int), ('z', float, 0.0)],
            namespace='',
        )

    Foo = optree.dataclasses.make_dataclass(  # noqa: N806
        'Foo',
        [('x', int), ('y', float)],
        namespace=GLOBAL_NAMESPACE,
    )
    foo = Foo(1, 2.0)
    accessors, leaves, treespec = optree.tree_flatten_with_accessor(foo)
    assert optree.tree_unflatten(treespec, leaves) == foo
    assert accessors == [
        optree.PyTreeAccessor((optree.DataclassEntry('x', Foo, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.DataclassEntry('y', Foo, optree.PyTreeKind.CUSTOM),)),
    ]
    assert [a(foo) for a in accessors] == [1, 2.0]
    assert leaves == [1, 2.0]
    assert treespec.namespace == ''
    assert treespec.kind == optree.PyTreeKind.CUSTOM
    assert treespec.type is Foo

    with pytest.raises(TypeError, match='The namespace must be a string'):
        optree.dataclasses.make_dataclass(
            'Bar1',
            ['x', ('y', int), ('z', float, 0.0)],
            ns=1,
            namespace=None,
        )

    with pytest.raises(ValueError, match=re.escape('The namespace cannot be an empty string.')):
        optree.dataclasses.make_dataclass(
            'Foo2',
            ['x', ('y', int), ('z', float, 0.0)],
            ns='',
            namespace={},
        )

    Bar = optree.dataclasses.make_dataclass(  # noqa: N806
        'Bar',
        [('x', int), ('y', float)],
        ns=GLOBAL_NAMESPACE,
        namespace={},
    )
    bar = Bar(1, 2.0)
    accessors, leaves, treespec = optree.tree_flatten_with_accessor(bar)
    assert optree.tree_unflatten(treespec, leaves) == bar
    assert accessors == [
        optree.PyTreeAccessor((optree.DataclassEntry('x', Bar, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.DataclassEntry('y', Bar, optree.PyTreeKind.CUSTOM),)),
    ]
    assert [a(bar) for a in accessors] == [1, 2.0]
    assert leaves == [1, 2.0]
    assert treespec.namespace == ''
    assert treespec.kind == optree.PyTreeKind.CUSTOM
    assert treespec.type is Bar
