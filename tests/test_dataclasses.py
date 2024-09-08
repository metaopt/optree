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

# pylint: disable=missing-function-docstring,invalid-name,wrong-import-order

import dataclasses
import inspect
import re
import sys
from collections import OrderedDict

import pytest

import optree


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
    if sys.version_info >= (3, 8):
        assert OrderedDict(
            [
                (name, (param.name, param.kind, param.default))
                for name, param in field_parameters.items()
            ][: len(field_original_parameters)],
        ) == OrderedDict(
            (
                name,
                (
                    param.name,
                    (
                        param.kind
                        if param.kind != inspect.Parameter.POSITIONAL_ONLY
                        else inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                    param.default,
                ),
            )
            for name, param in field_original_parameters.items()
        )
    else:
        assert OrderedDict(
            [
                (name, (param.name, param.kind, param.default))
                for name, param in field_parameters.items()
            ][: len(field_original_parameters)],
        ) == OrderedDict(
            (name.lstrip('_'), (param.name.lstrip('_'), param.kind, param.default))
            for name, param in field_original_parameters.items()
        )

    dataclass_parameters = inspect.signature(optree.dataclasses.dataclass).parameters.copy()
    dataclass_original_parameters = inspect.signature(dataclasses.dataclass).parameters.copy()
    assert len(dataclass_parameters) >= len(dataclass_original_parameters) + 1
    assert next(reversed(dataclass_parameters)) == 'namespace'
    assert dataclass_parameters['namespace'].kind == inspect.Parameter.KEYWORD_ONLY
    assert dataclass_parameters['namespace'].default is inspect.Parameter.empty
    dataclass_parameters.pop('namespace')
    if sys.version_info >= (3, 8):
        assert OrderedDict(
            [
                (name, (param.name, param.kind, param.default))
                for name, param in dataclass_parameters.items()
            ][: len(dataclass_original_parameters)],
        ) == OrderedDict(
            (
                name,
                (
                    param.name,
                    (
                        param.kind
                        if param.kind != inspect.Parameter.POSITIONAL_ONLY
                        else inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                    param.default,
                ),
            )
            for name, param in dataclass_original_parameters.items()
        )
    else:
        assert OrderedDict(
            [
                (name, (param.name, param.kind, param.default))
                for name, param in dataclass_parameters.items()
            ][: len(dataclass_original_parameters)],
        ) == OrderedDict(
            (name.lstrip('_'), (param.name.lstrip('_'), param.kind, param.default))
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
    if sys.version_info >= (3, 8):
        assert OrderedDict(
            [
                (
                    {'ns': 'namespace'}.get(name, name),
                    ({'ns': 'namespace'}.get(param.name, param.name), param.kind, param.default),
                )
                for name, param in make_dataclass_parameters.items()
            ][: len(make_dataclass_original_parameters)],
        ) == OrderedDict(
            (
                name,
                (
                    param.name,
                    (
                        param.kind
                        if param.kind != inspect.Parameter.POSITIONAL_ONLY
                        else inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                    param.default,
                ),
            )
            for name, param in make_dataclass_original_parameters.items()
        )
    else:
        assert OrderedDict(
            [
                (
                    {'ns': 'namespace'}.get(name, name),
                    ({'ns': 'namespace'}.get(param.name, param.name), param.kind, param.default),
                )
                for name, param in make_dataclass_parameters.items()
            ][: len(make_dataclass_original_parameters)],
        ) == OrderedDict(
            (name.lstrip('_'), (param.name.lstrip('_'), param.kind, param.default))
            for name, param in make_dataclass_original_parameters.items()
        )


def test_invalid_parameters():
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

    with pytest.raises(
        TypeError,
        match=re.escape("dataclass() missing 1 required keyword-only argument: 'namespace'"),
    ):
        optree.dataclasses.dataclass()
    optree.dataclasses.dataclass(namespace='some-namespace')
    dataclasses.dataclass()
    if sys.version_info >= (3, 11):
        optree.dataclasses.dataclass(weakref_slot=True, namespace='some-namespace')
        dataclasses.dataclass(weakref_slot=True)
        optree.dataclasses.dataclass(weakref_slot=False, namespace='some-namespace')
        dataclasses.dataclass(weakref_slot=False)
    else:
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.dataclass(weakref_slot=True, namespace='some-namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(weakref_slot=True)
        optree.dataclasses.dataclass(weakref_slot=False, namespace='some-namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(weakref_slot=False)
    if sys.version_info >= (3, 10):
        optree.dataclasses.dataclass(match_args=True, namespace='some-namespace')
        dataclasses.dataclass(match_args=True)
        optree.dataclasses.dataclass(match_args=False, namespace='some-namespace')
        dataclasses.dataclass(match_args=False)
        optree.dataclasses.dataclass(kw_only=True, namespace='some-namespace')
        dataclasses.dataclass(kw_only=True)
        optree.dataclasses.dataclass(kw_only=False, namespace='some-namespace')
        dataclasses.dataclass(kw_only=False)
        optree.dataclasses.dataclass(slots=True, namespace='some-namespace')
        dataclasses.dataclass(slots=True)
        optree.dataclasses.dataclass(slots=False, namespace='some-namespace')
        dataclasses.dataclass(slots=False)
    else:
        optree.dataclasses.dataclass(match_args=True, namespace='some-namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(match_args=True)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.dataclass(match_args=False, namespace='some-namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(match_args=False)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.dataclass(kw_only=True, namespace='some-namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(kw_only=True)
        optree.dataclasses.dataclass(kw_only=False, namespace='some-namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(kw_only=False)
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            optree.dataclasses.dataclass(slots=True, namespace='some-namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(slots=True)
        optree.dataclasses.dataclass(slots=False, namespace='some-namespace')
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            dataclasses.dataclass(slots=False)


def test_init_args():
    @optree.dataclasses.dataclass(namespace='some-namespace')
    class Foo:
        a: int
        b: int = 2
        c: int = optree.dataclasses.field(init=False, pytree_node=False)
        d: float = dataclasses.field(init=True, default=42.0)
        e: int = dataclasses.field(init=False, metadata={'pytree_node': False})
        f: float = optree.dataclasses.field(init=True, default=6.0, metadata={'pytree_node': True})

        def __post_init__(self):
            self.c = self.a + self.b
            self.e = self.d * self.f

    foo = Foo(1, d=4.5, f=3.0)
    leaves, treespec = optree.tree_flatten(foo)
    assert leaves == [foo]
    assert treespec.is_leaf()
    leaves, treespec = optree.tree_flatten(foo, namespace='some-namespace')
    assert leaves == [1, 2, 4.5, 3.0]
    assert foo == optree.tree_unflatten(treespec, leaves)
