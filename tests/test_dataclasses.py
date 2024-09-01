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
import sys
from collections import OrderedDict

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
    assert len(field_parameters) == len(field_original_parameters) + 1
    assert next(reversed(field_parameters)) == 'pytree_node'
    assert field_parameters['pytree_node'].kind == inspect.Parameter.KEYWORD_ONLY
    assert field_parameters['pytree_node'].default is True
    field_parameters.pop('pytree_node')
    assert OrderedDict(
        (name, (param.name, param.kind, param.default)) for name, param in field_parameters.items()
    ) == OrderedDict(
        (name, (param.name, param.kind, param.default))
        for name, param in field_original_parameters.items()
    )

    dataclass_parameters = inspect.signature(optree.dataclasses.dataclass).parameters.copy()
    dataclass_original_parameters = inspect.signature(dataclasses.dataclass).parameters.copy()
    assert len(dataclass_parameters) == len(dataclass_original_parameters) + 1
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
            (name, (param.name, param.kind, param.default))
            for name, param in dataclass_original_parameters.items()
        )

    make_dataclass_parameters = inspect.signature(
        optree.dataclasses.make_dataclass,
    ).parameters.copy()
    make_dataclass_original_parameters = inspect.signature(
        dataclasses.make_dataclass,
    ).parameters.copy()
    assert len(make_dataclass_parameters) == len(make_dataclass_original_parameters) + 1
    assert next(reversed(make_dataclass_parameters)) == 'namespace'
    assert make_dataclass_parameters['namespace'].kind == inspect.Parameter.KEYWORD_ONLY
    assert make_dataclass_parameters['namespace'].default is inspect.Parameter.empty
    make_dataclass_parameters.pop('namespace')
    assert 'ns' in make_dataclass_parameters
    assert 'ns' not in make_dataclass_original_parameters
    ns_index = list(make_dataclass_parameters).index('ns')
    make_dataclass_parameters_items = list(make_dataclass_parameters.items())
    make_dataclass_parameters_items[ns_index] = (
        'namespace',
        make_dataclass_parameters['ns'].replace(name='namespace'),
    )
    make_dataclass_parameters = dict(make_dataclass_parameters_items)
    if sys.version_info >= (3, 8):
        assert dict(
            [
                (name, (param.name, param.kind, param.default))
                for name, param in make_dataclass_parameters.items()
            ][: len(make_dataclass_original_parameters)],
        ) == {
            name: (
                param.name,
                (
                    param.kind
                    if param.kind != inspect.Parameter.POSITIONAL_ONLY
                    else inspect.Parameter.POSITIONAL_OR_KEYWORD
                ),
                param.default,
            )
            for name, param in make_dataclass_original_parameters.items()
        }
    else:
        assert dict(
            [
                (name, (param.name, param.kind, param.default))
                for name, param in make_dataclass_parameters.items()
            ][: len(make_dataclass_original_parameters)],
        ) == {
            name: (param.name, param.kind, param.default)
            for name, param in make_dataclass_original_parameters.items()
        }
