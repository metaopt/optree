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
"""PyTree integration with :mod:`dataclasses`."""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import types
from dataclasses import *  # noqa: F401,F403,RUF100 # pylint: disable=wildcard-import,unused-wildcard-import
from typing import Any, Callable, TypeVar, overload
from typing_extensions import dataclass_transform  # Python 3.11+

from optree.accessor import DataclassEntry
from optree.registry import register_pytree_node


__all__ = [*dataclasses.__all__]


_FIELDS = '__optree_dataclass_fields__'
_PYTREE_NODE_DEFAULT: bool = True


def field(  # type: ignore[no-redef] # pylint: disable=function-redefined,too-many-arguments
    *,
    default: Any = dataclasses.MISSING,
    default_factory: Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    hash: bool | None = None,  # pylint: disable=redefined-builtin
    compare: bool = True,
    metadata: dict[str, Any] | None = None,
    kw_only: bool = dataclasses.MISSING,  # type: ignore[assignment] # Python 3.10+
    pytree_node: bool = _PYTREE_NODE_DEFAULT,
) -> dataclasses.Field:
    """Field factory for :func:`dataclass`."""
    metadata = metadata or {}
    metadata['pytree_node'] = pytree_node

    kwargs = {
        'default': default,
        'default_factory': default_factory,
        'init': init,
        'repr': repr,
        'hash': hash,
        'compare': compare,
        'metadata': metadata,
    }
    if sys.version_info >= (3, 10):
        kwargs['kw_only'] = kw_only

    return dataclasses.field(**kwargs)  # pylint: disable=invalid-field-call


_T = TypeVar('_T')
_U = TypeVar('_U')
_TypeT = TypeVar('_TypeT', bound=type)


@overload  # type: ignore[no-redef]
@dataclass_transform(field_specifiers=(field,))
def dataclass(  # pylint: disable=too-many-arguments
    cls: None,
    *,
    namespace: str,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,  # Python 3.10+
    kw_only: bool = False,  # Python 3.10+
    slots: bool = False,  # Python 3.10+
    weakref_slot: bool = False,  # Python 3.11+
) -> Callable[[_TypeT], _TypeT]:
    """Dataclass decorator with PyTree integration."""


@overload
@dataclass_transform(field_specifiers=(field,))
def dataclass(  # pylint: disable=too-many-arguments
    cls: _TypeT,
    *,
    namespace: str,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,  # Python 3.10+
    kw_only: bool = False,  # Python 3.10+
    slots: bool = False,  # Python 3.10+
    weakref_slot: bool = False,  # Python 3.11+
) -> _TypeT: ...


@dataclass_transform(field_specifiers=(field,))
def dataclass(  # noqa: C901 # pylint: disable=function-redefined,too-many-arguments,too-many-locals
    cls: _TypeT | None = None,
    *,
    namespace: str,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,  # Python 3.10+
    kw_only: bool = False,  # Python 3.10+
    slots: bool = False,  # Python 3.10+
    weakref_slot: bool = False,  # Python 3.11+
) -> _TypeT | Callable[[_TypeT], _TypeT]:
    """Dataclass decorator with PyTree integration."""
    kwargs = {
        'init': init,
        'repr': repr,
        'eq': eq,
        'order': order,
        'unsafe_hash': unsafe_hash,
        'frozen': frozen,
    }
    if sys.version_info >= (3, 10):
        kwargs['match_args'] = match_args
        kwargs['kw_only'] = kw_only
        kwargs['slots'] = slots
    if sys.version_info >= (3, 11):
        kwargs['weakref_slot'] = weakref_slot

    if cls is None:

        def decorator(cls: _TypeT) -> _TypeT:
            return dataclass(cls, namespace=namespace, **kwargs)  # type: ignore[call-overload]

        return decorator

    if not isinstance(cls, type):
        raise TypeError(f'@{__name__}.dataclass() can only be used with classes, not {cls!r}.')
    if _FIELDS in cls.__dict__:
        raise TypeError(
            f'@{__name__}.dataclass() cannot be applied to {cls.__name__} more than once.',
        )

    cls = dataclasses.dataclass(cls, **kwargs)  # type: ignore[assignment]

    children_fields = {}
    metadata_fields = {}
    for f in dataclasses.fields(cls):
        if f.metadata.get('pytree_node', _PYTREE_NODE_DEFAULT):
            if not f.init:
                raise TypeError(f'PyTree node field {f.name!r} must be included in __init__.')
            children_fields[f.name] = f
        elif f.init:
            metadata_fields[f.name] = f

    children_fields = types.MappingProxyType(children_fields)
    metadata_fields = types.MappingProxyType(metadata_fields)
    setattr(cls, _FIELDS, (children_fields, metadata_fields))

    def flatten_func(
        obj: _T,
    ) -> tuple[
        tuple[_U, ...],
        tuple[tuple[str, Any], ...],
        tuple[str, ...],
    ]:
        children = tuple(getattr(obj, name) for name in children_fields)
        metadata = tuple((name, getattr(obj, name)) for name in metadata_fields)
        return children, metadata, tuple(children_fields)

    def unflatten_func(metadata: tuple[tuple[str, Any], ...], children: tuple[_U, ...]) -> _T:  # type: ignore[type-var]
        return cls(*children, **dict(metadata))

    return register_pytree_node(  # type: ignore[return-value]
        cls,
        flatten_func,
        unflatten_func,  # type: ignore[arg-type]
        path_entry_type=DataclassEntry,
        namespace=namespace,
    )


def make_dataclass(  # type: ignore[no-redef] # pylint: disable=function-redefined,too-many-arguments,too-many-locals
    cls_name: str,
    fields: dict[str, Any],  # pylint: disable=redefined-outer-name
    *,
    namespace: str,
    bases: tuple[type, ...] = (),
    ns: dict[str, Any] | None = None,  # redirect to `namespace` to `dataclasses.make_dataclass()`
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,  # Python 3.10+
    kw_only: bool = False,  # Python 3.10+
    slots: bool = False,  # Python 3.10+
    weakref_slot: bool = False,  # Python 3.11+
    module: str | None = None,
) -> type:
    """Make a dataclass with PyTree integration."""
    # pylint: disable-next=import-outside-toplevel
    from optree.registry import __GLOBAL_NAMESPACE as GLOBAL_NAMESPACE

    if (isinstance(namespace, dict) or namespace is None) and (  # type: ignore[unreachable]
        isinstance(ns, str) or ns is GLOBAL_NAMESPACE  # type: ignore[unreachable]
    ):
        ns, namespace = namespace, ns  # type: ignore[unreachable]
    kwargs = {
        'bases': bases,
        'namespace': ns,
        'init': init,
        'repr': repr,
        'eq': eq,
        'order': order,
        'unsafe_hash': unsafe_hash,
        'frozen': frozen,
    }
    if sys.version_info >= (3, 10):
        kwargs['match_args'] = match_args
        kwargs['kw_only'] = kw_only
        kwargs['slots'] = slots
    if sys.version_info >= (3, 11):
        kwargs['weakref_slot'] = weakref_slot
    if sys.version_info >= (3, 12):
        if module is None:
            try:
                # pylint: disable-next=protected-access
                module = sys._getframemodulename(1) or '__main__'  # type: ignore[attr-defined]
            except AttributeError:
                with contextlib.suppress(AttributeError, ValueError):
                    # pylint: disable-next=protected-access
                    module = sys._getframe(1).f_globals.get('__name__', '__main__')
        kwargs['module'] = module

    cls = dataclasses.make_dataclass(cls_name, fields=fields, **kwargs)  # type: ignore[arg-type]
    return dataclass(cls, namespace=namespace)  # type: ignore[call-overload]
