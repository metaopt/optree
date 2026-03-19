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
"""Integration with :mod:`attrs`.

This module implements PyTree integration with :mod:`attrs` by providing :func:`field`,
:func:`define`, :func:`frozen`, and :func:`register` functions. The :func:`field` and
:func:`define` functions wrap the corresponding :mod:`attrs` functions with an additional
``pytree_node`` parameter for controlling which fields are tree children versus metadata.
The :func:`register` function allows registering existing :mod:`attrs` classes as pytree nodes.

The PyTree integration allows attrs classes to be flattened and unflattened recursively. The fields
are stored in a special attribute named ``__optree_attrs_fields__`` in the attrs class.

>>> import optree
... from optree.integrations import attrs
...
>>> @attrs.define(namespace='my_module')
... class Point:
...     x: float
...     y: float
...     z: float = 0.0
...
>>> point = Point(2.0, 6.0, 3.0)
>>> point
Point(x=2.0, y=6.0, z=3.0)
>>> # Flatten without specifying the namespace
>>> optree.tree_flatten(point)  # `Point`s are leaf nodes
([Point(x=2.0, y=6.0, z=3.0)], PyTreeSpec(*))
>>> # Flatten with the namespace
>>> optree.tree_flatten(point, namespace='my_module')
([2.0, 6.0, 3.0], PyTreeSpec(CustomTreeNode(Point[()], [*, *, *]), namespace='my_module'))
>>> treespec = optree.tree_structure(point, namespace='my_module')
>>> point == optree.tree_unflatten(treespec, [2.0, 6.0, 3.0])
True
"""

# pragma: attrs cover file
# pylint: disable=import-error

from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

import attrs
from attrs import (
    NOTHING,
    Attribute,
    Factory,
    asdict,
    astuple,
    cmp_using,
    converters,
    evolve,
    exceptions,
    fields,
    fields_dict,
    filters,
    has,
    resolve_types,
    setters,
    validate,
    validators,
)
from attrs import make_class as _attrs_make_class

from optree.accessors import GetAttrEntry


if TYPE_CHECKING:
    from typing import ClassVar


__all__ = [
    'AttrsEntry',
    # Redefine `field`, `define`, `frozen`, `mutable`, and `make_class`.
    'field',
    'define',
    'frozen',
    'mutable',
    'make_class',
    'register',
    # Re-export commonly used APIs from the original package.
    'NOTHING',
    'Attribute',
    'Factory',
    'asdict',
    'astuple',
    'cmp_using',
    'converters',
    'evolve',
    'exceptions',
    'fields',
    'fields_dict',
    'filters',
    'has',
    'resolve_types',
    'setters',
    'validate',
    'validators',
]


_FIELDS = '__optree_attrs_fields__'
_PYTREE_NODE_DEFAULT: bool = True


_T = TypeVar('_T')
_U = TypeVar('_U')
_TypeT = TypeVar('_TypeT', bound=type)


class AttrsEntry(GetAttrEntry):
    """A path entry class for attrs classes."""

    __slots__: ClassVar[tuple[()]] = ()

    entry: str | int  # type: ignore[assignment]

    @property
    def fields(self, /) -> tuple[str, ...]:
        """Get all field names."""
        return tuple(a.name for a in self.type.__attrs_attrs__)  # type: ignore[attr-defined]

    @property
    def init_fields(self, /) -> tuple[str, ...]:
        """Get the init field names."""
        return tuple(a.name for a in self.type.__attrs_attrs__ if a.init)  # type: ignore[attr-defined]

    @property
    def field(self, /) -> str:
        """Get the field name."""
        if isinstance(self.entry, int):
            return self.init_fields[self.entry]
        return self.entry

    @property
    def name(self, /) -> str:
        """Get the attribute name."""
        return self.field

    def __repr__(self, /) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(field={self.field!r}, type={self.type!r})'


def field(**kwargs: Any) -> Any:
    """Field factory for :func:`define`.

    This factory function is used to define the fields in an attrs class. It is similar to
    :func:`attrs.field`, but with an additional ``pytree_node`` parameter. If ``pytree_node`` is
    :data:`True` (default), the field will be considered a child node in the PyTree structure which
    can be recursively flattened and unflattened. Otherwise, the field will be considered as PyTree
    metadata.

    Setting ``pytree_node`` in the field factory is equivalent to setting a key ``'pytree_node'`` in
    ``metadata``. The ``pytree_node`` value can be accessed using ``field.metadata['pytree_node']``.
    If ``pytree_node`` is :data:`None`, the value ``metadata.get('pytree_node', True)`` will be
    used.

    .. note::
        If a field is considered a child node, it must be included in the argument list of the
        :meth:`__init__` method, i.e., passes ``init=True`` in the field factory.

    Args:
        pytree_node (bool or None, optional): Whether the field is a PyTree node.
        **kwargs (optional): Optional keyword arguments passed to :func:`attrs.field`.

    Returns:
        The field defined using the provided arguments with ``metadata['pytree_node']`` set.
    """
    pytree_node = kwargs.pop('pytree_node', None)
    metadata = dict(kwargs.pop('metadata', None) or {})
    if pytree_node is None:
        pytree_node = metadata.get('pytree_node', _PYTREE_NODE_DEFAULT)
    metadata['pytree_node'] = pytree_node

    init = kwargs.get('init', True)
    if not init and pytree_node:
        raise TypeError(
            '`pytree_node=True` is not allowed for non-init fields. '
            f'Please explicitly set `{__name__}.field(init=False, pytree_node=False)`.',
        )

    return attrs.field(metadata=metadata, **kwargs)


@overload
def define(
    *,
    namespace: str,
    **kwargs: Any,
) -> Callable[[_TypeT], _TypeT]: ...


@overload
def define(
    cls: _TypeT,
    /,
    *,
    namespace: str,
    **kwargs: Any,
) -> _TypeT: ...


def define(  # pylint: disable=function-redefined
    cls: _TypeT | None = None,
    /,
    *,
    namespace: str,
    **kwargs: Any,
) -> _TypeT | Callable[[_TypeT], _TypeT]:
    """Attrs class decorator with PyTree integration.

    This is a wrapper around :func:`attrs.define` that also registers the class as a pytree node.

    Args:
        cls (type or None, optional): The class to decorate. If :data:`None`, return a decorator.
        namespace (str): The registry namespace used for the PyTree registration.
        **kwargs (optional): Optional keyword arguments passed to :func:`attrs.define`.

    Returns:
        type or callable: The decorated class with PyTree integration or decorator function.
    """
    if cls is None:

        def decorator(cls: _TypeT) -> _TypeT:
            return define(cls, namespace=namespace, **kwargs)

        return decorator

    if not inspect.isclass(cls):
        raise TypeError(f'@{__name__}.define() can only be used with classes, not {cls!r}.')

    cls = attrs.define(cls, **kwargs)
    return register(cls, namespace=namespace)


@overload
def frozen(
    *,
    namespace: str,
    **kwargs: Any,
) -> Callable[[_TypeT], _TypeT]: ...


@overload
def frozen(
    cls: _TypeT,
    /,
    *,
    namespace: str,
    **kwargs: Any,
) -> _TypeT: ...


def frozen(  # pylint: disable=function-redefined
    cls: _TypeT | None = None,
    /,
    *,
    namespace: str,
    **kwargs: Any,
) -> _TypeT | Callable[[_TypeT], _TypeT]:
    """Frozen attrs class decorator with PyTree integration.

    This is a convenience wrapper around :func:`define` with ``frozen=True``.

    Args:
        cls (type or None, optional): The class to decorate. If :data:`None`, return a decorator.
        namespace (str): The registry namespace used for the PyTree registration.
        **kwargs (optional): Optional keyword arguments passed to :func:`attrs.define`.

    Returns:
        type or callable: The decorated class with PyTree integration or decorator function.
    """
    kwargs.setdefault('frozen', True)
    kwargs.setdefault('on_setattr', None)
    return define(cls, namespace=namespace, **kwargs)  # type: ignore[type-var,return-value]


mutable = define
"""Alias for :func:`define`."""


def make_class(  # pylint: disable=redefined-outer-name
    name: str,
    attrs: Any,
    /,
    *,
    namespace: str,
    **kwargs: Any,
) -> type:
    """Create a new attrs class and register it as a pytree node.

    This is a wrapper around :func:`attrs.make_class` that also registers the class as a pytree
    node.

    Args:
        name (str): The name for the new class.
        attrs: A list of names or a dictionary of mappings of names to :func:`attrs.field` calls.
        namespace (str): The registry namespace used for the PyTree registration.
        **kwargs (optional): Optional keyword arguments passed to :func:`attrs.make_class`.

    Returns:
        type: A new attrs class registered as a pytree node.
    """
    cls = _attrs_make_class(name, attrs, **kwargs)
    return register(cls, namespace=namespace)


@overload
def register(
    cls: str | None = None,
    /,
    *,
    namespace: str | None = None,
) -> Callable[[_TypeT], _TypeT]: ...


@overload
def register(
    cls: _TypeT,
    /,
    *,
    namespace: str,
) -> _TypeT: ...


def register(  # noqa: C901 # pylint: disable=function-redefined,too-many-branches
    cls: _TypeT | str | None = None,
    /,
    *,
    namespace: str | None = None,
) -> _TypeT | Callable[[_TypeT], _TypeT]:
    """Register an existing attrs class as a pytree node.

    This function takes an existing :func:`attrs.define`-decorated class and registers it as a
    pytree node. It can be used as a direct function call or as a decorator.

    Fields with ``metadata['pytree_node']`` set to :data:`True` (or not set, defaulting to
    :data:`True`) are treated as children, while init fields with ``metadata['pytree_node']`` set
    to :data:`False` are treated as metadata.

    Usage::

        # Direct function call
        register(Point, namespace='my-namespace')

        # As a decorator
        @register(namespace='my-namespace')
        @attrs.define
        class Point:
            x: float
            y: float

    Args:
        cls (type, optional): An existing attrs-decorated class. If :data:`None`, return a
            decorator.
        namespace (str): The registry namespace used for the PyTree registration.

    Returns:
        type or callable: The same class, now registered as a pytree node, or a decorator function.
    """
    # pylint: disable-next=import-outside-toplevel
    from optree.registry import __GLOBAL_NAMESPACE as GLOBAL_NAMESPACE

    if cls is GLOBAL_NAMESPACE or isinstance(cls, str):
        if namespace is not None:
            raise ValueError('Cannot specify `namespace` when the first argument is a string.')
        if cls == '':
            raise ValueError('The namespace cannot be an empty string.')
        cls, namespace = None, cls

    if namespace is None:
        raise ValueError('Must specify `namespace` when the first argument is a class.')

    if cls is None:

        def decorator(cls: _TypeT, /) -> _TypeT:
            return register(cls, namespace=namespace)  # type: ignore[arg-type]

        return decorator

    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls!r}.')
    if not attrs.has(cls):
        raise TypeError(f'{cls!r} is not an attrs-decorated class.')
    if _FIELDS in cls.__dict__:
        raise TypeError(
            f'Cannot register {cls.__name__} as a pytree node more than once.',
        )
    if namespace is not GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}.')
    if namespace == '':
        namespace = GLOBAL_NAMESPACE

    children_fields = {}
    metadata_fields = {}
    for a in attrs.fields(cls):
        if a.metadata.get('pytree_node', _PYTREE_NODE_DEFAULT):
            if not a.init:
                raise TypeError(
                    f'PyTree node field {a.name!r} must be included in `__init__()`. '
                    f'Or you can explicitly set `{__name__}.field(init=False, pytree_node=False)`.',
                )
            children_fields[a.name] = a
        elif a.init:
            metadata_fields[a.name] = a

    children_field_names = tuple(children_fields)
    children_aliases = tuple(a.alias for a in children_fields.values())
    children_fields_proxy = MappingProxyType(children_fields)
    metadata_fields_proxy = MappingProxyType(metadata_fields)
    setattr(cls, _FIELDS, (children_fields_proxy, metadata_fields_proxy))

    def flatten_func(
        obj: _T,
        /,
    ) -> tuple[
        tuple[_U, ...],
        tuple[tuple[str, Any], ...],
        tuple[str, ...],
    ]:
        children = tuple(getattr(obj, name) for name in children_field_names)
        metadata = tuple((a.alias, getattr(obj, a.name)) for a in metadata_fields.values())
        return children, metadata, children_field_names

    # pylint: disable-next=line-too-long
    def unflatten_func(metadata: tuple[tuple[str, Any], ...], children: tuple[_U, ...], /) -> _T:  # type: ignore[type-var]
        kwargs = dict(zip(children_aliases, children))
        kwargs.update(metadata)
        return cls(**kwargs)

    from optree.registry import register_pytree_node  # pylint: disable=import-outside-toplevel

    register_pytree_node(
        cls,
        flatten_func,
        unflatten_func,  # type: ignore[arg-type]
        path_entry_type=AttrsEntry,
        namespace=namespace,
    )
    return cls
