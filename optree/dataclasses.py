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
"""PyTree integration with :mod:`dataclasses`.

This module implements PyTree integration with :mod:`dataclasses` by redefining the :func:`field`,
:func:`dataclass`, and :func:`make_dataclass` functions. The :func:`register_node` function allows
registering existing :func:`dataclasses.dataclass`-decorated classes as pytree nodes. Other APIs
are re-exported from the original :mod:`dataclasses` module.

The PyTree integration allows dataclasses to be flattened and unflattened recursively. The fields
are stored in a special attribute named ``__optree_dataclass_fields__`` in the dataclass.

>>> import math
... import optree
...
>>> @optree.dataclasses.dataclass(namespace='my_module')
... class Point:
...     x: float
...     y: float
...     z: float = 0.0
...     norm: float = optree.dataclasses.field(init=False, pytree_node=False)
...
...     def __post_init__(self) -> None:
...         self.norm = math.hypot(self.x, self.y, self.z)
...
>>> point = Point(2.0, 6.0, 3.0)
>>> point
Point(x=2.0, y=6.0, z=3.0, norm=7.0)
>>> # Flatten without specifying the namespace
>>> optree.tree_flatten(point)  # `Point`s are leaf nodes
([Point(x=2.0, y=6.0, z=3.0, norm=7.0)], PyTreeSpec(*))
>>> # Flatten with the namespace
>>> accessors, leaves, treespec = optree.tree_flatten_with_accessor(point, namespace='my_module')
>>> accessors, leaves, treespec  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
(
    [
        PyTreeAccessor(*.x, (DataclassEntry(field='x', type=<class '...Point'>),)),
        PyTreeAccessor(*.y, (DataclassEntry(field='y', type=<class '...Point'>),)),
        PyTreeAccessor(*.z, (DataclassEntry(field='z', type=<class '...Point'>),))
    ],
    [2.0, 6.0, 3.0],
    PyTreeSpec(CustomTreeNode(Point[()], [*, *, *]), namespace='my_module')
)
>>> point == optree.tree_unflatten(treespec, leaves)
True
"""

# pylint: disable=too-many-arguments

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import sys
import warnings
from dataclasses import *  # noqa: F401,F403,RUF100 # pylint: disable=wildcard-import,unused-wildcard-import
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, overload
from typing_extensions import dataclass_transform  # Python 3.11+

from optree.accessors import DataclassEntry


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = [
    'DataclassEntry',
    'register_node',
    # Redefine `field`, `dataclass`, and `make_dataclass`.
    # The remaining APIs are re-exported from the original package.
    *dataclasses.__all__,
]


_FIELDS = '__optree_dataclass_fields__'
_PYTREE_NODE_DEFAULT: bool = True


_T = TypeVar('_T')
_U = TypeVar('_U')
_TypeT = TypeVar('_TypeT', bound=type)


@overload  # type: ignore[no-redef]
def field(
    *,
    default: _T,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    hash: bool | None = None,  # pylint: disable=redefined-builtin
    compare: bool = True,
    metadata: dict[Any, Any] | None = None,
    kw_only: bool | Literal[dataclasses.MISSING] = dataclasses.MISSING,  # type: ignore[valid-type]
    doc: str | None = None,  # Python 3.14+
    pytree_node: bool | None = None,
) -> _T: ...


@overload
def field(
    *,
    default_factory: Callable[[], _T],
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    hash: bool | None = None,  # pylint: disable=redefined-builtin
    compare: bool = True,
    metadata: dict[Any, Any] | None = None,
    kw_only: bool | Literal[dataclasses.MISSING] = dataclasses.MISSING,  # type: ignore[valid-type]
    doc: str | None = None,  # Python 3.14+
    pytree_node: bool | None = None,
) -> _T: ...


@overload
def field(
    *,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    hash: bool | None = None,  # pylint: disable=redefined-builtin
    compare: bool = True,
    metadata: dict[Any, Any] | None = None,
    kw_only: bool | Literal[dataclasses.MISSING] = dataclasses.MISSING,  # type: ignore[valid-type]
    doc: str | None = None,  # Python 3.14+
    pytree_node: bool | None = None,
) -> Any: ...


def field(  # noqa: D417 # pylint: disable=function-redefined
    *,
    default: Any = dataclasses.MISSING,
    default_factory: Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    hash: bool | None = None,  # pylint: disable=redefined-builtin
    compare: bool = True,
    metadata: dict[Any, Any] | None = None,
    kw_only: bool | Literal[dataclasses.MISSING] = dataclasses.MISSING,  # type: ignore[valid-type]
    doc: str | None = None,  # Python 3.14+
    pytree_node: bool | None = None,
) -> Any:
    """Field factory for :func:`dataclass`.

    This factory function is used to define the fields in a dataclass. It is similar to the field
    factory :func:`dataclasses.field`, but with an additional ``pytree_node`` parameter. If
    ``pytree_node`` is :data:`True` (default), the field will be considered a child node in the
    PyTree structure which can be recursively flattened and unflattened. Otherwise, the field will
    be considered as PyTree metadata.

    Setting ``pytree_node`` in the field factory is equivalent to setting a key ``'pytree_node'`` in
    ``metadata`` in the original field factory. The ``pytree_node`` value can be accessed using
    ``field.metadata['pytree_node']``. If ``pytree_node`` is :data:`None`, the value
    ``metadata.get('pytree_node', True)`` will be used.

    .. note::
        If a field is considered a child node, it must be included in the argument list of the
        :meth:`__init__` method, i.e., passes ``init=True`` in the field factory.

    Args:
        pytree_node (bool or None, optional): Whether the field is a PyTree node.
        **kwargs (optional): Optional keyword arguments passed to :func:`dataclasses.field`.

    Returns:
        dataclasses.Field: The field defined using the provided arguments with
        ``field.metadata['pytree_node']`` set.
    """
    metadata = (metadata or {}).copy()
    if pytree_node is None:
        pytree_node = metadata.get('pytree_node', _PYTREE_NODE_DEFAULT)
    metadata['pytree_node'] = pytree_node

    kwargs = {
        'default': default,
        'default_factory': default_factory,
        'init': init,
        'repr': repr,
        'hash': hash,
        'compare': compare,
        'metadata': metadata,
        'kw_only': kw_only,
    }

    if sys.version_info >= (3, 14):  # pragma: >=3.14 cover
        kwargs['doc'] = doc
    elif doc is not None:  # pragma: <3.14 cover
        raise TypeError("field() got an unexpected keyword argument 'doc'")

    if not init and pytree_node:
        raise TypeError(
            '`pytree_node=True` is not allowed for non-init fields. '
            f'Please explicitly set `{__name__}.field(init=False, pytree_node=False)`.',
        )

    return dataclasses.field(**kwargs)  # pylint: disable=invalid-field-call


@overload  # type: ignore[no-redef]
def dataclass(
    *,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,  # Python 3.11+
    namespace: str,
) -> Callable[[_TypeT], _TypeT]: ...


@overload
def dataclass(
    cls: _TypeT,
    /,
    *,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,  # Python 3.11+
    namespace: str,
) -> _TypeT: ...


@dataclass_transform(field_specifiers=(field,))
def dataclass(  # noqa: D417 # pylint: disable=function-redefined
    cls: _TypeT | None = None,
    /,
    *,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,  # Python 3.11+
    namespace: str,
) -> _TypeT | Callable[[_TypeT], _TypeT]:
    """Dataclass decorator with PyTree integration.

    Args:
        cls (type or None, optional): The class to decorate. If :data:`None`, return a decorator.
        namespace (str): The registry namespace used for the PyTree registration.
        **kwargs (optional): Optional keyword arguments passed to :func:`dataclasses.dataclass`.

    Returns:
        type or callable: The decorated class with PyTree integration or decorator function.
    """
    # pylint: disable-next=import-outside-toplevel
    from optree.registry import __GLOBAL_NAMESPACE as GLOBAL_NAMESPACE

    kwargs = {
        'init': init,
        'repr': repr,
        'eq': eq,
        'order': order,
        'unsafe_hash': unsafe_hash,
        'frozen': frozen,
        'match_args': match_args,
        'kw_only': kw_only,
        'slots': slots,
    }

    if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
        kwargs['weakref_slot'] = weakref_slot
    elif weakref_slot is not False:  # pragma: <3.11 cover
        raise TypeError("dataclass() got an unexpected keyword argument 'weakref_slot'")

    if cls is None:

        def decorator(cls: _TypeT) -> _TypeT:
            return dataclass(cls, namespace=namespace, **kwargs)  # type: ignore[call-overload]

        return decorator

    if not inspect.isclass(cls):
        raise TypeError(f'@{__name__}.dataclass() can only be used with classes, not {cls!r}.')
    if _FIELDS in cls.__dict__:
        raise TypeError(
            f'@{__name__}.dataclass() cannot be applied to {cls.__name__} more than once.',
        )
    if namespace is not GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}.')
    if namespace == '':
        namespace = GLOBAL_NAMESPACE

    cls = dataclasses.dataclass(cls, **kwargs)  # type: ignore[assignment]
    return register_node(cls, namespace=namespace)


class _DataclassDecorator(Protocol[_TypeT]):  # pylint: disable=too-few-public-methods
    def __call__(  # pylint: disable=arguments-differ
        self,
        cls: _TypeT,
        /,
        *,
        init: bool = True,
        repr: bool = True,  # pylint: disable=redefined-builtin
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        match_args: bool = True,
        kw_only: bool = False,
        slots: bool = False,
        weakref_slot: bool = False,
    ) -> _TypeT:
        raise NotImplementedError


# pylint: disable-next=function-redefined,too-many-locals,too-many-branches
def make_dataclass(  # type: ignore[no-redef] # noqa: C901,D417
    cls_name: str,
    # pylint: disable-next=redefined-outer-name
    fields: Iterable[str | tuple[str, Any] | tuple[str, Any, Any]],
    *,
    bases: tuple[type, ...] = (),
    ns: dict[str, Any] | None = None,  # redirect to `namespace` to `dataclasses.make_dataclass()`
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,  # Python 3.11+
    module: str | None = None,  # Python 3.12+
    decorator: _DataclassDecorator[_TypeT] = dataclasses.dataclass,  # type: ignore[assignment] # Python 3.14+
    namespace: str,  # the PyTree registration namespace
) -> _TypeT:
    """Make a new dynamically created dataclass with PyTree integration.

    The dataclass name will be ``cls_name``. ``fields`` is an iterable of either (name), (name, type),
    or (name, type, Field) objects. If type is omitted, use the string :data:`typing.Any`. Field
    objects are created by the equivalent of calling :func:`field` (name, type [, Field-info]).

    The ``namespace`` parameter is the PyTree registration namespace which should be a string. The
    ``namespace`` in the original :func:`dataclasses.make_dataclass` function is renamed to ``ns``
    to avoid conflicts.

    The remaining parameters are passed to :func:`dataclasses.make_dataclass`.
    See :func:`dataclasses.make_dataclass` for more information.

    Args:
        cls_name: The name of the dataclass.
        fields (Iterable[str | tuple[str, Any] | tuple[str, Any, Any]]): An iterable of either
            (name), (name, type), or (name, type, Field) objects.
        namespace (str): The registry namespace used for the PyTree registration.
        ns (dict or None, optional): The namespace used in dynamic type creation.
            See :func:`dataclasses.make_dataclass` and the builtin :func:`type` function for more
            information.
        **kwargs (optional): Optional keyword arguments passed to :func:`dataclasses.make_dataclass`.

    Returns:
        type: The dynamically created dataclass with PyTree integration.
    """
    # pylint: disable-next=import-outside-toplevel
    from optree.registry import __GLOBAL_NAMESPACE as GLOBAL_NAMESPACE

    if isinstance(namespace, dict) or namespace is None:  # type: ignore[unreachable]
        if ns is GLOBAL_NAMESPACE or isinstance(ns, str):  # type: ignore[unreachable]
            ns, namespace = namespace, ns
        elif ns is None:
            raise TypeError("make_dataclass() missing 1 required keyword-only argument: 'ns'")
    if namespace is not GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}.')
    if namespace == '':
        namespace = GLOBAL_NAMESPACE

    dataclass_kwargs = {
        'init': init,
        'repr': repr,
        'eq': eq,
        'order': order,
        'unsafe_hash': unsafe_hash,
        'frozen': frozen,
        'match_args': match_args,
        'kw_only': kw_only,
        'slots': slots,
    }
    make_dataclass_kwargs = {
        'bases': bases,
        'namespace': ns,
    }

    if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
        dataclass_kwargs['weakref_slot'] = weakref_slot
    elif weakref_slot is not False:  # pragma: <3.11 cover
        raise TypeError("make_dataclass() got an unexpected keyword argument 'weakref_slot'")

    if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
        if module is None:
            try:
                # pylint: disable-next=protected-access
                module = sys._getframemodulename(1) or '__main__'  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover
                with contextlib.suppress(AttributeError, ValueError):
                    # pylint: disable-next=protected-access
                    module = sys._getframe(1).f_globals.get('__name__', '__main__')
        make_dataclass_kwargs['module'] = module
    elif module is not None:  # pragma: <3.12 cover
        raise TypeError("make_dataclass() got an unexpected keyword argument 'module'")

    registered_by_decorator = False
    if sys.version_info >= (3, 14):  # pragma: >=3.14 cover
        if decorator in (dataclasses.dataclass, dataclass):
            decorator = functools.partial(dataclass, namespace=namespace)
            registered_by_decorator = True
        make_dataclass_kwargs['decorator'] = decorator
    elif decorator is not dataclasses.dataclass:  # pragma: <3.14 cover
        raise TypeError("make_dataclass() got an unexpected keyword argument 'decorator'")

    cls: _TypeT = dataclasses.make_dataclass(  # type: ignore[assignment]
        cls_name,
        fields=fields,
        **dataclass_kwargs,  # type: ignore[arg-type]
        **make_dataclass_kwargs,  # type: ignore[arg-type]
    )
    if not registered_by_decorator:  # pragma: <3.14 cover
        cls = register_node(cls, namespace=namespace)
    return cls


@overload
def register_node(
    cls: str | None = None,
    /,
    *,
    namespace: str | None = None,
) -> Callable[[_TypeT], _TypeT]: ...


@overload
def register_node(
    cls: _TypeT,
    /,
    *,
    namespace: str,
) -> _TypeT: ...


def register_node(  # noqa: C901 # pylint: disable=function-redefined,too-many-branches
    cls: _TypeT | str | None = None,
    /,
    *,
    namespace: str | None = None,
) -> _TypeT | Callable[[_TypeT], _TypeT]:
    """Register an existing dataclass as a pytree node.

    This function takes an existing :func:`dataclasses.dataclass`-decorated class and registers it
    as a pytree node. It can be used as a direct function call or as a decorator.

    Fields with ``metadata['pytree_node']`` set to :data:`True` (or not set, defaulting to
    :data:`True`) are treated as children, while init fields with ``metadata['pytree_node']`` set
    to :data:`False` are treated as metadata.

    Usage::

        # Direct function call
        register_node(Point, namespace='my-namespace')

        # As a decorator
        @register_node(namespace='my-namespace')
        @dataclasses.dataclass
        class Point:
            x: float
            y: float

    Args:
        cls (type, optional): An existing dataclass. If :data:`None`, return a decorator.
        namespace (str): The registry namespace used for the PyTree registration.

    Returns:
        type or callable: The same class, now registered as a pytree node, or a decorator function.

    .. versionadded:: 0.20.0
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
            return register_node(cls, namespace=namespace)  # type: ignore[arg-type]

        return decorator

    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls!r}.')
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f'{cls!r} is not a dataclass.')
    if _FIELDS in cls.__dict__:
        raise TypeError(
            f'Cannot register {cls.__name__} as a pytree node more than once.',
        )
    if namespace is not GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}.')
    if namespace == '':
        namespace = GLOBAL_NAMESPACE
    if not cls.__dataclass_params__.init:  # type: ignore[attr-defined]
        warnings.warn(
            f'Dataclass {cls.__name__!r} was defined with `init=False`. '
            '`tree_unflatten()` may fail because '
            f'`{__name__}.register_node()` reconstructs instances with `cls(**kwargs)`.',
            UserWarning,
            stacklevel=2,
        )

    children_fields = {}
    metadata_fields = {}
    for f in dataclasses.fields(cls):
        if f.metadata.get('pytree_node', _PYTREE_NODE_DEFAULT):
            if not f.init:
                raise TypeError(
                    f'PyTree node field {f.name!r} must be included in `__init__()`. '
                    f'Or you can explicitly set `{__name__}.field(init=False, pytree_node=False)`.',
                )
            children_fields[f.name] = f
        elif f.init:
            metadata_fields[f.name] = f

    children_field_names = tuple(children_fields)
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
        metadata = tuple((name, getattr(obj, name)) for name in metadata_fields)
        return children, metadata, children_field_names

    # pylint: disable-next=line-too-long
    def unflatten_func(metadata: tuple[tuple[str, Any], ...], children: tuple[_U, ...], /) -> _T:  # type: ignore[type-var]
        kwargs = dict(zip(children_field_names, children, strict=True))
        kwargs.update(metadata)
        return cls(**kwargs)  # type: ignore[return-value]

    from optree.registry import register_pytree_node  # pylint: disable=import-outside-toplevel

    register_pytree_node(
        cls,  # type: ignore[arg-type]
        flatten_func,
        unflatten_func,  # type: ignore[arg-type]
        path_entry_type=DataclassEntry,
        namespace=namespace,
    )
    return cls
