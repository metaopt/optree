# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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
"""Typing utilities for OpTree."""

# mypy: no-warn-unused-ignores

from __future__ import annotations

import types
from typing import (
    Any,
    Callable,
    DefaultDict,
    Deque,
    Dict,
    ForwardRef,
    Generic,
    Hashable,
    Iterable,
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import OrderedDict  # Generic OrderedDict: Python 3.7.2+
from typing_extensions import TypeAlias  # Python 3.10+
from typing_extensions import Protocol, runtime_checkable  # Python 3.8+

from optree import _C


try:
    # pylint: disable-next=ungrouped-imports
    from typing_extensions import NamedTuple  # Generic NamedTuple: Python 3.11+
except ImportError:
    from typing import NamedTuple  # type: ignore[assignment]


__all__ = [
    'PyTreeSpec',
    'PyTreeDef',
    'PyTree',
    'PyTreeTypeVar',
    'CustomTreeNode',
    'Children',
    'MetaData',
    'FlattenFunc',
    'UnflattenFunc',
    'is_namedtuple',
    'is_namedtuple_class',
    'namedtuple_fields',
    'is_structseq',
    'is_structseq_class',
    'structseq_fields',
    'T',
    'S',
    'U',
    'KT',
    'VT',
    'Iterable',
    'Sequence',
    'List',
    'Tuple',
    'NamedTuple',
    'Dict',
    'OrderedDict',
    'DefaultDict',
    'Deque',
]


PyTreeSpec = _C.PyTreeSpec
PyTreeDef = PyTreeSpec  # alias

T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
KT = TypeVar('KT')
VT = TypeVar('VT')


Children = Sequence[T]
_MetaData = TypeVar('_MetaData', bound=Hashable)
MetaData = Optional[_MetaData]


@runtime_checkable
class CustomTreeNode(Protocol[T]):
    """The abstract base class for custom pytree nodes."""

    def tree_flatten(
        self,
    ) -> (
        # Use `range(num_children)` as path entries
        tuple[Children[T], MetaData]
        |
        # With optionally implemented path entries
        tuple[Children[T], MetaData, Iterable[Any] | None]
    ):
        """Flatten the custom pytree node into children and auxiliary data."""

    @classmethod
    def tree_unflatten(cls, metadata: MetaData, children: Children[T]) -> CustomTreeNode[T]:
        """Unflatten the children and auxiliary data into the custom pytree node."""


_GenericAlias = type(Union[int, str])


def _tp_cache(func: Callable) -> Callable:
    import functools  # pylint: disable=import-outside-toplevel

    cached = functools.lru_cache()(func)

    @functools.wraps(func)
    def inner(*args: Any, **kwds: Any) -> Any:
        try:
            return cached(*args, **kwds)
        except TypeError:
            # All real errors (not unhashable args) are raised below.
            return func(*args, **kwds)

    return inner


class PyTree(Generic[T]):  # pylint: disable=too-few-public-methods
    """Generic PyTree type.

    >>> import torch
    >>> from optree.typing import PyTree
    >>> TensorTree = PyTree[torch.Tensor]
    >>> TensorTree  # doctest: +NORMALIZE_WHITESPACE
    typing.Union[torch.Tensor,
                 typing.Tuple[ForwardRef('PyTree[torch.Tensor]'), ...],
                 typing.List[ForwardRef('PyTree[torch.Tensor]')],
                 typing.Dict[typing.Any, ForwardRef('PyTree[torch.Tensor]')],
                 typing.Deque[ForwardRef('PyTree[torch.Tensor]')],
                 optree.typing.CustomTreeNode[ForwardRef('PyTree[torch.Tensor]')]]
    """

    @_tp_cache
    def __class_getitem__(
        cls,
        item: T | tuple[T] | tuple[T, str | None],
    ) -> TypeAlias:
        """Instantiate a PyTree type with the given type."""
        if not isinstance(item, tuple):
            item = (item, None)
        if len(item) != 2:
            raise TypeError(
                f'{cls.__name__}[...] only supports a tuple of 2 items, '
                f'a parameter and a string of type name. Got {item!r}.',
            )
        param, name = item
        if name is not None and not isinstance(name, str):
            raise TypeError(
                f'{cls.__name__}[...] only supports a tuple of 2 items, '
                f'a parameter and a string of type name. Got {item!r}.',
            )

        if (
            isinstance(param, _GenericAlias)
            and param.__origin__ is Union  # type: ignore[attr-defined]
            and hasattr(param, '__pytree_args__')
        ):
            return param  # PyTree[PyTree[T]] -> PyTree[T]

        if name is not None:
            recurse_ref = ForwardRef(name)
        elif isinstance(param, TypeVar):
            recurse_ref = ForwardRef(f'{cls.__name__}[{param.__name__}]')
        elif isinstance(param, type):
            if param.__module__ == 'builtins':
                typename = param.__qualname__
            else:
                try:
                    typename = f'{param.__module__}.{param.__qualname__}'
                except AttributeError:
                    typename = f'{param.__module__}.{param.__name__}'
            recurse_ref = ForwardRef(f'{cls.__name__}[{typename}]')
        else:
            recurse_ref = ForwardRef(f'{cls.__name__}[{param!r}]')

        pytree_alias = Union[
            param,  # type: ignore[valid-type]
            Tuple[recurse_ref, ...],  # type: ignore[valid-type] # Tuple, NamedTuple, PyStructSequence
            List[recurse_ref],  # type: ignore[valid-type]
            Dict[Any, recurse_ref],  # type: ignore[valid-type] # Dict, OrderedDict, DefaultDict
            Deque[recurse_ref],  # type: ignore[valid-type]
            CustomTreeNode[recurse_ref],  # type: ignore[valid-type]
        ]
        pytree_alias.__pytree_args__ = item  # type: ignore[attr-defined]
        return pytree_alias

    def __new__(cls) -> NoReturn:  # pylint: disable=arguments-differ
        """Prohibit instantiation."""
        raise TypeError('Cannot instantiate special typing classes.')

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError('Cannot subclass special typing classes.')

    def __copy__(self) -> PyTree:
        """Immutable copy."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> PyTree:
        """Immutable copy."""
        return self


class PyTreeTypeVar:
    """Type variable for PyTree.

    >>> import torch
    >>> from optree.typing import PyTreeTypeVar
    >>> TensorTree = PyTreeTypeVar('TensorTree', torch.Tensor)
    >>> TensorTree  # doctest: +NORMALIZE_WHITESPACE
    typing.Union[torch.Tensor,
                 typing.Tuple[ForwardRef('TensorTree'), ...],
                 typing.List[ForwardRef('TensorTree')],
                 typing.Dict[typing.Any, ForwardRef('TensorTree')],
                 typing.Deque[ForwardRef('TensorTree')],
                 optree.typing.CustomTreeNode[ForwardRef('TensorTree')]]
    """

    @_tp_cache
    def __new__(cls, name: str, param: type) -> TypeAlias:
        """Instantiate a PyTree type variable with the given name and parameter."""
        if not isinstance(name, str):
            raise TypeError(f'{cls.__name__} only supports a string of type name. Got {name!r}.')
        return PyTree[param, name]  # type: ignore[misc,valid-type]

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError('Cannot subclass special typing classes.')

    def __copy__(self) -> TypeAlias:
        """Immutable copy."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> TypeAlias:
        """Immutable copy."""
        return self


FlattenFunc = Callable[[CustomTreeNode[T]], Tuple[Children[T], MetaData]]
UnflattenFunc = Callable[[MetaData, Children[T]], CustomTreeNode[T]]


def is_namedtuple(obj: object | type) -> bool:
    """Return whether the object is an instance of namedtuple or a subclass of namedtuple."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_namedtuple_class(cls)


def is_namedtuple_class(cls: type) -> bool:
    """Return whether the class is a subclass of namedtuple."""
    return (
        isinstance(cls, type)
        and issubclass(cls, tuple)
        and isinstance(getattr(cls, '_fields', None), tuple)
        and all(isinstance(field, str) for field in cls._fields)  # type: ignore[attr-defined]
    )


def namedtuple_fields(obj: tuple | type[tuple]) -> tuple[str, ...]:
    """Return the field names of a namedtuple."""
    if isinstance(obj, type):
        cls = obj
        if not is_namedtuple_class(cls):
            raise TypeError(f'Expected a collections.namedtuple type, got {cls!r}.')
    else:
        cls = type(obj)
        if not is_namedtuple_class(cls):
            raise TypeError(f'Expected an instance of collections.namedtuple type, got {obj!r}.')
    return cls._fields  # type: ignore[attr-defined]


def is_structseq(obj: object | type) -> bool:
    """Return whether the object is an instance of PyStructSequence or a class of PyStructSequence."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_structseq_class(cls)


# Set if the type allows subclassing (see CPython's Include/object.h)
Py_TPFLAGS_BASETYPE = _C.Py_TPFLAGS_BASETYPE  # (1UL << 10)


def is_structseq_class(cls: type) -> bool:
    """Return whether the class is a class of PyStructSequence."""
    return (
        isinstance(cls, type)
        # Check direct inheritance from `tuple` rather than `issubclass(cls, tuple)`
        and cls.__base__ is tuple
        # Check PyStructSequence members
        and isinstance(getattr(cls, 'n_sequence_fields', None), int)
        and isinstance(getattr(cls, 'n_fields', None), int)
        and isinstance(getattr(cls, 'n_unnamed_fields', None), int)
        # Check the type does not allow subclassing
        and not (cls.__flags__ & Py_TPFLAGS_BASETYPE)
    )


def structseq_fields(obj: tuple | type[tuple]) -> tuple[str, ...]:
    """Return the field names of a PyStructSequence."""
    if isinstance(obj, type):
        cls = obj
        if not is_structseq_class(cls):
            raise TypeError(f'Expected a PyStructSequence type, got {cls!r}.')
    else:
        cls = type(obj)
        if not is_structseq_class(cls):
            raise TypeError(f'Expected an instance of PyStructSequence type, got {obj!r}.')

    n_sequence_fields: int = cls.n_sequence_fields  # type: ignore[attr-defined]
    fields: list[str] = []
    for name, member in vars(cls).items():
        if len(fields) >= n_sequence_fields:
            break
        if isinstance(member, types.MemberDescriptorType):
            fields.append(name)
    return tuple(fields)


# Ensure that the behavior is consistent with C++ implementation
# pylint: disable-next=wrong-import-position,ungrouped-imports
from optree._C import (
    is_namedtuple,
    is_namedtuple_class,
    is_structseq,
    is_structseq_class,
    namedtuple_fields,
    structseq_fields,
)
