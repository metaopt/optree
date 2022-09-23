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
"""Typing utilities for OpTree."""

# mypy: no-warn-unused-ignores
# pylint: disable=missing-class-docstring,missing-function-docstring

from typing import (
    Any,
    Dict,
    ForwardRef,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from optree import _C


__all__ = [
    'PyTreeDef',
    'PyTree',
    'CustomTreeNode',
    'Children',
    'AuxData',
    'is_namedtuple',
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
]

# pylint: disable=unused-import
try:
    from typing_extensions import NamedTuple  # type: ignore[attr-defined]
except ImportError:
    from typing import NamedTuple  # type: ignore[assignment]

try:
    from typing import OrderedDict  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import OrderedDict  # type: ignore[assignment]

try:
    from typing import DefaultDict  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import DefaultDict  # type: ignore[assignment]

try:
    from typing import Protocol  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Protocol  # type: ignore[assignment]
# pylint: enable=unused-import

PyTreeDef = _C.PyTreeDef

T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
KT = TypeVar('KT')
VT = TypeVar('VT')


Children = Sequence[T]
_AuxData = TypeVar('_AuxData', bound=Hashable)
AuxData = Optional[_AuxData]


class CustomTreeNode(Protocol[T]):
    """The abstract base class for custom pytree nodes."""

    def tree_flatten(self) -> Tuple[Children[T], AuxData]:
        """Flattens the custom pytree node into children and auxiliary data."""

    @classmethod
    def tree_unflatten(cls, aux_data: AuxData, children: Children[T]) -> 'CustomTreeNode[T]':
        """Unflattens the children and auxiliary data into the custom pytree node."""


_GenericAlias = type(Union[int, str])


def _tp_cache(func):
    import functools  # pylint: disable=import-outside-toplevel

    cached = functools.lru_cache()(func)

    @functools.wraps(func)
    def inner(*args, **kwds):
        try:
            return cached(*args, **kwds)
        except TypeError:
            pass  # All real errors (not unhashable args) are raised below.
        return func(*args, **kwds)

    return inner


class PyTree(Generic[T]):  # pylint: disable=too-few-public-methods
    """Generic PyTree type.

    Examples:
        >>> import torch
        >>> from optree.typing import PyTree
        >>> TensorTree = PyTree[torch.Tensor]
        >>> TensorTree
        typing.Union[
            torch.Tensor,
            typing.Tuple[ForwardRef('PyTree[torch.Tensor]'), ...],
            typing.List[ForwardRef('PyTree[torch.Tensor]')],
            typing.Dict[typing.Any, ForwardRef('PyTree[torch.Tensor]')],
            optree.typing.CustomTreeNode[ForwardRef('PyTree[torch.Tensor]')]
        ]
    """

    @_tp_cache
    def __class_getitem__(cls, item: Union[T, Tuple[T]]):
        """Instantiate a PyTree type with the given item type."""
        if not isinstance(item, tuple):
            item = (item,)
        if len(item) != 1:
            raise TypeError(f'{cls.__name__}[...] only supports 1 parameter. Got {item!r}.')
        param = item[0]

        if (
            isinstance(param, _GenericAlias)
            and param.__origin__ is Union  # type: ignore[attr-defined]
            and hasattr(param, '__pytree_args__')
        ):
            return param  # PyTree[PyTree[T]] -> PyTree[T]

        if isinstance(param, TypeVar):
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
            Tuple[recurse_ref, ...],  # type: ignore[valid-type] # Tuple, NamedTuple
            List[recurse_ref],  # type: ignore[valid-type]
            Dict[Any, recurse_ref],  # type: ignore[valid-type] # Dict, OrderedDict, DefaultDict
            CustomTreeNode[recurse_ref],  # type: ignore[valid-type]
        ]
        pytree_alias.__pytree_args__ = item  # type: ignore[attr-defined]
        return pytree_alias


def is_namedtuple(obj: object) -> bool:
    """Return whether the object is a namedtuple."""
    return isinstance(obj, tuple) and hasattr(obj, '_fields')
