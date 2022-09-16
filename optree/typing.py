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

from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

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


PyTree = Union[
    T,
    Tuple[T, ...],  # Tuple, NamedTuple
    List[T],
    Dict[Any, T],  # Dict, OrderedDict, DefaultDict
    CustomTreeNode[T],
]


def is_namedtuple(obj: object) -> bool:
    """Return whether the object is a namedtuple."""
    return isinstance(obj, tuple) and hasattr(obj, '_fields')
