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
"""OpTree: Optimized PyTree."""

# mypy: no-warn-unused-ignores
# pylint: disable=missing-class-docstring,missing-function-docstring

from typing import (
    Any,
    Dict,
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
    from typing_extensions import NamedTuple
except ImportError:
    from typing import NamedTuple  # type: ignore[assignment]

try:
    from typing import OrderedDict
except ImportError:
    from typing_extensions import OrderedDict  # type: ignore[assignment]

try:
    from typing import DefaultDict
except ImportError:
    from typing_extensions import DefaultDict  # type: ignore[assignment]

try:
    from typing import Protocol
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


class CustomTreeNode(Protocol, Generic[T]):
    def tree_flatten(self) -> Tuple[Children[T], AuxData]:
        ...

    @classmethod
    def tree_unflatten(cls, aux_data: AuxData, children: Children[T]) -> 'CustomTreeNode[T]':
        ...


PyTree = Union[
    T,
    Tuple[T, ...],  # Tuple, NamedTuple
    List[T],
    Dict[Any, T],  # Dict, OrderedDict, DefaultDict
    CustomTreeNode[T],
]


def is_namedtuple(obj: object) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, '_fields')
