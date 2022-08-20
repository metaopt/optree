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

# pylint: disable=missing-class-docstring,missing-function-docstring

from collections import OrderedDict, defaultdict
from typing import Any, Hashable, NamedTuple, Sequence, Tuple, TypeVar, Union

from optree import _optree as pytree


try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore[misc]


PyTreeDef = pytree.PyTreeDef

Children = TypeVar('Children', bound=Sequence[Any])
AuxData = TypeVar('AuxData', bound=Hashable)


class CustomTreeNode(Protocol):
    def tree_flatten(self) -> Tuple[Children, AuxData]:
        ...

    @classmethod
    def tree_unflatten(cls, aux_data: AuxData, children: Children) -> 'CustomTreeNode':
        ...


PyTree = TypeVar(
    'PyTree',
    bound=Union[
        tuple, NamedTuple, list, dict, Sequence, defaultdict, OrderedDict, CustomTreeNode, None, Any
    ],
)
