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

# pylint: disable=all
# isort: off

from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    List,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from optree.typing import T, Children, AuxData, PyTree, T, CustomTreeNode, U

version: int

def flatten(
    tree: PyTree[T],
    leaf_predicate: Optional[Callable[[T], bool]] = None,
) -> Tuple[List[T], 'PyTreeDef']: ...
def tuple(treedefs: Sequence['PyTreeDef']) -> 'PyTreeDef': ...
def all_leaves(iterable: Iterable[T]) -> bool: ...

class PyTreeDef:
    num_nodes: int
    num_leaves: int
    def unflatten(self, leaves: Iterable[T]) -> PyTree[T]: ...
    def flatten_up_to(self, full_tree: PyTree[T]) -> List[PyTree[T]]: ...
    def compose(self, inner_treedef: 'PyTreeDef') -> 'PyTreeDef': ...
    def walk(
        self,
        f_node: Callable[[Tuple[U, ...], AuxData], U],
        f_leaf: Optional[Callable[[T], U]],
        leaves: Iterable[T],
    ) -> U: ...
    def from_iterable_tree(self, subtrees: Iterable[PyTree[T]]): ...
    def children(self) -> List['PyTreeDef']: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def register_node(
    nodetype: Type[CustomTreeNode[T]],
    to_iterable: Callable[[CustomTreeNode[T]], Tuple[Children[T], AuxData]],
    from_iterable: Callable[[AuxData, Children[T]], CustomTreeNode[T]],
) -> None: ...
