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

from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Sequence, Tuple, Type

if TYPE_CHECKING:
    from optree.typing import MetaData, Children, CustomTreeNode, PyTree, T, U

version: int

def flatten(
    tree: PyTree[T],
    leaf_predicate: Optional[Callable[[T], bool]] = None,
    node_is_leaf: bool = False,
) -> Tuple[List[T], 'PyTreeSpec']: ...
def all_leaves(iterable: Iterable[T], node_is_leaf: bool = False) -> bool: ...
def leaf(node_is_leaf: bool = False) -> 'PyTreeSpec': ...
def none(node_is_leaf: bool = False) -> 'PyTreeSpec': ...
def tuple(treespecs: Sequence['PyTreeSpec'], node_is_leaf: bool = False) -> 'PyTreeSpec': ...

class PyTreeSpec:
    num_nodes: int
    num_leaves: int
    none_is_leaf: bool
    def unflatten(self, leaves: Iterable[T]) -> PyTree[T]: ...
    def flatten_up_to(self, full_tree: PyTree[T]) -> List[PyTree[T]]: ...
    def compose(self, inner_treespec: 'PyTreeSpec') -> 'PyTreeSpec': ...
    def walk(
        self,
        f_node: Callable[[Tuple[U, ...], MetaData], U],
        f_leaf: Optional[Callable[[T], U]],
        leaves: Iterable[T],
    ) -> U: ...
    def children(self) -> List['PyTreeSpec']: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def register_node(
    type: Type[CustomTreeNode[T]],
    to_iterable: Callable[[CustomTreeNode[T]], Tuple[Children[T], MetaData]],
    from_iterable: Callable[[MetaData, Children[T]], CustomTreeNode[T]],
) -> None: ...
