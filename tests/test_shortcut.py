# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
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

import optree


def test_pytree_reexports():
    tree_operations = [name[len('tree_') :] for name in optree.__all__ if name.startswith('tree_')]
    assert optree.pytree.__all__ == tree_operations

    for name in optree.pytree.__all__:
        assert getattr(optree.pytree, name) is getattr(optree, f'tree_{name}')


def test_treespec_reexports():
    treespec_operations = [
        name[len('treespec_') :] for name in optree.__all__ if name.startswith('treespec_')
    ]
    treespec_all_set = set(optree.treespec.__all__)
    assert treespec_all_set.issubset(treespec_operations)
    assert optree.treespec.__all__ == [
        name for name in treespec_operations if name in treespec_all_set
    ]

    for name in optree.treespec.__all__:
        assert getattr(optree.treespec, name) is getattr(optree, f'treespec_{name}')
