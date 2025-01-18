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

# pylint: disable=missing-function-docstring,wrong-import-position,wrong-import-order

import random
import warnings

import pytest


pytest.importorskip('torch')

import torch

import optree
from helpers import LEAVES, TREES, parametrize


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    torch.tensor(0.0)


@parametrize(tree=list(TREES + LEAVES))
def test_tree_ravel(tree):
    random.seed(0)

    def replace_leaf(_):
        candidates = [
            torch.tensor(random.randint(-100, 100)),
            torch.tensor(random.uniform(-100.0, 100.0)),
        ]

        shapes = [
            (),
            (random.randint(1, 10),),
            (random.randint(1, 10), random.randint(1, 10)),
            (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        ]
        dtypes = [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]
        for dtype in dtypes:
            candidates.extend(
                (5.0 * (2.0 * torch.randn(size=shape) - 1.0)).to(dtype) for shape in shapes
            )

        return random.choice(candidates)

    tree = optree.tree_map(replace_leaf, tree)
    flat, unravel_func = optree.integration.torch.tree_ravel(tree)

    leaves, treespec = optree.tree_flatten(tree)
    assert flat.numel() == sum(leaf.numel() for leaf in leaves)
    assert flat.shape == (flat.numel(),)

    reconstructed = unravel_func(flat)
    reconstructed_leaves, reconstructed_treespec = optree.tree_flatten(reconstructed)
    assert reconstructed_treespec == treespec
    assert len(leaves) == len(reconstructed_leaves)
    for leaf, reconstructed_leaf in zip(leaves, reconstructed_leaves):
        assert torch.is_tensor(leaf)
        assert torch.is_tensor(reconstructed_leaf)
        assert torch.allclose(leaf, reconstructed_leaf)
        assert leaf.dtype == reconstructed_leaf.dtype
        assert leaf.shape == reconstructed_leaf.shape

    with pytest.raises(ValueError, match=r'Expected a tensor to unravel, got .*\.'):
        unravel_func(1)

    with pytest.raises(
        ValueError,
        match=r'The unravel function expected a tensor of shape .*, got .*\.',
    ):
        unravel_func(flat.reshape((-1, 1)))
    with pytest.raises(
        ValueError,
        match=r'The unravel function expected a tensor of shape .*, got .*\.',
    ):
        unravel_func(torch.cat([flat, torch.zeros((1,))]))

    if all(leaf.dtype == flat.dtype for leaf in leaves):
        unravel_func(flat.to(torch.complex128))
    else:
        with pytest.raises(
            ValueError,
            match=r'The unravel function expected a tensor of dtype .*, got dtype .*\.',
        ):
            unravel_func(flat.to(torch.complex128))


@parametrize(tree=list(TREES + LEAVES))
def test_tree_ravel_single_dtype(tree):
    random.seed(0)
    dtype = torch.float16
    default_dtype = torch.tensor([]).dtype

    def replace_leaf(_):
        candidates = []
        shapes = [
            (),
            (random.randint(1, 10),),
            (random.randint(1, 10), random.randint(1, 10)),
            (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        ]
        candidates.extend(
            (5.0 * (2.0 * torch.randn(size=shape) - 1.0)).to(dtype) for shape in shapes
        )
        return random.choice(candidates)

    tree = optree.tree_map(replace_leaf, tree)
    flat, unravel_func = optree.integration.torch.tree_ravel(tree)

    leaves, treespec = optree.tree_flatten(tree)
    assert flat.dtype == dtype if leaves else default_dtype
    assert flat.numel() == sum(leaf.numel() for leaf in leaves)
    assert flat.shape == (flat.numel(),)

    reconstructed = unravel_func(flat)
    reconstructed_leaves, reconstructed_treespec = optree.tree_flatten(reconstructed)
    assert reconstructed_treespec == treespec
    assert len(leaves) == len(reconstructed_leaves)
    for leaf, reconstructed_leaf in zip(leaves, reconstructed_leaves):
        assert torch.is_tensor(leaf)
        assert torch.is_tensor(reconstructed_leaf)
        assert torch.allclose(leaf, reconstructed_leaf)
        assert leaf.dtype == reconstructed_leaf.dtype
        assert leaf.shape == reconstructed_leaf.shape

    with pytest.raises(ValueError, match=r'Expected a tensor to unravel, got .*\.'):
        unravel_func(1)

    with pytest.raises(
        ValueError,
        match=r'The unravel function expected a tensor of shape .*, got .*\.',
    ):
        unravel_func(flat.reshape((-1, 1)))
    with pytest.raises(
        ValueError,
        match=r'The unravel function expected a tensor of shape .*, got .*\.',
    ):
        unravel_func(torch.cat([flat, torch.zeros((1,))]))

    unravel_func(flat.to(torch.complex128))


def test_tree_ravel_non_tensor():
    with pytest.raises(ValueError, match=r'All leaves must be tensors\.'):
        optree.integration.torch.tree_ravel(1)

    with pytest.raises(ValueError, match=r'All leaves must be tensors\.'):
        optree.integration.torch.tree_ravel((1, 2))

    with pytest.raises(ValueError, match=r'All leaves must be tensors\.'):
        optree.integration.torch.tree_ravel((torch.tensor(1), 2))

    optree.integration.torch.tree_ravel((torch.tensor(1), torch.tensor(2)))
