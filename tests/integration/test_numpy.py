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

import pytest


pytest.importorskip('numpy')

import numpy as np

import optree
from helpers import LEAVES, TREES, parametrize


@parametrize(tree=list(TREES + LEAVES))
def test_tree_ravel(tree):
    random.seed(0)
    rng = np.random.default_rng(0)

    def replace_leaf(_):
        candidates = [random.randint(-100, 100), random.uniform(-100.0, 100.0)]

        shapes = [
            (),
            (random.randint(1, 10),),
            (random.randint(1, 10), random.randint(1, 10)),
            (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        ]
        dtypes = [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
        ]
        for dtype in dtypes:
            candidates.extend(
                rng.uniform(low=-5.0, high=5.0, size=shape).astype(dtype) for shape in shapes
            )

        return random.choice(candidates)

    tree = optree.tree_map(replace_leaf, tree)
    flat, unravel_func = optree.integration.numpy.tree_ravel(tree)

    leaves, treespec = optree.tree_flatten(tree)
    assert np.size(flat) == sum(np.size(leaf) for leaf in leaves)
    assert np.shape(flat) == (np.size(flat),)

    reconstructed = unravel_func(flat)
    reconstructed_leaves, reconstructed_treespec = optree.tree_flatten(reconstructed)
    assert reconstructed_treespec == treespec
    assert len(leaves) == len(reconstructed_leaves)
    for leaf, reconstructed_leaf in zip(leaves, reconstructed_leaves):
        assert np.allclose(leaf, reconstructed_leaf)
        leaf = np.asarray(leaf)
        reconstructed_leaf = np.asarray(reconstructed_leaf)
        assert leaf.dtype == reconstructed_leaf.dtype
        assert leaf.shape == reconstructed_leaf.shape

    with pytest.raises(
        ValueError,
        match=r'The unravel function expected an array of shape .*, got .*\.',
    ):
        unravel_func(flat.reshape((-1, 1)))
    with pytest.raises(
        ValueError,
        match=r'The unravel function expected an array of shape .*, got .*\.',
    ):
        unravel_func(np.concatenate([flat, np.zeros((1,))]))

    if all(np.result_type(leaf) == flat.dtype for leaf in leaves):
        unravel_func(flat.astype(np.complex128))
    else:
        with pytest.raises(
            ValueError,
            match=r'The unravel function expected an array of dtype .*, got dtype .*\.',
        ):
            unravel_func(flat.astype(np.complex128))


@parametrize(tree=list(TREES + LEAVES))
def test_tree_ravel_single_dtype(tree):
    random.seed(0)
    rng = np.random.default_rng(0)
    dtype = np.float16
    default_dtype = np.array([]).dtype

    def replace_leaf(_):
        candidates = []
        shapes = [
            (),
            (random.randint(1, 10),),
            (random.randint(1, 10), random.randint(1, 10)),
            (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        ]
        candidates.extend(
            rng.uniform(low=-5.0, high=5.0, size=shape).astype(dtype) for shape in shapes
        )
        return random.choice(candidates)

    tree = optree.tree_map(replace_leaf, tree)
    flat, unravel_func = optree.integration.numpy.tree_ravel(tree)

    leaves, treespec = optree.tree_flatten(tree)
    assert flat.dtype == dtype if leaves else default_dtype
    assert np.size(flat) == sum(np.size(leaf) for leaf in leaves)
    assert np.shape(flat) == (np.size(flat),)

    reconstructed = unravel_func(flat)
    reconstructed_leaves, reconstructed_treespec = optree.tree_flatten(reconstructed)
    assert reconstructed_treespec == treespec
    assert len(leaves) == len(reconstructed_leaves)
    for leaf, reconstructed_leaf in zip(leaves, reconstructed_leaves):
        assert np.allclose(leaf, reconstructed_leaf)
        leaf = np.asarray(leaf)
        reconstructed_leaf = np.asarray(reconstructed_leaf)
        assert leaf.dtype == reconstructed_leaf.dtype
        assert leaf.shape == reconstructed_leaf.shape

    with pytest.raises(
        ValueError,
        match=r'The unravel function expected an array of shape .*, got .*\.',
    ):
        unravel_func(flat.reshape((-1, 1)))
    with pytest.raises(
        ValueError,
        match=r'The unravel function expected an array of shape .*, got .*\.',
    ):
        unravel_func(np.concatenate([flat, np.zeros((1,))]))

    unravel_func(flat.astype(np.complex128))
