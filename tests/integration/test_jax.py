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


pytest.importorskip('jax')

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import dtypes

import optree
from helpers import LEAVES, TREES, parametrize


@parametrize(tree=list(TREES + LEAVES))
def test_tree_ravel(tree):
    jax.config.update('jax_enable_x64', True)
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
            jnp.float32,
            jnp.float64,
            jnp.int32,
            jnp.int64,
        ]
        for dtype in dtypes:
            candidates.extend(
                jnp.array(rng.uniform(low=-5.0, high=5.0, size=shape), dtype=dtype)
                for shape in shapes
            )

        return random.choice(candidates)

    tree = optree.tree_map(replace_leaf, tree)
    flat, unravel_func = optree.integration.jax.tree_ravel(tree)

    leaves, treespec = optree.tree_flatten(tree)
    assert jnp.size(flat) == sum(jnp.size(leaf) for leaf in leaves)
    assert jnp.shape(flat) == (jnp.size(flat),)

    reconstructed = unravel_func(flat)
    reconstructed_leaves, reconstructed_treespec = optree.tree_flatten(reconstructed)
    assert reconstructed_treespec == treespec
    assert len(leaves) == len(reconstructed_leaves)
    for leaf, reconstructed_leaf in zip(leaves, reconstructed_leaves):
        assert jnp.allclose(leaf, reconstructed_leaf)
        leaf = jnp.asarray(leaf)
        reconstructed_leaf = jnp.asarray(reconstructed_leaf)
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
        unravel_func(jnp.concatenate([flat, jnp.zeros((1,))]))

    if all(dtypes.dtype(leaf) == dtypes.dtype(flat) for leaf in leaves):
        unravel_func(lax.convert_element_type(flat, jnp.complex128))
    else:
        with pytest.raises(
            ValueError,
            match=r'The unravel function expected an array of dtype .*, got dtype .*\.',
        ):
            unravel_func(lax.convert_element_type(flat, jnp.complex128))


@parametrize(tree=list(TREES + LEAVES))
def test_tree_ravel_single_dtype(tree):
    jax.config.update('jax_enable_x64', True)
    random.seed(0)
    rng = np.random.default_rng(0)
    jax_dtype = jnp.float32

    def replace_leaf(_):
        candidates = []
        shapes = [
            (),
            (random.randint(1, 10),),
            (random.randint(1, 10), random.randint(1, 10)),
            (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        ]
        candidates.extend(
            jnp.array(rng.uniform(low=-5.0, high=5.0, size=shape), dtype=jax_dtype)
            for shape in shapes
        )
        return random.choice(candidates)

    tree = optree.tree_map(replace_leaf, tree)
    flat, unravel_func = optree.integration.jax.tree_ravel(tree)

    leaves, treespec = optree.tree_flatten(tree)
    assert flat.dtype == jax_dtype if leaves else jnp.float64
    assert jnp.size(flat) == sum(jnp.size(leaf) for leaf in leaves)
    assert jnp.shape(flat) == (jnp.size(flat),)

    reconstructed = unravel_func(flat)
    reconstructed_leaves, reconstructed_treespec = optree.tree_flatten(reconstructed)
    assert reconstructed_treespec == treespec
    assert len(leaves) == len(reconstructed_leaves)
    for leaf, reconstructed_leaf in zip(leaves, reconstructed_leaves):
        assert jnp.allclose(leaf, reconstructed_leaf)
        leaf = jnp.asarray(leaf)
        reconstructed_leaf = jnp.asarray(reconstructed_leaf)
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
        unravel_func(jnp.concatenate([flat, jnp.zeros((1,))]))

    unravel_func(flat.astype(jnp.complex128))
