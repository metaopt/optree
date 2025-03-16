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

import types

import pytest

import optree


def test_imports():
    assert dir(optree.integration) == ['SUBMODULES', 'jax', 'numpy', 'torch']

    with pytest.raises(AttributeError):
        optree.integration.abc  # noqa: B018

    try:
        import jax  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            optree.integration.jax  # noqa: B018
    else:
        assert isinstance(optree.integration.jax, types.ModuleType)

    try:
        import numpy as np  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            optree.integration.numpy  # noqa: B018
    else:
        assert isinstance(optree.integration.numpy, types.ModuleType)

    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            optree.integration.torch  # noqa: B018
    else:
        assert isinstance(optree.integration.torch, types.ModuleType)
