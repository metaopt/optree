# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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
"""Integration with third-party libraries."""

import sys
from typing import Any


current_module = sys.modules[__name__]


SUBMODULES = frozenset({'jax', 'numpy', 'torch'})


# pylint: disable-next=too-few-public-methods
class _LazyModule(type(current_module)):  # type: ignore[misc]
    def __getattribute__(self, name: str) -> Any:  # noqa: N804
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name in SUBMODULES:
                import importlib  # pylint: disable=import-outside-toplevel

                submodule = importlib.import_module(f'{__name__}.{name}')
                setattr(self, name, submodule)
                return submodule
            raise


current_module.__class__ = _LazyModule

del sys, Any, current_module, _LazyModule
