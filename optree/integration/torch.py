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
"""Integration with PyTorch."""

# pragma: no cover file

import sys
import warnings

import optree.integrations.torch
from optree.integrations.torch import *  # noqa: F403 # pylint: disable=wildcard-import,unused-wildcard-import
from optree.integrations.torch import __all__  # noqa: F401


# pylint: disable-next=fixme
# TODO: remove this file in version 0.18.0
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning, module=__name__, append=False)

    warnings.warn(
        f'The {__name__!r} module is deprecated and will be removed in version 0.18.0. '
        f'Please use {optree.integrations.torch.__name__!r} instead.',
        FutureWarning,
        stacklevel=2,
    )


sys.modules[__name__] = optree.integrations.torch

del sys, warnings, optree
