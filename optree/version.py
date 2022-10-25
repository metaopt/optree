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
"""OpTree: Optimized PyTree Utilities."""

__version__ = '0.3.0'
__license__ = 'Apache License, Version 2.0'
__author__ = 'OpTree Contributors'
__release__ = False

if not __release__:
    import os
    import subprocess

    try:
        prefix, sep, suffix = (
            subprocess.check_output(
                ['git', 'describe', '--abbrev=7'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            )
            .strip()
            .lstrip('v')
            .replace('-', '.dev', 1)
            .replace('-', '+', 1)
            .partition('.dev')
        )
        if sep:
            version_prefix, dot, version_tail = prefix.rpartition('.')
            prefix = f'{version_prefix}{dot}{int(version_tail) + 1}'
            __version__ = sep.join((prefix, suffix))
            del version_prefix, dot, version_tail
        else:
            __version__ = prefix
        del prefix, sep, suffix
    except (OSError, subprocess.CalledProcessError):
        pass

    del os, subprocess
