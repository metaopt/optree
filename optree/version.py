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
"""OpTree: Optimized PyTree Utilities."""

# pylint: disable=invalid-name

__version__ = '0.19.0'
__license__ = 'Apache-2.0'
__author__ = 'OpTree Contributors'
__release__ = False

if not __release__:
    import subprocess
    from pathlib import Path

    root_dir = Path(__file__).absolute().parent.parent
    try:
        prefix, sep, suffix = (
            subprocess.check_output(  # noqa: S603
                [  # noqa: S607
                    'git',
                    f'--git-dir={root_dir / ".git"}',
                    'describe',
                    '--abbrev=7',
                ],
                cwd=root_dir,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding='utf-8',
                timeout=120.0,
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
            __version__ = f'{prefix}{sep}{suffix}'
            del version_prefix, dot, version_tail
        else:
            __version__ = prefix
        del prefix, sep, suffix
    except (OSError, RuntimeError, subprocess.SubprocessError):
        pass

    del Path, subprocess, root_dir
