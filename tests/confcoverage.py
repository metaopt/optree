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

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from coverage import CoveragePlugin


if TYPE_CHECKING:
    from coverage.config import CoverageConfig
    from coverage.plugin_support import Plugins


TEST_ROOT = Path(__file__).absolute().parent


def is_importable(mod: str) -> bool:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(('PYTHON', 'PYTEST', 'COV_'))
    }
    try:
        subprocess.check_call(
            [
                sys.executable,
                '-c',
                textwrap.dedent(
                    f"""
                    try:
                        import {mod}
                    except ImportError:
                        exit(1)
                    exit(0)
                    """,
                ).strip(),
            ],
            cwd=TEST_ROOT,
            env=env,
        )
    except subprocess.CalledProcessError:
        return False
    return True


class OptionalModuleConfigurer(CoveragePlugin):
    def __init__(self, optional_modules: str = '') -> None:
        self._optional_modules = sorted(filter(None, set(optional_modules.split())))

    def configure(self, config: CoverageConfig) -> None:
        for k, v in (
            (
                'report:exclude_lines',
                [
                    # package specific no cover
                    *self._package_pragmas(),
                ],
            ),
            (
                'report:partial_branches',
                [
                    # package specific no cover
                    rf'# pragma: ({"|".join(self._optional_modules)}) (no )?cover\b',
                ],
            ),
        ):
            before = set(config.get_option(k) or ())
            before.update(v)
            config.set_option(k, sorted(before))

        config.set_option('report:fail_under', 0.0)

    def _package_pragmas(self) -> list[str]:
        importable_modules = set(filter(is_importable, self._optional_modules))
        unimportable_modules = set(self._optional_modules).difference(importable_modules)
        importable_modules = sorted(map(re.escape, importable_modules))
        unimportable_modules = sorted(map(re.escape, unimportable_modules))
        return [
            *(rf'# pragma: {mod} cover\b' for mod in unimportable_modules),
            *(rf'\A(?s:.*# pragma: {mod} cover file\b.*)\Z' for mod in unimportable_modules),
            *(
                rf'# pragma: {mod} cover begin\b'
                rf'(?s:.)*?'
                rf'# pragma: {mod} cover end\b'
                for mod in unimportable_modules
            ),
            *(rf'# pragma: {mod} no cover\b' for mod in importable_modules),
            *(rf'\A(?s:.*# pragma: {mod} no cover file\b.*)\Z' for mod in importable_modules),
            *(
                rf'# pragma: {mod} no cover begin\b'
                rf'(?s:.)*?'
                rf'# pragma: {mod} no cover end\b'
                for mod in importable_modules
            ),
        ]


def coverage_init(reg: Plugins, options: dict[str, str]) -> None:
    reg.add_configurer(OptionalModuleConfigurer(**options))
