#!/usr/bin/env python3

# pylint: disable=missing-module-docstring

import re
from pathlib import Path


ROOT = Path(__file__).absolute().parents[2]

PYPROJECT_FILE = ROOT / 'pyproject.toml'
PYBIND11_GIT_URL = 'https://github.com/XuehaiPan/pybind11.git@subinterp-call-once-and-store'


if __name__ == '__main__':
    PYPROJECT_CONTENT = PYPROJECT_FILE.read_text(encoding='utf-8')

    PYPROJECT_FILE.write_text(
        data=re.sub(
            r'(requires\s*=\s*\[.*"\s*)\bpybind11\b[^"]*(\s*".*\])',
            rf'\g<1>pybind11 @ git+{PYBIND11_GIT_URL}#egg=pybind11\g<2>',
            string=PYPROJECT_CONTENT,
        ),
        encoding='utf-8',
    )
