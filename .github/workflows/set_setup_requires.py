#!/usr/bin/env python3

# pylint: disable=missing-module-docstring

import re
from pathlib import Path


ROOT = Path(__file__).absolute().parents[2]

PYPROJECT_FILE = ROOT / 'pyproject.toml'


if __name__ == '__main__':
    PYPROJECT_CONTENT = PYPROJECT_FILE.read_text(encoding='utf-8')

    PYPROJECT_FILE.write_text(
        data=re.sub(
            r'(requires\s*=\s*\[.*"\s*)\bpybind11\b[^"]*(\s*".*\])',
            r'\g<1>pybind11 @ git+https://github.com/pybind/pybind11.git#egg=pybind11\g<2>',
            string=PYPROJECT_CONTENT,
        ),
        encoding='utf-8',
    )
