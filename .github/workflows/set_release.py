#!/usr/bin/env python3

# pylint: disable=missing-module-docstring

import re
from pathlib import Path


ROOT = Path(__file__).absolute().parents[2]

VERSION_FILE = ROOT / 'optree' / 'version.py'


if __name__ == '__main__':
    VERSION_CONTENT = VERSION_FILE.read_text(encoding='utf-8')

    VERSION_FILE.write_text(
        data=re.sub(
            r'__release__\s*=.*',
            '__release__ = True',
            string=VERSION_CONTENT,
        ),
        encoding='utf-8',
    )
