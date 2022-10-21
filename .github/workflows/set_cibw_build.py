import os
import sys


CIBW_BUILD = 'CIBW_BUILD=*p%d%d-*' % sys.version_info[:2]  # pylint: disable=consider-using-f-string

print(CIBW_BUILD)
with open(os.getenv('GITHUB_ENV'), mode='at', encoding='UTF-8') as file:
    print(CIBW_BUILD, file=file)
