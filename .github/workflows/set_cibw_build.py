import os
import platform


# pylint: disable-next=consider-using-f-string
CIBW_BUILD = 'CIBW_BUILD=*%sp%s%s-*' % (
    platform.python_implementation().lower()[0],
    *platform.python_version_tuple()[:2],
)

print(CIBW_BUILD)
with open(os.getenv('GITHUB_ENV'), mode='at', encoding='UTF-8') as file:
    print(CIBW_BUILD, file=file)
