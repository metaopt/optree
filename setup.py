import os
import pathlib
import platform
import re
import shutil
import sys
import sysconfig

from setuptools import setup


try:
    from pybind11.setup_helpers import Pybind11Extension as Extension
    from pybind11.setup_helpers import build_ext
except ImportError:
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).absolute().parent
VERSION_FILE = HERE / 'optree' / 'version.py'

sys.path.insert(0, str(VERSION_FILE.parent))
import version  # noqa


class CMakeExtension(Extension):
    def __init__(self, name, source_dir='.', target=None, **kwargs):
        super().__init__(name, sources=[], **kwargs)
        self.source_dir = os.path.abspath(source_dir)
        self.target = target if target is not None else name.rpartition('.')[-1]


class cmake_build_ext(build_ext):  # noqa: N801
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        cmake = shutil.which('cmake')
        if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

        ext_path = pathlib.Path(self.get_ext_fullpath(ext.name)).absolute()
        build_temp = pathlib.Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            f'-DCMAKE_BUILD_TYPE={config}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={ext_path.parent}',
            f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{config.upper()}={build_temp}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPYTHON_INCLUDE_DIR={sysconfig.get_path("platinclude")}',
        ]

        if platform.system() == 'Darwin':
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r'-arch (\S+)', os.environ.get('ARCHFLAGS', ''))
            if archs:
                cmake_args.append(f'-DCMAKE_OSX_ARCHITECTURES={";".join(archs)}')

        try:
            import pybind11

            cmake_args.append(f'-DPYBIND11_CMAKE_DIR={pybind11.get_cmake_dir()}')
        except ImportError:
            pass

        build_args = ['--config', config]
        if (
            'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ
            and hasattr(self, 'parallel')
            and self.parallel
        ):
            build_args.extend(['--parallel', str(self.parallel)])
        else:
            build_args.append('--parallel')

        build_args.extend(['--target', ext.target, '--'])

        try:
            os.chdir(build_temp)
            self.spawn([cmake, ext.source_dir, *cmake_args])
            if not self.dry_run:
                self.spawn([cmake, '--build', '.', *build_args])
        finally:
            os.chdir(HERE)


VERSION_CONTENT = None

try:
    if not version.__release__:
        try:
            VERSION_CONTENT = VERSION_FILE.read_text(encoding='UTF-8')
            VERSION_FILE.write_text(
                data=re.sub(
                    r"""__version__\s*=\s*('[^']+'|"[^"]+")""",
                    f'__version__ = {version.__version__!r}',
                    string=VERSION_CONTENT,
                ),
                encoding='UTF-8',
            )
        except OSError:
            VERSION_CONTENT = None

    setup(
        name='optree',
        version=version.__version__,
        package_data={'sharedlib': ['*.so', '*.pyd']},
        include_package_data=True,
        cmdclass={'build_ext': cmake_build_ext},
        ext_modules=[CMakeExtension('optree._C', source_dir=HERE)],
    )
finally:
    if VERSION_CONTENT is not None:
        with VERSION_FILE.open(mode='wt', encoding='UTF-8', newline='') as file:
            file.write(VERSION_CONTENT)
