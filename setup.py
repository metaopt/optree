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
"""The setup script for the :mod:`optree` package."""

# pylint: disable=missing-class-docstring,missing-function-docstring

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING, Any

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    from packaging.version import Version
else:
    try:
        from packaging.version import Version
    except ImportError:
        from setuptools.dist import Version


HERE = Path(__file__).absolute().parent
CMAKE_MINIMUM_VERSION = '3.18'


@contextlib.contextmanager
def unset_python_path() -> Generator[str | None]:
    python_path = None
    python_no_user_site = None
    try:
        # pip's build environment pseudo-isolation sets `PYTHONPATH`.
        # It may break console scripts (e.g., `cmake` installed from PyPI).
        python_path = os.environ.pop('PYTHONPATH', None)  # unset `PYTHONPATH`
        python_no_user_site = os.environ.pop('PYTHONNOUSERSITE', None)  # unset `PYTHONNOUSERSITE`
        yield python_path
    finally:
        if python_path is not None:
            os.environ['PYTHONPATH'] = python_path
        if python_no_user_site is not None:
            os.environ['PYTHONNOUSERSITE'] = python_no_user_site


def cmake_context(
    cmake: os.PathLike[str] | str,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> contextlib.AbstractContextManager[str | None]:
    if dry_run:
        return contextlib.nullcontext()

    cmake = os.fspath(cmake)
    spawn_context = contextlib.nullcontext
    output = ''
    try:
        # System CMake or CMake in the build environment
        output = subprocess.check_output(  # noqa: S603
            [cmake, '--version'],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        print(
            f'Could not run `{cmake}` directly. '
            'Unset the `PYTHONPATH` environment variable in the build environment.',
            file=sys.stderr,
        )
        spawn_context = unset_python_path  # type: ignore[assignment]
        with unset_python_path():
            # CMake in the parent virtual environment
            output = subprocess.check_output(  # noqa: S603
                [cmake, '--version'],
                stderr=subprocess.STDOUT,
                text=True,
            ).strip()

    if verbose and output:
        print(output, file=sys.stderr)

    return spawn_context()


# pylint: disable-next=too-few-public-methods
class CMakeExtension(Extension):
    def __init__(
        self,
        name: str,
        source_dir: os.PathLike[str] | str = '.',
        target: str | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, sources=[], language=language, **kwargs)
        self.source_dir = Path(source_dir).absolute()
        self.target = target if target is not None else name.rpartition('.')[-1]

    @classmethod
    def cmake_executable(
        cls,
        *,
        minimum_version: Version | str | None = None,
        verbose: bool = False,
    ) -> str | None:
        cmake = os.getenv('CMAKE_COMMAND') or os.getenv('CMAKE_EXECUTABLE')
        if not cmake:
            cmake = shutil.which('cmake')
        if cmake and minimum_version is not None:
            with cmake_context(cmake, verbose=verbose):
                try:
                    cmake_capabilities = json.loads(
                        subprocess.check_output(  # noqa: S603
                            [cmake, '-E', 'capabilities'],
                            stderr=subprocess.DEVNULL,
                            text=True,
                        ),
                    )
                except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
                    cmake_capabilities = {}
            cmake_version = Version(cmake_capabilities.get('version', {}).get('string', '0.0.0'))
            if isinstance(minimum_version, str):
                minimum_version = Version(minimum_version)
            if cmake_version < minimum_version:
                cmake = None
        return cmake


# pylint: disable-next=invalid-name
class cmake_build_ext(build_ext):  # noqa: N801
    # pylint: disable-next=too-many-branches
    def build_extension(self, ext: Extension) -> None:  # noqa: C901
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        cmake = ext.cmake_executable()
        if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

        ext_path = Path(self.get_ext_fullpath(ext.name)).absolute()
        build_temp = Path(self.build_temp).absolute()

        config = os.getenv('CMAKE_BUILD_TYPE', '') or ('Debug' if self.debug else 'Release')
        cmake_args = [
            f'-DCMAKE_BUILD_TYPE={config}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={ext_path.parent}',
            f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{config.upper()}={build_temp}',
            f'-DPython_EXECUTABLE={sys.executable}',
            f'-DPython_INCLUDE_DIR={sysconfig.get_path("platinclude")}',
        ]
        if self.include_dirs:
            cmake_args.append(f'-DPython_EXTRA_INCLUDE_DIRS={";".join(self.include_dirs)}')
        if self.library_dirs:
            cmake_args.append(f'-DPython_EXTRA_LIBRARY_DIRS={";".join(self.library_dirs)}')
        if self.libraries:
            cmake_args.append(f'-DPython_EXTRA_LIBRARIES={";".join(self.libraries)}')

        # Cross-compilation support
        if platform.system() == 'Darwin':
            # macOS - respect ARCHFLAGS if set
            archs = re.findall(r'-arch\s+(\S+)', os.getenv('ARCHFLAGS', ''))
            if archs:
                cmake_args.append(f'-DCMAKE_OSX_ARCHITECTURES={";".join(archs)}')
        elif platform.system() == 'Windows':
            # Windows - set correct CMAKE_GENERATOR_PLATFORM
            cmake_generator_platform = os.getenv('CMAKE_GENERATOR_PLATFORM')
            if not cmake_generator_platform:
                cmake_generator_platform = {
                    'win32': 'Win32',
                    'win-amd64': 'x64',
                    'win-arm32': 'ARM',
                    'win-arm64': 'ARM64',
                }.get(self.plat_name)
            if cmake_generator_platform:
                cmake_args.append(f'-A={cmake_generator_platform}')

        pybind11_dir = os.getenv('pybind11_DIR', '')  # noqa: SIM112
        if pybind11_dir:
            cmake_args.append(f'-Dpybind11_DIR={pybind11_dir}')
        else:
            with contextlib.suppress(ImportError):
                import pybind11  # pylint: disable=import-outside-toplevel

                cmake_args.append(f'-Dpybind11_DIR={pybind11.get_cmake_dir()}')

        build_args = ['--config', config]
        if 'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ and bool(getattr(self, 'parallel', 0)):
            build_args.extend(['--parallel', str(self.parallel)])
        else:
            build_args.append('--parallel')

        build_args.extend(['--target', ext.target, '--'])

        self.mkpath(str(build_temp))
        with cmake_context(cmake, dry_run=self.dry_run, verbose=True):
            self.spawn([cmake, '-S', str(ext.source_dir), '-B', str(build_temp), *cmake_args])
            self.spawn([cmake, '--build', str(build_temp), *build_args])


@contextlib.contextmanager
def vcs_version(name: str, path: os.PathLike[str] | str) -> Generator[ModuleType]:
    path = Path(path).absolute()
    assert path.is_file()
    module_spec = spec_from_file_location(name=name, location=path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = sys.modules.get(name)
    if module is None:
        module = module_from_spec(module_spec)
        sys.modules[name] = module
    module_spec.loader.exec_module(module)

    if module.__release__:
        yield module
        return

    content = None
    try:
        try:
            content = path.read_text(encoding='utf-8')
            path.write_text(
                data=re.sub(
                    r"""__version__\s*=\s*('[^']+'|"[^"]+")""",
                    f'__version__ = {module.__version__!r}',
                    string=content,
                ),
                encoding='utf-8',
            )
        except OSError:
            content = None

        yield module
    finally:
        if content is not None:
            with path.open(mode='wt', encoding='utf-8', newline='') as file:
                file.write(content)


if __name__ == '__main__':
    with vcs_version(name='optree.version', path=HERE / 'optree' / 'version.py') as version:
        setup(
            name='optree',
            version=version.__version__,
            cmdclass={'build_ext': cmake_build_ext},
            ext_modules=[CMakeExtension('optree._C', source_dir=HERE, language='c++')],
            setup_requires=(
                [f'cmake >= {CMAKE_MINIMUM_VERSION}']
                if CMakeExtension.cmake_executable(minimum_version=CMAKE_MINIMUM_VERSION) is None
                else []
            ),
        )
