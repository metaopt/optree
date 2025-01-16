import contextlib
import os
import pathlib
import platform
import re
import shutil
import sys
import sysconfig
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


HERE = pathlib.Path(__file__).absolute().parent


class CMakeExtension(Extension):
    def __init__(self, name, source_dir='.', target=None, **kwargs):
        super().__init__(name, sources=[], **kwargs)
        self.source_dir = os.path.abspath(source_dir)
        self.target = target if target is not None else name.rpartition('.')[-1]


class cmake_build_ext(build_ext):  # noqa: N801
    def build_extension(self, ext):  # noqa: C901
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        cmake = shutil.which('cmake')
        if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

        ext_path = pathlib.Path(self.get_ext_fullpath(ext.name)).absolute()
        build_temp = pathlib.Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

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
            archs = re.findall(r'-arch (\S+)', os.getenv('ARCHFLAGS', ''))
            if archs:
                cmake_args.append(f'-DCMAKE_OSX_ARCHITECTURES={";".join(archs)}')
        elif platform.system() == 'Windows':
            # Windows - set correct CMAKE_GENERATOR_PLATFORM
            cmake_generator_platform = os.getenv('CMAKE_GENERATOR_PLATFORM', '')
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
                import pybind11

                cmake_args.append(f'-Dpybind11_DIR={pybind11.get_cmake_dir()}')

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

        cwd = os.getcwd()
        try:
            os.chdir(build_temp)
            self.spawn([cmake, ext.source_dir, *cmake_args])
            if not self.dry_run:
                self.spawn([cmake, '--build', '.', *build_args])
        finally:
            os.chdir(cwd)


@contextlib.contextmanager
def vcs_version(name, path):
    path = pathlib.Path(path).absolute()
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


with vcs_version(name='optree.version', path=(HERE / 'optree' / 'version.py')) as version:
    setup(
        name='optree',
        version=version.__version__,
        cmdclass={'build_ext': cmake_build_ext},
        ext_modules=[CMakeExtension('optree._C', source_dir=HERE)],
    )
