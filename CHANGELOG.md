# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [Unreleased]

### Added

- Improve typing support for generic `PyTree[T]` and registry lookup / register functions by [@XuehaiPan](https://github.com/XuehaiPan) in [#160](https://github.com/metaopt/optree/pull/160).

### Changed

-

### Fixed

-

### Removed

-

------

## [0.13.0] - 2024-10-03

### Added

- Add Python 3.13t support by [@XuehaiPan](https://github.com/XuehaiPan) in [#137](https://github.com/metaopt/optree/pull/137).
- Expose Python implementation for C utilities for `namedtuple` and `PyStructSequence` by [@XuehaiPan](https://github.com/XuehaiPan) in [#157](https://github.com/metaopt/optree/pull/157).
- Add `dataclasses` integration by [@XuehaiPan](https://github.com/XuehaiPan) in [#142](https://github.com/metaopt/optree/pull/142).
- Add Python 3.13 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#156](https://github.com/metaopt/optree/pull/156).
- Respect cmake variable `pybind11_DIR` by [@XuehaiPan](https://github.com/XuehaiPan) in [#155](https://github.com/metaopt/optree/pull/155).
- Add tests with PyDebug enabled in CI by [@XuehaiPan](https://github.com/XuehaiPan) in [#150](https://github.com/metaopt/optree/pull/150).

### Changed

- Split implementation files and add more `inline` / `constexpr` / `noexcept` qualifiers by [@XuehaiPan](https://github.com/XuehaiPan) in [#159](https://github.com/metaopt/optree/pull/159).
- Use `cmake`'s `FindPython` module by [@XuehaiPan](https://github.com/XuehaiPan) in [#151](https://github.com/metaopt/optree/pull/151).

### Fixed

- Fix potential segmentation fault for `structseq_fields` cache support by [@XuehaiPan](https://github.com/XuehaiPan) in [#150](https://github.com/metaopt/optree/pull/150).

------

## [0.12.1] - 2024-07-06

### Fixed

- Fix warning regression during import when launch with strict warning filters by [@XuehaiPan](https://github.com/XuehaiPan) in [#149](https://github.com/metaopt/optree/pull/149).

------

## [0.12.0] - 2024-07-05

### Added

- Add context manager to temporarily set the dictionary sorting mode by [@XuehaiPan](https://github.com/XuehaiPan) in [#147](https://github.com/metaopt/optree/pull/147).
- Add PyPy support by [@XuehaiPan](https://github.com/XuehaiPan) in [#145](https://github.com/metaopt/optree/pull/145).
- Add 32-bit wheels for Linux and Windows by [@XuehaiPan](https://github.com/XuehaiPan) in [#141](https://github.com/metaopt/optree/pull/141).
- Add Linux ppc64le and s390x wheels by [@XuehaiPan](https://github.com/XuehaiPan) in [#138](https://github.com/metaopt/optree/pull/138).
- Add accessor APIs `tree_flatten_with_accessor` and `PyTreeSpec.accessors` by [@XuehaiPan](https://github.com/XuehaiPan) in [#108](https://github.com/metaopt/optree/pull/108).
- Add submodule `optree.functools` by [@XuehaiPan](https://github.com/XuehaiPan) in [#134](https://github.com/metaopt/optree/pull/134).

### Changed

- Use `stable` tag instead of 2.12.0 for `pybind11` version by [@XuehaiPan](https://github.com/XuehaiPan) in [#146](https://github.com/metaopt/optree/pull/146).
- Refactor the raw import statement in `setup.py` with `importlib` utilities by [@XuehaiPan](https://github.com/XuehaiPan) in [#135](https://github.com/metaopt/optree/pull/135).
- Update minimal version of `typing-extensions` to 4.5.0 for `typing_extensions.deprecated` by [@XuehaiPan](https://github.com/XuehaiPan) in [#134](https://github.com/metaopt/optree/pull/134).
- Update string representation for `OrderedDict` by [@XuehaiPan](https://github.com/XuehaiPan) in [#133](https://github.com/metaopt/optree/pull/133).

### Fixed

- Fix gc for self-referential case by implementing `tp_traverse` by [@XuehaiPan](https://github.com/XuehaiPan) in [#144](https://github.com/metaopt/optree/pull/144).
- Fix potential segmentation fault for pickling support by [@XuehaiPan](https://github.com/XuehaiPan) in [#143](https://github.com/metaopt/optree/pull/143).
- Update CI runner image for Python 3.7 on macOS by [@XuehaiPan](https://github.com/XuehaiPan) in [#135](https://github.com/metaopt/optree/pull/135).

### Removed

- Deprecate key path APIs by [@XuehaiPan](https://github.com/XuehaiPan) in [#108](https://github.com/metaopt/optree/pull/108).
- Deprecate `optree.Partial` and replace with `optree.functools.partial` by [@XuehaiPan](https://github.com/XuehaiPan) in [#134](https://github.com/metaopt/optree/pull/134).

------

## [0.11.0] - 2024-03-26

### Added

- Add function `is_namedtuple_instance` and `is_structseq_instance` and result caches by [@XuehaiPan](https://github.com/XuehaiPan) in [#121](https://github.com/metaopt/optree/pull/121).
- Add `tree_iter` function by [@XuehaiPan](https://github.com/XuehaiPan) in [#130](https://github.com/metaopt/optree/pull/130).
- Add API to unregister node type in the registry by [@XuehaiPan](https://github.com/XuehaiPan) in [#124](https://github.com/metaopt/optree/pull/124).
- Add tree map functions with transposed outputs `tree_transpose_map` and `tree_transpose_map_with_path` by [@XuehaiPan](https://github.com/XuehaiPan) in [#127](https://github.com/metaopt/optree/pull/127).
- Add static constructors to create `PyTreeSpec` instances by [@XuehaiPan](https://github.com/XuehaiPan) in [#120](https://github.com/metaopt/optree/pull/120).
- Cache intermediate `str` objects in `PyObject_GetAttr` calls by [@XuehaiPan](https://github.com/XuehaiPan) in [#106](https://github.com/metaopt/optree/pull/106) and [#109](https://github.com/metaopt/optree/pull/109).
- Install `clang-format` and `clang-tidy` from PyPI by [@XuehaiPan](https://github.com/XuehaiPan) in [#107](https://github.com/metaopt/optree/pull/107).
- Also check `_make` and `_asdict` in function `is_namedtuple_class` by [@XuehaiPan](https://github.com/XuehaiPan) in [#105](https://github.com/metaopt/optree/pull/105).

### Changed

- Set recursion limit to 1000 for all platforms by [@XuehaiPan](https://github.com/XuehaiPan) in [#121](https://github.com/metaopt/optree/pull/121).
- Allow types to be registered in both the global namespace and custom namespaces by [@XuehaiPan](https://github.com/XuehaiPan) in [#124](https://github.com/metaopt/optree/pull/124).
- Set `treespec_is_leaf` as strict by default by [@XuehaiPan](https://github.com/XuehaiPan) in [#120](https://github.com/metaopt/optree/pull/120).
- Reorder functions for better code correspondence between C++ and Python by [@XuehaiPan](https://github.com/XuehaiPan) in [#117](https://github.com/metaopt/optree/pull/117).
- Standardize `py::handle` and `py::object` usage in function signature by [@XuehaiPan](https://github.com/XuehaiPan) in [#115](https://github.com/metaopt/optree/pull/115).
- Reorder cases for `namedtuple` and `PyStructSequence` types by [@XuehaiPan](https://github.com/XuehaiPan) in [#111](https://github.com/metaopt/optree/pull/111).
- Use `__bases__` rather than `__base__` in function `is_structseq_class` by [@XuehaiPan](https://github.com/XuehaiPan) in [#104](https://github.com/metaopt/optree/pull/104).

### Fixed

- Fix potential segmentation fault when modifying `treespec.entries()` by [@XuehaiPan](https://github.com/XuehaiPan) in [#116](https://github.com/metaopt/optree/pull/116).

------

## [0.10.0] - 2023-11-07

### Added

- Add `tree_ravel` function for JAX/NumPy/PyTorch array/tensor tree manipulation by [@XuehaiPan](https://github.com/XuehaiPan) in [#100](https://github.com/metaopt/optree/pull/100).
- Expose node kind enum for `PyTreeSpec` by [@XuehaiPan](https://github.com/XuehaiPan) in [#98](https://github.com/metaopt/optree/pull/98).
- Expose function `tree_flatten_one_level` by [@XuehaiPan](https://github.com/XuehaiPan) in [#101](https://github.com/metaopt/optree/pull/101).
- Add tree broadcast functions `broadcast_common`, `tree_broadcast_common`, `tree_broadcast_map`, and `tree_broadcast_map_with_path` by [@XuehaiPan](https://github.com/XuehaiPan) in [#87](https://github.com/metaopt/optree/pull/87).
- Add function `tree_is_leaf` and add `is_leaf` argument to function `all_leaves` by [@XuehaiPan](https://github.com/XuehaiPan) in [#93](https://github.com/metaopt/optree/pull/93).
- Add methods `PyTreeSpec.entry` and `PyTreeSpec.child` by [@XuehaiPan](https://github.com/XuehaiPan) in [#88](https://github.com/metaopt/optree/pull/88).
- Add Python 3.12 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#90](https://github.com/metaopt/optree/pull/90).
- Allow passing third-party dependency version from environment variable by [@XuehaiPan](https://github.com/XuehaiPan) in [#80](https://github.com/metaopt/optree/pull/80).

### Changed

- Set recursion limit to 2000 for all platforms by [@XuehaiPan](https://github.com/XuehaiPan) in [#97](https://github.com/metaopt/optree/pull/97).
- Make `PyTreeSpec.is_prefix` to be consistent with `PyTreeSpec.flatten_up_to` by [@XuehaiPan](https://github.com/XuehaiPan) in [#94](https://github.com/metaopt/optree/pull/94).
- Decrease the `MAX_RECURSION_DEPTH` to 2000 on Windows by [@XuehaiPan](https://github.com/XuehaiPan) in [#85](https://github.com/metaopt/optree/pull/85).
- Bump `abseil-cpp` version to 20230802.1 by [@XuehaiPan](https://github.com/XuehaiPan) in [#80](https://github.com/metaopt/optree/pull/80).

### Fixed

- Memorize ongoing `repr` / `hash` calls to resolve infinite recursion under self-referential case by [@XuehaiPan](https://github.com/XuehaiPan) and [@JieRen98](https://github.com/JieRen98) in [#82](https://github.com/metaopt/optree/pull/82).

### Removed

- Remove dependence on `abseil-cpp` by [@XuehaiPan](https://github.com/XuehaiPan) in [#85](https://github.com/metaopt/optree/pull/85).

------

## [0.9.2] - 2023-09-18

### Changed

- Bump `pybind11` version to 2.11.1 and add initial Python 3.12 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#78](https://github.com/metaopt/optree/pull/78).
- Bump `abseil-cpp` version to 20230802.0 by [@XuehaiPan](https://github.com/XuehaiPan) in [#79](https://github.com/metaopt/optree/pull/79).

### Fixed

- Fix empty paths when flatten with custom `is_leaf` function by [@XuehaiPan](https://github.com/XuehaiPan) in [#76](https://github.com/metaopt/optree/pull/76).

------

## [0.9.1] - 2023-05-23

### Changed

- Use `py::type::handle_of(obj)` rather than deprecated `obj.get_type()` by [@XuehaiPan](https://github.com/XuehaiPan) in [#49](https://github.com/metaopt/optree/pull/49).
- Bump `abseil-cpp` version to 20230125.3 by [@XuehaiPan](https://github.com/XuehaiPan) in [#57](https://github.com/metaopt/optree/pull/57).

### Fixed

- Add `@runtime_checkable` decorator for `CustomTreeNode` protocol class by [@XuehaiPan](https://github.com/XuehaiPan) in [#56](https://github.com/metaopt/optree/pull/56).

------

## [0.9.0] - 2023-03-23

### Added

- Preserve dict key order in the output of `tree_unflatten`, `tree_map`, and `tree_map_with_path` by [@XuehaiPan](https://github.com/XuehaiPan) in [#46](https://github.com/metaopt/optree/pull/46).

### Changed

- Change keyword argument `initializer` back to `initial` for `tree_reduce` to align with `functools.reduce` C implementation by [@XuehaiPan](https://github.com/XuehaiPan) in [#47](https://github.com/metaopt/optree/pull/47).

------

## [0.8.0] - 2023-03-14

### Added

- Add methods `PyTreeSpec.paths` and `PyTreeSpec.entries` by [@XuehaiPan](https://github.com/XuehaiPan) in [#43](https://github.com/metaopt/optree/pull/43).
- Allow tree-map with mixed inputs of ordered and unordered dictionaries by [@XuehaiPan](https://github.com/XuehaiPan) in [#42](https://github.com/metaopt/optree/pull/42).
- Add more utility functions for `namedtuple` and `PyStructSequence` type by [@XuehaiPan](https://github.com/XuehaiPan) in [#41](https://github.com/metaopt/optree/pull/41).
- Add methods `PyTreeSpec.is_prefix` and `PyTreeSpec.is_suffix` and function `tree_broadcast_prefix` by [@XuehaiPan](https://github.com/XuehaiPan) in [#40](https://github.com/metaopt/optree/pull/40).
- Add tree reduce functions `tree_sum`, `tree_max`, and `tree_min` by [@XuehaiPan](https://github.com/XuehaiPan) in [#39](https://github.com/metaopt/optree/pull/39).
- Test dict key equality with `PyDict_Contains` ($O (n)$) rather than sorting ($O (n \log n)$) by [@XuehaiPan](https://github.com/XuehaiPan) in [#37](https://github.com/metaopt/optree/pull/37).
- Make error message more clear when value mismatch by [@XuehaiPan](https://github.com/XuehaiPan) in [#36](https://github.com/metaopt/optree/pull/36).
- Add `ruff` and `flake8` plugins integration by [@XuehaiPan](https://github.com/XuehaiPan) in [#33](https://github.com/metaopt/optree/pull/33) and [#34](https://github.com/metaopt/optree/pull/34).

### Changed

- Allow tree-map with mixed inputs of ordered and unordered dictionaries by [@XuehaiPan](https://github.com/XuehaiPan) in [#42](https://github.com/metaopt/optree/pull/42).
- Use more appropriate exception handling (e.g., change `ValueError` to `TypeError` in `structseq_fields`) by [@XuehaiPan](https://github.com/XuehaiPan) in [#41](https://github.com/metaopt/optree/pull/41).
- Inherit `optree._C.InternalError` from `SystemError` rather than `RuntimeError` by [@XuehaiPan](https://github.com/XuehaiPan) in [#41](https://github.com/metaopt/optree/pull/41).
- Change keyword argument `initial` to `initializer` for `tree_reduce` to align with `functools.reduce` by [@XuehaiPan](https://github.com/XuehaiPan) in [#39](https://github.com/metaopt/optree/pull/39).

------

## [0.7.0] - 2023-02-07

### Added

- Add `PyStructSequence` types as internal node types by [@XuehaiPan](https://github.com/XuehaiPan) in [#30](https://github.com/metaopt/optree/pull/30).

### Changed

- Add `PyStructSequence` types as internal node types by [@XuehaiPan](https://github.com/XuehaiPan) in [#30](https://github.com/metaopt/optree/pull/30).
- Use postponed evaluation of annotations by [@XuehaiPan](https://github.com/XuehaiPan) in [#28](https://github.com/metaopt/optree/pull/28).

------

## [0.6.0] - 2023-02-02

### Added

- Add Linux AArch64 and Windows ARM64 wheels by [@XuehaiPan](https://github.com/XuehaiPan) in [#25](https://github.com/metaopt/optree/pull/25).
- Add property `PyTreeSpec.type` and method `PyTreeSpec.is_leaf` by [@XuehaiPan](https://github.com/XuehaiPan) in [#26](https://github.com/metaopt/optree/pull/26).
- Raise a warning when registering subclasses of `namedtuple` by [@XuehaiPan](https://github.com/XuehaiPan) in [#24](https://github.com/metaopt/optree/pull/24).
- Add `clang-tidy` integration and update code style by [@XuehaiPan](https://github.com/XuehaiPan) in [#20](https://github.com/metaopt/optree/pull/20).

### Fixed

- Add `doctest` integration and fix docstring by [@XuehaiPan](https://github.com/XuehaiPan) in [#23](https://github.com/metaopt/optree/pull/23).

------

## [0.5.1] - 2023-01-21

### Added

- Add property `PyTreeSpec.num_children` by [@XuehaiPan](https://github.com/XuehaiPan).
- Update docstring and documentation by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [0.5.0] - 2022-11-30

### Added

- Add custom exceptions for internal error handling by [@XuehaiPan](https://github.com/XuehaiPan).

### Fixed

- Fix `PyTreeSpec` equality test and hash by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [0.4.2] - 2022-11-27

### Changed

- Better internal error handling by [@XuehaiPan](https://github.com/XuehaiPan) in [#17](https://github.com/metaopt/optree/pull/17).
- Use static raw pointers for global imports by [@XuehaiPan](https://github.com/XuehaiPan) in [#16](https://github.com/metaopt/optree/pull/16).

------

## [0.4.1] - 2022-11-25

### Fixed

- Fix segmentation fault error for global imports [@XuehaiPan](https://github.com/XuehaiPan) in [#14](https://github.com/metaopt/optree/pull/14).

------

## [0.4.0] - 2022-11-25

### Added

- Add namespace support for custom node type registry by [@XuehaiPan](https://github.com/XuehaiPan) in [#12](https://github.com/metaopt/optree/pull/12).
- Add tree flatten and tree map functions with extra paths by [@XuehaiPan](https://github.com/XuehaiPan) in [#11](https://github.com/metaopt/optree/pull/11).
- Add in-place version of tree-map function `tree_map_` by [@XuehaiPan](https://github.com/XuehaiPan).
- Add macOS ARM64 wheels by [@XuehaiPan](https://github.com/XuehaiPan) in [#9](https://github.com/metaopt/optree/pull/9).
- Add Python 3.11 support by [@XuehaiPan](https://github.com/XuehaiPan).

### Changed

- Use shallow clone for third-party Git repos by [@XuehaiPan](https://github.com/XuehaiPan).
- Use cmake FetchContent rather than Git submodules by [@XuehaiPan](https://github.com/XuehaiPan).

### Removed

- Drop Python 3.6 support by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [0.3.0] - 2022-10-26

### Added

- Add Read the Docs integration by [@XuehaiPan](https://github.com/XuehaiPan).
- Add benchmark script and results by [@XuehaiPan](https://github.com/XuehaiPan).
- Support both "`None` is Node" and "`None` is Leaf" by [@XuehaiPan](https://github.com/XuehaiPan).
- Add `OrderedDict` and `defaultdict` and `deque` as builtin support by [@XuehaiPan](https://github.com/XuehaiPan).

### Changed

- Reorganize code structure and rename `PyTreeDef` to `PyTreeSpec` by [@XuehaiPan](https://github.com/XuehaiPan).

### Fixed

- Fix Python 3.6 support by [@XuehaiPan](https://github.com/XuehaiPan).
- Fix generic `NamedTuple` for Python 3.8-3.10 by [@XuehaiPan](https://github.com/XuehaiPan).
- Fix builds for Python 3.8-3.10 on Windows by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [0.2.0] - 2022-09-24

### Added

- Add `cibuildwheel` integration for building wheels by [@XuehaiPan](https://github.com/XuehaiPan).
- Add full type annotations by [@XuehaiPan](https://github.com/XuehaiPan).

### Changed

- Improve custom tree node representation by [@XuehaiPan](https://github.com/XuehaiPan).

### Fixed

- Fix cross-platform compatibility by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [0.1.0] - 2022-09-16

### Added

- The first beta release of OpTree by [@XuehaiPan](https://github.com/XuehaiPan).
- OpTree with Linux / Windows / macOS x64 support by [@XuehaiPan](https://github.com/XuehaiPan).

------

[Unreleased]: https://github.com/metaopt/optree/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/metaopt/optree/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/metaopt/optree/compare/v0.11.0...v0.12.1
[0.12.0]: https://github.com/metaopt/optree/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/metaopt/optree/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/metaopt/optree/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/metaopt/optree/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/metaopt/optree/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/metaopt/optree/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/metaopt/optree/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/metaopt/optree/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/metaopt/optree/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/metaopt/optree/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/metaopt/optree/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/metaopt/optree/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/metaopt/optree/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/metaopt/optree/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/metaopt/optree/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/metaopt/optree/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/metaopt/optree/releases/tag/v0.1.0
