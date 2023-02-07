# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [Unreleased]

### Added

- Add `PyStructSequence` types as internal node types by [@XuehaiPan](https://github.com/XuehaiPan) in [#30](https://github.com/metaopt/optree/pull/30).

### Changed

- Add `PyStructSequence` types as internal node types by [@XuehaiPan](https://github.com/XuehaiPan) in [#30](https://github.com/metaopt/optree/pull/30).
- Use postponed evaluation of annotations by [@XuehaiPan](https://github.com/XuehaiPan) in [#28](https://github.com/metaopt/optree/pull/28).

### Fixed

-

### Removed

-

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

[Unreleased]: https://github.com/metaopt/optree/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/metaopt/optree/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/metaopt/optree/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/metaopt/optree/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/metaopt/optree/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/metaopt/optree/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/metaopt/optree/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/metaopt/optree/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/metaopt/optree/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/metaopt/optree/releases/tag/v0.1.0
