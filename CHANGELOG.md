# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [Unreleased]

### Added

- Upload PyEmscripten/WASM wheels to PyPI by [@XuehaiPan](https://github.com/XuehaiPan) in [#295](https://github.com/metaopt/optree/pull/295).

### Changed

-

### Fixed

- Fix an abort during subinterpreter finalization when garbage-collecting `PyTreeSpec` or `PyTreeIter` objects by [@XuehaiPan](https://github.com/XuehaiPan) in [#294](https://github.com/metaopt/optree/pull/294).

### Removed

- Drop Python 3.9 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#272](https://github.com/metaopt/optree/pull/272).

------

## [0.20.0] - 2026-08-20

### Added

- Preserve dict insertion order on `unflatten` for the Python `tree_flatten_one_level` path mirroring the C++ path by [@XuehaiPan](https://github.com/XuehaiPan) in [#280](https://github.com/metaopt/optree/pull/280).
- Add built-in `frozendict` support for Python 3.15+ (PEP 814), with `PyTreeKind.FROZENDICT`, `optree.treespec_frozendict()`, `optree.treespec.frozendict()`, and `optree.typing.FrozenDict` by [@XuehaiPan](https://github.com/XuehaiPan) in [#274](https://github.com/metaopt/optree/pull/274).
- Migrate to `Python_FIND_ABI` for CMake's `FindPython` by [@XuehaiPan](https://github.com/XuehaiPan) in [#292](https://github.com/metaopt/optree/pull/292).
- Add Python 3.15 and Python 3.15t support by [@XuehaiPan](https://github.com/XuehaiPan) in [#293](https://github.com/metaopt/optree/pull/293).

### Changed

- Update minimal version of `typing-extensions` to 4.10.0 for `typing_extensions.TypeIs` by [@XuehaiPan](https://github.com/XuehaiPan) in [#285](https://github.com/metaopt/optree/pull/285).

### Fixed

- Define `Py_GIL_DISABLED` for free-threaded debug builds on Windows when building the C extension to work around an upstream CMake `FindPython` bug by [@XuehaiPan](https://github.com/XuehaiPan) in [#285](https://github.com/metaopt/optree/pull/285).
- Fix a deadlock when registering or unregistering a pytree node concurrently with flattening, caused by releasing the GIL while holding the registry lock by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix a deadlock when a duplicate registration or an unregistration of an unregistered type formats its error message, which runs the type's `__repr__` while the registry lock is held and wedges any thread that is flattening by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix registering an already-registered `collections.namedtuple` subclass or `PyStructSequence` emitting the override warning even though the registration is rejected, which raised `UserWarning` instead of `ValueError` under warnings-as-errors by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix a deadlock when unregistering a pytree node drops the last reference to its flatten or unflatten function, so a `__del__` or weakref callback re-entered the registry while its lock was held by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix a deadlock while flattening on free-threading builds, caused by acquiring the non-recursive dictionary insertion order lock twice by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix a spurious `SystemError` from `optree._C.get_registry_size()` when a concurrent registration slipped between its two reads by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `optree.tree_iter()` hanging uninterruptibly when the `is_leaf` predicate or a custom flatten function advances the same iterator, now reported as a `RuntimeError` by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix the type caches handing an interpreter a value owned by another one, by keying them on the interpreter in addition to the type address by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix the type caches retaining entries for a finalized interpreter, which leaked immortal keys and could be inherited by a fresh interpreter reusing the same ID by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix a deadlock in the `PyStructSequence` field cache, caused by re-acquiring the GIL while still holding the cache lock by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix the type caches publishing an entry that a failed cleanup registration would orphan, leaving a value owned by an interpreter with no callback able to evict it, whether the publish preceded that registration or raced another thread still performing it by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `optree.structseq_fields()` and the struct sequence accessors reporting the trailing hidden field names for unnamed sequence slots, by mapping each field to the slot its member offset names; the pure Python fallback, which cannot read those offsets, recovers the positions from a probe instance instead of assuming the unnamed slots trail the named ones by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix the treespec repr printing the raw unnamed-field marker as if it were a keyword argument, now rendered as `<unnamed@N>` by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `StructSequenceEntry` reporting an unnamed sequence slot as `field='unnamed field'`, which reads as a real field name, now reported by its index by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `PyTreeSpec.__setstate__()` accepting a malformed state, which could read out of bounds or abort the interpreter when the treespec was later used by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `PyTreeSpec.__getstate__()` returning the treespec's internal mutable containers, so mutating the pickled state corrupted an otherwise immutable treespec by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `PyTreeSpec.__setstate__()` borrowing the key lists from the state it was given rather than copying them, so mutating that state afterwards silently reordered the restored treespec by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix pickling a `PyTreeSpec` with protocol 0 or 1 aborting the interpreter, by reducing through `copyreg.__newobj__` by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `PyTreeSpec.compose()`, `PyTreeSpec.broadcast_to_common_suffix()`, `treespec_transform()`, and `treespec_from_collection()` silently rebinding a custom node to a different registration when an empty namespace adopted a non-empty one by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `PyTreeSpec.broadcast_to_common_suffix()` sorting the argument treespec's dictionary keys in place while building its key-mismatch error message, corrupting a treespec the caller still holds by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix the dictionary key order being lost when a treespec built under the global namespace is promoted to an insertion-ordered namespace by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `treespec_from_collection()` on a leaf reporting an escalated `UserWarning` as a confusing `SystemError`, by checking the return value of `PyErr_WarnEx()` by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `treespec_from_collection()` keeping the caller's namespace on a leaf or `None` root, which made otherwise identical treespecs compare unequal by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `treespec_is_prefix()` and `treespec_is_suffix()` comparing against a stale subtree when a dictionary node's keys had been reordered by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix the tree iterator never reporting or clearing its `is_leaf` predicate to the garbage collector, leaking any reference cycle that passes through it by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix a list shrunk by an `is_leaf` predicate part way through flattening being read out of bounds on Python versions before 3.13.0a4, now raising `IndexError` by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix a reference cycle passing through a registered custom type not being collectable once the registry no longer holds the registration, by reporting the registration's members to the garbage collector when a treespec's own nodes hold every reference to it; a cycle spanning several treespecs remains uncollectable by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix a deeply nested treespec overflowing the native stack and crashing in `treespec_paths()`, `treespec_accessors()`, and `PyTreeSpec.broadcast_to_common_suffix()` instead of raising `RecursionError` by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `DataclassEntry` and `AttrsEntry` resolving an integer entry against every declared field rather than the children the registration emitted, which returned the wrong attribute for a class holding a metadata or non-`init` field, now read from the record `register_node()` leaves on the class itself by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `optree.dataclasses.register_node()` and `optree.integrations.attrs.register_node()` leaving a class in a half-registered state that could never be registered again: the registry entry and the field marker are now committed together, and a failure of either rolls the other back by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `optree.dataclasses.register_node()` silently dropping `InitVar` pseudo-fields, which are neither children nor metadata and cannot round-trip, now rejected with a pointer to the generic API by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `tree_broadcast_common()` and `broadcast_common()` applying the caller's `is_leaf` predicate to an internal sentinel tree, which could raise from the predicate or collapse a filled subtree and under-replicate by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `PyTreeEntry` equality and hashing comparing the bytecode of `__call__()` and `codify()` rather than the methods themselves, so two entry classes that happened to share an implementation compared equal by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `PyTreeAccessor` overriding `__eq__()` while inheriting `tuple.__ne__()`, so comparing one with a plain tuple of equal entries reported `False` for both `==` and `!=` by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `GetAttrEntry.codify()` emitting invalid attribute access for a field name that is not an identifier, now rendered as a `getattr()` call by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `prefix_errors()` raising `AssertionError` instead of reporting an error for a custom node whose per-instance entries differ while its metadata does not, a pair `broadcast_prefix()` accepts by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `prefix_errors()` raising `TypeError` while formatting a dictionary key mismatch when the keys have different types, now ordered with `total_order_sorted()` by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `tree_transpose_map()`, `tree_transpose_map_with_path()`, `tree_transpose_map_with_accessor()`, and `tree_partition()` rejecting a leafless outer structure when an explicit `inner_treespec` leaves nothing to infer by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix the `tree_broadcast_map()` family flattening a custom node twice when called with a single input tree, which broke a one-shot flatten function and contradicted the documented equivalence with `tree_map()` by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `treespec_entry()` and `treespec_child()` annotating their index as `int` while the runtime accepts any `SupportsInt` or `SupportsIndex` object by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `optree.utils.total_order_sorted()` mistaking a `TypeError` raised by the caller's `key` callback for a comparison failure and silently returning the sequence unsorted, and calling the callback twice per element by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix a dictionary whose keys cannot be compared being flattened in a partially sorted order instead of insertion order, when a comparison raised part way through the sort by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).
- Fix `tree_broadcast_common()` documenting that its results share one structure and `broadcast_common()` documenting that it returns two pytrees rather than two lists of leaves by [@XuehaiPan](https://github.com/XuehaiPan) in [#290](https://github.com/metaopt/optree/pull/290).

------

## [0.19.1] - 2026-05-05

### Added

- Add `attrs` integration module `optree.integrations.attrs` with `field`, `define`, `frozen`, `mutable`, `make_class`, `register_node`, and `AttrsEntry` by [@XuehaiPan](https://github.com/XuehaiPan) in [#273](https://github.com/metaopt/optree/pull/273).
- Add `optree.dataclasses.register_node` to register existing dataclasses as pytree nodes by [@XuehaiPan](https://github.com/XuehaiPan) in [#273](https://github.com/metaopt/optree/pull/273).
- Extend `GetAttrEntry` to support dotted attribute paths for traversing nested attributes (e.g., `a.b.c`) by [@XuehaiPan](https://github.com/XuehaiPan).
- Add `functools.Placeholder` support and re-export for `optree.functools.partial` (Python 3.14+) by [@XuehaiPan](https://github.com/XuehaiPan) in [#276](https://github.com/metaopt/optree/pull/276).

### Fixed

- Fix Windows wheels crashing when another extension preloads an older `msvcp140.dll` by disabling MSVC's constexpr `std::mutex` constructor by [@XuehaiPan](https://github.com/XuehaiPan) in [#279](https://github.com/metaopt/optree/pull/279). Issued by [@jmkolbe](https://github.com/jmkolbe) in [#278](https://github.com/metaopt/optree/issues/278).

------

## [0.19.0] - 2026-02-23

### Added

- Add subinterpreters support for Python 3.14+ by [@XuehaiPan](https://github.com/XuehaiPan) in [#245](https://github.com/metaopt/optree/pull/245).

### Fixed

- Polish docstrings and fix grammars by [@XuehaiPan](https://github.com/XuehaiPan) in [#256](https://github.com/metaopt/optree/pull/256).

------

## [0.18.0] - 2025-11-14

### Added

- Use ARM-based GHA runners to build ARM wheels by [@XuehaiPan](https://github.com/XuehaiPan) in [#234](https://github.com/metaopt/optree/pull/234).
- Add Android support by [@XuehaiPan](https://github.com/XuehaiPan) in [#236](https://github.com/metaopt/optree/pull/236).
- Add `manylinux-riscv64` wheels by [@XuehaiPan](https://github.com/XuehaiPan) in [#243](https://github.com/metaopt/optree/pull/243).
- Add `cp{313,314}-ios` / `cp{313,314}-android` / `cp{312,313}-pyodide` wheels by [@XuehaiPan](https://github.com/XuehaiPan) in [#242](https://github.com/metaopt/optree/pull/242) and [#244](https://github.com/metaopt/optree/pull/244).
- Add support for method pair `(__tree_flatten__, __tree_unflatten__)` for `register_pytree_node_class(cls)` by [@XuehaiPan](https://github.com/XuehaiPan) in [#253](https://github.com/metaopt/optree/pull/253).

### Removed

- Remove CPython 3.9/3.10 wheels for Windows ARM64 due to unavailability of official Python distribution by [@XuehaiPan](https://github.com/XuehaiPan) in [#234](https://github.com/metaopt/optree/pull/234).
- Remove previously deprecated singular-named modules in [#209](https://github.com/metaopt/optree/pull/209) by [@XuehaiPan](https://github.com/XuehaiPan) in [#238](https://github.com/metaopt/optree/pull/238).
- Remove PyPy 3.10 (EOL) wheels by [@dependabot](https://docs.github.com/en/code-security/dependabot) in [#241](https://github.com/metaopt/optree/pull/241).

------

## [0.17.0] - 2025-07-25

### Added

- Add WASM support by [@XuehaiPan](https://github.com/XuehaiPan) in [#226](https://github.com/metaopt/optree/pull/226).
- Bump `cibuildwheel` from 2.23 to 3.0 by [@dependabot](https://docs.github.com/en/code-security/dependabot) in [#228](https://github.com/metaopt/optree/pull/228).
- Add iOS support by [@XuehaiPan](https://github.com/XuehaiPan) in [#232](https://github.com/metaopt/optree/pull/232).
- Build Python 3.14 and 3.14t wheels in CI by [@XuehaiPan](https://github.com/XuehaiPan) in [#233](https://github.com/metaopt/optree/pull/233).

### Changed

- Build wheels against `pybind11` 3.0.0 by [@XuehaiPan](https://github.com/XuehaiPan) in [#231](https://github.com/metaopt/optree/pull/231).

### Fixed

- Handle `pybind11` macro defined as 0 instead of non-exist by [@XuehaiPan](https://github.com/XuehaiPan) in [#227](https://github.com/metaopt/optree/pull/227).

### Removed

- Remove PyPy 3.9 (EOL) wheels by [@dependabot](https://docs.github.com/en/code-security/dependabot) in [#228](https://github.com/metaopt/optree/pull/228).

------

## [0.16.0] - 2025-05-28

### Added

- Explicitly set recursion limit for recursion tests by [@XuehaiPan](https://github.com/XuehaiPan) in [#207](https://github.com/metaopt/optree/pull/207).
- Dump build-time meta-information to C extension by [@XuehaiPan](https://github.com/XuehaiPan) in [#215](https://github.com/metaopt/optree/pull/215).
- Use `pybind11::native_enum` to create enum class `PyTreeKind` if available by [@XuehaiPan](https://github.com/XuehaiPan) in [#214](https://github.com/metaopt/optree/pull/214).
- Enable `pybind11::smart_holder` to create class `PyTreeSpec` and `PyTreeIter` if available by [@XuehaiPan](https://github.com/XuehaiPan) in [#217](https://github.com/metaopt/optree/pull/217).
- Implement optional `tp_clear` for class `PyTreeSpec` and `PyTreeIter` by [@XuehaiPan](https://github.com/XuehaiPan) in [#218](https://github.com/metaopt/optree/pull/218).
- Add function `tree_partition` by [@pfackeldey](https://github.com/pfackeldey) in [#222](https://github.com/metaopt/optree/pull/222).
- Add Python 3.14 and Python 3.14t support by [@XuehaiPan](https://github.com/XuehaiPan) in [#216](https://github.com/metaopt/optree/pull/216).

### Changed

- Enforce naming convention of packages with singular and plural: `optree.{accessor,integration}` -> `optree.{accessors,integrations}` by [@XuehaiPan](https://github.com/XuehaiPan) in [#209](https://github.com/metaopt/optree/pull/209).
- Allow creating dataclass types in the global namespace by [@XuehaiPan](https://github.com/XuehaiPan) in [#212](https://github.com/metaopt/optree/pull/212).
- Migrate to `setuptools>=77` for PEP-639 by [@XuehaiPan](https://github.com/XuehaiPan) in [#208](https://github.com/metaopt/optree/pull/208).
- Update minimal version of `typing-extensions` to 4.6.0 for `typing_extensions.TypeAliasType` by [@XuehaiPan](https://github.com/XuehaiPan) in [#216](https://github.com/metaopt/optree/pull/216).

### Fixed

- Never call `PyType_Ready` twice and use `PyType_Modified` instead by [@XuehaiPan](https://github.com/XuehaiPan) in [#214](https://github.com/metaopt/optree/pull/214).
- Fix `optree.typing.PyTree[T]` for Python 3.14 due to immutable `UnionType` by [@XuehaiPan](https://github.com/XuehaiPan) in [#216](https://github.com/metaopt/optree/pull/216).

### Removed

- Drop Python 3.8 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#206](https://github.com/metaopt/optree/pull/206).
- Retire benchmark script by [@XuehaiPan](https://github.com/XuehaiPan) in [#211](https://github.com/metaopt/optree/pull/211).

------

## [0.15.0] - 2025-04-06

### Added

- Add method `PyTreeSpec.traverse` by [@XuehaiPan](https://github.com/XuehaiPan) in [#197](https://github.com/metaopt/optree/pull/197).
- Include test suites in SDist by [@XuehaiPan](https://github.com/XuehaiPan) in [#201](https://github.com/metaopt/optree/pull/201).
- Include branch coverage and add conditional pragmas by [@XuehaiPan](https://github.com/XuehaiPan) in [#204](https://github.com/metaopt/optree/pull/204).
- Detect `cmake` version and guard minimum version in `setup.py` by [@XuehaiPan](https://github.com/XuehaiPan) in [#205](https://github.com/metaopt/optree/pull/205).

### Removed

- Remove deprecated key path APIs by [@XuehaiPan](https://github.com/XuehaiPan) in [#195](https://github.com/metaopt/optree/pull/195).
- Remove deprecated `optree.Partial` by [@XuehaiPan](https://github.com/XuehaiPan) in [#196](https://github.com/metaopt/optree/pull/196).
- Remove duplicate lint checks by [@XuehaiPan](https://github.com/XuehaiPan) in [#202](https://github.com/metaopt/optree/pull/202).

------

## [0.14.1] - 2025-03-02

### Added

- Support using system `cmake` executable during setup by [@mgorny](https://github.com/mgorny) in [#188](https://github.com/metaopt/optree/pull/188).
- Add shortcut module `optree.pytree` and `optree.treespec` by [@lqhuang](https://github.com/lqhuang) in [#189](https://github.com/metaopt/optree/pull/189).
- Support lookup all registry entries in a `namespace` via `register_pytree_node.get()` by [@XuehaiPan](https://github.com/XuehaiPan) in [#190](https://github.com/metaopt/optree/pull/190).
- Add PyPy 3.11 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#194](https://github.com/metaopt/optree/pull/194).

### Changed

- Enable CXX11 ABI in C++ extension by [@XuehaiPan](https://github.com/XuehaiPan) in [#184](https://github.com/metaopt/optree/pull/184).

------

## [0.14.0] - 2025-01-17

### Added

- Add method `PyTreeSpec.one_level` and `PyTreeSpec.is_one_level` by [@XuehaiPan](https://github.com/XuehaiPan) in [#179](https://github.com/metaopt/optree/pull/179).
- Add method `PyTreeSpec.transform` by [@XuehaiPan](https://github.com/XuehaiPan) in [#177](https://github.com/metaopt/optree/pull/177).

### Changed

- Mark some arguments as positional-only as of Python 3.8+ by [@XuehaiPan](https://github.com/XuehaiPan) in [#178](https://github.com/metaopt/optree/pull/178).

### Fixed

- Fix cross-compiling for ARM64 on x64 Windows by [@XuehaiPan](https://github.com/XuehaiPan) in [#183](https://github.com/metaopt/optree/pull/183).

### Removed

- Drop Python 3.7 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#161](https://github.com/metaopt/optree/pull/161).

------

## [0.13.1] - 2024-11-12

### Added

- Upload coverage / JUnit results / core dumps in CI workflows by [@XuehaiPan](https://github.com/XuehaiPan) in [#170](https://github.com/metaopt/optree/pull/170) and [#172](https://github.com/metaopt/optree/pull/172).
- Add more info to `tree_flatten_one_level` by [@XuehaiPan](https://github.com/XuehaiPan) in [#168](https://github.com/metaopt/optree/pull/168).
- Improve typing support for generic `PyTree[T]` and registry lookup / register functions by [@XuehaiPan](https://github.com/XuehaiPan) in [#160](https://github.com/metaopt/optree/pull/160) and [#166](https://github.com/metaopt/optree/pull/166).

### Changed

- Move include directory `include/{ => optree}/*.h` by [@XuehaiPan](https://github.com/XuehaiPan) in [#167](https://github.com/metaopt/optree/pull/167).

### Fixed

- Improve typing support for `optree.dataclasses.dataclass` and `optree.dataclasses.field` by [@manulari](https://github.com/manulari) in [#165](https://github.com/metaopt/optree/pull/165).

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

[Unreleased]: https://github.com/metaopt/optree/compare/v0.20.0...HEAD
[0.20.0]: https://github.com/metaopt/optree/compare/v0.19.1...v0.20.0
[0.19.1]: https://github.com/metaopt/optree/compare/v0.19.0...v0.19.1
[0.19.0]: https://github.com/metaopt/optree/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/metaopt/optree/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/metaopt/optree/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/metaopt/optree/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/metaopt/optree/compare/v0.14.1...v0.15.0
[0.14.1]: https://github.com/metaopt/optree/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/metaopt/optree/compare/v0.13.1...v0.14.0
[0.13.1]: https://github.com/metaopt/optree/compare/v0.13.0...v0.13.1
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
