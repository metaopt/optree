# Copyright 2022-2026 MetaOPT Team. All Rights Reserved.
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
"""Shared test fixtures for pytree traversal, registration, and interpreter behavior."""

# pylint: disable=invalid-name

import contextlib
import dataclasses
import functools
import gc
import itertools
import os
import platform
import re
import subprocess
import sys
import sysconfig
import textwrap
import time
import types
from collections import OrderedDict, UserDict, defaultdict, deque, namedtuple
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, ParamSpec, TypeVar
from typing_extensions import Self

import pytest

import optree
from optree._C import (
    OPTREE_HAS_FROZENDICT,
    OPTREE_HAS_SUBINTERPRETER_SUPPORT,
    PYBIND11_HAS_NATIVE_ENUM,
    PYBIND11_HAS_SUBINTERPRETER_SUPPORT,
    Py_DEBUG,
    Py_GIL_DISABLED,
    get_registry_size,
)
from optree.ops import STANDARD_DICT_TYPES as STANDARD_DICT_TYPES
from optree.registry import __GLOBAL_NAMESPACE as GLOBAL_NAMESPACE
from optree.registry import _NODETYPE_REGISTRY as NODETYPE_REGISTRY


TEST_ROOT = Path(__file__).absolute().parent
P = ParamSpec('P')
R = TypeVar('R')


INITIAL_REGISTRY_SIZE = get_registry_size()
assert (9 if sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT else 8) == INITIAL_REGISTRY_SIZE
assert len(NODETYPE_REGISTRY) == INITIAL_REGISTRY_SIZE + 2

_ = PYBIND11_HAS_NATIVE_ENUM
_ = PYBIND11_HAS_SUBINTERPRETER_SUPPORT
_ = OPTREE_HAS_SUBINTERPRETER_SUPPORT

if sysconfig.get_config_var('Py_DEBUG') is None:
    assert Py_DEBUG == hasattr(sys, 'gettotalrefcount')
else:
    assert Py_DEBUG == bool(int(sysconfig.get_config_var('Py_DEBUG') or '0'))
skipif_pydebug = pytest.mark.skipif(
    Py_DEBUG,
    reason='Py_DEBUG is enabled which causes too much overhead',
)

assert Py_GIL_DISABLED == bool(int(sysconfig.get_config_var('Py_GIL_DISABLED') or '0'))
skipif_freethreading = pytest.mark.skipif(
    Py_GIL_DISABLED,
    reason='Py_GIL_DISABLED is set',
)

# Free-threaded builds before 3.14 hold deferred references to type objects in per-thread caches, so
# `gc_collect()` cannot force a heap type to be reclaimed there.
HAS_DEFERRED_TYPE_REFS = Py_GIL_DISABLED and sys.version_info < (3, 14)
skipif_deferred_type_refs = pytest.mark.skipif(
    HAS_DEFERRED_TYPE_REFS,
    reason='free-threaded builds before 3.14 keep deferred references to type objects',
)

PYPY = platform.python_implementation() == 'PyPy'
skipif_pypy = pytest.mark.skipif(
    PYPY,
    reason='PyPy does not support weakref and refcount correctly',
)

IOS = sys.platform.startswith(('ios', 'iphoneos', 'iphonesimulator'))
skipif_ios = pytest.mark.skipif(
    IOS,
    reason='iOS does not support subprocesses',
)

ANDROID = sys.platform.startswith('android')
skipif_android = pytest.mark.skipif(
    ANDROID,
    reason='Android testbed returns wrong `sys.executable` which breaks subprocesses',
)

WASM = sys.platform.startswith(('emscripten', 'wasi'))
skipif_wasm = pytest.mark.skipif(
    WASM,
    reason='WASM does not support threading and multiprocessing and has limited stack size',
)


NUM_GC_REPEAT = 10 if Py_GIL_DISABLED else 5


def gc_collect(repeat: int = NUM_GC_REPEAT) -> None:
    """Run repeated collections so reference-count checks see reclaimed objects."""
    for _ in range(repeat):
        gc.collect()


def getrefcount(obj: object = None) -> int:
    """Return the reference count after collecting garbage."""
    gc_collect()
    return sys.getrefcount(obj)


def parametrize(**argvalues: Iterable[object]) -> pytest.MarkDecorator:
    """Parametrize a test over the Cartesian product of the named arguments."""
    arguments = list(argvalues)
    combinations = list(itertools.product(*argvalues.values()))

    def represent(value: object) -> str:
        if isinstance(value, types.FunctionType):
            return f"<function '{value.__module__}.{value.__qualname__}'>"
        return repr(value)

    ids = tuple(
        '-'.join(f'{arg}({represent(value)})' for arg, value in zip(arguments, values, strict=True))
        for values in combinations
    )

    return pytest.mark.parametrize(arguments, combinations, ids=ids)


@contextlib.contextmanager
def systrace(function: Callable[[types.FrameType, str, Any], Any] | None) -> Iterator[None]:
    """Temporarily replace the interpreter's trace function."""
    old_trace = sys.gettrace()
    sys.settrace(function)
    try:
        yield
    finally:
        sys.settrace(old_trace)


@contextlib.contextmanager
def recursionlimit(limit: int) -> Iterator[None]:
    """Temporarily cap the recursion limit with tracing disabled."""
    with systrace(None):
        gc_collect()
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(min(old_limit, limit))
        try:
            yield
        finally:
            sys.setrecursionlimit(old_limit)


def disable_systrace(func: Callable[P, R]) -> Callable[P, R]:
    """Disable tracing while the decorated function runs."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with systrace(None):
            return func(*args, **kwargs)

    return wrapper


class CalledProcessError(subprocess.CalledProcessError):
    """Include captured process output in test failure messages."""

    def __str__(self) -> str:
        """Append any captured output and errors to the process failure."""
        return ''.join(
            (
                super().__str__(),
                f'\nOutput:\n{self.output}' if self.output is not None else '',
                f'\nStderr:\n{self.stderr}' if self.stderr is not None else '',
            ),
        )


def check_script_in_subprocess(
    script: str,
    /,
    *,
    output: str | re.Pattern[str] | None,
    timeout: float = 300.0,
    cwd: str | Path = TEST_ROOT,
    env: Mapping[str, str] | None = None,
    rerun: int = 1,
    rstrip: bool = False,
) -> str:
    """Run the test script in a fresh interpreter for each repetition and check its output."""
    script = textwrap.dedent(script).strip()
    result = ''
    for _ in range(rerun):
        try:
            result = subprocess.check_output(  # noqa: S603
                [sys.executable, '-u', '-X', 'dev', '-Walways', '-Werror', '-c', script],
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                timeout=timeout,
                cwd=cwd,
                env={
                    key: value
                    for key, value in (env if env is not None else os.environ).items()
                    if (
                        not key.startswith(('PYTHON', 'PYTEST', 'COV_'))
                        or key in ('PYTHON_GIL', 'PYTHONDEVMODE', 'PYTHONHASHSEED')
                    )
                },
            )
        except subprocess.CalledProcessError as ex:
            raise CalledProcessError(ex.returncode, ex.cmd, ex.output, ex.stderr) from None
        if rstrip:
            result = result.rstrip()
        if output is not None:
            if isinstance(output, re.Pattern):
                assert output.search(result) is not None, (
                    f'output {result!r} does not match pattern {output.pattern!r}'
                )
            else:
                assert result == output
    return result


MISSING = object()


def assert_equal_type_and_value(
    actual: Any,
    expected: Any = MISSING,
    *,
    expected_type: type | None = None,
) -> None:
    """Assert the exact type and, if supplied, the value including individual accessor entries."""
    if expected_type is None:
        assert expected is not MISSING
        expected_type = type(expected)
    assert type(actual) is expected_type

    if expected is MISSING:
        return

    assert actual == expected
    if isinstance(expected, optree.PyTreeAccessor):
        assert hash(actual) == hash(expected)
        for i, j in zip(actual, expected, strict=True):
            assert_equal_type_and_value(i, j)


def is_tuple(tup: Any) -> bool:
    """Treat tuples as leaves."""
    return isinstance(tup, tuple)


def is_list(lst: Any) -> bool:
    """Treat lists as leaves."""
    return isinstance(lst, list)


def is_dict(dct: Any) -> bool:
    """Treat dictionaries as leaves."""
    return isinstance(dct, dict)


def is_primitive_collection(obj: Any) -> bool:
    """Treat supported collections of primitive values as leaves."""
    if type(obj) in {tuple, list, deque}:
        return all(isinstance(item, (int, float, str, bool, type(None))) for item in obj)
    if type(obj) in STANDARD_DICT_TYPES:
        return all(isinstance(value, (int, float, str, bool, type(None))) for value in obj.values())
    return False


def is_none(none: Any) -> bool:
    """Treat only ``None`` as a leaf."""
    return none is None


def is_not_none(none: Any) -> bool:
    """Treat every value except ``None`` as a leaf."""
    return none is not None


def always(obj: Any) -> bool:  # pylint: disable=unused-argument
    """Treat every object as a leaf."""
    return True


def never(obj: Any) -> bool:  # pylint: disable=unused-argument
    """Defer all leaf decisions to the registry."""
    return False


IS_LEAF_FUNCTIONS = (
    is_tuple,
    is_list,
    is_dict,
    is_primitive_collection,
    is_none,
    is_not_none,
    always,
    never,
)


CustomTuple = namedtuple('CustomTuple', ('foo', 'bar'))  # noqa: PYI024


class CustomNamedTupleSubclass(CustomTuple):
    """Exercise named tuple recognition through inheritance."""


class EmptyTuple(NamedTuple):
    """A named tuple with no fields."""


# sys.float_info(max=*, max_exp=*, max_10_exp=*, min=*, min_exp=*, min_10_exp=*, dig=*, mant_dig=*, epsilon=*, radix=*, rounds=*)
SysFloatInfoType = type(sys.float_info)
# time.struct_time(tm_year=*, tm_mon=*, tm_mday=*, tm_hour=*, tm_min=*, tm_sec=*, tm_wday=*, tm_yday=*, tm_isdst=*)
TimeStructTimeType = time.struct_time

if PYPY:
    SysFloatInfoType.__module__ = 'sys'
    TimeStructTimeType.__module__ = 'time'


class Vector3D:
    """A registered node with two children and a third coordinate stored as metadata."""

    def __init__(self, x: Any, y: Any, z: Any) -> None:
        """Store the three coordinates."""
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other: object) -> bool:
        """Compare the three coordinates."""
        return (
            isinstance(other, Vector3D)
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    def __hash__(self) -> int:
        """Hash the coordinates together."""
        return hash((self.x, self.y, self.z))

    def __repr__(self) -> str:
        """Show the class name and all three coordinates."""
        return f'{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})'


class Vector3DEntry(optree.PyTreeEntry):
    """Access a vector child by its position in the flattened node."""

    def __call__(self, obj: Vector3D) -> Any:
        """Select the x or y coordinate."""
        assert self.entry in {0, 1}
        return obj.x if self.entry == 0 else obj.y


optree.register_pytree_node(
    Vector3D,
    lambda o: ((o.x, o.y), o.z),
    lambda z, xy: Vector3D(xy[0], xy[1], z),
    path_entry_type=Vector3DEntry,
    namespace=GLOBAL_NAMESPACE,
)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
class Vector2D:
    """A two-child node registered through its flatten and unflatten methods."""

    def __init__(self, x: Any, y: Any) -> None:
        """Store the two coordinates."""
        self.x = x
        self.y = y

    def __getitem__(self, index: int) -> Any:
        """Select the x or y coordinate."""
        assert index in {0, 1}
        return self.x if index == 0 else self.y

    def __eq__(self, other: object) -> bool:
        """Compare both coordinates of two vectors."""
        return isinstance(other, Vector2D) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self) -> int:
        """Hash the coordinates together."""
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        """Show the class name and both coordinates."""
        return f'{self.__class__.__name__}(x={self.x}, y={self.y})'

    def tree_flatten(self) -> tuple[tuple[Any, Any], None]:
        """Return both coordinates as children with no metadata."""
        return (self.x, self.y), None

    @classmethod
    def tree_unflatten(cls, metadata: None, children: Iterable[Any]) -> Self:  # pylint: disable=unused-argument
        """Rebuild the vector from its children."""
        return cls(*children)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
@dataclasses.dataclass
class MyDataclass:
    """A four-child dataclass that supplies field names as path entries."""

    alpha: Any
    beta: Any
    gamma: Any
    delta: Any

    def __tree_flatten__(self) -> tuple[tuple[Any, Any, Any, Any], None, tuple[str, str, str, str]]:
        """Return the field values and their path entries in declaration order."""
        return (
            (self.alpha, self.beta, self.gamma, self.delta),
            None,
            ('alpha', 'beta', 'gamma', 'delta'),
        )

    @classmethod
    def __tree_unflatten__(cls, metadata: None, children: Iterable[Any]) -> Self:
        """Rebuild all four fields from the children."""
        return cls(*children)


@optree.register_pytree_node_class(path_entry_type=optree.GetAttrEntry, namespace=GLOBAL_NAMESPACE)
@dataclasses.dataclass
class MyOtherDataclass:
    """A dataclass with two children and two metadata fields."""

    a: Any
    b: Any
    c: Any
    d: Any

    def tree_flatten(self) -> tuple[tuple[Any, Any], tuple[Any, Any], tuple[str, str]]:
        """Return fields a and c as children and fields b and d as metadata."""
        return (
            (self.a, self.c),
            (self.b, self.d),
            ('a', 'c'),
        )

    @classmethod
    def tree_unflatten(cls, metadata: tuple[Any, Any], children: Iterable[Any]) -> Self:
        """Restore the original field order from the children and metadata."""
        a, c = children
        b, d = metadata
        return cls(a, b, c, d)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
@dataclasses.dataclass
class MyAnotherDataclass:
    """A three-child dataclass that uses positional path entries."""

    x: Any
    y: Any
    z: Any

    def __tree_flatten__(self) -> tuple[tuple[Any, Any, Any], None]:
        """Return all three fields as children."""
        return (self.x, self.y, self.z), None

    @classmethod
    def __tree_unflatten__(cls, metadata: None, children: Iterable[Any]) -> Self:
        """Restore the three fields from the children."""
        return cls(*children)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
class FlatCache:
    """Cache a tree's flattened representation and rebuild its structure on demand."""

    TREE_PATH_ENTRY_TYPE = optree.GetItemEntry

    def __init__(
        self,
        structured: Any,
        *,
        leaves: Sequence[Any] | None = None,
        treespec: optree.PyTreeSpec | None = None,
    ) -> None:
        """Use the supplied leaves and treespec or flatten the structured value."""
        if treespec is None:
            flattened, treespec = optree.tree_flatten(structured)
            if leaves is not None:
                if list(leaves) != flattened:
                    raise ValueError('`leaves` does not match the flattened `structured` value.')
            else:
                leaves = flattened
        elif leaves is None:
            raise ValueError('Provide `leaves` when passing a `treespec`.')
        self._structured = structured
        self.treespec = treespec
        self.leaves = leaves

    def __getitem__(self, index: int) -> Any:
        """Return a cached leaf by index."""
        return self.leaves[index]

    def __eq__(self, other: object) -> bool:
        """Compare the structured values of two caches."""
        return isinstance(other, FlatCache) and self.structured == other.structured

    def __hash__(self) -> int:
        """Hash the structured value."""
        return hash(self.structured)

    def __repr__(self) -> str:
        """Show the structured value held by the cache."""
        return f'{self.__class__.__name__}({self.structured!r})'

    @property
    def structured(self) -> Any:
        """Return the structured value, rebuilding it on first access if needed."""
        if self._structured is None:
            self._structured = optree.tree_unflatten(self.treespec, self.leaves)
        return self._structured

    def __tree_flatten__(self) -> tuple[Sequence[Any], optree.PyTreeSpec]:
        """Return cached leaves with their treespec as metadata."""
        return self.leaves, self.treespec

    @classmethod
    def __tree_unflatten__(cls, metadata: optree.PyTreeSpec, children: Sequence[Any]) -> Self:
        """Build the cache from children, flattening any newly introduced subtrees."""
        if not optree.all_leaves(children):
            children, metadata = optree.tree_flatten(optree.tree_unflatten(metadata, children))
        return cls(structured=None, leaves=children, treespec=metadata)


@optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
class MyDict(UserDict):
    """A registered mapping whose children follow reverse key order."""

    TREE_PATH_ENTRY_TYPE = optree.MappingEntry

    def __tree_flatten__(self) -> tuple[list[Any], list[Any], list[Any]]:
        """Return the values with their keys as both metadata and path entries."""
        reversed_keys = sorted(self.keys(), reverse=True)
        return [self[key] for key in reversed_keys], reversed_keys, reversed_keys

    @classmethod
    def __tree_unflatten__(cls, metadata: Iterable[Any], children: Iterable[Any]) -> Self:
        """Pair each key with its reconstructed value."""
        return cls(zip(metadata, children, strict=True))

    def __repr__(self) -> str:
        """Include the custom mapping class in its representation."""
        return f'{self.__class__.__name__}({super().__repr__()})'


@optree.register_pytree_node_class('namespace')
class MyAnotherDict(MyDict):
    """A custom mapping registered in a separate namespace."""


class Counter:
    """Mutable integer used to track visits to tree nodes."""

    def __init__(self, start: int = 0) -> None:
        """Initialize the count."""
        self.count = start

    def increment(self, n: int = 1) -> int:
        """Add to the count and return the updated value."""
        self.count += n
        return self.count

    def __int__(self) -> int:
        """Return the current count."""
        return self.count

    def __eq__(self, other: object) -> bool:
        """Compare the current values of two counters."""
        return isinstance(other, Counter) and self.count == other.count

    def __hash__(self) -> int:
        """Hash the current count."""
        return hash(self.count)

    def __repr__(self) -> str:
        """Show the current count."""
        return f'Counter({self.count})'

    def __next__(self) -> int:
        """Increment the count by one and return it."""
        return self.increment()


NAMESPACED_TREE = MyAnotherDict([('baz', 101), ('foo', MyDict(a=1, b=2, c=None))])


# pylint: disable=line-too-long
TREES = (
    1,
    None,
    (None,),
    (1, None),
    (),
    [],
    ([()]),
    (1, 2),
    ((1, 'foo'), ['bar', (3, None, 7)]),
    [3],
    EmptyTuple(),
    [3, CustomTuple(foo=(3, CustomTuple(foo=3, bar=None)), bar={'baz': 34})],
    TimeStructTimeType((*range(1, 3), None, *range(3, 9))),
    SysFloatInfoType(
        (*range(1, 10), None, TimeStructTimeType((*range(10, 15), None, *range(15, 20)))),
    ),
    [Vector3D(3, None, [4, 'foo'])],
    Vector2D(2, 3.0),
    {},
    {'a': 1, 'b': 2},
    {'b': (2, 3), 'a': 1, 'c': None, 'd': {'f': 4, 'e': None}},
    OrderedDict(),
    OrderedDict([('foo', 34), ('baz', 101), ('something', -42)]),
    OrderedDict([('foo', 34), ('baz', 101), ('something', deque([None, 2, 3]))]),
    OrderedDict([('foo', 34), ('baz', 101), ('something', deque([None, None, 3], maxlen=2))]),
    OrderedDict([('foo', 34), ('baz', 101), ('something', deque([None, 2, 3], maxlen=2))]),
    defaultdict(),
    defaultdict(int),
    defaultdict(dict, [('foo', 34), ('baz', 101), ('something', -42)]),
    deque(),
    deque(maxlen=0),
    deque([None, 2, 3]),
    deque([None, None, 3], maxlen=2),
    MyDict([('baz', 101), ('foo', MyDict(a=1, b=2, c=None))]),
    NAMESPACED_TREE,
    CustomNamedTupleSubclass(foo='hello', bar=3.5),
    MyDataclass(2, None, 3, 5),
    MyOtherDataclass(7, 11, None, 13),
    MyAnotherDataclass(MyDataclass(2, 3, None, 5), 7, MyOtherDataclass(11, None, 13, 19)),
    FlatCache(None),
    FlatCache(1),
    FlatCache({'a': [1, 2]}),
)


TREE_PATHS_NONE_IS_NODE = [
    [()],
    [],
    [],
    [(0,)],
    [],
    [],
    [],
    [(0,), (1,)],
    [(0, 0), (0, 1), (1, 0), (1, 1, 0), (1, 1, 2)],
    [(0,)],
    [],
    [(0,), (1, 0, 0), (1, 0, 1, 0), (1, 1, 'baz')],
    [(0,), (1,), (3,), (4,), (5,), (6,), (7,), (8,)],
    [
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (8,),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 6),
        (10, 7),
        (10, 8),
    ],
    [(0, 0)],
    [(0,), (1,)],
    [],
    [('a',), ('b',)],
    [('a',), ('b', 0), ('b', 1), ('d', 'f')],
    [],
    [('foo',), ('baz',), ('something',)],
    [('foo',), ('baz',), ('something', 1), ('something', 2)],
    [('foo',), ('baz',), ('something', 1)],
    [('foo',), ('baz',), ('something', 0), ('something', 1)],
    [],
    [],
    [('baz',), ('foo',), ('something',)],
    [],
    [],
    [(1,), (2,)],
    [(1,)],
    [('foo', 'b'), ('foo', 'a'), ('baz',)],
    [()],
    [(0,), (1,)],
    [('alpha',), ('gamma',), ('delta',)],
    [('a',)],
    [(0, 'alpha'), (0, 'beta'), (0, 'delta'), (1,), (2, 'a'), (2, 'c')],
    [],
    [(0,)],
    [(0,), (1,)],
]

TREE_PATHS_NONE_IS_LEAF = [
    [()],
    [()],
    [(0,)],
    [(0,), (1,)],
    [],
    [],
    [],
    [(0,), (1,)],
    [(0, 0), (0, 1), (1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2)],
    [(0,)],
    [],
    [(0,), (1, 0, 0), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 'baz')],
    [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)],
    [
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (8,),
        (9,),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 5),
        (10, 6),
        (10, 7),
        (10, 8),
    ],
    [(0, 0), (0, 1)],
    [(0,), (1,)],
    [],
    [('a',), ('b',)],
    [('a',), ('b', 0), ('b', 1), ('c',), ('d', 'e'), ('d', 'f')],
    [],
    [('foo',), ('baz',), ('something',)],
    [('foo',), ('baz',), ('something', 0), ('something', 1), ('something', 2)],
    [('foo',), ('baz',), ('something', 0), ('something', 1)],
    [('foo',), ('baz',), ('something', 0), ('something', 1)],
    [],
    [],
    [('baz',), ('foo',), ('something',)],
    [],
    [],
    [(0,), (1,), (2,)],
    [(0,), (1,)],
    [('foo', 'c'), ('foo', 'b'), ('foo', 'a'), ('baz',)],
    [()],
    [(0,), (1,)],
    [('alpha',), ('beta',), ('gamma',), ('delta',)],
    [('a',), ('c',)],
    [(0, 'alpha'), (0, 'beta'), (0, 'gamma'), (0, 'delta'), (1,), (2, 'a'), (2, 'c')],
    [],
    [(0,)],
    [(0,), (1,)],
]

TREE_ACCESSORS_NONE_IS_NODE = [
    [optree.PyTreeAccessor()],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
    ],
    [],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),)),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
    ],
    [optree.PyTreeAccessor((optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),))],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),)),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(1, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.MappingEntry('baz', dict, optree.PyTreeKind.DICT),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(0, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(1, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(3, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(4, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(5, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(6, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(7, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(8, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(0, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(1, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(2, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(3, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(4, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(5, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(6, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(7, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(8, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(0, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(1, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(2, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(3, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(4, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(6, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(7, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(8, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                Vector3DEntry(0, Vector3D, optree.PyTreeKind.CUSTOM),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor((optree.FlattenedEntry(0, Vector2D, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.FlattenedEntry(1, Vector2D, optree.PyTreeKind.CUSTOM),)),
    ],
    [],
    [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
    ],
    [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('d', dict, optree.PyTreeKind.DICT),
                optree.MappingEntry('f', dict, optree.PyTreeKind.DICT),
            ),
        ),
    ],
    [],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(2, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [],
    [],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('something', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
    ],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(2, deque, optree.PyTreeKind.DEQUE),)),
    ],
    [optree.PyTreeAccessor((optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),))],
    [
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('b', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor((optree.MappingEntry('baz', MyDict, optree.PyTreeKind.CUSTOM),)),
    ],
    [optree.PyTreeAccessor()],
    [
        optree.PyTreeAccessor(
            (optree.NamedTupleEntry(0, CustomNamedTupleSubclass, optree.PyTreeKind.NAMEDTUPLE),),
        ),
        optree.PyTreeAccessor(
            (optree.NamedTupleEntry(1, CustomNamedTupleSubclass, optree.PyTreeKind.NAMEDTUPLE),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.DataclassEntry('alpha', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('gamma', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('delta', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.GetAttrEntry('a', MyOtherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('alpha', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('beta', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('delta', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry(1, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(2, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.GetAttrEntry('a', MyOtherDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(2, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.GetAttrEntry('c', MyOtherDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
    ],
    [],
    [optree.PyTreeAccessor((optree.GetItemEntry(0, FlatCache, optree.PyTreeKind.CUSTOM),))],
    [
        optree.PyTreeAccessor((optree.GetItemEntry(0, FlatCache, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.GetItemEntry(1, FlatCache, optree.PyTreeKind.CUSTOM),)),
    ],
]

TREE_ACCESSORS_NONE_IS_LEAF = [
    [optree.PyTreeAccessor()],
    [optree.PyTreeAccessor()],
    [optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),))],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),)),
    ],
    [],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),)),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.SequenceEntry(2, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
    ],
    [optree.PyTreeAccessor((optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),))],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),)),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(0, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
                optree.NamedTupleEntry(1, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(1, list, optree.PyTreeKind.LIST),
                optree.NamedTupleEntry(1, CustomTuple, optree.PyTreeKind.NAMEDTUPLE),
                optree.MappingEntry('baz', dict, optree.PyTreeKind.DICT),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(0, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(1, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(2, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(3, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(4, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(5, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(6, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(7, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(8, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(0, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(1, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(2, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(3, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(4, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(5, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(6, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(7, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(8, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (optree.StructSequenceEntry(9, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(0, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(1, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(2, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(3, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(4, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(5, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(6, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(7, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.StructSequenceEntry(10, SysFloatInfoType, optree.PyTreeKind.STRUCTSEQUENCE),
                optree.StructSequenceEntry(8, TimeStructTimeType, optree.PyTreeKind.STRUCTSEQUENCE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                Vector3DEntry(0, Vector3D, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.SequenceEntry(0, list, optree.PyTreeKind.LIST),
                Vector3DEntry(1, Vector3D, optree.PyTreeKind.CUSTOM),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor((optree.FlattenedEntry(0, Vector2D, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.FlattenedEntry(1, Vector2D, optree.PyTreeKind.CUSTOM),)),
    ],
    [],
    [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor((optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),)),
    ],
    [
        optree.PyTreeAccessor((optree.MappingEntry('a', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('b', dict, optree.PyTreeKind.DICT),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor((optree.MappingEntry('c', dict, optree.PyTreeKind.DICT),)),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('d', dict, optree.PyTreeKind.DICT),
                optree.MappingEntry('e', dict, optree.PyTreeKind.DICT),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('d', dict, optree.PyTreeKind.DICT),
                optree.MappingEntry('f', dict, optree.PyTreeKind.DICT),
            ),
        ),
    ],
    [],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(2, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', OrderedDict, optree.PyTreeKind.ORDEREDDICT),),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('something', OrderedDict, optree.PyTreeKind.ORDEREDDICT),
                optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),
            ),
        ),
    ],
    [],
    [],
    [
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('foo', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('something', defaultdict, optree.PyTreeKind.DEFAULTDICT),),
        ),
    ],
    [],
    [],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(2, deque, optree.PyTreeKind.DEQUE),)),
    ],
    [
        optree.PyTreeAccessor((optree.SequenceEntry(0, deque, optree.PyTreeKind.DEQUE),)),
        optree.PyTreeAccessor((optree.SequenceEntry(1, deque, optree.PyTreeKind.DEQUE),)),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('c', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('b', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor((optree.MappingEntry('baz', MyDict, optree.PyTreeKind.CUSTOM),)),
    ],
    [optree.PyTreeAccessor()],
    [
        optree.PyTreeAccessor(
            (optree.NamedTupleEntry(0, CustomNamedTupleSubclass, optree.PyTreeKind.NAMEDTUPLE),),
        ),
        optree.PyTreeAccessor(
            (optree.NamedTupleEntry(1, CustomNamedTupleSubclass, optree.PyTreeKind.NAMEDTUPLE),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.DataclassEntry('alpha', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('beta', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('gamma', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry('delta', MyDataclass, optree.PyTreeKind.CUSTOM),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (optree.GetAttrEntry('a', MyOtherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (optree.GetAttrEntry('c', MyOtherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
    ],
    [
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('alpha', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('beta', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('gamma', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(0, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.DataclassEntry('delta', MyDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (optree.DataclassEntry(1, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(2, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.GetAttrEntry('a', MyOtherDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.DataclassEntry(2, MyAnotherDataclass, optree.PyTreeKind.CUSTOM),
                optree.GetAttrEntry('c', MyOtherDataclass, optree.PyTreeKind.CUSTOM),
            ),
        ),
    ],
    [],
    [optree.PyTreeAccessor((optree.GetItemEntry(0, FlatCache, optree.PyTreeKind.CUSTOM),))],
    [
        optree.PyTreeAccessor((optree.GetItemEntry(0, FlatCache, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor((optree.GetItemEntry(1, FlatCache, optree.PyTreeKind.CUSTOM),)),
    ],
]
TREE_STRINGS_NONE_IS_NODE = (
    'PyTreeSpec(*)',
    'PyTreeSpec(None)',
    'PyTreeSpec((None,))',
    'PyTreeSpec((*, None))',
    'PyTreeSpec(())',
    'PyTreeSpec([])',
    'PyTreeSpec([()])',
    'PyTreeSpec((*, *))',
    'PyTreeSpec(((*, *), [*, (*, None, *)]))',
    'PyTreeSpec([*])',
    'PyTreeSpec(EmptyTuple())',
    "PyTreeSpec([*, CustomTuple(foo=(*, CustomTuple(foo=*, bar=None)), bar={'baz': *})])",
    'PyTreeSpec(time.struct_time(tm_year=*, tm_mon=*, tm_mday=None, tm_hour=*, tm_min=*, tm_sec=*, tm_wday=*, tm_yday=*, tm_isdst=*))',
    'PyTreeSpec(sys.float_info(max=*, max_exp=*, max_10_exp=*, min=*, min_exp=*, min_10_exp=*, dig=*, mant_dig=*, epsilon=*, radix=None, rounds=time.struct_time(tm_year=*, tm_mon=*, tm_mday=*, tm_hour=*, tm_min=*, tm_sec=None, tm_wday=*, tm_yday=*, tm_isdst=*)))',
    "PyTreeSpec([CustomTreeNode(Vector3D[[4, 'foo']], [*, None])])",
    'PyTreeSpec(CustomTreeNode(Vector2D[None], [*, *]))',
    'PyTreeSpec({})',
    "PyTreeSpec({'a': *, 'b': *})",
    "PyTreeSpec({'a': *, 'b': (*, *), 'c': None, 'd': {'e': None, 'f': *}})",
    'PyTreeSpec(OrderedDict())',
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': *}))",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([None, *, *])}))",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([None, *], maxlen=2)}))",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([*, *], maxlen=2)}))",
    'PyTreeSpec(defaultdict(None, {}))',
    "PyTreeSpec(defaultdict(<class 'int'>, {}))",
    "PyTreeSpec(defaultdict(<class 'dict'>, {'baz': *, 'foo': *, 'something': *}))",
    'PyTreeSpec(deque())',
    'PyTreeSpec(deque(maxlen=0))',
    'PyTreeSpec(deque([None, *, *]))',
    'PyTreeSpec(deque([None, *], maxlen=2))',
    "PyTreeSpec(CustomTreeNode(MyDict[['foo', 'baz']], [CustomTreeNode(MyDict[['c', 'b', 'a']], [None, *, *]), *]))",
    'PyTreeSpec(*)',
    'PyTreeSpec(CustomNamedTupleSubclass(foo=*, bar=*))',
    'PyTreeSpec(CustomTreeNode(MyDataclass[None], [*, None, *, *]))',
    'PyTreeSpec(CustomTreeNode(MyOtherDataclass[(11, 13)], [*, None]))',
    'PyTreeSpec(CustomTreeNode(MyAnotherDataclass[None], [CustomTreeNode(MyDataclass[None], [*, *, None, *]), *, CustomTreeNode(MyOtherDataclass[(None, 19)], [*, *])]))',
    'PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec(None)], []))',
    'PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec(*)], [*]))',
    "PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec({'a': [*, *]})], [*, *]))",
)

TREE_STRINGS_NONE_IS_LEAF = (
    'PyTreeSpec(*, NoneIsLeaf)',
    'PyTreeSpec(*, NoneIsLeaf)',
    'PyTreeSpec((*,), NoneIsLeaf)',
    'PyTreeSpec((*, *), NoneIsLeaf)',
    'PyTreeSpec((), NoneIsLeaf)',
    'PyTreeSpec([], NoneIsLeaf)',
    'PyTreeSpec([()], NoneIsLeaf)',
    'PyTreeSpec((*, *), NoneIsLeaf)',
    'PyTreeSpec(((*, *), [*, (*, *, *)]), NoneIsLeaf)',
    'PyTreeSpec([*], NoneIsLeaf)',
    'PyTreeSpec(EmptyTuple(), NoneIsLeaf)',
    "PyTreeSpec([*, CustomTuple(foo=(*, CustomTuple(foo=*, bar=*)), bar={'baz': *})], NoneIsLeaf)",
    'PyTreeSpec(time.struct_time(tm_year=*, tm_mon=*, tm_mday=*, tm_hour=*, tm_min=*, tm_sec=*, tm_wday=*, tm_yday=*, tm_isdst=*), NoneIsLeaf)',
    'PyTreeSpec(sys.float_info(max=*, max_exp=*, max_10_exp=*, min=*, min_exp=*, min_10_exp=*, dig=*, mant_dig=*, epsilon=*, radix=*, rounds=time.struct_time(tm_year=*, tm_mon=*, tm_mday=*, tm_hour=*, tm_min=*, tm_sec=*, tm_wday=*, tm_yday=*, tm_isdst=*)), NoneIsLeaf)',
    "PyTreeSpec([CustomTreeNode(Vector3D[[4, 'foo']], [*, *])], NoneIsLeaf)",
    'PyTreeSpec(CustomTreeNode(Vector2D[None], [*, *]), NoneIsLeaf)',
    'PyTreeSpec({}, NoneIsLeaf)',
    "PyTreeSpec({'a': *, 'b': *}, NoneIsLeaf)",
    "PyTreeSpec({'a': *, 'b': (*, *), 'c': *, 'd': {'e': *, 'f': *}}, NoneIsLeaf)",
    'PyTreeSpec(OrderedDict(), NoneIsLeaf)',
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': *}), NoneIsLeaf)",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([*, *, *])}), NoneIsLeaf)",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([*, *], maxlen=2)}), NoneIsLeaf)",
    "PyTreeSpec(OrderedDict({'foo': *, 'baz': *, 'something': deque([*, *], maxlen=2)}), NoneIsLeaf)",
    'PyTreeSpec(defaultdict(None, {}), NoneIsLeaf)',
    "PyTreeSpec(defaultdict(<class 'int'>, {}), NoneIsLeaf)",
    "PyTreeSpec(defaultdict(<class 'dict'>, {'baz': *, 'foo': *, 'something': *}), NoneIsLeaf)",
    'PyTreeSpec(deque(), NoneIsLeaf)',
    'PyTreeSpec(deque(maxlen=0), NoneIsLeaf)',
    'PyTreeSpec(deque([*, *, *]), NoneIsLeaf)',
    'PyTreeSpec(deque([*, *], maxlen=2), NoneIsLeaf)',
    "PyTreeSpec(CustomTreeNode(MyDict[['foo', 'baz']], [CustomTreeNode(MyDict[['c', 'b', 'a']], [*, *, *]), *]), NoneIsLeaf)",
    'PyTreeSpec(*, NoneIsLeaf)',
    'PyTreeSpec(CustomNamedTupleSubclass(foo=*, bar=*), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(MyDataclass[None], [*, *, *, *]), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(MyOtherDataclass[(11, 13)], [*, *]), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(MyAnotherDataclass[None], [CustomTreeNode(MyDataclass[None], [*, *, *, *]), *, CustomTreeNode(MyOtherDataclass[(None, 19)], [*, *])]), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec(None)], []), NoneIsLeaf)',
    'PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec(*)], [*]), NoneIsLeaf)',
    "PyTreeSpec(CustomTreeNode(FlatCache[PyTreeSpec({'a': [*, *]})], [*, *]), NoneIsLeaf)",
)

if sys.version_info >= (3, 15) and OPTREE_HAS_FROZENDICT:
    from builtins import frozendict  # type: ignore[import]

    TREES = (  # type: ignore[no-redef]
        *TREES,
        frozendict(),
        frozendict({'a': 1, 'b': 2}),
        frozendict({'baz': 101, 'foo': -42, 'something': 7}),
    )
    TREE_PATHS_NONE_IS_NODE = [
        *TREE_PATHS_NONE_IS_NODE,
        [],
        [('a',), ('b',)],
        [('baz',), ('foo',), ('something',)],
    ]
    TREE_PATHS_NONE_IS_LEAF = [
        *TREE_PATHS_NONE_IS_LEAF,
        [],
        [('a',), ('b',)],
        [('baz',), ('foo',), ('something',)],
    ]
    TREE_ACCESSORS_NONE_IS_NODE = [
        *TREE_ACCESSORS_NONE_IS_NODE,
        [],
        [
            optree.PyTreeAccessor(
                (optree.MappingEntry('a', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
            optree.PyTreeAccessor(
                (optree.MappingEntry('b', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
        ],
        [
            optree.PyTreeAccessor(
                (optree.MappingEntry('baz', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
            optree.PyTreeAccessor(
                (optree.MappingEntry('foo', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
            optree.PyTreeAccessor(
                (optree.MappingEntry('something', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
        ],
    ]
    TREE_ACCESSORS_NONE_IS_LEAF = [
        *TREE_ACCESSORS_NONE_IS_LEAF,
        [],
        [
            optree.PyTreeAccessor(
                (optree.MappingEntry('a', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
            optree.PyTreeAccessor(
                (optree.MappingEntry('b', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
        ],
        [
            optree.PyTreeAccessor(
                (optree.MappingEntry('baz', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
            optree.PyTreeAccessor(
                (optree.MappingEntry('foo', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
            optree.PyTreeAccessor(
                (optree.MappingEntry('something', frozendict, optree.PyTreeKind.FROZENDICT),),
            ),
        ],
    ]
    TREE_STRINGS_NONE_IS_NODE = (
        *TREE_STRINGS_NONE_IS_NODE,
        'PyTreeSpec(frozendict())',
        "PyTreeSpec(frozendict({'a': *, 'b': *}))",
        "PyTreeSpec(frozendict({'baz': *, 'foo': *, 'something': *}))",
    )
    TREE_STRINGS_NONE_IS_LEAF = (
        *TREE_STRINGS_NONE_IS_LEAF,
        'PyTreeSpec(frozendict(), NoneIsLeaf)',
        "PyTreeSpec(frozendict({'a': *, 'b': *}), NoneIsLeaf)",
        "PyTreeSpec(frozendict({'baz': *, 'foo': *, 'something': *}), NoneIsLeaf)",
    )


TREE_PATHS = {
    optree.NONE_IS_NODE: TREE_PATHS_NONE_IS_NODE,
    optree.NONE_IS_LEAF: TREE_PATHS_NONE_IS_LEAF,
}
TREE_ACCESSORS = {
    optree.NONE_IS_NODE: TREE_ACCESSORS_NONE_IS_NODE,
    optree.NONE_IS_LEAF: TREE_ACCESSORS_NONE_IS_LEAF,
}
TREE_STRINGS = {
    optree.NONE_IS_NODE: TREE_STRINGS_NONE_IS_NODE,
    optree.NONE_IS_LEAF: TREE_STRINGS_NONE_IS_LEAF,
}

assert all(
    len(table[none_is_leaf]) == len(TREES)
    for table in (TREE_PATHS, TREE_ACCESSORS, TREE_STRINGS)
    for none_is_leaf in (optree.NONE_IS_NODE, optree.NONE_IS_LEAF)
)


LEAVES = (
    'foo',
    0.1,
    1,
    object(),
)
