# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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
"""Access support for pytrees."""

from __future__ import annotations

import dataclasses
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)
from typing_extensions import Self  # Python 3.11+

from optree import _C


if TYPE_CHECKING:
    import builtins

    from optree.typing import PyTree, PyTreeKind


__all__ = [
    'PyTreeEntry',
    'GetItemEntry',
    'GetAttrEntry',
    'FlattenedEntry',
    'SequenceEntry',
    'MappingEntry',
    'NamedTupleEntry',
    'StructSequenceEntry',
    'DataclassEntry',
    'PyTreeAccessor',
]


SLOTS = {'slots': True} if sys.version_info >= (3, 10) else {}  # Python 3.10+


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True, **SLOTS)
class PyTreeEntry:
    """Base class for path entries."""

    entry: Any
    type: builtins.type
    kind: PyTreeKind

    def __post_init__(self) -> None:
        """Post-initialize the path entry."""
        from optree.typing import PyTreeKind  # pylint: disable=import-outside-toplevel

        if self.kind is PyTreeKind.LEAF:
            raise ValueError('Cannot create a leaf path entry.')
        if self.kind is PyTreeKind.NONE:
            raise ValueError('Cannot create a path entry for None.')

    def __call__(self, obj: Any) -> Any:
        """Get the child object."""
        try:
            return obj[self.entry]  # should be overridden
        except TypeError as ex:
            raise TypeError(
                f'{self.__class__!r} cannot access through {obj!r} via entry {self.entry!r}',
            ) from ex

    def __add__(self, other: object) -> PyTreeAccessor:
        """Join the path entry with another path entry or accessor."""
        if isinstance(other, PyTreeEntry):
            return PyTreeAccessor((self, other))
        if isinstance(other, PyTreeAccessor):
            return PyTreeAccessor((self, *other))
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Check if the path entries are equal."""
        return (
            isinstance(other, PyTreeEntry)
            and self.entry == other.entry
            and self.type == other.type
            and self.kind == other.kind
            and self.__class__.__call__ is other.__class__.__call__
        )

    def __hash__(self) -> int:
        """Get the hash of the path entry."""
        return hash((self.__class__, self.entry, self.type, self.kind))

    def __repr__(self) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(entry={self.entry!r}, type={self.type!r})'

    def pprint(self, root: str = '') -> str:
        """Pretty name of the path entry."""
        return f'{root}[<flat index {self.entry!r}>]'  # should be overridden


del SLOTS


_T_co = TypeVar('_T_co', covariant=True)
_KT_co = TypeVar('_KT_co', covariant=True)
_VT_co = TypeVar('_VT_co', covariant=True)


class GetItemEntry(PyTreeEntry):
    """A generic path entry class for nodes that access their children by :meth:`__getitem__`."""

    def __call__(self, obj: Any) -> Any:
        """Get the child object."""
        return obj[self.entry]

    def pprint(self, root: str = '') -> str:
        """Pretty name of the path entry."""
        return f'{root}[{self.entry!r}]'


class GetAttrEntry(PyTreeEntry):
    """A generic path entry class for nodes that access their children by :meth:`__getattr__`."""

    entry: str

    def __call__(self, obj: Any) -> Any:
        """Get the child object."""
        return getattr(obj, self.entry)

    def pprint(self, root: str = '') -> str:
        """Pretty name of the path entry."""
        return f'{root}.{self.entry}'


class FlattenedEntry(PyTreeEntry):  # pylint: disable=too-few-public-methods
    """A fallback path entry class for flattened objects."""


class SequenceEntry(GetItemEntry, Generic[_T_co]):
    """A path entry class for sequences."""

    entry: int

    @property
    def index(self) -> int:
        """Get the index."""
        return self.entry

    def __call__(self, obj: Sequence[_T_co]) -> _T_co:
        """Get the child object."""
        return obj[self.index]

    def __repr__(self) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(index={self.index!r}, type={self.type!r})'


class MappingEntry(GetItemEntry, Generic[_KT_co, _VT_co]):
    """A path entry class for mappings."""

    entry: _KT_co

    @property
    def key(self) -> _KT_co:
        """Get the key."""
        return self.entry

    def __call__(self, obj: Mapping[_KT_co, _VT_co]) -> _VT_co:
        """Get the child object."""
        return obj[self.key]

    def __repr__(self) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(key={self.key!r}, type={self.type!r})'


class NamedTupleEntry(SequenceEntry):
    """A path entry class for namedtuple objects."""

    entry: int

    @property
    def fields(self) -> tuple[str, ...]:
        """Get the field names."""
        from optree.typing import namedtuple_fields  # pylint: disable=import-outside-toplevel

        return namedtuple_fields(self.type)

    @property
    def field(self) -> str:
        """Get the field name."""
        return self.fields[self.entry]

    def __repr__(self) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(field={self.field!r}, type={self.type!r})'

    def pprint(self, root: str = '') -> str:
        """Pretty name of the path entry."""
        return f'{root}.{self.field}'


class StructSequenceEntry(NamedTupleEntry):
    """A path entry class for PyStructSequence objects."""

    entry: int

    @property
    def fields(self) -> tuple[str, ...]:
        """Get the field names."""
        from optree.typing import structseq_fields  # pylint: disable=import-outside-toplevel

        return structseq_fields(self.type)


class DataclassEntry(GetAttrEntry):
    """A path entry class for dataclasses."""

    entry: str

    @property
    def fields(self) -> tuple[str, ...]:
        """Get the field names."""
        return tuple(f.name for f in dataclasses.fields(self.type))

    @property
    def field(self) -> str:
        """Get the field name."""
        return self.entry

    def __repr__(self) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(field={self.entry!r}, type={self.type!r})'


class PyTreeAccessor(Tuple[PyTreeEntry, ...]):
    """A path class for PyTrees."""

    @property
    def path(self) -> tuple[Any, ...]:
        """Get the path of the accessor."""
        return tuple(e.entry for e in self)

    def __new__(cls, path: Iterable[PyTreeEntry] = ()) -> Self:
        """Create a new accessor instance."""
        if not isinstance(path, (list, tuple)):
            path = tuple(path)
        if not all(isinstance(p, PyTreeEntry) for p in path):
            raise TypeError(f'Expected a path of Entry, got {path!r}.')
        return super().__new__(cls, path)

    def __call__(self, obj: PyTree) -> Any:
        """Get the child object."""
        for entry in self:
            obj = entry(obj)
        return obj

    @overload  # type: ignore[override]
    def __getitem__(self, index: int) -> PyTreeEntry:  # noqa: D105,RUF100
        ...

    @overload
    def __getitem__(self, index: slice) -> PyTreeAccessor:  # noqa: D105,RUF100
        ...

    def __getitem__(self, index: int | slice) -> PyTreeEntry | PyTreeAccessor:
        """Get the child path entry or an accessor for a subpath."""
        if isinstance(index, slice):
            return PyTreeAccessor(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: object) -> PyTreeAccessor:
        """Join the accessor with another path entry or accessor."""
        if isinstance(other, PyTreeEntry):
            return PyTreeAccessor((*self, other))
        if isinstance(other, PyTreeAccessor):
            return PyTreeAccessor((*self, *other))
        return NotImplemented

    def __mul__(self, value: int) -> PyTreeAccessor:  # type: ignore[override]
        """Repeat the accessor."""
        return PyTreeAccessor(super().__mul__(value))

    def __rmul__(self, value: int) -> PyTreeAccessor:  # type: ignore[override]
        """Repeat the accessor."""
        return PyTreeAccessor(super().__rmul__(value))

    def __eq__(self, other: object) -> bool:
        """Check if the accessors are equal."""
        return isinstance(other, PyTreeAccessor) and super().__eq__(other)

    def __repr__(self) -> str:
        """Get the representation of the accessor."""
        return f'{self.__class__.__name__}({self.pprint()}, {super().__repr__()})'

    def __str__(self) -> str:
        """Get the string representation of the accessor."""
        return f'{self.__class__.__name__}({self.pprint()}'

    def pprint(self, root: str = '*') -> str:
        """Pretty name of the path."""
        string = root
        for entry in self:
            string = entry.pprint(string)
        return string


# These classes are used internally in the C++ side for accessor APIs
setattr(_C, 'PyTreeEntry', PyTreeEntry)  # noqa: B010
setattr(_C, 'GetItemEntry', GetItemEntry)  # noqa: B010
setattr(_C, 'GetAttrEntry', GetAttrEntry)  # noqa: B010
setattr(_C, 'FlattenedEntry', FlattenedEntry)  # noqa: B010
setattr(_C, 'SequenceEntry', SequenceEntry)  # noqa: B010
setattr(_C, 'MappingEntry', MappingEntry)  # noqa: B010
setattr(_C, 'NamedTupleEntry', NamedTupleEntry)  # noqa: B010
setattr(_C, 'StructSequenceEntry', StructSequenceEntry)  # noqa: B010
setattr(_C, 'DataclassEntry', DataclassEntry)  # noqa: B010
setattr(_C, 'PyTreeAccessor', PyTreeAccessor)  # noqa: B010
