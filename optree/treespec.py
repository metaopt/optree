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
"""The :mod:`optree.treespec` namespace contains constructors for ``TreeSpec`` class.

>>> import optree.treespec as ts
>>> ts.leaf()
PyTreeSpec(*)
>>> ts.none()
PyTreeSpec(None)
>>> ts.dict({'a': ts.leaf(), 'b': ts.leaf()})
PyTreeSpec({'a': *, 'b': *})

.. versionadded:: 0.14.1
"""

# pylint: disable=too-many-lines

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Collection, Iterable, Mapping

import optree._C as _C
import optree.ops as ops


if TYPE_CHECKING:
    # NOTE: Avoid conflict with `tuple` function in the current module.
    #       Linters are smart enough to derive the alias `_tuple` to `tuple` correctly.
    from builtins import tuple as _tuple

    from optree.typing import NamedTuple, PyTreeSpec
    from optree.typing import structseq as StructSequence  # noqa: N812


__all__ = [
    'leaf',
    'none',
    'tuple',
    'list',
    'dict',
    'namedtuple',
    'ordereddict',
    'defaultdict',
    'deque',
    'structseq',
    'from_collection',
]


def leaf(
    *,
    none_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec:
    """Make a treespec representing a leaf node.

    See also :func:`pytree.structure`, :func:`treespec.none`, and :func:`treespec.tuple`.

    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> treespec.leaf()
    PyTreeSpec(*)
    >>> treespec.leaf(none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec.leaf(none_is_leaf=False) == treespec.leaf(none_is_leaf=True)
    False
    >>> treespec.leaf() == pytree.structure(1)
    True
    >>> treespec.leaf(none_is_leaf=True) == pytree.structure(1, none_is_leaf=True)
    True
    >>> treespec.leaf(none_is_leaf=True) == pytree.structure(None, none_is_leaf=True)
    True
    >>> treespec.leaf(none_is_leaf=True) == pytree.structure(None, none_is_leaf=False)
    False
    >>> treespec.leaf(none_is_leaf=True) == treespec.none(none_is_leaf=True)
    True
    >>> treespec.leaf(none_is_leaf=True) == treespec.none(none_is_leaf=False)
    False
    >>> treespec.leaf(none_is_leaf=False) == treespec.none(none_is_leaf=True)
    False
    >>> treespec.leaf(none_is_leaf=False) == treespec.none(none_is_leaf=False)
    False

    Args:
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a leaf node.

    .. versionadded:: 0.14.1
    """
    return _C.make_leaf(
        none_is_leaf,
        namespace,  # unused
    )


def none(
    *,
    none_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec:
    """Make a treespec representing a :data:`None` node.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.tuple`.

    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> treespec.none()
    PyTreeSpec(None)
    >>> treespec.none(none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec.none(none_is_leaf=False) == treespec.none(none_is_leaf=True)
    False
    >>> treespec.none() == pytree.structure(None)
    True
    >>> treespec.none() == pytree.structure(1)
    False
    >>> treespec.none(none_is_leaf=True) == pytree.structure(1, none_is_leaf=True)
    True
    >>> treespec.none(none_is_leaf=True) == pytree.structure(None, none_is_leaf=True)
    True
    >>> treespec.none(none_is_leaf=True) == pytree.structure(None, none_is_leaf=False)
    False
    >>> treespec.none(none_is_leaf=True) == treespec.leaf(none_is_leaf=True)
    True
    >>> treespec.none(none_is_leaf=False) == treespec.leaf(none_is_leaf=True)
    False
    >>> treespec.none(none_is_leaf=True) == treespec.leaf(none_is_leaf=False)
    False
    >>> treespec.none(none_is_leaf=False) == treespec.leaf(none_is_leaf=False)
    False

    Args:
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a :data:`None` node.

    .. versionadded:: 0.14.1
    """
    return _C.make_none(
        none_is_leaf,
        namespace,  # unused
    )


def tuple(
    iterable: Iterable[PyTreeSpec] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a tuple treespec from an iterable of child treespecs.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.none`.

    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> treespec.tuple([treespec.leaf(), treespec.leaf()])
    PyTreeSpec((*, *))
    >>> treespec.tuple([treespec.leaf(), treespec.leaf(), treespec.none()])
    PyTreeSpec((*, *, None))
    >>> treespec.tuple()
    PyTreeSpec(())
    >>> treespec.tuple([treespec.leaf(), treespec.tuple([treespec.leaf(), treespec.leaf()])])
    PyTreeSpec((*, (*, *)))
    >>> treespec.tuple([treespec.leaf(), pytree.structure({'a': 1, 'b': 2})])
    PyTreeSpec((*, {'a': *, 'b': *}))
    >>> treespec.tuple([treespec.leaf(), pytree.structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a tuple node with the given children.

    .. versionadded:: 0.14.1
    """
    return ops.treespec_tuple(
        iterable,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def list(
    iterable: Iterable[PyTreeSpec] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a list treespec from an iterable of child treespecs.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.none`.

    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> treespec.list([treespec.leaf(), treespec.leaf()])
    PyTreeSpec([*, *])
    >>> treespec.list([treespec.leaf(), treespec.leaf(), treespec.none()])
    PyTreeSpec([*, *, None])
    >>> treespec.list()
    PyTreeSpec([])
    >>> treespec.list([treespec.leaf(), treespec.tuple([treespec.leaf(), treespec.leaf()])])
    PyTreeSpec([*, (*, *)])
    >>> treespec.list([treespec.leaf(), pytree.structure({'a': 1, 'b': 2})])
    PyTreeSpec([*, {'a': *, 'b': *}])
    >>> treespec.list([treespec.leaf(), pytree.structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a list node with the given children.

    .. versionadded:: 0.14.1
    """
    return ops.treespec_list(
        iterable,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def dict(
    mapping: Mapping[Any, PyTreeSpec] | Iterable[_tuple[Any, PyTreeSpec]] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    """Make a dict treespec from a dict of child treespecs.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.none`.

    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> treespec.dict({'a': treespec.leaf(), 'b': treespec.leaf()})
    PyTreeSpec({'a': *, 'b': *})
    >>> treespec.dict([('b', treespec.leaf()), ('c', treespec.leaf()), ('a', treespec.none())])
    PyTreeSpec({'a': None, 'b': *, 'c': *})
    >>> treespec.dict()
    PyTreeSpec({})
    >>> treespec.dict(a=treespec.leaf(), b=treespec.tuple([treespec.leaf(), treespec.leaf()]))
    PyTreeSpec({'a': *, 'b': (*, *)})
    >>> treespec.dict({'a': treespec.leaf(), 'b': pytree.structure([1, 2])})
    PyTreeSpec({'a': *, 'b': [*, *]})
    >>> treespec.dict({'a': treespec.leaf(), 'b': pytree.structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)
        **kwargs (PyTreeSpec, optional): Additional child treespecs to add to the mapping.

    Returns:
        A treespec representing a dict node with the given children.

    .. versionadded:: 0.14.1
    """
    return ops.treespec_dict(
        mapping,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
        **kwargs,
    )


def namedtuple(
    namedtuple: NamedTuple[PyTreeSpec],  # type: ignore[type-arg]
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a namedtuple treespec from a namedtuple of child treespecs.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.none`.

    >>> from collections import namedtuple
    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> treespec.namedtuple(Point(x=treespec.leaf(), y=treespec.leaf()))
    PyTreeSpec(Point(x=*, y=*))
    >>> treespec.namedtuple(Point(x=treespec.leaf(), y=treespec.tuple([treespec.leaf(), treespec.leaf()])))
    PyTreeSpec(Point(x=*, y=(*, *)))
    >>> treespec.namedtuple(Point(x=treespec.leaf(), y=pytree.structure([1, 2])))
    PyTreeSpec(Point(x=*, y=[*, *]))
    >>> treespec.namedtuple(Point(x=treespec.leaf(), y=pytree.structure([1, 2], none_is_leaf=True)))
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        namedtuple (namedtuple of PyTreeSpec): A namedtuple of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a dict node with the given children.

    .. versionadded:: 0.14.1
    """
    return ops.treespec_namedtuple(
        namedtuple,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def ordereddict(
    mapping: Mapping[Any, PyTreeSpec] | Iterable[_tuple[Any, PyTreeSpec]] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    """Make an OrderedDict treespec from an OrderedDict of child treespecs.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.none`.

    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> treespec.ordereddict({'a': treespec.leaf(), 'b': treespec.leaf()})
    PyTreeSpec(OrderedDict({'a': *, 'b': *}))
    >>> treespec.ordereddict([('b', treespec.leaf()), ('c', treespec.leaf()), ('a', treespec.none())])
    PyTreeSpec(OrderedDict({'b': *, 'c': *, 'a': None}))
    >>> treespec.ordereddict()
    PyTreeSpec(OrderedDict())
    >>> treespec.ordereddict(a=treespec.leaf(), b=treespec.tuple([treespec.leaf(), treespec.leaf()]))
    PyTreeSpec(OrderedDict({'a': *, 'b': (*, *)}))
    >>> treespec.ordereddict({'a': treespec.leaf(), 'b': pytree.structure([1, 2])})
    PyTreeSpec(OrderedDict({'a': *, 'b': [*, *]}))
    >>> treespec.ordereddict({'a': treespec.leaf(), 'b': pytree.structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)
        **kwargs (PyTreeSpec, optional): Additional child treespecs to add to the mapping.

    Returns:
        A treespec representing an OrderedDict node with the given children.

    .. versionadded:: 0.14.1
    """
    return ops.treespec_ordereddict(
        mapping,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
        **kwargs,
    )


def defaultdict(
    default_factory: Callable[[], Any] | None = None,
    mapping: Mapping[Any, PyTreeSpec] | Iterable[_tuple[Any, PyTreeSpec]] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    """Make a defaultdict treespec from a defaultdict of child treespecs.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.none`.

    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> treespec.defaultdict(int, {'a': treespec.leaf(), 'b': treespec.leaf()})
    PyTreeSpec(defaultdict(<class 'int'>, {'a': *, 'b': *}))
    >>> treespec.defaultdict(int, [('b', treespec.leaf()), ('c', treespec.leaf()), ('a', treespec.none())])
    PyTreeSpec(defaultdict(<class 'int'>, {'a': None, 'b': *, 'c': *}))
    >>> treespec.defaultdict()
    PyTreeSpec(defaultdict(None, {}))
    >>> treespec.defaultdict(int)
    PyTreeSpec(defaultdict(<class 'int'>, {}))
    >>> treespec.defaultdict(int, a=treespec.leaf(), b=treespec.tuple([treespec.leaf(), treespec.leaf()]))
    PyTreeSpec(defaultdict(<class 'int'>, {'a': *, 'b': (*, *)}))
    >>> treespec.defaultdict(int, {'a': treespec.leaf(), 'b': pytree.structure([1, 2])})
    PyTreeSpec(defaultdict(<class 'int'>, {'a': *, 'b': [*, *]}))
    >>> treespec.defaultdict(int, {'a': treespec.leaf(), 'b': pytree.structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        default_factory (callable or None, optional): A factory function that will be used to create
            a missing value. (default: :data:`None`)
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)
        **kwargs (PyTreeSpec, optional): Additional child treespecs to add to the mapping.

    Returns:
        A treespec representing a defaultdict node with the given children.

    .. versionadded:: 0.14.1
    """
    return ops.treespec_defaultdict(
        default_factory,
        mapping,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
        **kwargs,
    )


def deque(
    iterable: Iterable[PyTreeSpec] = (),
    /,
    maxlen: int | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a deque treespec from a deque of child treespecs.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.none`.

    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> treespec.deque([treespec.leaf(), treespec.leaf()])
    PyTreeSpec(deque([*, *]))
    >>> treespec.deque([treespec.leaf(), treespec.leaf(), treespec.none()], maxlen=5)
    PyTreeSpec(deque([*, *, None], maxlen=5))
    >>> treespec.deque()
    PyTreeSpec(deque([]))
    >>> treespec.deque([treespec.leaf(), treespec.tuple([treespec.leaf(), treespec.leaf()])])
    PyTreeSpec(deque([*, (*, *)]))
    >>> treespec.deque([treespec.leaf(), pytree.structure({'a': 1, 'b': 2})], maxlen=5)
    PyTreeSpec(deque([*, {'a': *, 'b': *}], maxlen=5))
    >>> treespec.deque([treespec.leaf(), pytree.structure({'a': 1, 'b': 2}, none_is_leaf=True)], maxlen=5)
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        maxlen (int or None, optional): The maximum size of a deque or :data:`None` if unbounded.
            (default: :data:`None`)
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a deque node with the given children.

    .. versionadded:: 0.14.1
    """
    return ops.treespec_deque(
        iterable,
        maxlen=maxlen,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def structseq(
    structseq: StructSequence[PyTreeSpec],
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a PyStructSequence treespec from a PyStructSequence of child treespecs.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.none`.

    Args:
        structseq (PyStructSequence of PyTreeSpec): A PyStructSequence of child treespecs. They must
            have the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a PyStructSequence node with the given children.

    .. versionadded:: 0.14.1
    """
    return ops.treespec_structseq(
        structseq,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def from_collection(
    collection: Collection[PyTreeSpec],
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a treespec from a collection of child treespecs.

    See also :func:`pytree.structure`, :func:`treespec.leaf`, and :func:`treespec.none`.

    >>> from collections import deque
    >>> import optree.pytree as pytree
    >>> import optree.treespec as treespec
    >>> treespec.from_collection(None)
    PyTreeSpec(None)
    >>> treespec.from_collection(None, none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec.from_collection(object())
    PyTreeSpec(*)
    >>> treespec.from_collection([treespec.leaf(), treespec.none()])
    PyTreeSpec([*, None])
    >>> treespec.from_collection({'a': treespec.leaf(), 'b': treespec.none()})
    PyTreeSpec({'a': *, 'b': None})
    >>> treespec.from_collection(deque([treespec.leaf(), pytree.structure({'a': 1, 'b': 2})], maxlen=5))
    PyTreeSpec(deque([*, {'a': *, 'b': *}], maxlen=5))
    >>> treespec.from_collection({'a': treespec.leaf(), 'b': (treespec.leaf(), treespec.none())})
    Traceback (most recent call last):
        ...
    ValueError: Expected a(n) dict of PyTreeSpec(s), got {'a': PyTreeSpec(*), 'b': (PyTreeSpec(*), PyTreeSpec(None))}.
    >>> treespec.from_collection([treespec.leaf(), pytree.structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.


    Args:
        collection (collection of PyTreeSpec): A collection of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing the same structure of the collection with the given children.

    .. versionadded:: 0.14.1
    """
    return _C.make_from_collection(collection, none_is_leaf, namespace)
