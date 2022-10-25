# Copyright 2022 MetaOPT Team. All Rights Reserved.
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
"""OpTree: Optimized PyTree Utilities."""

import functools
from collections import OrderedDict, defaultdict, deque
from operator import methodcaller
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Type

import optree._C as _C
from optree.typing import KT, VT, Children, CustomTreeNode, DefaultDict, MetaData
from optree.typing import OrderedDict as GenericOrderedDict
from optree.typing import PyTree, PyTreeSpec, T
from optree.utils import safe_zip, unzip2


__all__ = [
    'register_pytree_node',
    'register_pytree_node_class',
    'Partial',
    'register_keypaths',
    'AttributeKeyPathEntry',
    'GetitemKeyPathEntry',
]


PyTreeNodeRegistryEntry = NamedTuple(
    'PyTreeNodeRegistryEntry',
    [
        ('to_iter', Callable[[CustomTreeNode[T]], Tuple[Children[T], MetaData]]),
        ('from_iter', Callable[[MetaData, Children[T]], CustomTreeNode[T]]),
    ],
)


def register_pytree_node(
    type: Type[CustomTreeNode[T]],  # pylint: disable=redefined-builtin
    flatten_func: Callable[[CustomTreeNode[T]], Tuple[Children[T], MetaData]],
    unflatten_func: Callable[[MetaData, Children[T]], CustomTreeNode[T]],
) -> Type[CustomTreeNode[T]]:
    """Extends the set of types that are considered internal nodes in pytrees.

    Args:
        type: A Python type to treat as an internal pytree node.
        flatten_func: A function to be used during flattening, taking a value of type ``type`` and
            returning a pair, with (1) an iterable for the children to be flattened recursively, and
            (2) some hashable auxiliary data to be stored in the treespec and to be passed to the
            ``unflatten_func``.
        unflatten_func: A function taking two arguments: the auxiliary data that was returned by
            ``flatten_func`` and stored in the treespec, and the unflattened children. The function
            should return an instance of ``type``.
    """
    _C.register_node(type, flatten_func, unflatten_func)
    CustomTreeNode.register(type)  # pylint: disable=no-member
    _nodetype_registry[type] = PyTreeNodeRegistryEntry(flatten_func, unflatten_func)
    return type


def register_pytree_node_class(cls: Type[CustomTreeNode[T]]) -> Type[CustomTreeNode[T]]:
    """Extends the set of types that are considered internal nodes in pytrees.

    This function is a thin wrapper around :func:`register_pytree_node`, and provides a
    class-oriented interface::

        @register_pytree_node_class
        class Special:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def tree_flatten(self):
                return ((self.x, self.y), None)

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(*children)
    """
    register_pytree_node(cls, methodcaller('tree_flatten'), cls.tree_unflatten)
    return cls


def _sorted_items(items: Iterable[Tuple[KT, VT]]) -> Iterable[Tuple[KT, VT]]:  # pragma: no cover
    try:
        # Sort directly if possible (do not use `key` for performance reasons)
        return sorted(items)
    except TypeError:  # the keys are not comparable
        try:
            # Add `{obj.__class__.__module__}.{obj.__class__.__qualname__}` to the key order to make
            # it sortable between different types (e.g. `int` vs. `str`)
            return sorted(
                items,
                # pylint: disable-next=consider-using-f-string
                key=lambda kv: ('{0.__module__}.{0.__qualname__}'.format(kv[0].__class__), kv),
            )
        except TypeError:  # cannot sort the keys (e.g. user-defined types)
            return items  # fallback to insertion order


def _sorted_keys(dct: Dict[KT, VT]) -> Iterable[KT]:  # pragma: no cover
    try:
        # Sort directly if possible (do not use `key` for performance reasons)
        return sorted(dct)  # type: ignore[type-var]
    except TypeError:  # the keys are not comparable
        try:
            # Add `{obj.__class__.__module__}.{obj.__class__.__qualname__}` to the key order to make
            # it sortable between different types (e.g. `int` vs. `str`)
            return sorted(
                dct,
                # pylint: disable-next=consider-using-f-string
                key=lambda o: ('{0.__module__}.{0.__qualname__}'.format(o.__class__), o),
            )
        except TypeError:  # cannot sort the keys (e.g. user-defined types)
            return dct  # fallback to insertion order


def _dict_flatten(dct: Dict[KT, VT]) -> Tuple[Tuple[VT, ...], Tuple[KT, ...]]:  # pragma: no cover
    keys, values = unzip2(_sorted_items(dct.items()))
    return values, keys


def _ordereddict_flatten(
    dct: GenericOrderedDict[KT, VT]
) -> Tuple[Tuple[VT, ...], Tuple[KT, ...]]:  # pragma: no cover
    keys, values = unzip2(dct.items())
    return values, keys


def _defaultdict_flatten(
    dct: DefaultDict[KT, VT]
) -> Tuple[Tuple[VT, ...], Tuple[Optional[Callable[[], VT]], Tuple[KT, ...]]]:  # pragma: no cover
    values, keys = _dict_flatten(dct)
    return values, (dct.default_factory, keys)


# pylint: disable=all
_nodetype_registry: Dict[Type, PyTreeNodeRegistryEntry] = {
    type(None): PyTreeNodeRegistryEntry(
        lambda n: ((), None),
        lambda _, n: None,  # type: ignore[arg-type,return-value]
    ),
    tuple: PyTreeNodeRegistryEntry(
        lambda t: (t, None),  # type: ignore[arg-type,return-value]
        lambda _, t: tuple(t),  # type: ignore[arg-type,return-value]
    ),
    list: PyTreeNodeRegistryEntry(
        lambda l: (l, None),  # type: ignore[arg-type,return-value]
        lambda _, l: list(l),  # type: ignore[arg-type,return-value]
    ),
    dict: PyTreeNodeRegistryEntry(
        _dict_flatten,  # type: ignore[arg-type]
        lambda keys, values: dict(safe_zip(keys, values)),  # type: ignore[arg-type,return-value]
    ),
    OrderedDict: PyTreeNodeRegistryEntry(
        _ordereddict_flatten,  # type: ignore[arg-type]
        lambda keys, values: OrderedDict(safe_zip(keys, values)),  # type: ignore[arg-type,return-value]
    ),
    defaultdict: PyTreeNodeRegistryEntry(
        _defaultdict_flatten,  # type: ignore[arg-type]
        lambda metadata, values: defaultdict(metadata[0], safe_zip(metadata[1], values)),  # type: ignore[arg-type,return-value,index]
    ),
    deque: PyTreeNodeRegistryEntry(
        lambda d: (list(d), d.maxlen),  # type: ignore[call-overload,attr-defined]
        lambda maxlen, d: deque(d, maxlen=maxlen),  # type: ignore[arg-type,return-value]
    ),
}
# pylint: enable=all


# pylint: disable-next=line-too-long
register_pytree_node.get: Callable[[Type[CustomTreeNode[T]]], PyTreeNodeRegistryEntry] = _nodetype_registry.get  # type: ignore[attr-defined,misc]


class _HashablePartialShim:
    """Object that delegates :meth:`__call__`, :meth:`__hash__`, and :meth:`__eq__` to another object."""

    func: Callable[..., Any]
    args: Tuple[Any, ...]
    keywords: Dict[str, Any]

    def __init__(self, partial_func: functools.partial) -> None:
        self.partial_func: functools.partial = partial_func

    def __call__(self, *args, **kwargs) -> Any:
        return self.partial_func(*args, **kwargs)

    def __hash__(self) -> int:
        return hash(self.partial_func)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _HashablePartialShim):
            return self.partial_func == other.partial_func
        return self.partial_func == other


@register_pytree_node_class
class Partial(functools.partial, CustomTreeNode[Any]):  # pylint: disable=too-few-public-methods
    """A version of :func:`functools.partial` that works in pytrees.

    Use it for partial function evaluation in a way that is compatible with JAX's transformations,
    e.g., ``Partial(func, *args, **kwargs)``.

    (You need to explicitly opt-in to this behavior because we did not want to give
    :func:`functools.partial` different semantics than normal function closures.)

    For example, here is a basic usage of :class:`Partial` in a manner similar to
    :func:`functools.partial`:

    >>> import jax.numpy as jnp
    >>> add_one = Partial(jnp.add, 1)
    >>> add_one(2)
    DeviceArray(3, dtype=int32, weak_type=True)

    Pytree compatibility means that the resulting partial function can be passed as an argument
    within transformed JAX functions, which is not possible with a standard :func:`functools.partial`
    function:

    >>> from jax import jit
    >>> @jit
    ... def call_func(f, *args):
    ...   return f(*args)
    ...
    >>> call_func(add_one, 2)
    DeviceArray(3, dtype=int32, weak_type=True)

    Passing zero arguments to :class:`Partial` effectively wraps the original function, making it a
    valid argument in JAX transformed functions:

    >>> call_func(Partial(jnp.add), 1, 2)
    DeviceArray(3, dtype=int32, weak_type=True)

    Had we passed :func:`jnp.add` to ``call_func`` directly, it would have resulted in a
    :class:`TypeError`.

    Note that if the result of :class:`Partial` is used in the context where the value is traced, it
    results in all bound arguments being traced when passed to the partially-evaluated function:

    >>> print_zero = Partial(print, 0)
    >>> print_zero()
    0
    >>> call_func(print_zero)
    Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
    """

    func: Callable[..., Any]
    args: Tuple[Any, ...]
    keywords: Dict[str, Any]

    def __new__(cls, func: Callable[..., Any], *args, **keywords) -> 'Partial':
        """Creates a new :class:`Partial` instance."""
        # In Python 3.10+, if func is itself a functools.partial instance, functools.partial.__new__
        # would merge the arguments of this Partial instance with the arguments of the func. We box
        # func in a class that does not (yet) have a `func` attribute to defeat this optimization,
        # since we care exactly which arguments are considered part of the pytree.
        if isinstance(func, functools.partial):
            original_func = func
            func = _HashablePartialShim(original_func)
            assert not hasattr(func, 'func')
            out = super().__new__(cls, func, *args, **keywords)
            func.func = original_func.func
            func.args = original_func.args
            func.keywords = original_func.keywords
            return out

        return super().__new__(cls, func, *args, **keywords)

    def tree_flatten(self) -> Tuple[Tuple[Tuple[Any, ...], Dict[str, Any]], Callable[..., Any]]:
        """Flattens the :class:`Partial` instance to children and auxiliary data."""
        return (self.args, self.keywords), self.func

    @classmethod
    def tree_unflatten(  # type: ignore[override] # pylint: disable=arguments-renamed
        cls,
        func: Callable[..., Any],
        args: Tuple[Tuple[Any, ...], Dict[str, Any]],
    ) -> 'Partial':
        """Unflattens the children and auxiliary data into a :class:`Partial` instance."""
        return cls(func, *args[0], **args[1])


class KeyPathEntry(NamedTuple):
    key: Any

    def pprint(self) -> str:
        """Pretty name of the key path entry."""
        return NotImplemented


class KeyPath(NamedTuple):
    keys: Tuple[KeyPathEntry, ...] = ()

    def __add__(self, other: object) -> 'KeyPath':
        if isinstance(other, KeyPathEntry):
            return KeyPath(self.keys + (other,))
        raise TypeError(type(other))

    def pprint(self) -> str:
        """Pretty name of the key path."""
        if not self.keys:
            return ' tree root'
        return ''.join(k.pprint() for k in self.keys)


class GetitemKeyPathEntry(KeyPathEntry):
    """The key path entry class for sequences and dictionaries."""

    def pprint(self) -> str:
        """Pretty name of the key path entry."""
        return f'[{repr(self.key)}]'


class AttributeKeyPathEntry(KeyPathEntry):
    """The key path entry class for namedtuples."""

    def pprint(self) -> str:
        """Pretty name of the key path entry."""
        return f'.{self.key}'


class FlattenedKeyPathEntry(KeyPathEntry):  # fallback
    """The fallback key path entry class."""

    def pprint(self) -> str:
        """Pretty name of the key path entry."""
        return f'[<flat index {self.key}>]'


KeyPathHandler = Callable[[PyTree], List[KeyPathEntry]]
_keypath_registry: Dict[Type[CustomTreeNode], KeyPathHandler] = {}


def register_keypaths(
    type: Type[CustomTreeNode[T]],  # pylint: disable=redefined-builtin
    handler: KeyPathHandler,
) -> KeyPathHandler:
    """Registers a key path handler for a custom pytree node type."""
    _keypath_registry[type] = handler
    return handler


register_keypaths(tuple, lambda tup: list(map(GetitemKeyPathEntry, range(len(tup)))))  # type: ignore[arg-type]
register_keypaths(list, lambda lst: list(map(GetitemKeyPathEntry, range(len(lst)))))  # type: ignore[arg-type]
register_keypaths(dict, lambda dct: list(map(GetitemKeyPathEntry, _sorted_keys(dct))))  # type: ignore[arg-type]

# pylint: disable-next=line-too-long
register_keypaths.get: Callable[[Type[CustomTreeNode[T]]], KeyPathHandler] = _keypath_registry.get  # type: ignore[attr-defined,misc]
