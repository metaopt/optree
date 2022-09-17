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
"""OpTree: Optimized PyTree."""

# pylint: disable=missing-class-docstring,missing-function-docstring

import collections
import functools
from operator import methodcaller
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Tuple, Type

import optree._C as _C
from optree.typing import KT, VT, AuxData, Children, CustomTreeNode, PyTree, PyTreeDef, T
from optree.utils import safe_zip, unzip2


__all__ = [
    'AttributeKeyPathEntry',
    'GetitemKeyPathEntry',
    'Partial',
    'PyTreeDef',
    'register_keypaths',
    'register_pytree_node',
    'register_pytree_node_class',
]


PyTreeNodeRegistryEntry = NamedTuple(
    'PyTreeNodeRegistryEntry',
    [
        ('to_iter', Callable[[CustomTreeNode[T]], Tuple[Children[T], AuxData]]),
        ('from_iter', Callable[[AuxData, Children[T]], CustomTreeNode[T]]),
    ],
)


def register_pytree_node(
    nodetype: Type[CustomTreeNode[T]],
    flatten_func: Callable[[CustomTreeNode[T]], Tuple[Children[T], AuxData]],
    unflatten_func: Callable[[AuxData, Children[T]], CustomTreeNode[T]],
) -> Type[CustomTreeNode[T]]:
    """Extend the set of types that are considered internal nodes in pytrees.

    Args:
        nodetype: a Python type to treat as an internal pytree node.
        flatten_func: a function to be used during flattening, taking a value of
            type ``nodetype`` and returning a pair, with (1) an iterable for the
            children to be flattened recursively, and (2) some hashable auxiliary
            data to be stored in the treedef and to be passed to the
            ``unflatten_func``.
        unflatten_func: a function taking two arguments: the auxiliary data that
            was returned by ``flatten_func`` and stored in the treedef, and the
            unflattened children. The function should return an instance of
            ``nodetype``.
    """
    _C.register_node(nodetype, flatten_func, unflatten_func)
    CustomTreeNode.register(nodetype)  # pylint: disable=no-member
    _nodetype_registry[nodetype] = PyTreeNodeRegistryEntry(flatten_func, unflatten_func)
    return nodetype


def register_pytree_node_class(cls: Type[CustomTreeNode[T]]) -> Type[CustomTreeNode[T]]:
    """Extend the set of types that are considered internal nodes in pytrees.

    This function is a thin wrapper around ``register_pytree_node``, and provides
    a class-oriented interface::

        @register_pytree_node_class
        class Special:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def tree_flatten(self):
                return ((self.x, self.y), None)
            @classmethod
            def tree_unflatten(cls, aux_data, children):
                return cls(*children)
    """
    register_pytree_node(cls, methodcaller('tree_flatten'), cls.tree_unflatten)
    return cls


def _sorted_items(items: Iterable[Tuple[KT, VT]]) -> Iterable[Tuple[KT, VT]]:
    try:
        # Sort directly if possible (do not use `key` for performance reasons)
        return sorted(items)
    except TypeError:  # the keys are not comparable
        try:
            # Add `obj.__class__.__qualname__` to the key order to make it sortable
            # between different types (e.g. `int` vs. `str`)
            return sorted(items, key=lambda kv: (kv[0].__class__.__qualname__, kv))
        except TypeError:  # cannot sort the keys (e.g. user-defined types)
            return items  # fallback to insertion order


def _sorted_keys(dct: Dict[KT, VT]) -> Iterable[KT]:
    try:
        # Sort directly if possible (do not use `key` for performance reasons)
        return sorted(dct)  # type: ignore[type-var]
    except TypeError:  # the keys are not comparable
        try:
            # Add `obj.__class__.__qualname__` to the key order to make it sortable
            # between different types (e.g. `int` vs. `str`)
            return sorted(dct, key=lambda o: (o.__class__.__qualname__, o))
        except TypeError:  # cannot sort the keys (e.g. user-defined types)
            return dct  # fallback to insertion order


# pylint: disable=line-too-long
_nodetype_registry: Dict[Type, PyTreeNodeRegistryEntry] = {
    tuple: PyTreeNodeRegistryEntry(
        lambda xs: (xs, None),  # type: ignore[arg-type,return-value]
        lambda _, xs: tuple(xs),  # type: ignore[arg-type,return-value]
    ),
    list: PyTreeNodeRegistryEntry(
        lambda xs: (xs, None),  # type: ignore[arg-type,return-value]
        lambda _, xs: list(xs),  # type: ignore[arg-type,return-value]
    ),
    dict: PyTreeNodeRegistryEntry(
        lambda d: unzip2(_sorted_items(d.items()))[::-1],  # type: ignore[attr-defined]
        lambda keys, values: dict(zip(keys, values)),  # type: ignore[arg-type,return-value]
    ),
    type(None): PyTreeNodeRegistryEntry(
        lambda xs: ((), None),
        lambda _, xs: None,  # type: ignore[arg-type,return-value]
    ),
}
register_pytree_node(
    collections.OrderedDict,  # type: ignore[arg-type]
    lambda od: (tuple(od.values()), tuple(od.keys())),  # type: ignore[attr-defined]
    lambda keys, values: collections.OrderedDict(safe_zip(keys, values)),  # type: ignore[arg-type,return-value]
)
register_pytree_node(
    collections.defaultdict,  # type: ignore[arg-type]
    lambda dd: (tuple(dd.values()), (dd.default_factory, tuple(dd.keys()))),  # type: ignore[attr-defined]
    lambda aux, values: collections.defaultdict(aux[0], safe_zip(aux[1], values)),  # type: ignore[arg-type,return-value,index]
)

register_pytree_node.get: Callable[[Type[CustomTreeNode[T]]], PyTreeNodeRegistryEntry] = _nodetype_registry.get  # type: ignore[attr-defined,misc]
# pylint: enable=line-too-long


class _HashableCallableShim:
    """Object that delegates __call__, __hash__, and __eq__ to another object."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func: Callable[..., Any] = func
        self.args: Tuple[Any, ...] = None  # type: ignore[assignment]
        self.keywords: Dict[str, Any] = None  # type: ignore[assignment]

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    def __hash__(self) -> int:
        return hash(self.func)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _HashableCallableShim):
            return self.func == other.func
        return self.func == other


@register_pytree_node_class
class Partial(functools.partial, CustomTreeNode):  # pylint: disable=too-few-public-methods
    """A version of functools.partial that works in pytrees.

    Use it for partial function evaluation in a way that is compatible with JAX's
    transformations, e.g., ``Partial(func, *args, **kwargs)``.

    (You need to explicitly opt-in to this behavior because we did not want to give
    functools.partial different semantics than normal function closures.)

    For example, here is a basic usage of ``Partial`` in a manner similar to
    ``functools.partial``:

    >>> import jax.numpy as jnp
    >>> add_one = Partial(jnp.add, 1)
    >>> add_one(2)
    DeviceArray(3, dtype=int32, weak_type=True)

    Pytree compatibility means that the resulting partial function can be passed
    as an argument within transformed JAX functions, which is not possible with a
    standard ``functools.partial`` function:

    >>> from jax import jit
    >>> @jit
    ... def call_func(f, *args):
    ...   return f(*args)
    ...
    >>> call_func(add_one, 2)
    DeviceArray(3, dtype=int32, weak_type=True)

    Passing zero arguments to ``Partial`` effectively wraps the original function,
    making it a valid argument in JAX transformed functions:

    >>> call_func(Partial(jnp.add), 1, 2)
    DeviceArray(3, dtype=int32, weak_type=True)

    Had we passed ``jnp.add`` to ``call_func`` directly, it would have resulted in a
    ``TypeError``.

    Note that if the result of ``Partial`` is used in the context where the
    value is traced, it results in all bound arguments being traced when passed
    to the partially-evaluated function:

    >>> print_zero = Partial(print, 0)
    >>> print_zero()
    0
    >>> call_func(print_zero)
    Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
    """

    def __new__(cls, func: Callable[..., Any], *args, **keywords) -> 'Partial':
        """Create a new :class:`Partial` instance."""
        # In Python 3.10+, if func is itself a functools.partial instance,
        # functools.partial.__new__ would merge the arguments of this Partial
        # instance with the arguments of the func. We box func in a class that does
        # not (yet) have a `func` attribute to defeat this optimization, since we
        # care exactly which arguments are considered part of the pytree.
        if isinstance(func, functools.partial):
            original_func = func
            func = _HashableCallableShim(original_func)
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
    nodetype: Type[CustomTreeNode[T]],
    handler: KeyPathHandler,
) -> KeyPathHandler:
    """Register a key path handler for a custom pytree node type."""
    _keypath_registry[nodetype] = handler
    return handler


# pylint: disable=line-too-long
register_keypaths(tuple, lambda tup: list(map(GetitemKeyPathEntry, range(len(tup)))))  # type: ignore[arg-type]
register_keypaths(list, lambda lst: list(map(GetitemKeyPathEntry, range(len(lst)))))  # type: ignore[arg-type]
register_keypaths(dict, lambda dct: list(map(GetitemKeyPathEntry, _sorted_keys(dct))))  # type: ignore[arg-type]

register_keypaths.get: Callable[[Type[CustomTreeNode[T]]], KeyPathHandler] = _keypath_registry.get  # type: ignore[attr-defined,misc]
# pylint: enable=line-too-long
