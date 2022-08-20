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

import optree._optree as pytree
from optree.typing import AuxData, Children, CustomTreeNode, PyTree, PyTreeDef
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
        ('to_iter', Callable[[PyTree], Tuple[Children, AuxData]]),
        ('from_iter', Callable[[AuxData, Children], PyTree]),
    ],
)


def register_pytree_node(
    nodetype: Type[PyTree],
    flatten_func: Callable[[PyTree], Tuple[Children, AuxData]],
    unflatten_func: Callable[[AuxData, Children], PyTree],
) -> Type[PyTree]:
    """Extends the set of types that are considered internal nodes in pytrees.

    See :ref:`example usage <pytrees>`.

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
    pytree.register_node(nodetype, flatten_func, unflatten_func)
    CustomTreeNode.register(nodetype)
    _nodetype_registry[nodetype] = PyTreeNodeRegistryEntry(flatten_func, unflatten_func)
    return nodetype


def register_pytree_node_class(cls: Type) -> Type:
    """Extends the set of types that are considered internal nodes in pytrees.

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


def _sorted_items(items: Iterable[Tuple[Any, Any]]) -> Iterable[Tuple[Any, Any]]:
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


def _sorted_keys(dct: Dict[Any, Any]) -> Iterable[Any]:
    try:
        # Sort directly if possible (do not use `key` for performance reasons)
        return sorted(dct)
    except TypeError:  # the keys are not comparable
        try:
            # Add `obj.__class__.__qualname__` to the key order to make it sortable
            # between different types (e.g. `int` vs. `str`)
            return sorted(dct, key=lambda o: (o.__class__.__qualname__, o))
        except TypeError:  # cannot sort the keys (e.g. user-defined types)
            return dct  # fallback to insertion order


_nodetype_registry: Dict[Type, PyTreeNodeRegistryEntry] = {
    tuple: PyTreeNodeRegistryEntry(lambda xs: (xs, None), lambda _, xs: tuple(xs)),
    list: PyTreeNodeRegistryEntry(lambda xs: (xs, None), lambda _, xs: list(xs)),
    dict: PyTreeNodeRegistryEntry(
        lambda xs: unzip2(_sorted_items(xs.items()))[::-1], lambda keys, xs: dict(zip(keys, xs))
    ),
    type(None): PyTreeNodeRegistryEntry(lambda z: ((), None), lambda _, xs: None),
}
register_pytree_node(
    collections.OrderedDict,
    lambda x: (tuple(x.values()), tuple(x.keys())),
    lambda keys, values: collections.OrderedDict(safe_zip(keys, values)),
)
register_pytree_node(
    collections.defaultdict,
    lambda x: (tuple(x.values()), (x.default_factory, tuple(x.keys()))),
    lambda s, values: collections.defaultdict(s[0], safe_zip(s[1], values)),
)

register_pytree_node.get = _nodetype_registry.get  # type: ignore[attr-defined]


class _HashableCallableShim:
    """Object that delegates __call__, __hash__, and __eq__ to another object."""

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    def __hash__(self) -> int:
        return hash(self.func)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _HashableCallableShim):
            return self.func == other.func
        return self.func == other


@register_pytree_node_class
class Partial(functools.partial):  # pylint: disable=too-few-public-methods
    """A version of functools.partial that works in pytrees.

    Use it for partial function evaluation in a way that is compatible with JAX's
    transformations, e.g., ``Partial(func, *args, **kwargs)``.

    (You need to explicitly opt-in to this behavior because we didn't want to give
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

    def __new__(cls, func: Callable, *args, **keywords) -> 'Partial':
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

    def tree_flatten(self) -> Tuple[Children, AuxData]:
        return (self.args, self.keywords), self.func

    @classmethod
    def tree_unflatten(cls, func: Callable, args: Children) -> 'Partial':
        return cls(func, *args[0], **args[1])


class KeyPathEntry(NamedTuple):
    key: Any

    def pprint(self) -> str:
        assert False  # must override


class KeyPath(NamedTuple):
    keys: Tuple[KeyPathEntry, ...] = ()

    def __add__(self, other: object) -> 'KeyPath':
        if isinstance(other, KeyPathEntry):
            return KeyPath(self.keys + (other,))
        raise TypeError(type(other))

    def pprint(self) -> str:
        if not self.keys:
            return ' tree root'
        return ''.join(k.pprint() for k in self.keys)


class GetitemKeyPathEntry(KeyPathEntry):
    def pprint(self) -> str:
        return f'[{repr(self.key)}]'


class AttributeKeyPathEntry(KeyPathEntry):
    def pprint(self) -> str:
        return f'.{self.key}'


class FlattenedKeyPathEntry(KeyPathEntry):  # fallback
    def pprint(self) -> str:
        return f'[<flat index {self.key}>]'


_keypath_registry: Dict[Type, Callable[[PyTree], List[KeyPathEntry]]] = {}


def register_keypaths(
    nodetype: Type, handler: Callable[[PyTree], List[KeyPathEntry]]
) -> Callable[[PyTree], List[KeyPathEntry]]:
    _keypath_registry[nodetype] = handler
    return handler


register_keypaths(tuple, lambda tup: list(map(GetitemKeyPathEntry, range(len(tup)))))
register_keypaths(list, lambda lst: list(map(GetitemKeyPathEntry, range(len(lst)))))
register_keypaths(dict, lambda dct: list(map(GetitemKeyPathEntry, _sorted_keys(dct))))

register_keypaths.get = _keypath_registry.get  # type: ignore[attr-defined]
