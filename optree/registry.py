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
"""OpTree: Optimized PyTree Utilities."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import sys
from collections import OrderedDict, defaultdict, deque, namedtuple
from operator import methodcaller
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Sequence, overload
from typing_extensions import Self  # Python 3.11+

from optree import _C
from optree.typing import (
    KT,
    VT,
    CustomTreeNode,
    FlattenFunc,
    PyTree,
    T,
    UnflattenFunc,
    is_namedtuple_class,
    is_structseq_class,
    structseq,
)
from optree.utils import safe_zip, total_order_sorted, unzip2


if TYPE_CHECKING:
    import builtins


__all__ = [
    'register_pytree_node',
    'register_pytree_node_class',
    'unregister_pytree_node',
    'Partial',
    'register_keypaths',
    'AttributeKeyPathEntry',
    'GetitemKeyPathEntry',
]


SLOTS = {'slots': True} if sys.version_info >= (3, 10) else {}


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, **SLOTS)
class PyTreeNodeRegistryEntry:
    type: builtins.type
    flatten_func: FlattenFunc
    unflatten_func: UnflattenFunc
    namespace: str = ''


del SLOTS


# pylint: disable-next=missing-class-docstring,too-few-public-methods
class GlobalNamespace:  # pragma: no cover
    def __repr__(self) -> str:
        return '<GLOBAL NAMESPACE>'


__GLOBAL_NAMESPACE: str = GlobalNamespace()  # type: ignore[assignment]
__REGISTRY_LOCK: Lock = Lock()
del GlobalNamespace


def register_pytree_node(
    cls: type[CustomTreeNode[T]],
    flatten_func: FlattenFunc,
    unflatten_func: UnflattenFunc,
    namespace: str,
) -> type[CustomTreeNode[T]]:
    """Extend the set of types that are considered internal nodes in pytrees.

    See also :func:`register_pytree_node_class` and :func:`unregister_pytree_node`.

    The ``namespace`` argument is used to avoid collisions that occur when different libraries
    register the same Python type with different behaviors. It is recommended to add a unique prefix
    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify
    the same class in different namespaces for different use cases.

    .. warning::
        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is
        used to isolate the behavior of flattening and unflattening a pytree node type. This is to
        prevent accidental collisions between different libraries that may register the same type.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_func (callable): A function to be used during flattening, taking an instance of ``cls``
            and returning a triple or optionally a pair, with (1) an iterable for the children to be
            flattened recursively, and (2) some hashable auxiliary data to be stored in the treespec
            and to be passed to the ``unflatten_func``, and (3) (optional) an iterable for the tree
            path entries to the corresponding children. If the entries are not provided or given by
            :data:`None`, then `range(len(children))` will be used.
        unflatten_func (callable): A function taking two arguments: the auxiliary data that was
            returned by ``flatten_func`` and stored in the treespec, and the unflattened children.
            The function should return an instance of ``cls``.
        namespace (str): A non-empty string that uniquely identifies the namespace of the type registry.
            This is used to isolate the registry from other modules that might register a different
            custom behavior for the same type.

    Returns:
        The same type as the input ``cls``.

    Raises:
        TypeError: If the input type is not a class.
        TypeError: If the namespace is not a string.
        ValueError: If the namespace is an empty string.
        ValueError: If the type is already registered in the registry.

    Examples:
        >>> # Registry a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda _, children: set(children),
        ...     namespace='set',
        ... )
        <class 'set'>

        >>> # Register a Python type into a namespace
        >>> import torch
        >>> register_pytree_node(
        ...     torch.Tensor,
        ...     flatten_func=lambda tensor: (
        ...         (tensor.cpu().detach().numpy(),),
        ...         {'dtype': tensor.dtype, 'device': tensor.device, 'requires_grad': tensor.requires_grad},
        ...     ),
        ...     unflatten_func=lambda metadata, children: torch.tensor(children[0], **metadata),
        ...     namespace='torch2numpy',
        ... )
        <class 'torch.Tensor'>

        >>> # doctest: +SKIP
        >>> tree = {'weight': torch.ones(size=(1, 2)).cuda(), 'bias': torch.zeros(size=(2,))}
        >>> tree
        {'weight': tensor([[1., 1.]], device='cuda:0'), 'bias': tensor([0., 0.])}

        >>> # Flatten without specifying the namespace
        >>> tree_flatten(tree)  # `torch.Tensor`s are leaf nodes
        ([tensor([0., 0.]), tensor([[1., 1.]], device='cuda:0')], PyTreeSpec({'bias': *, 'weight': *}))

        >>> # Flatten with the namespace
        >>> tree_flatten(tree, namespace='torch2numpy')
        (
            [array([0., 0.], dtype=float32), array([[1., 1.]], dtype=float32)],
            PyTreeSpec(
                {
                    'bias': CustomTreeNode(Tensor[{'dtype': torch.float32, 'device': device(type='cpu'), 'requires_grad': False}], [*]),
                    'weight': CustomTreeNode(Tensor[{'dtype': torch.float32, 'device': device(type='cuda', index=0), 'requires_grad': False}], [*])
                },
                namespace='torch2numpy'
            )
        )

        >>> # Register the same type with a different namespace for different behaviors
        >>> def tensor2flatparam(tensor):
        ...     return [torch.nn.Parameter(tensor.reshape(-1))], tensor.shape, None
        ...
        ... def flatparam2tensor(metadata, children):
        ...     return children[0].reshape(metadata)
        ...
        ... register_pytree_node(
        ...     torch.Tensor,
        ...     flatten_func=tensor2flatparam,
        ...     unflatten_func=flatparam2tensor,
        ...     namespace='tensor2flatparam',
        ... )
        <class 'torch.Tensor'>

        >>> # Flatten with the new namespace
        >>> tree_flatten(tree, namespace='tensor2flatparam')
        (
            [
                Parameter containing: tensor([0., 0.], requires_grad=True),
                Parameter containing: tensor([1., 1.], device='cuda:0', requires_grad=True)
            ],
            PyTreeSpec(
                {
                    'bias': CustomTreeNode(Tensor[torch.Size([2])], [*]),
                    'weight': CustomTreeNode(Tensor[torch.Size([1, 2])], [*])
                },
                namespace='tensor2flatparam'
            )
        )
    """
    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls}.')
    if namespace is not __GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace}.')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')

    registration_key: type | tuple[str, type]
    if namespace is __GLOBAL_NAMESPACE:
        registration_key = cls
        namespace = ''
    else:
        registration_key = (namespace, cls)

    with __REGISTRY_LOCK:
        _C.register_node(cls, flatten_func, unflatten_func, namespace)
        _NODETYPE_REGISTRY[registration_key] = PyTreeNodeRegistryEntry(
            cls,
            flatten_func,
            unflatten_func,
            namespace=namespace,
        )
    return cls


@overload
def register_pytree_node_class(
    cls: str | None = None,
    *,
    namespace: str | None = None,
) -> Callable[[type[CustomTreeNode[T]]], type[CustomTreeNode[T]]]:  # pragma: no cover
    ...


@overload
def register_pytree_node_class(
    cls: type[CustomTreeNode[T]],
    *,
    namespace: str,
) -> type[CustomTreeNode[T]]:  # pragma: no cover
    ...


def register_pytree_node_class(
    cls: type[CustomTreeNode[T]] | str | None = None,
    *,
    namespace: str | None = None,
) -> type[CustomTreeNode[T]] | Callable[[type[CustomTreeNode[T]]], type[CustomTreeNode[T]]]:
    """Extend the set of types that are considered internal nodes in pytrees.

    See also :func:`register_pytree_node` and :func:`unregister_pytree_node`.

    The ``namespace`` argument is used to avoid collisions that occur when different libraries
    register the same Python type with different behaviors. It is recommended to add a unique prefix
    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify
    the same class in different namespaces for different use cases.

    .. warning::
        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is
        used to isolate the behavior of flattening and unflattening a pytree node type. This is to
        prevent accidental collisions between different libraries that may register the same type.

    Args:
        cls (type, optional): A Python type to treat as an internal pytree node.
        namespace (str, optional): A non-empty string that uniquely identifies the namespace of the
            type registry. This is used to isolate the registry from other modules that might
            register a different custom behavior for the same type.

    Returns:
        The same type as the input ``cls`` if the argument presents. Otherwise, return a decorator
        function that registers the class as a pytree node.

    Raises:
        TypeError: If the namespace is not a string.
        ValueError: If the namespace is an empty string.
        ValueError: If the type is already registered in the registry.

    This function is a thin wrapper around :func:`register_pytree_node`, and provides a
    class-oriented interface::

        @register_pytree_node_class(namespace='foo')
        class Special:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def tree_flatten(self):
                return ((self.x, self.y), None)

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(*children)

        @register_pytree_node_class('mylist')
        class MyList(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(*children)
    """
    if cls is __GLOBAL_NAMESPACE or isinstance(cls, str):
        if namespace is not None:
            raise ValueError('Cannot specify `namespace` when the first argument is a string.')
        if cls == '':
            raise ValueError('The namespace cannot be an empty string.')
        return functools.partial(register_pytree_node_class, namespace=cls)  # type: ignore[return-value]

    if namespace is None:
        raise ValueError('Must specify `namespace` when the first argument is a class.')
    if namespace is not __GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace}')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')

    if cls is None:
        return functools.partial(register_pytree_node_class, namespace=namespace)  # type: ignore[return-value]
    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls}.')
    register_pytree_node(cls, methodcaller('tree_flatten'), cls.tree_unflatten, namespace)
    return cls


def unregister_pytree_node(
    cls: type[CustomTreeNode[T]],
    *,
    namespace: str,
) -> PyTreeNodeRegistryEntry:
    """Remove a type from the pytree node registry.

    See also :func:`register_pytree_node` and :func:`register_pytree_node_class`.

    This function is the inverse operation of function :func:`register_pytree_node`.

    Args:
        cls (type): A Python type to remove from the pytree node registry.
        namespace (str): The namespace of the pytree node registry to remove the type from.

    Returns:
        The removed registry entry.

    Raises:
        TypeError: If the input type is not a class.
        TypeError: If the namespace is not a string.
        ValueError: If the namespace is an empty string.
        ValueError: If the type is a built-in type that cannot be unregistered.
        ValueError: If the type is not found in the registry.

    Examples:
        >>> # Register a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda _, children: set(children),
        ...     namespace='temp',
        ... )
        <class 'set'>

        >>> # Unregister the Python type
        >>> unregister_pytree_node(set, namespace='temp')
    """
    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls}.')
    if namespace is not __GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace}.')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')

    registration_key: type | tuple[str, type]
    if namespace is __GLOBAL_NAMESPACE:
        registration_key = cls
        namespace = ''
    else:
        registration_key = (namespace, cls)

    with __REGISTRY_LOCK:
        _C.unregister_node(cls, namespace)
        return _NODETYPE_REGISTRY.pop(registration_key)


def _sorted_items(items: Iterable[tuple[KT, VT]]) -> list[tuple[KT, VT]]:
    return total_order_sorted(items, key=lambda kv: kv[0])


def _sorted_keys(dct: dict[KT, VT]) -> list[KT]:
    return total_order_sorted(dct)


def _none_flatten(none: None) -> tuple[tuple[()], None]:
    return (), None


def _none_unflatten(_: None, children: Iterable[Any]) -> None:
    sentinel = object()
    if next(iter(children), sentinel) is not sentinel:
        raise ValueError('Expected no children.')
    return None  # noqa: RET501


def _tuple_flatten(tup: tuple[T, ...]) -> tuple[tuple[T, ...], None]:
    return tup, None


def _tuple_unflatten(_: None, children: Iterable[T]) -> tuple[T, ...]:
    return tuple(children)


def _list_flatten(lst: list[T]) -> tuple[list[T], None]:
    return lst, None


def _list_unflatten(_: None, children: Iterable[T]) -> list[T]:
    return list(children)


def _dict_flatten(dct: dict[KT, VT]) -> tuple[tuple[VT, ...], list[KT], tuple[KT, ...]]:
    keys, values = unzip2(_sorted_items(dct.items()))
    return values, list(keys), keys


def _dict_unflatten(keys: list[KT], values: Iterable[VT]) -> dict[KT, VT]:
    return dict(safe_zip(keys, values))


def _ordereddict_flatten(
    dct: OrderedDict[KT, VT],
) -> tuple[tuple[VT, ...], list[KT], tuple[KT, ...]]:
    keys, values = unzip2(dct.items())
    return values, list(keys), keys


def _ordereddict_unflatten(keys: list[KT], values: Iterable[VT]) -> OrderedDict[KT, VT]:
    return OrderedDict(safe_zip(keys, values))


def _defaultdict_flatten(
    dct: defaultdict[KT, VT],
) -> tuple[tuple[VT, ...], tuple[Callable[[], VT] | None, list[KT]], tuple[KT, ...]]:
    values, keys, entries = _dict_flatten(dct)
    return values, (dct.default_factory, list(keys)), entries


def _defaultdict_unflatten(
    metadata: tuple[Callable[[], VT], list[KT]],
    values: Iterable[VT],
) -> defaultdict[KT, VT]:
    default_factory, keys = metadata
    return defaultdict(default_factory, safe_zip(keys, values))


def _deque_flatten(deq: deque[T]) -> tuple[deque[T], int | None]:
    return deq, deq.maxlen


def _deque_unflatten(maxlen: int | None, children: Iterable[T]) -> deque[T]:
    return deque(children, maxlen=maxlen)


def _namedtuple_flatten(tup: NamedTuple[T]) -> tuple[tuple[T, ...], type[NamedTuple[T]]]:  # type: ignore[type-arg]
    return tup, type(tup)


def _namedtuple_unflatten(cls: type[NamedTuple[T]], children: Iterable[T]) -> NamedTuple[T]:  # type: ignore[type-arg]
    return cls(*children)  # type: ignore[call-overload]


def _structseq_flatten(seq: structseq[T]) -> tuple[tuple[T, ...], type[structseq[T]]]:
    return seq, type(seq)


def _structseq_unflatten(cls: type[structseq[T]], children: Iterable[T]) -> structseq[T]:
    return cls(children)


# pylint: disable=all
_NODETYPE_REGISTRY: dict[type | tuple[str, type], PyTreeNodeRegistryEntry] = {  # fmt: off
    type(None): PyTreeNodeRegistryEntry(type(None), _none_flatten, _none_unflatten),  # type: ignore[arg-type]
    tuple: PyTreeNodeRegistryEntry(tuple, _tuple_flatten, _tuple_unflatten),  # type: ignore[arg-type]
    list: PyTreeNodeRegistryEntry(list, _list_flatten, _list_unflatten),  # type: ignore[arg-type]
    dict: PyTreeNodeRegistryEntry(dict, _dict_flatten, _dict_unflatten),  # type: ignore[arg-type]
    namedtuple: PyTreeNodeRegistryEntry(namedtuple, _namedtuple_flatten, _namedtuple_unflatten),  # type: ignore[arg-type,dict-item] # noqa: PYI024
    OrderedDict: PyTreeNodeRegistryEntry(OrderedDict, _ordereddict_flatten, _ordereddict_unflatten),  # type: ignore[arg-type]
    defaultdict: PyTreeNodeRegistryEntry(defaultdict, _defaultdict_flatten, _defaultdict_unflatten),  # type: ignore[arg-type]
    deque: PyTreeNodeRegistryEntry(deque, _deque_flatten, _deque_unflatten),  # type: ignore[arg-type]
    structseq: PyTreeNodeRegistryEntry(structseq, _structseq_flatten, _structseq_unflatten),  # type: ignore[arg-type]
}
# pylint: enable=all


def _pytree_node_registry_get(
    cls: type,
    *,
    namespace: str = __GLOBAL_NAMESPACE,
) -> PyTreeNodeRegistryEntry | None:
    handler: PyTreeNodeRegistryEntry | None = _NODETYPE_REGISTRY.get(cls)
    if handler is not None:
        return handler
    handler = _NODETYPE_REGISTRY.get((namespace, cls))
    if handler is not None:
        return handler
    if is_structseq_class(cls):
        return _NODETYPE_REGISTRY.get(structseq)
    if is_namedtuple_class(cls):
        return _NODETYPE_REGISTRY.get(namedtuple)  # type: ignore[call-overload] # noqa: PYI024
    return None


register_pytree_node.get = _pytree_node_registry_get  # type: ignore[attr-defined]
del _pytree_node_registry_get


class _HashablePartialShim:
    """Object that delegates :meth:`__call__`, :meth:`__hash__`, and :meth:`__eq__` to another object."""

    func: Callable[..., Any]
    args: tuple[Any, ...]
    keywords: dict[str, Any]

    def __init__(self, partial_func: functools.partial) -> None:
        self.partial_func: functools.partial = partial_func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.partial_func(*args, **kwargs)

    def __hash__(self) -> int:
        return hash(self.partial_func)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _HashablePartialShim):
            return self.partial_func == other.partial_func
        return self.partial_func == other


@register_pytree_node_class(namespace=__GLOBAL_NAMESPACE)
class Partial(functools.partial, CustomTreeNode[Any]):  # pylint: disable=too-few-public-methods
    """A version of :func:`functools.partial` that works in pytrees.

    Use it for partial function evaluation in a way that is compatible with transformations,
    e.g., ``Partial(func, *args, **kwargs)``.

    (You need to explicitly opt-in to this behavior because we did not want to give
    :func:`functools.partial` different semantics than normal function closures.)

    For example, here is a basic usage of :class:`Partial` in a manner similar to
    :func:`functools.partial`:

    >>> import operator
    >>> import torch
    >>> add_one = Partial(operator.add, torch.ones(()))
    >>> add_one(torch.tensor([[1, 2], [3, 4]]))
    tensor([[2., 3.],
            [4., 5.]])

    Pytree compatibility means that the resulting partial function can be passed as an argument
    within tree-map functions, which is not possible with a standard :func:`functools.partial`
    function:

    >>> def call_func_on_cuda(f, *args, **kwargs):
    ...     f, args, kwargs = tree_map(lambda t: t.cuda(), (f, args, kwargs))
    ...     return f(*args, **kwargs)
    ...
    >>> # doctest: +SKIP
    >>> tree_map(lambda t: t.cuda(), add_one)
    Partial(<built-in function add>, tensor(1., device='cuda:0'))
    >>> call_func_on_cuda(add_one, torch.tensor([[1, 2], [3, 4]]))
    tensor([[2., 3.],
            [4., 5.]], device='cuda:0')

    Passing zero arguments to :class:`Partial` effectively wraps the original function, making it a
    valid argument in tree-map functions:

    >>> # doctest: +SKIP
    >>> call_func_on_cuda(Partial(torch.add), torch.tensor(1), torch.tensor(2))
    tensor(3, device='cuda:0')

    Had we passed :func:`operator.add` to ``call_func_on_cuda`` directly, it would have resulted in
    a :class:`TypeError` or :class:`AttributeError`.
    """

    func: Callable[..., Any]
    args: tuple[Any, ...]
    keywords: dict[str, Any]

    def __new__(cls, func: Callable[..., Any], *args: Any, **keywords: Any) -> Self:
        """Create a new :class:`Partial` instance."""
        # In Python 3.10+, if func is itself a functools.partial instance, functools.partial.__new__
        # would merge the arguments of this Partial instance with the arguments of the func. We box
        # func in a class that does not (yet) have a `func` attribute to defeat this optimization,
        # since we care exactly which arguments are considered part of the pytree.
        if isinstance(func, functools.partial):
            original_func = func
            func = _HashablePartialShim(original_func)
            assert not hasattr(func, 'func'), 'shimmed function should not have a `func` attribute'
            out = super().__new__(cls, func, *args, **keywords)
            func.func = original_func.func
            func.args = original_func.args
            func.keywords = original_func.keywords
            return out

        return super().__new__(cls, func, *args, **keywords)

    def tree_flatten(self) -> tuple[tuple[tuple[Any, ...], dict[str, Any]], Callable[..., Any]]:
        """Flatten the :class:`Partial` instance to children and auxiliary data."""
        return (self.args, self.keywords), self.func

    @classmethod
    def tree_unflatten(  # type: ignore[override]
        cls,
        metadata: Callable[..., Any],
        children: tuple[tuple[Any, ...], dict[str, Any]],
    ) -> Self:
        """Unflatten the children and auxiliary data into a :class:`Partial` instance."""
        args, keywords = children
        return cls(metadata, *args, **keywords)


class KeyPathEntry(NamedTuple):
    key: Any

    def __add__(self, other: object) -> KeyPath:
        if isinstance(other, KeyPathEntry):
            return KeyPath((self, other))
        if isinstance(other, KeyPath):
            return KeyPath((self, *other.keys))
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.key == other.key

    def pprint(self) -> str:
        """Pretty name of the key path entry."""
        raise NotImplementedError


class KeyPath(NamedTuple):
    keys: tuple[KeyPathEntry, ...] = ()

    def __add__(self, other: object) -> KeyPath:
        if isinstance(other, KeyPathEntry):
            return KeyPath((*self.keys, other))
        if isinstance(other, KeyPath):
            return KeyPath(self.keys + other.keys)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KeyPath) and self.keys == other.keys

    def pprint(self) -> str:
        """Pretty name of the key path."""
        if not self.keys:
            return ' tree root'
        return ''.join(k.pprint() for k in self.keys)


class GetitemKeyPathEntry(KeyPathEntry):
    """The key path entry class for sequences and dictionaries."""

    def pprint(self) -> str:
        """Pretty name of the key path entry."""
        return f'[{self.key!r}]'


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


KeyPathHandler = Callable[[PyTree], Sequence[KeyPathEntry]]
_KEYPATH_REGISTRY: dict[type[CustomTreeNode], KeyPathHandler] = {}


def register_keypaths(
    cls: type[CustomTreeNode[T]],
    handler: KeyPathHandler,
) -> KeyPathHandler:
    """Register a key path handler for a custom pytree node type."""
    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls}.')
    if cls in _KEYPATH_REGISTRY:
        raise ValueError(f'Key path handler for {cls} has already been registered.')

    _KEYPATH_REGISTRY[cls] = handler
    return handler


register_keypaths(tuple, lambda tup: list(map(GetitemKeyPathEntry, range(len(tup)))))  # type: ignore[arg-type]
register_keypaths(list, lambda lst: list(map(GetitemKeyPathEntry, range(len(lst)))))  # type: ignore[arg-type]
register_keypaths(dict, lambda dct: list(map(GetitemKeyPathEntry, _sorted_keys(dct))))  # type: ignore[arg-type]
register_keypaths(OrderedDict, lambda odct: list(map(GetitemKeyPathEntry, odct)))  # type: ignore[arg-type,call-overload]
register_keypaths(defaultdict, lambda ddct: list(map(GetitemKeyPathEntry, _sorted_keys(ddct))))  # type: ignore[arg-type]
register_keypaths(deque, lambda dq: list(map(GetitemKeyPathEntry, range(len(dq)))))  # type: ignore[arg-type]

register_keypaths.get = _KEYPATH_REGISTRY.get  # type: ignore[attr-defined]
