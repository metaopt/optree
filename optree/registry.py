# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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
import inspect
from collections import OrderedDict, defaultdict, deque
from operator import methodcaller
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

import optree._C as _C
from optree.typing import KT, VT, Children, CustomTreeNode, DefaultDict, MetaData
from optree.typing import OrderedDict as GenericOrderedDict
from optree.typing import PyTree, T
from optree.utils import safe_zip, unzip2


__all__ = [
    'register_pytree_node',
    'register_pytree_node_class',
    'Partial',
    'register_keypaths',
    'AttributeKeyPathEntry',
    'GetitemKeyPathEntry',
]


class PyTreeNodeRegistryEntry(NamedTuple):
    to_iterable: Callable[[CustomTreeNode[T]], Tuple[Children[T], MetaData]]
    from_iterable: Callable[[MetaData, Children[T]], CustomTreeNode[T]]


__GLOBAL_NAMESPACE: str = object()  # type: ignore[assignment]
__REGISTRY_LOCK: Lock = Lock()


def register_pytree_node(
    cls: Type[CustomTreeNode[T]],
    flatten_func: Callable[[CustomTreeNode[T]], Tuple[Children[T], MetaData]],
    unflatten_func: Callable[[MetaData, Children[T]], CustomTreeNode[T]],
    namespace: str,
) -> Type[CustomTreeNode[T]]:
    """Extend the set of types that are considered internal nodes in pytrees.

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
        flatten_func (callable): A function to be used during flattening, taking an instance of ``cls`` and
            returning a triple or optionally a pair, with (1) an iterable for the children to be
            flattened recursively, and (2) some hashable auxiliary data to be stored in the treespec
            and to be passed to the ``unflatten_func``, and (3) (optional) an iterable for the tree
            path entries to the corresponding children. If the entries are not provided or given by
            :data:`None`, then `range(len(children))` will be used.
        unflatten_func (callable): A function taking two arguments: the auxiliary data that was returned by
            ``flatten_func`` and stored in the treespec, and the unflattened children. The function
            should return an instance of ``cls``.
        namespace (str): A non-empty string that uniquely identifies the namespace of the type registry.
            This is used to isolate the registry from other modules that might register a different
            custom behavior for the same type.

    Returns:
        The same type as the input ``cls``.

    Example::

        >>> # Registry a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda _, children: set(children),
        ...     namespace='set',
        ... )

        >>> # Register a Python type into a namespace
        >>> import torch
        >>> register_pytree_node(
        ...     torch.Tensor,
        ...     flatten_func=lambda tensor: (
        ...         (tensor.cpu().numpy(),),
        ...         dict(dtype=tensor.dtype, device=tensor.device, requires_grad=tensor.requires_grad),
        ...     ),
        ...     unflatten_func=lambda metadata, children: torch.tensor(children[0], **metadata),
        ...     namespace='torch2numpy',
        ... )
        <class 'torch.Tensor'>

        >>> tree = {'weight': torch.ones(size=(1, 2)).cuda(), 'bias': torch.zeros(size=(2,))}
        >>> tree
        {'weight': tensor([[1., 1.]], device='cuda:0'), 'bias': tensor([0., 0.])}

        >>> # Flatten without specifying the namespace
        >>> tree_flatten(tree)  # `torch.Tensor`s are leaf nodes
        ([tensor([0., 0.]), tensor([[1., 1.]], device='cuda:0')], PyTreeSpec({'bias': *, 'weight': *}))

        >>> # Flatten with the namespace
        >>> optree.tree_flatten(tree, namespace='torch2numpy')
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
        >>> optree.tree_flatten(tree, namespace='tensor2flatparam')
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

    registration_key: Union[Type, Tuple[str, Type]]
    if namespace is __GLOBAL_NAMESPACE:
        registration_key = cls
        namespace = ''
    else:
        registration_key = (namespace, cls)

    with __REGISTRY_LOCK:
        _C.register_node(cls, flatten_func, unflatten_func, namespace)
        CustomTreeNode.register(cls)  # pylint: disable=no-member
        _nodetype_registry[registration_key] = PyTreeNodeRegistryEntry(flatten_func, unflatten_func)  # type: ignore[arg-type]
    return cls


@overload
def register_pytree_node_class(
    cls: Optional[str] = None,
    *,
    namespace: Optional[str] = None,
) -> Callable[[Type[CustomTreeNode[T]]], Type[CustomTreeNode[T]]]:  # pragma: no cover
    ...


@overload
def register_pytree_node_class(
    cls: Type[CustomTreeNode[T]],
    *,
    namespace: str,
) -> Type[CustomTreeNode[T]]:  # pragma: no cover
    ...


def register_pytree_node_class(
    cls: Optional[Union[Type[CustomTreeNode[T]], str]] = None,
    *,
    namespace: Optional[str] = None,
) -> Union[Type[CustomTreeNode[T]], Callable[[Type[CustomTreeNode[T]]], Type[CustomTreeNode[T]]]]:
    """Extend the set of types that are considered internal nodes in pytrees.

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
            def __init__(self, x, y):
                self.x = x
                self.y = y

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


def _dict_flatten(
    dct: Dict[KT, VT]
) -> Tuple[Tuple[VT, ...], Tuple[KT, ...], Tuple[KT, ...]]:  # pragma: no cover
    keys, values = unzip2(_sorted_items(dct.items()))
    return values, keys, keys


def _ordereddict_flatten(
    dct: GenericOrderedDict[KT, VT]
) -> Tuple[Tuple[VT, ...], Tuple[KT, ...], Tuple[KT, ...]]:  # pragma: no cover
    keys, values = unzip2(dct.items())
    return values, keys, keys


def _defaultdict_flatten(
    dct: DefaultDict[KT, VT]
) -> Tuple[
    Tuple[VT, ...], Tuple[Optional[Callable[[], VT]], Tuple[KT, ...]], Tuple[KT, ...]
]:  # pragma: no cover
    values, keys, entries = _dict_flatten(dct)
    return values, (dct.default_factory, keys), entries


# pylint: disable=all
_nodetype_registry: Dict[Union[Type, Tuple[str, Type]], PyTreeNodeRegistryEntry] = {
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


def _pytree_node_registry_get(
    type: Type, *, namespace: str = __GLOBAL_NAMESPACE
) -> Optional[PyTreeNodeRegistryEntry]:
    entry: Optional[PyTreeNodeRegistryEntry] = _nodetype_registry.get(type)
    if entry is not None or namespace is __GLOBAL_NAMESPACE or namespace == '':
        return entry
    return _nodetype_registry.get((namespace, type))


register_pytree_node.get = _pytree_node_registry_get  # type: ignore[attr-defined]
del _pytree_node_registry_get


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
    >>> tree_map(lambda t: t.cuda(), add_one)
    Partial(<built-in function add>, tensor(1., device='cuda:0'))
    >>> call_func_on_cuda(add_one, torch.tensor([[1, 2], [3, 4]]))
    tensor([[2., 3.],
            [4., 5.]], device='cuda:0')

    Passing zero arguments to :class:`Partial` effectively wraps the original function, making it a
    valid argument in tree-map functions:

    >>> call_func_on_cuda(Partial(torch.add), torch.tensor(1), torch.tensor(2))
    tensor(3, device='cuda:0')

    Had we passed :func:`operator.add` to ``call_func_on_cuda`` directly, it would have resulted in
    a :class:`TypeError` or :class:`AttributeError`.
    """

    func: Callable[..., Any]
    args: Tuple[Any, ...]
    keywords: Dict[str, Any]

    def __new__(cls, func: Callable[..., Any], *args, **keywords) -> 'Partial':
        """Create a new :class:`Partial` instance."""
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
        """Flatten the :class:`Partial` instance to children and auxiliary data."""
        return (self.args, self.keywords), self.func

    @classmethod
    def tree_unflatten(  # type: ignore[override] # pylint: disable=arguments-renamed
        cls,
        func: Callable[..., Any],
        args: Tuple[Tuple[Any, ...], Dict[str, Any]],
    ) -> 'Partial':
        """Unflatten the children and auxiliary data into a :class:`Partial` instance."""
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
    cls: Type[CustomTreeNode[T]],
    handler: KeyPathHandler,
) -> KeyPathHandler:
    """Register a key path handler for a custom pytree node type."""
    _keypath_registry[cls] = handler
    return handler


register_keypaths(tuple, lambda tup: list(map(GetitemKeyPathEntry, range(len(tup)))))  # type: ignore[arg-type]
register_keypaths(list, lambda lst: list(map(GetitemKeyPathEntry, range(len(lst)))))  # type: ignore[arg-type]
register_keypaths(dict, lambda dct: list(map(GetitemKeyPathEntry, _sorted_keys(dct))))  # type: ignore[arg-type]

# pylint: disable-next=line-too-long
register_keypaths.get: Callable[[Type[CustomTreeNode[T]]], KeyPathHandler] = _keypath_registry.get  # type: ignore[attr-defined,misc]
