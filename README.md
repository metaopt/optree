<!-- markdownlint-disable html -->

# OpTree

![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
[![PyPI](https://img.shields.io/pypi/v/optree?logo=pypi)](https://pypi.org/project/optree)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/metaopt/optree/build.yml?label=build&logo=github)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/metaopt/optree/tests.yml?label=tests&logo=github)
[![Codecov](https://img.shields.io/codecov/c/github/metaopt/optree/main?logo=codecov)](https://codecov.io/gh/metaopt/optree)
[![Documentation Status](https://img.shields.io/readthedocs/optree?logo=readthedocs)](https://optree.readthedocs.io)
[![Downloads](https://static.pepy.tech/personalized-badge/optree?period=total&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/optree)
[![GitHub Repo Stars](https://img.shields.io/github/stars/metaopt/optree?color=brightgreen&logo=github)](https://github.com/metaopt/optree/stargazers)

Optimized PyTree Utilities.

--------------------------------------------------------------------------------

### Table of Contents  <!-- omit in toc --> <!-- markdownlint-disable heading-increment -->

- [Installation](#installation)
- [PyTrees](#pytrees)
  - [Tree Nodes and Leaves](#tree-nodes-and-leaves)
    - [Built-in PyTree Node Types](#built-in-pytree-node-types)
    - [Registering a Container-like Custom Type as Non-leaf Nodes](#registering-a-container-like-custom-type-as-non-leaf-nodes)
    - [Notes about the PyTree Type Registry](#notes-about-the-pytree-type-registry)
  - [`None` is Non-leaf Node vs. `None` is Leaf](#none-is-non-leaf-node-vs-none-is-leaf)
  - [Key Ordering for Dictionaries](#key-ordering-for-dictionaries)
- [Benchmark](#benchmark)
  - [Tree Flatten](#tree-flatten)
  - [Tree UnFlatten](#tree-unflatten)
  - [Tree Flatten with Path](#tree-flatten-with-path)
  - [Tree Copy](#tree-copy)
  - [Tree Map](#tree-map)
  - [Tree Map (nargs)](#tree-map-nargs)
  - [Tree Map with Path](#tree-map-with-path)
  - [Tree Map with Path (nargs)](#tree-map-with-path-nargs)
- [Changelog](#changelog)
- [License](#license)

--------------------------------------------------------------------------------

## Installation

Install from PyPI ([![PyPI](https://img.shields.io/pypi/v/optree?logo=pypi)](https://pypi.org/project/optree) / ![Status](https://img.shields.io/pypi/status/optree)):

```bash
pip3 install --upgrade optree
```

Install from conda-forge ([![conda-forge](https://img.shields.io/conda/v/conda-forge/optree?logo=condaforge)](https://anaconda.org/conda-forge/optree)):

```bash
conda install -c conda-forge optree
```

Install the latest version from GitHub:

```bash
pip3 install git+https://github.com/metaopt/optree.git#egg=optree
```

Or, clone this repo and install manually:

```bash
git clone --depth=1 https://github.com/metaopt/optree.git
cd optree
pip3 install .
```

Compiling from the source requires Python 3.7+, a compiler (`gcc` / `clang` / `icc` / `cl.exe`) supports C++20 and a `cmake` installation.

--------------------------------------------------------------------------------

## PyTrees

A PyTree is a recursive structure that can be an arbitrarily nested Python container (e.g., `tuple`, `list`, `dict`, `OrderedDict`, `NamedTuple`, etc.) or an opaque Python object.
The key concepts of tree operations are tree flattening and its inverse (tree unflattening).
Additional tree operations can be performed based on these two basic functions (e.g., `tree_map = tree_unflatten ∘ map ∘ tree_flatten`).

Tree flattening is traversing the entire tree in a left-to-right depth-first manner and returning the leaves of the tree in a deterministic order.

```python
>>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': 5, 'd': 6}
>>> optree.tree_flatten(tree)
([1, 2, 3, 4, 5, 6], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}))
>>> optree.tree_flatten(1)
([1], PyTreeSpec(*))
>>> optree.tree_flatten(None)
([], PyTreeSpec(None))
```

This usually implies that the equal pytrees return equal lists of leaves and the same tree structure.
See also section [Key Ordering for Dictionaries](#key-ordering-for-dictionaries).

```python
>>> {'a': [1, 2], 'b': [3]} == {'b': [3], 'a': [1, 2]}
True
>>> optree.tree_leaves({'a': [1, 2], 'b': [3]}) == optree.tree_leaves({'b': [3], 'a': [1, 2]})
True
>>> optree.tree_structure({'a': [1, 2], 'b': [3]}) == optree.tree_structure({'b': [3], 'a': [1, 2]})
True
```

### Tree Nodes and Leaves

A tree is a collection of non-leaf nodes and leaf nodes, where the leaf nodes have no children to flatten.
`optree.tree_flatten(...)` will flatten the tree and return a list of leaf nodes while the non-leaf nodes will store in the tree specification.

#### Built-in PyTree Node Types

OpTree out-of-box supports the following Python container types in the registry:

- [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple)
- [`list`](https://docs.python.org/3/library/stdtypes.html#list)
- [`dict`](https://docs.python.org/3/library/stdtypes.html#dict)
- [`collections.namedtuple`](https://docs.python.org/3/library/collections.html#collections.namedtuple) and its subclasses
- [`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict)
- [`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict)
- [`collections.deque`](https://docs.python.org/3/library/collections.html#collections.deque)
- [`PyStructSequence`](https://docs.python.org/3/c-api/tuple.html#struct-sequence-objects) types created by C API [`PyStructSequence_NewType`](https://docs.python.org/3/c-api/tuple.html#c.PyStructSequence_NewType)

which are considered non-leaf nodes in the tree.
Python objects that the type is not registered will be treated as leaf nodes.
The registration lookup uses the `is` operator to determine whether the type is matched.
So subclasses will need to explicitly register in the registry, otherwise, an object of that type will be considered as a leaf.
The [`NoneType`](https://docs.python.org/3/library/constants.html#None) is a special case discussed in section [`None` is non-leaf Node vs. `None` is Leaf](#none-is-non-leaf-node-vs-none-is-leaf).

#### Registering a Container-like Custom Type as Non-leaf Nodes

A container-like Python type can be registered in the type registry with a pair of functions that specify:

- `flatten_func(container) -> (children, metadata, entries)`: convert an instance of the container type to a `(children, metadata, entries)` triple, where `children` is an iterable of subtrees and `entries` is an iterable of path entries of the container (e.g., indices or keys).
- `unflatten_func(metadata, children) -> container`: convert such a pair back to an instance of the container type.

The `metadata` is some necessary data apart from the children to reconstruct the container, e.g., the keys of the dictionary (the children are values).

The `entries` can be omitted (only returns a pair) or is optional to implement (returns `None`). If so, use `range(len(children))` (i.e., flat indices) as path entries of the current node. The function signature can be `flatten_func(container) -> (children, metadata)` or `flatten_func(container) -> (children, metadata, None)`.

The following examples show how to register custom types and utilize them for `tree_flatten` and `tree_map`. Please refer to section [Notes about the PyTree Type Registry](#notes-about-the-pytree-type-registry) for more information.

```python
# Registry a Python type with lambda functions
optree.register_pytree_node(
    set,
    # (set) -> (children, metadata, None)
    lambda s: (sorted(s), None, None),
    # (metadata, children) -> (set)
    lambda _, children: set(children),
    namespace='set',
)

# Register a Python type into a namespace
import torch

optree.register_pytree_node(
    torch.Tensor,
    # (tensor) -> (children, metadata)
    flatten_func=lambda tensor: (
        (tensor.cpu().numpy(),),
        dict(dtype=tensor.dtype, device=tensor.device, requires_grad=tensor.requires_grad),
    ),
    # (metadata, children) -> tensor
    unflatten_func=lambda metadata, children: torch.tensor(children[0], **metadata),
    namespace='torch2numpy',
)
```

```python
>>> tree = {'weight': torch.ones(size=(1, 2)).cuda(), 'bias': torch.zeros(size=(2,))}
>>> tree
{'weight': tensor([[1., 1.]], device='cuda:0'), 'bias': tensor([0., 0.])}

# Flatten without specifying the namespace
>>> tree_flatten(tree)  # `torch.Tensor`s are leaf nodes
([tensor([0., 0.]), tensor([[1., 1.]], device='cuda:0')], PyTreeSpec({'bias': *, 'weight': *}))

# Flatten with the namespace
>>> leaves, treespec = optree.tree_flatten(tree, namespace='torch2numpy')
>>> leaves, treespec
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

# `entries` are not defined and use `range(len(children))`
>>> optree.tree_paths(tree, namespace='torch2numpy')
[('bias', 0), ('weight', 0)]

# Unflatten back to a copy of the original object
>>> optree.tree_unflatten(treespec, leaves)
{'bias': tensor([0., 0.]), 'weight': tensor([[1., 1.]], device='cuda:0')}
```

Users can also extend the pytree registry by decorating the custom class and defining an instance method `tree_flatten` and a class method `tree_unflatten`.

```python
from collections import UserDict

@optree.register_pytree_node_class(namespace='mydict')
class MyDict(UserDict):
    def tree_flatten(self):  # -> (children, metadata, entries)
        reversed_keys = sorted(self.keys(), reverse=True)
        return (
            [self[key] for key in reversed_keys],  # children
            reversed_keys,  # metadata
            reversed_keys,  # entries
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(zip(metadata, children))
```

```python
>>> tree = MyDict(b=4, a=(2, 3), c=MyDict({'d': 5, 'f': 6}))

# Flatten without specifying the namespace
>>> optree.tree_flatten_with_path(tree)  # `MyDict`s are leaf nodes
(
    [()],
    [MyDict(b=4, a=(2, 3), c=MyDict({'d': 5, 'f': 6}))],
    PyTreeSpec(*)
)

# Flatten with the namespace
>>> optree.tree_flatten_with_path(tree, namespace='mydict')
(
    [('c', 'f'), ('c', 'd'), ('b',), ('a', 0), ('a', 1)],
    [6, 5, 4, 2, 3],
    PyTreeSpec(
        CustomTreeNode(MyDict[['c', 'b', 'a']], [CustomTreeNode(MyDict[['f', 'd']], [*, *]), *, (*, *)]),
        namespace='mydict'
    )
)
```

#### Notes about the PyTree Type Registry

There are several key attributes of the pytree type registry:

1. **The type registry is per-interpreter-dependent.** This means registering a custom type in the registry affects all modules that use OpTree.

    ```diff
    - !!! WARNING !!!
      For safety reasons, a `namespace` must be specified while registering a custom type. It is
      used to isolate the behavior of flattening and unflattening a pytree node type. This is to
      prevent accidental collisions between different libraries that may register the same type.
    ```

2. **The elements in the type registry are immutable.** Users can neither register the same type twice in the same namespace (i.e., update the type registry), nor remove a type from the type registry. To update the behavior of an already registered type, simply register it again with another `namespace`.

3. **Users cannot modify the behavior of already registered built-in types** listed in [Built-in PyTree Node Types](#built-in-pytree-node-types), such as key order sorting for `dict` and `collections.defaultdict`.

4. **Inherited subclasses are not implicitly registered.** The registration lookup uses `type(obj) is registered_type` rather than `isinstance(obj, registered_type)`. Users need to register the subclasses explicitly. To register all subclasses, it is easy to implement with [`metaclass`](https://docs.python.org/3/reference/datamodel.html#metaclasses) or [`__init_subclass__`](https://docs.python.org/3/reference/datamodel.html#customizing-class-creation), for example:

    ```python
    from collections import UserDict

    @optree.register_pytree_node_class(namespace='mydict')
    class MyDict(UserDict):
        def __init_subclass__(cls):  # define this in the base class
            super().__init_subclass__()
            # Register a subclass to namespace 'mydict'
            optree.register_pytree_node_class(cls, namespace='mydict')

        def tree_flatten(self):  # -> (children, metadata, entries)
            reversed_keys = sorted(self.keys(), reverse=True)
            return (
                [self[key] for key in reversed_keys],  # children
                reversed_keys,  # metadata
                reversed_keys,  # entries
            )

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(zip(metadata, children))

    # Subclasses will be automatically registered in namespace 'mydict'
    class MyAnotherDict(MyDict):
        pass
    ```

    ```python
    >>> tree = MyDict(b=4, a=(2, 3), c=MyAnotherDict({'d': 5, 'f': 6}))
    >>> optree.tree_flatten_with_path(tree, namespace='mydict')
    (
        [('c', 'f'), ('c', 'd'), ('b',), ('a', 0), ('a', 1)],
        [6, 5, 4, 2, 3],
        PyTreeSpec(
            CustomTreeNode(MyDict[['c', 'b', 'a']], [CustomTreeNode(MyAnotherDict[['f', 'd']], [*, *]), *, (*, *)]),
            namespace='mydict'
        )
    )
    ```

5. **Be careful about the potential infinite recursion of the custom flatten function.** The returned `children` from the custom flatten function are considered subtrees. They will be further flattened recursively. The `children` can have the same type as the current node. Users must design their termination condition carefully.

    ```python
    import numpy as np
    import torch

    optree.register_pytree_node(
        np.ndarray,
        # Children are nest lists of Python objects
        lambda array: (np.atleast_1d(array).tolist(), array.ndim == 0),
        lambda scalar, rows: np.asarray(rows) if not scalar else np.asarray(rows[0]),
        namespace='numpy1',
    )

    optree.register_pytree_node(
        np.ndarray,
        # Children are Python objects
        lambda array: (
            list(array.ravel()),  # list(1DArray[T]) -> List[T]
            dict(shape=array.shape, dtype=array.dtype)
        ),
        lambda metadata, children: np.asarray(children, dtype=metadata['dtype']).reshape(metadata['shape']),
        namespace='numpy2',
    )

    optree.register_pytree_node(
        np.ndarray,
        # Returns a list of `np.ndarray`s without termination condition
        lambda array: ([array.ravel()], array.dtype),
        lambda shape, children: children[0].reshape(shape),
        namespace='numpy3',
    )

    optree.register_pytree_node(
        torch.Tensor,
        # Children are nest lists of Python objects
        lambda tensor: (torch.atleast_1d(tensor).tolist(), tensor.ndim == 0),
        lambda scalar, rows: torch.tensor(rows) if not scalar else torch.tensor(rows[0])),
        namespace='torch1',
    )

    optree.register_pytree_node(
        torch.Tensor,
        # Returns a list of `torch.Tensor`s without termination condition
        lambda tensor: (
            list(tensor.view(-1)),  # list(1DTensor[T]) -> List[0DTensor[T]] (STILL TENSORS!)
            tensor.shape
        ),
        lambda shape, children: torch.stack(children).reshape(shape),
        namespace='torch2',
    )
    ```

    ```python
    >>> optree.tree_flatten(np.arange(9).reshape(3, 3), namespace='numpy1')
    (
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        PyTreeSpec(
            CustomTreeNode(ndarray[False], [[*, *, *], [*, *, *], [*, *, *]]),
            namespace='numpy1'
        )
    )
    # Implicitly casts `float`s to `np.float64`
    >>> optree.tree_map(lambda x: x + 1.5, np.arange(9).reshape(3, 3), namespace='numpy1')
    array([[1.5, 2.5, 3.5],
           [4.5, 5.5, 6.5],
           [7.5, 8.5, 9.5]])

    >>> optree.tree_flatten(np.arange(9).reshape(3, 3), namespace='numpy2')
    (
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        PyTreeSpec(
            CustomTreeNode(ndarray[{'shape': (3, 3), 'dtype': dtype('int64')}], [*, *, *, *, *, *, *, *, *]),
            namespace='numpy2'
        )
    )
    # Explicitly casts `float`s to `np.int64`
    >>> optree.tree_map(lambda x: x + 1.5, np.arange(9).reshape(3, 3), namespace='numpy2')
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    # Children are also `np.ndarray`s, recurse without termination condition.
    >>> optree.tree_flatten(np.arange(9).reshape(3, 3), namespace='numpy3')
    Traceback (most recent call last):
        ...
    RecursionError: Maximum recursion depth exceeded during flattening the tree.

    >>> optree.tree_flatten(torch.arange(9).reshape(3, 3), namespace='torch1')
    (
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        PyTreeSpec(
            CustomTreeNode(Tensor[False], [[*, *, *], [*, *, *], [*, *, *]]),
            namespace='torch1'
        )
    )
    # Implicitly casts `float`s to `torch.float32`
    >>> optree.tree_map(lambda x: x + 1.5, torch.arange(9).reshape(3, 3), namespace='torch1')
    tensor([[1.5000, 2.5000, 3.5000],
            [4.5000, 5.5000, 6.5000],
            [7.5000, 8.5000, 9.5000]])

    # Children are also `torch.Tensor`s, recurse without termination condition.
    >>> optree.tree_flatten(torch.arange(9).reshape(3, 3), namespace='torch2')
    Traceback (most recent call last):
        ...
    RecursionError: Maximum recursion depth exceeded during flattening the tree.
    ```

### `None` is Non-leaf Node vs. `None` is Leaf

The [`None`](https://docs.python.org/3/library/constants.html#None) object is a special object in the Python language.
It serves some of the same purposes as `null` (a pointer does not point to anything) in other programming languages, which denotes a variable is empty or marks default parameters.
However, the `None` object is a singleton object rather than a pointer.
It may also serve as a sentinel value.
In addition, if a function has returned without any return value or the return statement is omitted, the function will also implicitly return the `None` object.

By default, the `None` object is considered a non-leaf node in the tree with arity 0, i.e., _**a non-leaf node that has no children**_.
This is like the behavior of an empty tuple.
While flattening a tree, it will remain in the tree structure definitions rather than in the leaves list.

```python
>>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
>>> optree.tree_flatten(tree)
([1, 2, 3, 4, 5], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *}))
>>> optree.tree_flatten(tree, none_is_leaf=True)
([1, 2, 3, 4, None, 5], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf))
>>> optree.tree_flatten(1)
([1], PyTreeSpec(*))
>>> optree.tree_flatten(None)
([], PyTreeSpec(None))
>>> optree.tree_flatten(None, none_is_leaf=True)
([None], PyTreeSpec(*, NoneIsLeaf))
```

OpTree provides a keyword argument `none_is_leaf` to determine whether to consider the `None` object as a leaf, like other opaque objects.
If `none_is_leaf=True`, the `None` object will place in the leaves list.
Otherwise, the `None` object will remain in the tree specification (structure).

```python
>>> import torch

>>> linear = torch.nn.Linear(in_features=3, out_features=2, bias=False)
>>> linear._parameters  # a container has None
OrderedDict([
    ('weight', Parameter containing:
               tensor([[-0.6677,  0.5209,  0.3295],
                       [-0.4876, -0.3142,  0.1785]], requires_grad=True)),
    ('bias', None)
])

>>> optree.tree_map(torch.zeros_like, linear._parameters)
OrderedDict([
    ('weight', tensor([[0., 0., 0.],
                       [0., 0., 0.]])),
    ('bias', None)
])

>>> optree.tree_map(torch.zeros_like, linear._parameters, none_is_leaf=True)
Traceback (most recent call last):
    ...
TypeError: zeros_like(): argument 'input' (position 1) must be Tensor, not NoneType

>>> optree.tree_map(lambda t: torch.zeros_like(t) if t is not None else 0, linear._parameters, none_is_leaf=True)
OrderedDict([
    ('weight', tensor([[0., 0., 0.],
                       [0., 0., 0.]])),
    ('bias', 0)
])
```

### Key Ordering for Dictionaries

The built-in Python dictionary (i.e., [`builtins.dict`](https://docs.python.org/3/library/stdtypes.html#dict)) is an unordered mapping that holds the keys and values.
The leaves of a dictionary are the values. Although since Python 3.6, the built-in dictionary is insertion ordered ([PEP 468](https://peps.python.org/pep-0468)).
The dictionary equality operator (`==`) does not check for key ordering.
To ensure [referential transparency](https://en.wikipedia.org/wiki/Referential_transparency) that "equal `dict`" implies "equal ordering of leaves", the order of values of the dictionary is sorted by the keys.
This behavior is also applied to [`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict).

```python
>>> optree.tree_flatten({'a': [1, 2], 'b': [3]})
([1, 2, 3], PyTreeSpec({'a': [*, *], 'b': [*]}))
>>> optree.tree_flatten({'b': [3], 'a': [1, 2]})
([1, 2, 3], PyTreeSpec({'a': [*, *], 'b': [*]}))
```

If users want to keep the values in the insertion order in pytree traversal, they should use [`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict), which will take the order of keys under consideration:

```python
>>> OrderedDict([('a', [1, 2]), ('b', [3])]) == OrderedDict([('b', [3]), ('a', [1, 2])])
False
>>> optree.tree_flatten(OrderedDict([('a', [1, 2]), ('b', [3])]))
([1, 2, 3], PyTreeSpec(OrderedDict([('a', [*, *]), ('b', [*])])))
>>> optree.tree_flatten(OrderedDict([('b', [3]), ('a', [1, 2])]))
([3, 1, 2], PyTreeSpec(OrderedDict([('b', [*]), ('a', [*, *])])))
```

Since OpTree v0.9.0, the key order of the reconstructed output dictionaries from `tree_unflatten` is guaranteed to be consistent with the key order of the input dictionaries in `tree_flatten`.

```python
>>> leaves, treespec = optree.tree_flatten({'b': [3], 'a': [1, 2]})
>>> leaves, treespec
([1, 2, 3], PyTreeSpec({'a': [*, *], 'b': [*]}))
>>> optree.tree_unflatten(treespec, leaves)
{'b': [3], 'a': [1, 2]}
>>> optree.tree_map(lambda x: x, {'b': [3], 'a': [1, 2]})
{'b': [3], 'a': [1, 2]}
>>> optree.tree_map(lambda x: x + 1, {'b': [3], 'a': [1, 2]})
{'b': [4], 'a': [2, 3]}
```

This property is also preserved during serialization/deserialization.

```python
>>> leaves, treespec = optree.tree_flatten({'b': [3], 'a': [1, 2]})
>>> leaves, treespec
([1, 2, 3], PyTreeSpec({'a': [*, *], 'b': [*]}))
>>> restored_treespec = pickle.loads(pickle.dumps(treespec))
>>> optree.tree_unflatten(treespec, leaves)
{'b': [3], 'a': [1, 2]}
>>> optree.tree_unflatten(restored_treespec, leaves)
{'b': [3], 'a': [1, 2]}
```

> Note that there are no restrictions on the `dict` to require the keys are comparable (sortable).
> There can be multiple types of keys in the dictionary.
> The keys are sorted in ascending order by `key=lambda k: k` first if capable otherwise fallback to `key=lambda k: (k.__class__.__qualname__, k)`. This handles most cases.
>
> ```python
> >>> sorted({1: 2, 1.5: 1}.keys())
> [1, 1.5]
> >>> sorted({'a': 3, 1: 2, 1.5: 1}.keys())
> Traceback (most recent call last):
>     ...
> TypeError: '<' not supported between instances of 'int' and 'str'
> >>> sorted({'a': 3, 1: 2, 1.5: 1}.keys(), key=lambda k: (k.__class__.__qualname__, k))
> [1.5, 1, 'a']
> ```

--------------------------------------------------------------------------------

## Benchmark

We benchmark the performance of:

- tree flatten
- tree unflatten
- tree copy (i.e., `unflatten(flatten(...))`)
- tree map

compared with the following libraries:

- OpTree ([`@v0.9.0`](https://github.com/metaopt/optree/tree/v0.9.0))
- JAX XLA ([`jax[cpu] == 0.4.6`](https://pypi.org/project/jax/0.4.6))
- PyTorch ([`torch == 2.0.0`](https://pypi.org/project/torch/2.0.0))
- DM-Tree ([`dm-tree == 0.1.8`](https://pypi.org/project/dm-tree/0.1.8))

| Average Time Cost (↓)      | OpTree (v0.9.0) | JAX XLA (v0.4.6) | PyTorch (v2.0.0) | DM-Tree (v0.1.8) |
| :------------------------- | --------------: | ---------------: | ---------------: | ---------------: |
| Tree Flatten               |           x1.00 |             2.33 |            22.05 |             1.12 |
| Tree UnFlatten             |           x1.00 |             2.69 |             4.28 |            16.23 |
| Tree Flatten with Path     |           x1.00 |            16.16 |    Not Supported |            27.59 |
| Tree Copy                  |           x1.00 |             2.56 |             9.97 |            11.02 |
| Tree Map                   |           x1.00 |             2.56 |             9.58 |            10.62 |
| Tree Map (nargs)           |           x1.00 |             2.89 |    Not Supported |            31.33 |
| Tree Map with Path         |           x1.00 |             7.23 |    Not Supported |            19.66 |
| Tree Map with Path (nargs) |           x1.00 |             6.56 |    Not Supported |            29.61 |

All results are reported on a workstation with an AMD Ryzen 9 5950X CPU @ 4.45GHz in an isolated virtual environment with Python 3.10.9.
Run with the following commands:

```bash
conda create --name optree-benchmark anaconda::python=3.10 --yes --no-default-packages
conda activate optree-benchmark
python3 -m pip install --editable '.[benchmark]' --extra-index-url https://download.pytorch.org/whl/cpu
python3 benchmark.py --number=10000 --repeat=5
```

The test inputs are nested containers (i.e., pytrees) extracted from `torch.nn.Module` objects.
They are:

```python
tiny_mlp = nn.Sequential(
    nn.Linear(1, 1, bias=True),
    nn.BatchNorm1d(1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Linear(1, 1, bias=False),
    nn.Sigmoid(),
)
```

and AlexNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, VisionTransformerH14 (ViT-H/14), and SwinTransformerB (Swin-B) from [`torchvsion`](https://github.com/pytorch/vision).
Please refer to [`benchmark.py`](https://github.com/metaopt/optree/blob/HEAD/benchmark.py) for more details.

### Tree Flatten

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       29.70 |        71.06 |       583.66 |        31.32 |            2.39 |           19.65 |            1.05 |
| AlexNet   |   188 |      103.92 |       262.56 |      2304.36 |       119.61 |            2.53 |           22.17 |            1.15 |
| ResNet18  |   698 |      368.06 |       852.69 |      8440.31 |       420.43 |            2.32 |           22.93 |            1.14 |
| ResNet34  |  1242 |      644.96 |      1461.55 |     14498.81 |       712.81 |            2.27 |           22.48 |            1.11 |
| ResNet50  |  1702 |      919.95 |      2080.58 |     20995.96 |      1006.42 |            2.26 |           22.82 |            1.09 |
| ResNet101 |  3317 |     1806.36 |      3996.90 |     40314.12 |      1955.48 |            2.21 |           22.32 |            1.08 |
| ResNet152 |  4932 |     2656.92 |      5812.38 |     57775.53 |      2826.92 |            2.19 |           21.75 |            1.06 |
| ViT-H/14  |  3420 |     1863.50 |      4418.24 |     41334.64 |      2128.71 |            2.37 |           22.18 |            1.14 |
| Swin-B    |  2881 |     1631.06 |      3944.13 |     36131.54 |      2032.77 |            2.42 |           22.15 |            1.25 |
|           |       |             |              |              |  **Average** |        **2.33** |       **22.05** |        **1.12** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/227140610-dce44f1b-3a91-43e6-85b5-7566ae4c8769.png" width="90%" />
</div>

### Tree UnFlatten

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       55.13 |       152.07 |       231.94 |       940.11 |            2.76 |            4.21 |           17.05 |
| AlexNet   |   188 |      226.29 |       678.29 |       972.90 |      4195.04 |            3.00 |            4.30 |           18.54 |
| ResNet18  |   698 |      766.54 |      1953.26 |      3137.86 |     12049.88 |            2.55 |            4.09 |           15.72 |
| ResNet34  |  1242 |     1309.22 |      3526.12 |      5759.16 |     20966.75 |            2.69 |            4.40 |           16.01 |
| ResNet50  |  1702 |     1914.96 |      5002.83 |      8369.43 |     29597.10 |            2.61 |            4.37 |           15.46 |
| ResNet101 |  3317 |     3672.61 |      9633.29 |     15683.16 |     57240.20 |            2.62 |            4.27 |           15.59 |
| ResNet152 |  4932 |     5407.58 |     13970.88 |     23074.68 |     82072.54 |            2.58 |            4.27 |           15.18 |
| ViT-H/14  |  3420 |     4013.18 |     11146.31 |     17633.07 |     66723.58 |            2.78 |            4.39 |           16.63 |
| Swin-B    |  2881 |     3595.34 |      9505.31 |     15054.88 |     57310.03 |            2.64 |            4.19 |           15.94 |
|           |       |             |              |              |  **Average** |        **2.69** |        **4.28** |       **16.23** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/227140674-1edc9fc5-f8db-481a-817d-a40b93c12b32.png" width="90%" />
</div>

### Tree Flatten with Path

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       36.49 |       543.67 |          N/A |       919.13 |           14.90 |             N/A |           25.19 |
| AlexNet   |   188 |      115.44 |      2185.21 |          N/A |      3752.11 |           18.93 |             N/A |           32.50 |
| ResNet18  |   698 |      431.84 |      7106.55 |          N/A |     12286.70 |           16.46 |             N/A |           28.45 |
| ResNet34  |  1242 |      845.61 |     13431.99 |          N/A |     22860.48 |           15.88 |             N/A |           27.03 |
| ResNet50  |  1702 |     1166.27 |     18426.52 |          N/A |     31225.05 |           15.80 |             N/A |           26.77 |
| ResNet101 |  3317 |     2312.77 |     34770.49 |          N/A |     59346.86 |           15.03 |             N/A |           25.66 |
| ResNet152 |  4932 |     3304.74 |     50557.25 |          N/A |     85847.91 |           15.30 |             N/A |           25.98 |
| ViT-H/14  |  3420 |     2235.25 |     37473.53 |          N/A |     64105.24 |           16.76 |             N/A |           28.68 |
| Swin-B    |  2881 |     1970.25 |     32205.83 |          N/A |     55177.50 |           16.35 |             N/A |           28.01 |
|           |       |             |              |              |  **Average** |       **16.16** |             N/A |       **27.59** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/227140719-d0040671-57f8-4dee-a0b8-02ee6d008723.png" width="90%" />
</div>

### Tree Copy

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       89.81 |       232.26 |       845.20 |       981.48 |            2.59 |            9.41 |           10.93 |
| AlexNet   |   188 |      334.58 |       959.32 |      3360.46 |      4316.05 |            2.87 |           10.04 |           12.90 |
| ResNet18  |   698 |     1128.11 |      2840.71 |     11471.07 |     12297.07 |            2.52 |           10.17 |           10.90 |
| ResNet34  |  1242 |     2160.57 |      5333.10 |     20563.06 |     21901.91 |            2.47 |            9.52 |           10.14 |
| ResNet50  |  1702 |     2746.84 |      6823.88 |     29705.99 |     28927.88 |            2.48 |           10.81 |           10.53 |
| ResNet101 |  3317 |     5762.05 |     13481.45 |     56968.78 |     60115.93 |            2.34 |            9.89 |           10.43 |
| ResNet152 |  4932 |     8151.21 |     20805.61 |     81024.06 |     84079.57 |            2.55 |            9.94 |           10.31 |
| ViT-H/14  |  3420 |     5963.61 |     15665.91 |     59813.52 |     68377.82 |            2.63 |           10.03 |           11.47 |
| Swin-B    |  2881 |     5401.59 |     14255.33 |     53361.77 |     62317.07 |            2.64 |            9.88 |           11.54 |
|           |       |             |              |              |  **Average** |        **2.56** |        **9.97** |       **11.02** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/227140744-d87eedf8-6fa8-44ad-9bac-7475a5a73f5e.png" width="90%" />
</div>

### Tree Map

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       95.13 |       243.86 |       867.34 |      1026.99 |            2.56 |            9.12 |           10.80 |
| AlexNet   |   188 |      348.44 |       987.57 |      3398.32 |      4354.81 |            2.83 |            9.75 |           12.50 |
| ResNet18  |   698 |     1190.62 |      2982.66 |     11719.94 |     12559.01 |            2.51 |            9.84 |           10.55 |
| ResNet34  |  1242 |     2205.87 |      5417.60 |     20935.72 |     22308.51 |            2.46 |            9.49 |           10.11 |
| ResNet50  |  1702 |     3128.48 |      7579.55 |     30372.71 |     31638.67 |            2.42 |            9.71 |           10.11 |
| ResNet101 |  3317 |     6173.05 |     14846.57 |     59167.85 |     60245.42 |            2.41 |            9.58 |            9.76 |
| ResNet152 |  4932 |     8641.22 |     22000.74 |     84018.65 |     86182.21 |            2.55 |            9.72 |            9.97 |
| ViT-H/14  |  3420 |     6211.79 |     17077.49 |     59790.25 |     69763.86 |            2.75 |            9.63 |           11.23 |
| Swin-B    |  2881 |     5673.66 |     14339.69 |     53309.17 |     59764.61 |            2.53 |            9.40 |           10.53 |
|           |       |             |              |              |  **Average** |        **2.56** |        **9.58** |       **10.62** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/227140788-6bb37706-f441-46c8-8897-a778e8679e05.png" width="90%" />
</div>

### Tree Map (nargs)

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |      137.06 |       389.96 |          N/A |      3908.77 |            2.85 |             N/A |           28.52 |
| AlexNet   |   188 |      467.24 |      1496.96 |          N/A |     15395.13 |            3.20 |             N/A |           32.95 |
| ResNet18  |   698 |     1603.79 |      4534.01 |          N/A |     50323.76 |            2.83 |             N/A |           31.38 |
| ResNet34  |  1242 |     2907.64 |      8435.33 |          N/A |     90389.23 |            2.90 |             N/A |           31.09 |
| ResNet50  |  1702 |     4183.77 |     11382.51 |          N/A |    121777.01 |            2.72 |             N/A |           29.11 |
| ResNet101 |  3317 |     7721.13 |     22247.85 |          N/A |    238755.17 |            2.88 |             N/A |           30.92 |
| ResNet152 |  4932 |    11508.05 |     31429.39 |          N/A |    360257.74 |            2.73 |             N/A |           31.30 |
| ViT-H/14  |  3420 |     8294.20 |     24524.86 |          N/A |    270514.87 |            2.96 |             N/A |           32.61 |
| Swin-B    |  2881 |     7074.62 |     20854.80 |          N/A |    241120.41 |            2.95 |             N/A |           34.08 |
|           |       |             |              |              |  **Average** |        **2.89** |             N/A |       **31.33** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/227140815-754fd476-0dee-42df-a809-40c953d7aff5.png" width="90%" />
</div>

### Tree Map with Path

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |      109.82 |       778.30 |          N/A |      2186.40 |            7.09 |             N/A |           19.91 |
| AlexNet   |   188 |      365.16 |      2939.36 |          N/A |      8355.37 |            8.05 |             N/A |           22.88 |
| ResNet18  |   698 |     1308.26 |      9529.58 |          N/A |     25758.24 |            7.28 |             N/A |           19.69 |
| ResNet34  |  1242 |     2527.21 |     18084.89 |          N/A |     45942.32 |            7.16 |             N/A |           18.18 |
| ResNet50  |  1702 |     3226.03 |     22935.53 |          N/A |     61275.34 |            7.11 |             N/A |           18.99 |
| ResNet101 |  3317 |     6663.52 |     46878.89 |          N/A |    126642.14 |            7.04 |             N/A |           19.01 |
| ResNet152 |  4932 |     9378.19 |     66136.44 |          N/A |    176981.01 |            7.05 |             N/A |           18.87 |
| ViT-H/14  |  3420 |     7033.69 |     50418.37 |          N/A |    142508.11 |            7.17 |             N/A |           20.26 |
| Swin-B    |  2881 |     6078.15 |     43173.22 |          N/A |    116612.71 |            7.10 |             N/A |           19.19 |
|           |       |             |              |              |  **Average** |        **7.23** |             N/A |       **19.66** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/227140830-ab8dfb6e-ea59-449e-af86-ae89897258be.png" width="90%" />
</div>

### Tree Map with Path (nargs)

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |      146.05 |       917.00 |          N/A |      3940.61 |            6.28 |             N/A |           26.98 |
| AlexNet   |   188 |      489.27 |      3560.76 |          N/A |     15434.71 |            7.28 |             N/A |           31.55 |
| ResNet18  |   698 |     1712.79 |     11171.44 |          N/A |     50219.86 |            6.52 |             N/A |           29.32 |
| ResNet34  |  1242 |     3112.83 |     21024.58 |          N/A |     95505.71 |            6.75 |             N/A |           30.68 |
| ResNet50  |  1702 |     4220.70 |     26600.82 |          N/A |    121897.57 |            6.30 |             N/A |           28.88 |
| ResNet101 |  3317 |     8631.34 |     54372.37 |          N/A |    236555.54 |            6.30 |             N/A |           27.41 |
| ResNet152 |  4932 |    12710.49 |     77643.13 |          N/A |    353600.32 |            6.11 |             N/A |           27.82 |
| ViT-H/14  |  3420 |     8753.09 |     58712.71 |          N/A |    286365.36 |            6.71 |             N/A |           32.72 |
| Swin-B    |  2881 |     7359.29 |     50112.23 |          N/A |    228866.66 |            6.81 |             N/A |           31.10 |
|           |       |             |              |              |  **Average** |        **6.56** |             N/A |       **29.61** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/227140850-bd3744aa-363d-46a7-9e92-4279d14d9be6.png" width="90%" />
</div>

--------------------------------------------------------------------------------

## Changelog

See [CHANGELOG.md](https://github.com/metaopt/optree/blob/HEAD/CHANGELOG.md).

--------------------------------------------------------------------------------

## License

OpTree is released under the Apache License 2.0.

OpTree is heavily based on JAX's implementation of the PyTree utility, with deep refactoring and several improvements.
The original licenses can be found at [JAX's Apache License 2.0](https://github.com/google/jax/blob/HEAD/LICENSE) and [Tensorflow's Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/HEAD/LICENSE).
