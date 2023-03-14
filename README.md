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
To ensure that "equal `dict`" implies "equal ordering of leaves", the order of values of the dictionary is sorted by the keys.
This behavior is also applied to [`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict).

```python
>>> optree.tree_flatten({'a': [1, 2], 'b': [3]})
([1, 2, 3], PyTreeSpec({'a': [*, *], 'b': [*]}))
>>> optree.tree_flatten({'b': [3], 'a': [1, 2]})
([1, 2, 3], PyTreeSpec({'a': [*, *], 'b': [*]}))
```

Note that there are no restrictions on the `dict` to require the keys are comparable (sortable).
There can be multiple types of keys in the dictionary.
The keys are sorted in ascending order by `key=lambda k: k` first if capable otherwise fallback to `key=lambda k: (k.__class__.__qualname__, k)`. This handles most cases.

```python
>>> sorted({1: 2, 1.5: 1}.keys())
[1, 1.5]
>>> sorted({'a': 3, 1: 2, 1.5: 1}.keys())
Traceback (most recent call last):
    ...
TypeError: '<' not supported between instances of 'int' and 'str'
>>> sorted({'a': 3, 1: 2, 1.5: 1}.keys(), key=lambda k: (k.__class__.__qualname__, k))
[1.5, 1, 'a']
```

If users want to keep the values in the insertion order, they should use [`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict), which will take the order of keys under consideration:

```python
>>> OrderedDict([('a', [1, 2]), ('b', [3])]) == OrderedDict([('b', [3]), ('a', [1, 2])])
False
>>> optree.tree_flatten(OrderedDict([('a', [1, 2]), ('b', [3])]))
([1, 2, 3], PyTreeSpec(OrderedDict([('a', [*, *]), ('b', [*])])))
>>> optree.tree_flatten(OrderedDict([('b', [3]), ('a', [1, 2])]))
([3, 1, 2], PyTreeSpec(OrderedDict([('b', [*]), ('a', [*, *])])))
```

--------------------------------------------------------------------------------

## Benchmark

We benchmark the performance of:

- tree flatten
- tree unflatten
- tree copy (i.e., `unflatten(flatten(...))`)
- tree map

compared with the following libraries:

- OpTree ([`@v0.8.0`](https://github.com/metaopt/optree/tree/v0.8.0))
- JAX XLA ([`jax[cpu] == 0.4.6`](https://pypi.org/project/jax/0.4.6))
- PyTorch ([`torch == 1.13.1`](https://pypi.org/project/torch/1.13.1))
- DM-Tree ([`dm-tree == 0.1.8`](https://pypi.org/project/dm-tree/0.1.8))

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
| TinyMLP   |    53 |       30.18 |        68.69 |       577.79 |        31.55 |            2.28 |           19.15 |            1.05 |
| AlexNet   |   188 |       97.68 |       242.74 |      2102.67 |       118.22 |            2.49 |           21.53 |            1.21 |
| ResNet18  |   698 |      346.24 |       787.04 |      7769.05 |       407.51 |            2.27 |           22.44 |            1.18 |
| ResNet34  |  1242 |      663.70 |      1431.62 |     13989.08 |       712.72 |            2.16 |           21.08 |            1.07 |
| ResNet50  |  1702 |      882.40 |      1906.07 |     19243.43 |       966.05 |            2.16 |           21.81 |            1.09 |
| ResNet101 |  3317 |     1847.35 |      3953.69 |     39870.71 |      2031.28 |            2.14 |           21.58 |            1.10 |
| ResNet152 |  4932 |     2678.84 |      5588.23 |     56023.10 |      2874.76 |            2.09 |           20.91 |            1.07 |
| ViT-H/14  |  3420 |     1947.77 |      4467.48 |     40057.71 |      2195.20 |            2.29 |           20.57 |            1.13 |
| Swin-B    |  2881 |     1763.83 |      3985.11 |     35818.71 |      1968.06 |            2.26 |           20.31 |            1.12 |
|           |       |             |              |              |  **Average** |        **2.24** |       **21.04** |        **1.11** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/224939906-0b4fb41f-96d8-4e35-858a-563d7a4ba051.png" width="90%" />
</div>

### Tree UnFlatten

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       54.91 |       134.31 |       234.37 |       913.43 |            2.45 |            4.27 |           16.64 |
| AlexNet   |   188 |      210.76 |       565.19 |       929.61 |      3808.84 |            2.68 |            4.41 |           18.07 |
| ResNet18  |   698 |      703.22 |      1727.14 |      3184.54 |     11643.08 |            2.46 |            4.53 |           16.56 |
| ResNet34  |  1242 |     1312.01 |      3147.73 |      5762.68 |     20852.70 |            2.40 |            4.39 |           15.89 |
| ResNet50  |  1702 |     1758.62 |      4177.30 |      7891.72 |     27874.16 |            2.38 |            4.49 |           15.85 |
| ResNet101 |  3317 |     3753.81 |      8226.49 |     15362.37 |     53974.51 |            2.19 |            4.09 |           14.38 |
| ResNet152 |  4932 |     5313.30 |     12205.85 |     24068.88 |     80256.68 |            2.30 |            4.53 |           15.10 |
| ViT-H/14  |  3420 |     3994.53 |     10016.00 |     17411.04 |     66000.54 |            2.51 |            4.36 |           16.52 |
| Swin-B    |  2881 |     3584.82 |      8940.27 |     15582.13 |     56003.34 |            2.49 |            4.35 |           15.62 |
|           |       |             |              |              |  **Average** |        **2.42** |        **4.38** |       **16.07** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/224940012-0620acfe-9f11-4995-ba74-e91127763074.png" width="90%" />
</div>

### Tree Flatten with Path

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       35.55 |       522.00 |          N/A |       915.30 |           14.68 |             N/A |           25.75 |
| AlexNet   |   188 |      113.89 |      2005.85 |          N/A |      3503.25 |           17.61 |             N/A |           30.76 |
| ResNet18  |   698 |      432.31 |      7052.73 |          N/A |     12239.45 |           16.31 |             N/A |           28.31 |
| ResNet34  |  1242 |      812.18 |     12657.12 |          N/A |     21703.50 |           15.58 |             N/A |           26.72 |
| ResNet50  |  1702 |     1105.15 |     17173.43 |          N/A |     29293.02 |           15.54 |             N/A |           26.51 |
| ResNet101 |  3317 |     2182.68 |     33455.81 |          N/A |     56810.61 |           15.33 |             N/A |           26.03 |
| ResNet152 |  4932 |     3272.97 |     49550.72 |          N/A |     84535.23 |           15.14 |             N/A |           25.83 |
| ViT-H/14  |  3420 |     2287.13 |     37485.28 |          N/A |     63024.61 |           16.39 |             N/A |           27.56 |
| Swin-B    |  2881 |     2092.16 |     33942.08 |          N/A |     52744.88 |           16.22 |             N/A |           25.21 |
|           |       |             |              |              |  **Average** |       **15.87** |             N/A |       **26.96** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/224940097-877967ee-d554-45a9-b03a-7f1372caa4d3.png" width="90%" />
</div>

### Tree Copy

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       91.28 |       211.31 |       833.84 |       952.72 |            2.31 |            9.13 |           10.44 |
| AlexNet   |   188 |      320.20 |       825.66 |      3118.08 |      3938.46 |            2.58 |            9.74 |           12.30 |
| ResNet18  |   698 |     1113.83 |      2578.31 |     11325.44 |     12068.00 |            2.31 |           10.17 |           10.83 |
| ResNet34  |  1242 |     2050.00 |      4836.56 |     20324.52 |     22749.73 |            2.36 |            9.91 |           11.10 |
| ResNet50  |  1702 |     2897.93 |      6121.16 |     27563.39 |     28840.04 |            2.11 |            9.51 |            9.95 |
| ResNet101 |  3317 |     5456.58 |     12306.26 |     53733.62 |     56140.12 |            2.26 |            9.85 |           10.29 |
| ResNet152 |  4932 |     8044.33 |     18873.23 |     79896.03 |     83215.06 |            2.35 |            9.93 |           10.34 |
| ViT-H/14  |  3420 |     6046.78 |     14451.37 |     58204.01 |     70966.61 |            2.39 |            9.63 |           11.74 |
| Swin-B    |  2881 |     5173.48 |     13174.36 |     51701.17 |     60053.21 |            2.55 |            9.99 |           11.61 |
|           |       |             |              |              |  **Average** |        **2.36** |        **9.76** |       **10.96** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/224940198-ade2cf22-a1b7-4686-869f-cf2c587a77ae.png" width="90%" />
</div>

### Tree Map

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       99.25 |       229.26 |       848.37 |       968.28 |            2.31 |            8.55 |            9.76 |
| AlexNet   |   188 |      332.74 |       853.33 |      3142.39 |      3992.22 |            2.56 |            9.44 |           12.00 |
| ResNet18  |   698 |     1190.94 |      2760.28 |     11399.95 |     12294.02 |            2.32 |            9.57 |           10.32 |
| ResNet34  |  1242 |     2286.53 |      4925.70 |     20423.57 |     23204.74 |            2.15 |            8.93 |           10.15 |
| ResNet50  |  1702 |     2968.51 |      6622.94 |     27807.01 |     29259.40 |            2.23 |            9.37 |            9.86 |
| ResNet101 |  3317 |     5851.06 |     13132.59 |     53999.13 |     57251.12 |            2.24 |            9.23 |            9.78 |
| ResNet152 |  4932 |     8682.55 |     19346.59 |     80462.95 |     84364.39 |            2.23 |            9.27 |            9.72 |
| ViT-H/14  |  3420 |     6695.68 |     16045.45 |     58313.07 |     68415.82 |            2.40 |            8.71 |           10.22 |
| Swin-B    |  2881 |     5747.50 |     13757.05 |     52229.81 |     61017.78 |            2.39 |            9.09 |           10.62 |
|           |       |             |              |              |  **Average** |        **2.32** |        **9.13** |       **10.27** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/224940349-a7027403-49a2-4862-83c6-83200226dc8d.png" width="90%" />
</div>

### Tree Map (nargs)

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |      133.87 |       344.99 |          N/A |      3599.07 |            2.58 |             N/A |           26.89 |
| AlexNet   |   188 |      445.41 |      1310.77 |          N/A |     14207.10 |            2.94 |             N/A |           31.90 |
| ResNet18  |   698 |     1599.16 |      4239.56 |          N/A |     49255.49 |            2.65 |             N/A |           30.80 |
| ResNet34  |  1242 |     3066.14 |      8115.79 |          N/A |     88568.31 |            2.65 |             N/A |           28.89 |
| ResNet50  |  1702 |     3951.48 |     10557.52 |          N/A |    127232.92 |            2.67 |             N/A |           32.20 |
| ResNet101 |  3317 |     7801.80 |     20208.53 |          N/A |    235961.43 |            2.59 |             N/A |           30.24 |
| ResNet152 |  4932 |    11489.21 |     29375.98 |          N/A |    349007.54 |            2.56 |             N/A |           30.38 |
| ViT-H/14  |  3420 |     8319.66 |     23204.11 |          N/A |    266190.21 |            2.79 |             N/A |           32.00 |
| Swin-B    |  2881 |     7259.47 |     20098.17 |          N/A |    226166.17 |            2.77 |             N/A |           31.15 |
|           |       |             |              |              |  **Average** |        **2.69** |             N/A |       **30.49** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/224940450-ae265203-c1d9-476e-9b22-edf850065356.png" width="90%" />
</div>

### Tree Map with Path

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |      104.70 |       703.83 |          N/A |      1998.00 |            6.72 |             N/A |           19.08 |
| AlexNet   |   188 |      352.30 |      2668.73 |          N/A |      7681.19 |            7.58 |             N/A |           21.80 |
| ResNet18  |   698 |     1289.51 |      9342.79 |          N/A |     25497.31 |            7.25 |             N/A |           19.77 |
| ResNet34  |  1242 |     2366.46 |     16746.52 |          N/A |     45254.59 |            7.08 |             N/A |           19.12 |
| ResNet50  |  1702 |     3399.11 |     23574.46 |          N/A |     60494.27 |            6.94 |             N/A |           17.80 |
| ResNet101 |  3317 |     6329.82 |     43955.95 |          N/A |    118725.60 |            6.94 |             N/A |           18.76 |
| ResNet152 |  4932 |     9307.87 |     64777.45 |          N/A |    174764.97 |            6.96 |             N/A |           18.78 |
| ViT-H/14  |  3420 |     6705.10 |     48862.92 |          N/A |    139617.21 |            7.29 |             N/A |           20.82 |
| Swin-B    |  2881 |     5780.20 |     41703.04 |          N/A |    115003.61 |            7.21 |             N/A |           19.90 |
|           |       |             |              |              |  **Average** |        **7.11** |             N/A |       **19.54** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/224940726-a43101af-a2bb-496f-95bd-332513dff0a0.png" width="90%" />
</div>

### Tree Map with Path (nargs)

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |      138.19 |       828.53 |          N/A |      3599.32 |            6.00 |             N/A |           26.05 |
| AlexNet   |   188 |      461.91 |      3138.59 |          N/A |     14069.17 |            6.79 |             N/A |           30.46 |
| ResNet18  |   698 |     1702.79 |     10890.25 |          N/A |     49456.32 |            6.40 |             N/A |           29.04 |
| ResNet34  |  1242 |     3115.89 |     19356.46 |          N/A |     88955.96 |            6.21 |             N/A |           28.55 |
| ResNet50  |  1702 |     4422.25 |     26205.69 |          N/A |    121569.30 |            5.93 |             N/A |           27.49 |
| ResNet101 |  3317 |     8334.83 |     50909.37 |          N/A |    241862.38 |            6.11 |             N/A |           29.02 |
| ResNet152 |  4932 |    12208.52 |     75327.94 |          N/A |    351472.89 |            6.17 |             N/A |           28.79 |
| ViT-H/14  |  3420 |     9320.75 |     56869.65 |          N/A |    266430.78 |            6.10 |             N/A |           28.58 |
| Swin-B    |  2881 |     7472.11 |     49260.03 |          N/A |    233154.60 |            6.59 |             N/A |           31.20 |
|           |       |             |              |              |  **Average** |        **6.26** |             N/A |       **28.80** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/224940797-50d77985-1ffd-49f6-9a74-d6195f099371.png" width="90%" />
</div>

--------------------------------------------------------------------------------

## Changelog

See [CHANGELOG.md](https://github.com/metaopt/optree/blob/HEAD/CHANGELOG.md).

--------------------------------------------------------------------------------

## License

OpTree is released under the Apache License 2.0.

OpTree is heavily based on JAX's implementation of the PyTree utility, with deep refactoring and several improvements.
The original licenses can be found at [JAX's Apache License 2.0](https://github.com/google/jax/blob/HEAD/LICENSE) and [Tensorflow's Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/HEAD/LICENSE).
