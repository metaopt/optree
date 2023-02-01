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
  - [Tree Copy](#tree-copy)
  - [Tree Map](#tree-map)
  - [Tree Map (nargs)](#tree-map-nargs)
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
    RecursionError: maximum recursion depth exceeded during flattening the tree

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
    RecursionError: maximum recursion depth exceeded during flattening the tree
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

- OpTree ([`@v0.6.0`](https://github.com/metaopt/optree/tree/v0.6.0))
- JAX XLA ([`jax[cpu] == 0.4.2`](https://pypi.org/project/jax/0.4.2))
- PyTorch ([`torch == 1.13.1`](https://pypi.org/project/torch/1.13.1))
- DM-Tree ([`dm-tree == 0.1.8`](https://pypi.org/project/dm-tree/0.1.8))

All results are reported on a workstation with an AMD Ryzen 9 5950X CPU @ 4.45GHz in an isolated virtual environment with Python 3.10.4.
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
| TinyMLP   |    53 |       28.26 |        73.35 |       586.13 |        34.47 |            2.60 |           20.74 |            1.22 |
| AlexNet   |   188 |       92.96 |       267.17 |      2248.57 |       127.33 |            2.87 |           24.19 |            1.37 |
| ResNet18  |   698 |      326.79 |       852.62 |      8282.92 |       440.36 |            2.61 |           25.35 |            1.35 |
| ResNet34  |  1242 |      593.91 |      1482.44 |     14473.95 |       768.43 |            2.50 |           24.37 |            1.29 |
| ResNet50  |  1702 |      795.99 |      2017.89 |     20029.09 |      1037.20 |            2.54 |           25.16 |            1.30 |
| ResNet101 |  3317 |     1650.51 |      4233.15 |     41516.81 |      2154.67 |            2.56 |           25.15 |            1.31 |
| ResNet152 |  4932 |     2394.63 |      5993.18 |     58042.53 |      3058.69 |            2.50 |           24.24 |            1.28 |
| ViT-H/14  |  3420 |     1836.05 |      4770.52 |     44474.74 |      2379.66 |            2.60 |           24.22 |            1.30 |
| Swin-B    |  2881 |     1594.45 |      4121.95 |     37752.24 |      1978.64 |            2.59 |           23.68 |            1.24 |
|           |       |             |              |              |  **Average** |        **2.60** |       **24.12** |        **1.29** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/216264800-b444868f-cfc6-4c49-98da-93768b3412fb.png" width="90%" />
</div>

### Tree UnFlatten

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       63.55 |       158.87 |       256.59 |      1027.58 |            2.50 |            4.04 |           16.17 |
| AlexNet   |   188 |      235.44 |       663.60 |      1007.49 |      4205.17 |            2.82 |            4.28 |           17.86 |
| ResNet18  |   698 |      822.46 |      2010.86 |      3532.06 |     12858.07 |            2.44 |            4.29 |           15.63 |
| ResNet34  |  1242 |     1437.55 |      3538.67 |      6141.87 |     22288.63 |            2.46 |            4.27 |           15.50 |
| ResNet50  |  1702 |     1951.18 |      4710.27 |      8397.79 |     29803.80 |            2.41 |            4.30 |           15.27 |
| ResNet101 |  3317 |     3755.57 |      9186.90 |     16272.87 |     57796.31 |            2.45 |            4.33 |           15.39 |
| ResNet152 |  4932 |     5754.58 |     13886.80 |     25624.53 |     90946.36 |            2.41 |            4.45 |           15.80 |
| ViT-H/14  |  3420 |     4553.70 |     11966.30 |     19082.76 |     70107.41 |            2.63 |            4.19 |           15.40 |
| Swin-B    |  2881 |     3859.93 |     10072.91 |     16329.98 |     63275.47 |            2.61 |            4.23 |           16.39 |
|           |       |             |              |              |  **Average** |        **2.53** |        **4.27** |       **15.94** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/216264908-a10508e8-67bf-4f14-bd1c-f25aaddac54e.png" width="90%" />
</div>

### Tree Copy

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       96.30 |       239.47 |       869.65 |      1080.70 |            2.49 |            9.03 |           11.22 |
| AlexNet   |   188 |      340.20 |       964.23 |      3365.82 |      4351.26 |            2.83 |            9.89 |           12.79 |
| ResNet18  |   698 |     1236.77 |      3006.39 |     12439.50 |     12836.29 |            2.43 |           10.06 |           10.38 |
| ResNet34  |  1242 |     2075.53 |      5083.91 |     21165.74 |     23238.98 |            2.45 |           10.20 |           11.20 |
| ResNet50  |  1702 |     2798.09 |      6798.03 |     28834.38 |     30770.09 |            2.43 |           10.31 |           11.00 |
| ResNet101 |  3317 |     5621.68 |     13382.80 |     55883.92 |     60485.99 |            2.38 |            9.94 |           10.76 |
| ResNet152 |  4932 |     8747.99 |     20993.80 |     88719.78 |     89079.39 |            2.40 |           10.14 |           10.18 |
| ViT-H/14  |  3420 |     6182.13 |     15860.77 |     60967.97 |     72937.40 |            2.57 |            9.86 |           11.80 |
| Swin-B    |  2881 |     5605.43 |     14415.01 |     54938.24 |     63941.94 |            2.57 |            9.80 |           11.41 |
|           |       |             |              |              |  **Average** |        **2.51** |        **9.91** |       **11.19** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/216264959-420a5eb6-3561-4109-8a79-d2db7013a4c6.png" width="90%" />
</div>

### Tree Map

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |      105.15 |       255.75 |       890.21 |      1089.57 |            2.43 |            8.47 |           10.36 |
| AlexNet   |   188 |      352.12 |       976.09 |      3398.01 |      4397.15 |            2.77 |            9.65 |           12.49 |
| ResNet18  |   698 |     1230.57 |      2996.44 |     11905.56 |     13212.48 |            2.44 |            9.67 |           10.74 |
| ResNet34  |  1242 |     2253.72 |      5407.81 |     21334.26 |     23700.81 |            2.40 |            9.47 |           10.52 |
| ResNet50  |  1702 |     3030.14 |      7163.10 |     29178.82 |     33178.31 |            2.36 |            9.63 |           10.95 |
| ResNet101 |  3317 |     5875.25 |     14767.67 |     59539.43 |     60629.81 |            2.51 |           10.13 |           10.32 |
| ResNet152 |  4932 |     8782.90 |     22213.90 |     88817.16 |     90715.53 |            2.53 |           10.11 |           10.33 |
| ViT-H/14  |  3420 |     6400.49 |     16288.05 |     61549.03 |     73521.01 |            2.54 |            9.62 |           11.49 |
| Swin-B    |  2881 |     5832.84 |     14875.89 |     54969.73 |     66202.16 |            2.55 |            9.42 |           11.35 |
|           |       |             |              |              |  **Average** |        **2.50** |        **9.57** |       **10.95** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/216265029-8c1969d0-3628-4f28-a26f-265c1d4fe8ad.png" width="90%" />
</div>

### Tree Map (nargs)

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |      150.06 |       390.32 |          N/A |      4005.74 |            2.60 |             N/A |           26.69 |
| AlexNet   |   188 |      502.39 |      1505.90 |          N/A |     15807.33 |            3.00 |             N/A |           31.46 |
| ResNet18  |   698 |     1751.59 |      4665.96 |          N/A |     53674.66 |            2.66 |             N/A |           30.64 |
| ResNet34  |  1242 |     3264.14 |      8395.17 |          N/A |     96977.90 |            2.57 |             N/A |           29.71 |
| ResNet50  |  1702 |     4493.30 |     11929.97 |          N/A |    138695.26 |            2.66 |             N/A |           30.87 |
| ResNet101 |  3317 |     8241.38 |     21702.82 |          N/A |    255637.95 |            2.63 |             N/A |           31.02 |
| ResNet152 |  4932 |    12853.26 |     34334.12 |          N/A |    402458.13 |            2.67 |             N/A |           31.31 |
| ViT-H/14  |  3420 |     9036.95 |     25573.11 |          N/A |    290971.87 |            2.83 |             N/A |           32.20 |
| Swin-B    |  2881 |     8113.26 |     22950.82 |          N/A |    256514.59 |            2.83 |             N/A |           31.62 |
|           |       |             |              |              |  **Average** |        **2.71** |             N/A |       **30.61** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/216265082-cceccfe1-d9a7-48ec-9c41-d1b56b5169ba.png" width="90%" />
</div>

--------------------------------------------------------------------------------

## Changelog

See [CHANGELOG.md](https://github.com/metaopt/optree/blob/HEAD/CHANGELOG.md).

--------------------------------------------------------------------------------

## License

OpTree is released under the Apache License 2.0.

OpTree is heavily based on JAX's implementation of the PyTree utility, with deep refactoring and several improvements.
The original licenses can be found at [JAX's Apache License 2.0](https://github.com/google/jax/blob/HEAD/LICENSE) and [Tensorflow's Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/HEAD/LICENSE).
