<!-- markdownlint-disable html -->

# OpTree

![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
[![PyPI](https://img.shields.io/pypi/v/optree?logo=pypi)](https://pypi.org/project/optree)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/metaopt/optree/Build?label=build&logo=github)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/metaopt/optree/Tests?label=tests&logo=github)
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
register_pytree_node(
    set,
    # (set) -> (children, metadata, None)
    lambda s: (sorted(s), None, None),
    # (metadata, children) -> (set)
    lambda _, children: set(children),
    namespace='set',
)

# Register a Python type into a namespace
import torch

register_pytree_node(
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
            list(array.ravel()),  # list(NDArray[T]) -> List[T]
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
            list(tensor.view(-1)),  # list(NDTensor[T]) -> List[0DTensor[T]] (STILL TENSORS!)
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

If users want to keep the values in the insertion order, they should use [`collection.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict), which will take the order of keys under consideration:

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

- OpTree ([`@v0.3.0`](https://github.com/metaopt/optree/tree/v0.3.0))
- JAX XLA ([`jax[cpu] == 0.3.24`](https://pypi.org/project/jax/0.3.24))
- PyTorch ([`torch == 1.13.0`](https://pypi.org/project/torch/1.13.0))
- DM-Tree ([`dm-tree == 0.1.7`](https://pypi.org/project/dm-tree/0.1.7))

All results are reported on a workstation with an AMD Ryzen 9 5950X CPU @ 4.45GHz in an isolated virtual environment with Python 3.10.4.
Run with the following command:

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
| TinyMLP   |    53 |       26.40 |        68.19 |       586.87 |        34.14 |            2.58 |           22.23 |            1.29 |
| AlexNet   |   188 |       84.28 |       259.51 |      2182.07 |       125.12 |            3.08 |           25.89 |            1.48 |
| ResNet18  |   698 |      288.57 |       807.27 |      7881.69 |       429.39 |            2.80 |           27.31 |            1.49 |
| ResNet34  |  1242 |      580.75 |      1564.97 |     15082.84 |       819.02 |            2.69 |           25.97 |            1.41 |
| ResNet50  |  1702 |      791.18 |      2081.17 |     20982.82 |      1104.62 |            2.63 |           26.52 |            1.40 |
| ResNet101 |  3317 |     1603.93 |      3939.37 |     40382.14 |      2208.63 |            2.46 |           25.18 |            1.38 |
| ResNet152 |  4932 |     2446.56 |      6267.98 |     56892.36 |      3139.17 |            2.56 |           23.25 |            1.28 |
| ViT-H/14  |  3420 |     1681.48 |      4488.33 |     41703.16 |      2504.86 |            2.67 |           24.80 |            1.49 |
| Swin-B    |  2881 |     1565.41 |      4091.10 |     34241.99 |      1936.75 |            2.61 |           21.87 |            1.24 |
|           |       |             |              |              |  **Average** |        **2.68** |       **24.78** |        **1.38** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/200494435-fd5bb385-59f7-4811-b520-98bf5763ccf3.png" width="90%" />
</div>

### Tree UnFlatten

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       59.47 |       163.00 |       257.56 |       967.00 |            2.74 |            4.33 |           16.26 |
| AlexNet   |   188 |      234.68 |       701.56 |      1011.04 |      4000.43 |            2.99 |            4.31 |           17.05 |
| ResNet18  |   698 |      758.82 |      2036.76 |      3391.87 |     12060.09 |            2.68 |            4.47 |           15.89 |
| ResNet34  |  1242 |     1459.17 |      3886.79 |      6519.28 |     21435.14 |            2.66 |            4.47 |           14.69 |
| ResNet50  |  1702 |     2003.60 |      5137.90 |      8341.17 |     29067.89 |            2.56 |            4.16 |           14.51 |
| ResNet101 |  3317 |     4005.73 |     10203.31 |     17316.07 |     59531.47 |            2.55 |            4.32 |           14.86 |
| ResNet152 |  4932 |     5644.08 |     15153.87 |     25438.67 |     88626.45 |            2.68 |            4.51 |           15.70 |
| ViT-H/14  |  3420 |     4492.64 |     12544.41 |     18091.68 |     67876.19 |            2.79 |            4.03 |           15.11 |
| Swin-B    |  2881 |     3637.86 |      9973.78 |     15353.31 |     57655.54 |            2.74 |            4.22 |           15.85 |
|           |       |             |              |              |  **Average** |        **2.71** |        **4.31** |       **15.55** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/200494672-baa2c54c-87e1-427d-85a0-a79dd5715106.png" width="90%" />
</div>

### Tree Copy

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       90.43 |       234.08 |       871.84 |      1006.30 |            2.59 |            9.64 |           11.13 |
| AlexNet   |   188 |      324.33 |       931.61 |      3263.23 |      4106.04 |            2.87 |           10.06 |           12.66 |
| ResNet18  |   698 |     1111.82 |      2840.68 |     11836.55 |     12564.23 |            2.55 |           10.65 |           11.30 |
| ResNet34  |  1242 |     2029.24 |      5129.39 |     20888.04 |     23559.50 |            2.53 |           10.29 |           11.61 |
| ResNet50  |  1702 |     2884.39 |      7118.82 |     30239.69 |     29509.25 |            2.47 |           10.48 |           10.23 |
| ResNet101 |  3317 |     5773.17 |     14396.40 |     60021.18 |     62725.03 |            2.49 |           10.40 |           10.86 |
| ResNet152 |  4932 |     8552.95 |     21321.48 |     85857.53 |     86037.99 |            2.49 |           10.04 |           10.06 |
| ViT-H/14  |  3420 |     6116.61 |     16038.69 |     59993.87 |     70215.65 |            2.62 |            9.81 |           11.48 |
| Swin-B    |  2881 |     5466.03 |     14449.60 |     50528.12 |     60269.63 |            2.64 |            9.24 |           11.03 |
|           |       |             |              |              |  **Average** |        **2.58** |       **10.07** |       **11.15** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/200494749-c34969a4-03d0-4d76-8aae-35e9deff7cfa.png" width="90%" />
</div>

### Tree Map

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |       98.24 |       255.39 |       868.24 |      1032.07 |            2.60 |            8.84 |           10.51 |
| AlexNet   |   188 |      337.87 |      1004.53 |      3304.64 |      4099.77 |            2.97 |            9.78 |           12.13 |
| ResNet18  |   698 |     1171.69 |      3059.72 |     11921.16 |     12727.38 |            2.61 |           10.17 |           10.86 |
| ResNet34  |  1242 |     2267.61 |      5793.53 |     22222.92 |     22437.44 |            2.55 |            9.80 |            9.89 |
| ResNet50  |  1702 |     2961.05 |      7792.69 |     30132.32 |     31460.04 |            2.63 |           10.18 |           10.62 |
| ResNet101 |  3317 |     6101.05 |     14342.22 |     56480.19 |     61830.65 |            2.35 |            9.26 |           10.13 |
| ResNet152 |  4932 |     8568.48 |     21641.40 |     83021.19 |     87077.66 |            2.53 |            9.69 |           10.16 |
| ViT-H/14  |  3420 |     6735.93 |     18027.05 |     63986.88 |     75742.33 |            2.68 |            9.50 |           11.24 |
| Swin-B    |  2881 |     5756.71 |     14528.51 |     51052.90 |     60715.06 |            2.52 |            8.87 |           10.55 |
|           |       |             |              |              |  **Average** |        **2.61** |        **9.56** |       **10.68** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/200494856-9f8481d1-d805-408a-9b4f-9572eb77b6a4.png" width="90%" />
</div>

### Tree Map (nargs)

| Module    | Nodes | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :-------- | ----: | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| TinyMLP   |    53 |      144.61 |       391.05 |          N/A |      3774.79 |            2.70 |             N/A |           26.10 |
| AlexNet   |   188 |      480.23 |      1515.41 |          N/A |     15105.87 |            3.16 |             N/A |           31.46 |
| ResNet18  |   698 |     1690.19 |      4997.44 |          N/A |     52089.06 |            2.96 |             N/A |           30.82 |
| ResNet34  |  1242 |     3084.36 |      8572.54 |          N/A |     93923.27 |            2.78 |             N/A |           30.45 |
| ResNet50  |  1702 |     4441.17 |     11962.92 |          N/A |    126937.65 |            2.69 |             N/A |           28.58 |
| ResNet101 |  3317 |     8155.78 |     22232.67 |          N/A |    251333.88 |            2.73 |             N/A |           30.82 |
| ResNet152 |  4932 |    12862.88 |     33714.46 |          N/A |    368424.63 |            2.62 |             N/A |           28.64 |
| ViT-H/14  |  3420 |     9511.10 |     27920.13 |          N/A |    281245.95 |            2.94 |             N/A |           29.57 |
| Swin-B    |  2881 |     7628.29 |     22421.37 |          N/A |    238211.56 |            2.94 |             N/A |           31.23 |
|           |       |             |              |              |  **Average** |        **2.83** |             N/A |       **29.74** |

<div align="center">
  <img src="https://user-images.githubusercontent.com/16078332/200494913-b337517f-3ba7-4c9b-90a9-e9b1f8455be3.png" width="90%" />
</div>

```text
TinyMLP(num_nodes=53, num_leaves=16, treespec=PyTreeSpec([OrderedDict([('tenso...), buffers=OrderedDict([])))])]))
| Subject          | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :--------------- | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| Tree Flatten     |       26.40 |        68.19 |       586.87 |        34.14 |            2.58 |           22.23 |            1.29 |
| Tree UnFlatten   |       59.47 |       163.00 |       257.56 |       967.00 |            2.74 |            4.33 |           16.26 |
| Tree Copy        |       90.43 |       234.08 |       871.84 |      1006.30 |            2.59 |            9.64 |           11.13 |
| Tree Map         |       98.24 |       255.39 |       868.24 |      1032.07 |            2.60 |            8.84 |           10.51 |
| Tree Map (nargs) |      144.61 |       391.05 |          N/A |      3774.79 |            2.70 |             N/A |           26.10 |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :    26.40μs            <=  optree.tree_leaves(x)                      (None is Node)
✔ OpTree :    25.98μs -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :    26.19μs -- x0.99   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:    68.19μs -- x2.58   <=  jax.tree_util.tree_leaves(x)
  PyTorch:   586.87μs -- x22.23  <=  torch_utils_pytree.tree_flatten(x)[0]
  DM-Tree:    34.14μs -- x1.29   <=  dm_tree.flatten(x)

### Tree UnFlatten ###
✔ OpTree :    59.47μs            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :    59.71μs -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:   163.00μs -- x2.74   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:   257.56μs -- x4.33   <=  torch_utils_pytree.tree_unflatten(flat, spec)
  DM-Tree:   967.00μs -- x16.26  <=  dm_tree.unflatten_as(spec, flat)

### Tree Copy ###
✔ OpTree :    90.43μs            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :    91.51μs -- x1.01   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :    91.34μs -- x1.01   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:   234.08μs -- x2.59   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch:   871.84μs -- x9.64   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))
  DM-Tree:  1006.30μs -- x11.13  <=  dm_tree.unflatten_as(x, dm_tree.flatten(x))

### Tree Map ###
~ OpTree :    98.24μs            <=  optree.tree_map(fn1, x)                      (None is Node)
✔ OpTree :    97.62μs -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :    98.05μs -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   255.39μs -- x2.60   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch:   868.24μs -- x8.84   <=  torch_utils_pytree.tree_map(fn1, x)
  DM-Tree:  1032.07μs -- x10.51  <=  dm_tree.map_structure(fn1, x)

### Tree Map (nargs) ###
~ OpTree :   144.61μs            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree :   141.03μs -- x0.98   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :   142.79μs -- x0.99   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   391.05μs -- x2.70   <=  jax.tree_util.tree_map(fn3, x, y, z)
  DM-Tree:  3774.79μs -- x26.10  <=  dm_tree.map_structure_up_to(x, fn3, x, y, z)
```

```text
AlexNet(num_nodes=188, num_leaves=32, treespec=PyTreeSpec(OrderedDict([('featur...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :--------------- | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| Tree Flatten     |       84.28 |       259.51 |      2182.07 |       125.12 |            3.08 |           25.89 |            1.48 |
| Tree UnFlatten   |      234.68 |       701.56 |      1011.04 |      4000.43 |            2.99 |            4.31 |           17.05 |
| Tree Copy        |      324.33 |       931.61 |      3263.23 |      4106.04 |            2.87 |           10.06 |           12.66 |
| Tree Map         |      337.87 |      1004.53 |      3304.64 |      4099.77 |            2.97 |            9.78 |           12.13 |
| Tree Map (nargs) |      480.23 |      1515.41 |          N/A |     15105.87 |            3.16 |             N/A |           31.46 |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
✔ OpTree :    84.28μs            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :    84.64μs -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :    85.54μs -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   259.51μs -- x3.08   <=  jax.tree_util.tree_leaves(x)
  PyTorch:  2182.07μs -- x25.89  <=  torch_utils_pytree.tree_flatten(x)[0]
  DM-Tree:   125.12μs -- x1.48   <=  dm_tree.flatten(x)

### Tree UnFlatten ###
~ OpTree :   234.68μs            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :   234.02μs -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:   701.56μs -- x2.99   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  1011.04μs -- x4.31   <=  torch_utils_pytree.tree_unflatten(flat, spec)
  DM-Tree:  4000.43μs -- x17.05  <=  dm_tree.unflatten_as(spec, flat)

### Tree Copy ###
~ OpTree :   324.33μs            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :   324.09μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :   323.18μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:   931.61μs -- x2.87   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch:  3263.23μs -- x10.06  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))
  DM-Tree:  4106.04μs -- x12.66  <=  dm_tree.unflatten_as(x, dm_tree.flatten(x))

### Tree Map ###
✔ OpTree :   337.87μs            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :   340.56μs -- x1.01   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :   338.36μs -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  1004.53μs -- x2.97   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch:  3304.64μs -- x9.78   <=  torch_utils_pytree.tree_map(fn1, x)
  DM-Tree:  4099.77μs -- x12.13  <=  dm_tree.map_structure(fn1, x)

### Tree Map (nargs) ###
~ OpTree :   480.23μs            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :   481.45μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :   479.35μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  1515.41μs -- x3.16   <=  jax.tree_util.tree_map(fn3, x, y, z)
  DM-Tree: 15105.87μs -- x31.46  <=  dm_tree.map_structure_up_to(x, fn3, x, y, z)
```

```text
ResNet18(num_nodes=698, num_leaves=244, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :--------------- | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| Tree Flatten     |      288.57 |       807.27 |      7881.69 |       429.39 |            2.80 |           27.31 |            1.49 |
| Tree UnFlatten   |      758.82 |      2036.76 |      3391.87 |     12060.09 |            2.68 |            4.47 |           15.89 |
| Tree Copy        |     1111.82 |      2840.68 |     11836.55 |     12564.23 |            2.55 |           10.65 |           11.30 |
| Tree Map         |     1171.69 |      3059.72 |     11921.16 |     12727.38 |            2.61 |           10.17 |           10.86 |
| Tree Map (nargs) |     1690.19 |      4997.44 |          N/A |     52089.06 |            2.96 |             N/A |           30.82 |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
✔ OpTree :   288.57μs            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :   294.01μs -- x1.02   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :   294.99μs -- x1.02   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   807.27μs -- x2.80   <=  jax.tree_util.tree_leaves(x)
  PyTorch:  7881.69μs -- x27.31  <=  torch_utils_pytree.tree_flatten(x)[0]
  DM-Tree:   429.39μs -- x1.49   <=  dm_tree.flatten(x)

### Tree UnFlatten ###
✔ OpTree :   758.82μs            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :   765.90μs -- x1.01   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  2036.76μs -- x2.68   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  3391.87μs -- x4.47   <=  torch_utils_pytree.tree_unflatten(flat, spec)
  DM-Tree: 12060.09μs -- x15.89  <=  dm_tree.unflatten_as(spec, flat)

### Tree Copy ###
~ OpTree :  1111.82μs            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  1103.80μs -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  1100.03μs -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  2840.68μs -- x2.55   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 11836.55μs -- x10.65  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))
  DM-Tree: 12564.23μs -- x11.30  <=  dm_tree.unflatten_as(x, dm_tree.flatten(x))

### Tree Map ###
✔ OpTree :  1171.69μs            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  1172.46μs -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  1181.17μs -- x1.01   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  3059.72μs -- x2.61   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 11921.16μs -- x10.17  <=  torch_utils_pytree.tree_map(fn1, x)
  DM-Tree: 12727.38μs -- x10.86  <=  dm_tree.map_structure(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :  1690.19μs            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  1760.86μs -- x1.04   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  1761.76μs -- x1.04   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4997.44μs -- x2.96   <=  jax.tree_util.tree_map(fn3, x, y, z)
  DM-Tree: 52089.06μs -- x30.82  <=  dm_tree.map_structure_up_to(x, fn3, x, y, z)
```

```text
ResNet34(num_nodes=1242, num_leaves=436, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :--------------- | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| Tree Flatten     |      580.75 |      1564.97 |     15082.84 |       819.02 |            2.69 |           25.97 |            1.41 |
| Tree UnFlatten   |     1459.17 |      3886.79 |      6519.28 |     21435.14 |            2.66 |            4.47 |           14.69 |
| Tree Copy        |     2029.24 |      5129.39 |     20888.04 |     23559.50 |            2.53 |           10.29 |           11.61 |
| Tree Map         |     2267.61 |      5793.53 |     22222.92 |     22437.44 |            2.55 |            9.80 |            9.89 |
| Tree Map (nargs) |     3084.36 |      8572.54 |          N/A |     93923.27 |            2.78 |             N/A |           30.45 |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :   580.75μs            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :   577.02μs -- x0.99   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :   571.14μs -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  1564.97μs -- x2.69   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 15082.84μs -- x25.97  <=  torch_utils_pytree.tree_flatten(x)[0]
  DM-Tree:   819.02μs -- x1.41   <=  dm_tree.flatten(x)

### Tree UnFlatten ###
✔ OpTree :  1459.17μs            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :  1465.07μs -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  3886.79μs -- x2.66   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  6519.28μs -- x4.47   <=  torch_utils_pytree.tree_unflatten(flat, spec)
  DM-Tree: 21435.14μs -- x14.69  <=  dm_tree.unflatten_as(spec, flat)

### Tree Copy ###
~ OpTree :  2029.24μs            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
✔ OpTree :  2021.45μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :  2024.29μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  5129.39μs -- x2.53   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 20888.04μs -- x10.29  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))
  DM-Tree: 23559.50μs -- x11.61  <=  dm_tree.unflatten_as(x, dm_tree.flatten(x))

### Tree Map ###
~ OpTree :  2267.61μs            <=  optree.tree_map(fn1, x)                      (None is Node)
✔ OpTree :  2257.85μs -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  2268.77μs -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  5793.53μs -- x2.55   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 22222.92μs -- x9.80   <=  torch_utils_pytree.tree_map(fn1, x)
  DM-Tree: 22437.44μs -- x9.89   <=  dm_tree.map_structure(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  3084.36μs            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  3080.29μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :  3072.45μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  8572.54μs -- x2.78   <=  jax.tree_util.tree_map(fn3, x, y, z)
  DM-Tree: 93923.27μs -- x30.45  <=  dm_tree.map_structure_up_to(x, fn3, x, y, z)
```

```text
ResNet50(num_nodes=1702, num_leaves=640, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :--------------- | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| Tree Flatten     |      791.18 |      2081.17 |     20982.82 |      1104.62 |            2.63 |           26.52 |            1.40 |
| Tree UnFlatten   |     2003.60 |      5137.90 |      8341.17 |     29067.89 |            2.56 |            4.16 |           14.51 |
| Tree Copy        |     2884.39 |      7118.82 |     30239.69 |     29509.25 |            2.47 |           10.48 |           10.23 |
| Tree Map         |     2961.05 |      7792.69 |     30132.32 |     31460.04 |            2.63 |           10.18 |           10.62 |
| Tree Map (nargs) |     4441.17 |     11962.92 |          N/A |    126937.65 |            2.69 |             N/A |           28.58 |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :   791.18μs            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :   791.38μs -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :   779.75μs -- x0.99   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  2081.17μs -- x2.63   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 20982.82μs -- x26.52  <=  torch_utils_pytree.tree_flatten(x)[0]
  DM-Tree:  1104.62μs -- x1.40   <=  dm_tree.flatten(x)

### Tree UnFlatten ###
~ OpTree :  2003.60μs            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  2000.10μs -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  5137.90μs -- x2.56   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  8341.17μs -- x4.16   <=  torch_utils_pytree.tree_unflatten(flat, spec)
  DM-Tree: 29067.89μs -- x14.51  <=  dm_tree.unflatten_as(spec, flat)

### Tree Copy ###
~ OpTree :  2884.39μs            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  2879.97μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  2868.84μs -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  7118.82μs -- x2.47   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 30239.69μs -- x10.48  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))
  DM-Tree: 29509.25μs -- x10.23  <=  dm_tree.unflatten_as(x, dm_tree.flatten(x))

### Tree Map ###
✔ OpTree :  2961.05μs            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  3079.33μs -- x1.04   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  3116.74μs -- x1.05   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  7792.69μs -- x2.63   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 30132.32μs -- x10.18  <=  torch_utils_pytree.tree_map(fn1, x)
  DM-Tree: 31460.04μs -- x10.62  <=  dm_tree.map_structure(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  4441.17μs            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree :  4430.87μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  4449.43μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 11962.92μs -- x2.69   <=  jax.tree_util.tree_map(fn3, x, y, z)
  DM-Tree: 126937.65μs -- x28.58  <=  dm_tree.map_structure_up_to(x, fn3, x, y, z)
```

```text
ResNet101(num_nodes=3317, num_leaves=1252, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :--------------- | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| Tree Flatten     |     1603.93 |      3939.37 |     40382.14 |      2208.63 |            2.46 |           25.18 |            1.38 |
| Tree UnFlatten   |     4005.73 |     10203.31 |     17316.07 |     59531.47 |            2.55 |            4.32 |           14.86 |
| Tree Copy        |     5773.17 |     14396.40 |     60021.18 |     62725.03 |            2.49 |           10.40 |           10.86 |
| Tree Map         |     6101.05 |     14342.22 |     56480.19 |     61830.65 |            2.35 |            9.26 |           10.13 |
| Tree Map (nargs) |     8155.78 |     22232.67 |          N/A |    251333.88 |            2.73 |             N/A |           30.82 |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1603.93μs            <=  optree.tree_leaves(x)                      (None is Node)
✔ OpTree :  1458.25μs -- x0.91   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :  1492.62μs -- x0.93   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  3939.37μs -- x2.46   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 40382.14μs -- x25.18  <=  torch_utils_pytree.tree_flatten(x)[0]
  DM-Tree:  2208.63μs -- x1.38   <=  dm_tree.flatten(x)

### Tree UnFlatten ###
~ OpTree :  4005.73μs            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  3957.47μs -- x0.99   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA: 10203.31μs -- x2.55   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 17316.07μs -- x4.32   <=  torch_utils_pytree.tree_unflatten(flat, spec)
  DM-Tree: 59531.47μs -- x14.86  <=  dm_tree.unflatten_as(spec, flat)

### Tree Copy ###
~ OpTree :  5773.17μs            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
✔ OpTree :  5741.73μs -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :  5759.01μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 14396.40μs -- x2.49   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 60021.18μs -- x10.40  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))
  DM-Tree: 62725.03μs -- x10.86  <=  dm_tree.unflatten_as(x, dm_tree.flatten(x))

### Tree Map ###
~ OpTree :  6101.05μs            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  6145.86μs -- x1.01   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  5709.67μs -- x0.94   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 14342.22μs -- x2.35   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 56480.19μs -- x9.26   <=  torch_utils_pytree.tree_map(fn1, x)
  DM-Tree: 61830.65μs -- x10.13  <=  dm_tree.map_structure(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  8155.78μs            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  8144.17μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :  8113.58μs -- x0.99   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 22232.67μs -- x2.73   <=  jax.tree_util.tree_map(fn3, x, y, z)
  DM-Tree: 251333.88μs -- x30.82  <=  dm_tree.map_structure_up_to(x, fn3, x, y, z)
```

```text
ResNet152(num_nodes=4932, num_leaves=1864, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :--------------- | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| Tree Flatten     |     2446.56 |      6267.98 |     56892.36 |      3139.17 |            2.56 |           23.25 |            1.28 |
| Tree UnFlatten   |     5644.08 |     15153.87 |     25438.67 |     88626.45 |            2.68 |            4.51 |           15.70 |
| Tree Copy        |     8552.95 |     21321.48 |     85857.53 |     86037.99 |            2.49 |           10.04 |           10.06 |
| Tree Map         |     8568.48 |     21641.40 |     83021.19 |     87077.66 |            2.53 |            9.69 |           10.16 |
| Tree Map (nargs) |    12862.88 |     33714.46 |          N/A |    368424.63 |            2.62 |             N/A |           28.64 |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  2446.56μs            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  2455.99μs -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  2429.96μs -- x0.99   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  6267.98μs -- x2.56   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 56892.36μs -- x23.25  <=  torch_utils_pytree.tree_flatten(x)[0]
  DM-Tree:  3139.17μs -- x1.28   <=  dm_tree.flatten(x)

### Tree UnFlatten ###
✔ OpTree :  5644.08μs            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :  5723.38μs -- x1.01   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA: 15153.87μs -- x2.68   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 25438.67μs -- x4.51   <=  torch_utils_pytree.tree_unflatten(flat, spec)
  DM-Tree: 88626.45μs -- x15.70  <=  dm_tree.unflatten_as(spec, flat)

### Tree Copy ###
~ OpTree :  8552.95μs            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  8531.50μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  8528.88μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 21321.48μs -- x2.49   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 85857.53μs -- x10.04  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))
  DM-Tree: 86037.99μs -- x10.06  <=  dm_tree.unflatten_as(x, dm_tree.flatten(x))

### Tree Map ###
~ OpTree :  8568.48μs            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  8569.48μs -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  8542.91μs -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 21641.40μs -- x2.53   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 83021.19μs -- x9.69   <=  torch_utils_pytree.tree_map(fn1, x)
  DM-Tree: 87077.66μs -- x10.16  <=  dm_tree.map_structure(fn1, x)

### Tree Map (nargs) ###
~ OpTree : 12862.88μs            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree : 12806.09μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree : 12909.94μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 33714.46μs -- x2.62   <=  jax.tree_util.tree_map(fn3, x, y, z)
  DM-Tree: 368424.63μs -- x28.64  <=  dm_tree.map_structure_up_to(x, fn3, x, y, z)
```

```text
ViT-H/14(num_nodes=3420, num_leaves=784, treespec=PyTreeSpec(OrderedDict([('conv_p...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :--------------- | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| Tree Flatten     |     1681.48 |      4488.33 |     41703.16 |      2504.86 |            2.67 |           24.80 |            1.49 |
| Tree UnFlatten   |     4492.64 |     12544.41 |     18091.68 |     67876.19 |            2.79 |            4.03 |           15.11 |
| Tree Copy        |     6116.61 |     16038.69 |     59993.87 |     70215.65 |            2.62 |            9.81 |           11.48 |
| Tree Map         |     6735.93 |     18027.05 |     63986.88 |     75742.33 |            2.68 |            9.50 |           11.24 |
| Tree Map (nargs) |     9511.10 |     27920.13 |          N/A |    281245.95 |            2.94 |             N/A |           29.57 |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
✔ OpTree :  1681.48μs            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1702.21μs -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :  1694.58μs -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4488.33μs -- x2.67   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 41703.16μs -- x24.80  <=  torch_utils_pytree.tree_flatten(x)[0]
  DM-Tree:  2504.86μs -- x1.49   <=  dm_tree.flatten(x)

### Tree UnFlatten ###
✔ OpTree :  4492.64μs            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :  4535.79μs -- x1.01   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA: 12544.41μs -- x2.79   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 18091.68μs -- x4.03   <=  torch_utils_pytree.tree_unflatten(flat, spec)
  DM-Tree: 67876.19μs -- x15.11  <=  dm_tree.unflatten_as(spec, flat)

### Tree Copy ###
~ OpTree :  6116.61μs            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
✔ OpTree :  6075.72μs -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :  6104.80μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 16038.69μs -- x2.62   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 59993.87μs -- x9.81   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))
  DM-Tree: 70215.65μs -- x11.48  <=  dm_tree.unflatten_as(x, dm_tree.flatten(x))

### Tree Map ###
~ OpTree :  6735.93μs            <=  optree.tree_map(fn1, x)                      (None is Node)
✔ OpTree :  6679.19μs -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  6726.99μs -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 18027.05μs -- x2.68   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 63986.88μs -- x9.50   <=  torch_utils_pytree.tree_map(fn1, x)
  DM-Tree: 75742.33μs -- x11.24  <=  dm_tree.map_structure(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  9511.10μs            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree :  9503.85μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  9550.25μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 27920.13μs -- x2.94   <=  jax.tree_util.tree_map(fn3, x, y, z)
  DM-Tree: 281245.95μs -- x29.57  <=  dm_tree.map_structure_up_to(x, fn3, x, y, z)
```

```text
Swin-B(num_nodes=2881, num_leaves=706, treespec=PyTreeSpec(OrderedDict([('featur...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (μs) | JAX XLA (μs) | PyTorch (μs) | DM-Tree (μs) | Speedup (J / O) | Speedup (P / O) | Speedup (D / O) |
| :--------------- | ----------: | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: |
| Tree Flatten     |     1565.41 |      4091.10 |     34241.99 |      1936.75 |            2.61 |           21.87 |            1.24 |
| Tree UnFlatten   |     3637.86 |      9973.78 |     15353.31 |     57655.54 |            2.74 |            4.22 |           15.85 |
| Tree Copy        |     5466.03 |     14449.60 |     50528.12 |     60269.63 |            2.64 |            9.24 |           11.03 |
| Tree Map         |     5756.71 |     14528.51 |     51052.90 |     60715.06 |            2.52 |            8.87 |           10.55 |
| Tree Map (nargs) |     7628.29 |     22421.37 |          N/A |    238211.56 |            2.94 |             N/A |           31.23 |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1565.41μs            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1565.91μs -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1550.64μs -- x0.99   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4091.10μs -- x2.61   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 34241.99μs -- x21.87  <=  torch_utils_pytree.tree_flatten(x)[0]
  DM-Tree:  1936.75μs -- x1.24   <=  dm_tree.flatten(x)

### Tree UnFlatten ###
~ OpTree :  3637.86μs            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  3596.18μs -- x0.99   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  9973.78μs -- x2.74   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 15353.31μs -- x4.22   <=  torch_utils_pytree.tree_unflatten(flat, spec)
  DM-Tree: 57655.54μs -- x15.85  <=  dm_tree.unflatten_as(spec, flat)

### Tree Copy ###
✔ OpTree :  5466.03μs            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  5467.68μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :  5469.55μs -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 14449.60μs -- x2.64   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 50528.12μs -- x9.24   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))
  DM-Tree: 60269.63μs -- x11.03  <=  dm_tree.unflatten_as(x, dm_tree.flatten(x))

### Tree Map ###
~ OpTree :  5756.71μs            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  5712.77μs -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  5706.22μs -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 14528.51μs -- x2.52   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 51052.90μs -- x8.87   <=  torch_utils_pytree.tree_map(fn1, x)
  DM-Tree: 60715.06μs -- x10.55  <=  dm_tree.map_structure(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  7628.29μs            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  7622.97μs -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :  7548.69μs -- x0.99   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 22421.37μs -- x2.94   <=  jax.tree_util.tree_map(fn3, x, y, z)
  DM-Tree: 238211.56μs -- x31.23  <=  dm_tree.map_structure_up_to(x, fn3, x, y, z)
```

--------------------------------------------------------------------------------

## License

OpTree is released under the Apache License 2.0.

OpTree is heavily based on JAX's implementation of the PyTree utility, with deep refactoring and several improvements.
The original licenses can be found at [JAX's Apache License 2.0](https://github.com/google/jax/blob/HEAD/LICENSE) and [Tensorflow's Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/HEAD/LICENSE).
