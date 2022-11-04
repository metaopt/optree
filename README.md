# OpTree

![Python 3.6+](https://img.shields.io/badge/Python-3.6%2B-brightgreen)
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
    - [Registering a Custom Container-like Type as Non-leaf Nodes](#registering-a-custom-container-like-type-as-non-leaf-nodes)
    - [Limitations of the PyTree Type Registry](#limitations-of-the-pytree-type-registry)
  - [`None` is non-leaf Node vs. `None` is Leaf](#none-is-non-leaf-node-vs-none-is-leaf)
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
git clone --depth=1 --recurse-submodules https://github.com/metaopt/optree.git
cd optree
pip3 install .
```

Compiling from the source requires Python 3.6+, a compiler (`gcc` / `clang` / `icc` / `cl.exe`) supports C++20 and a `cmake` installation.

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

This usually implies that the equal pytrees return equal lists of trees and the same tree structure.
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
So subclasses will need to explicitly register in the registration, otherwise, an object of that type will be considered as a leaf.
The `NoneType` is a special case discussed in section [`None` is non-leaf Node vs. `None` is Leaf](#none-is-non-leaf-node-vs-none-is-leaf).

#### Registering a Custom Container-like Type as Non-leaf Nodes

A container-like Python type can be registered in the container registry with a pair of functions that specify:

- `flatten_func(container) -> (children, metadata)`: convert an instance of the container type to a `(children, metadata)` pair, where `children` is an iterable of subtrees.
- `unflatten_func(metadata, children) -> container`: convert such a pair back to an instance of the container type.

The `metadata` is some necessary data apart from the children to reconstruct the container, e.g., the keys of the dictionary (the children are values).

```python
>>> import torch

>>> optree.register_pytree_node(
...     torch.Tensor,
...     flatten_func=lambda tensor: (
...         (tensor.cpu().numpy(),),
...         dict(dtype=tensor.dtype, device=tensor.device, requires_grad=tensor.requires_grad),
...     ),
...     unflatten_func=lambda metadata, children: torch.tensor(children[0], **metadata),
... )
<class 'torch.Tensor'>

>>> tree = {'weight': torch.ones(size=(1, 2)).cuda(), 'bias': torch.zeros(size=(2,))}
>>> tree
{'weight': tensor([[1., 1.]], device='cuda:0'), 'bias': tensor([0., 0.])}

>>> leaves, treespec = optree.tree_flatten(tree)
>>> leaves, treespec
(
    [array([0., 0.], dtype=float32), array([[1., 1.]], dtype=float32)],
    PyTreeSpec({
        'bias': CustomTreeNode(Tensor[{'dtype': torch.float32, 'device': device(type='cpu'), 'requires_grad': False}], [*]),
        'weight': CustomTreeNode(Tensor[{'dtype': torch.float32, 'device': device(type='cuda', index=0), 'requires_grad': False}], [*])
    })
)

>>> optree.tree_unflatten(treespec, leaves)
{'bias': tensor([0., 0.]), 'weight': tensor([[1., 1.]], device='cuda:0')}
```

Users can also extend the pytree registry by decorating the custom class and defining an instance method `tree_flatten` and a class method `tree_unflatten`.

```python
>>> from collections import UserDict
...
... @optree.register_pytree_node_class
... class MyDict(UserDict):
...     def tree_flatten(self):
...         reversed_keys = sorted(self.keys(), reverse=True)
...         return [self[key] for key in reversed_keys], reversed_keys
...
...     @classmethod
...     def tree_unflatten(metadata, children):
...         return MyDict(zip(metadata, children))

>>> optree.tree_flatten(MyDict(b=2, a=1, c=3))
([3, 2, 1], PyTreeSpec(CustomTreeNode(MyDict[['c', 'b', 'a']], [*, *, *])))
```

#### Limitations of the PyTree Type Registry

There are several limitations of the pytree type registry:

1. **The type registry is per-interpreter-dependent.** This means registering a custom type in the registry affects all modules that use OpTree. The type registry does not support per-module isolation such as namespaces.
2. **The elements in the type registry are immutable.** Users either cannot register the same type twice (i.e., update the type registry). Nor cannot remove a type from the type registry.
3. **Users cannot modify the behavior of already registered built-in types** listed [Built-in PyTree Node Types](#built-in-pytree-node-types), such as key order sorting for `dict` and `collections.defaultdict`.
4. **Inherited subclasses are not implicitly registered.** The registration lookup uses `type(obj) is registered_type` rather than `isinstance(obj, registered_type)`. Users need to register the subclasses explicitly.

### `None` is non-leaf Node vs. `None` is Leaf

The [`None`](https://docs.python.org/3/library/constants.html#None) object is a special object in the Python language.
It serves some of the same purposes as `null` (a pointer does not point to anything) in other programming languages, which denotes a variable is empty or marks default parameters.
However, the `None` object is a singleton object rather than a pointer.
It may also serve as a sentinel value.
In addition, if a function has returned without any return value, it also implicitly returns the `None` object.

By default, the `None` object is considered a non-leaf node in the tree with arity 0, i.e., _**a non-leaf node that has no children**_.
This is slightly different than the definition of a non-leaf node as discussed above.
While flattening a tree, it will remain in the tree structure definitions rather than in the leaves list.

```python
>>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
>>> optree.tree_flatten(tree)
([1, 2, 3, 4, 5], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *}))
>>> optree.tree_flatten(tree, none_is_leaf=True)
([1, 2, 3, 4, None, 5], PyTreeSpec(NoneIsLeaf, {'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}))
>>> optree.tree_flatten(1)
([1], PyTreeSpec(*))
>>> optree.tree_flatten(None)
([], PyTreeSpec(None))
>>> optree.tree_flatten(None, none_is_leaf=True)
([None], PyTreeSpec(NoneIsLeaf, *))
```

OpTree provides a keyword argument `none_is_leaf` to determine whether to consider the `None` object as a leaf, like other opaque objects.
If `none_is_leaf=True`, the `None` object will place in the leaves list.
Otherwise, the `None` object will remain in the tree specification (structure).

```python
>>> import torch

>>> linear = torch.nn.Linear(in_features=3, out_features=2, bias=False)
>>> linear._parameters
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
- JAX XLA ([`jax[cpu] == 0.3.23`](https://pypi.org/project/jax/0.3.23))
- PyTorch ([`torch == 1.13.0`](https://pypi.org/project/torch/1.13.0))

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
tiny_custom = nn.Sequential(
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

| Module               | Nodes | Leaves | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:---------------------|------:|-------:|------------:|-------------:|-------------:|----------------:|----------------:|
| TinyCustom           |    53 |     16 |       26.23 |        72.65 |       585.64 |            2.77 |           22.33 |
| AlexNet              |   188 |     32 |       94.66 |       266.28 |      2210.98 |            2.81 |           23.36 |
| ResNet18             |   698 |    244 |      313.95 |       800.61 |      8413.09 |            2.55 |           26.80 |
| ResNet34             |  1242 |    436 |      607.69 |      1475.81 |     14854.69 |            2.43 |           24.44 |
| ResNet50             |  1702 |    640 |      815.27 |      1949.11 |     20424.27 |            2.39 |           25.05 |
| ResNet101            |  3317 |   1252 |     1679.05 |      3880.55 |     39937.00 |            2.31 |           23.79 |
| ResNet152            |  4932 |   1864 |     2546.81 |      5876.13 |     60198.69 |            2.31 |           23.64 |
| VisionTransformerH14 |  3420 |    784 |     1823.95 |      4445.94 |     42621.66 |            2.44 |           23.37 |
| SwinTransformerB     |  2881 |    706 |     1553.94 |      3958.88 |     36197.56 |            2.55 |           23.29 |
|                      |       |        |             |              |  **Average** |       **x2.51** |      **x24.01** |

### Tree UnFlatten

| Module               | Nodes | Leaves | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:---------------------|------:|-------:|------------:|-------------:|-------------:|----------------:|----------------:|
| TinyCustom           |    53 |     16 |       61.01 |       121.27 |       226.04 |            1.99 |            3.70 |
| AlexNet              |   188 |     32 |      235.37 |       492.55 |       862.87 |            2.09 |            3.67 |
| ResNet18             |   698 |    244 |      771.62 |      1514.16 |      2981.51 |            1.96 |            3.86 |
| ResNet34             |  1242 |    436 |     1397.50 |      2768.87 |      5458.85 |            1.98 |            3.91 |
| ResNet50             |  1702 |    640 |     1884.59 |      3682.84 |      7494.53 |            1.95 |            3.98 |
| ResNet101            |  3317 |   1252 |     3844.82 |      7237.28 |     14820.25 |            1.88 |            3.85 |
| ResNet152            |  4932 |   1864 |     5682.82 |     10812.01 |     21647.86 |            1.90 |            3.81 |
| VisionTransformerH14 |  3420 |    784 |     4328.64 |      8727.60 |     16262.43 |            2.02 |            3.76 |
| SwinTransformerB     |  2881 |    706 |     3660.17 |      7357.22 |     13516.73 |            2.01 |            3.69 |
|                      |       |        |             |              |  **Average** |       **x1.98** |       **x3.80** |

### Tree Copy

| Module               | Nodes | Leaves | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:---------------------|------:|-------:|------------:|-------------:|-------------:|----------------:|----------------:|
| TinyCustom           |    53 |     16 |       95.68 |       202.91 |       824.76 |            2.12 |            8.62 |
| AlexNet              |   188 |     32 |      325.85 |       760.83 |      3163.96 |            2.33 |            9.71 |
| ResNet18             |   698 |    244 |     1131.25 |      2396.90 |     11425.73 |            2.12 |           10.10 |
| ResNet34             |  1242 |    436 |     2099.06 |      4321.82 |     20640.78 |            2.06 |            9.83 |
| ResNet50             |  1702 |    640 |     2828.82 |      5747.07 |     28492.64 |            2.03 |           10.07 |
| ResNet101            |  3317 |   1252 |     5658.26 |     11466.17 |     56537.20 |            2.03 |            9.99 |
| ResNet152            |  4932 |   1864 |     8321.20 |     16697.24 |     82391.62 |            2.01 |            9.90 |
| VisionTransformerH14 |  3420 |    784 |     6385.36 |     13552.72 |     61474.43 |            2.12 |            9.63 |
| SwinTransformerB     |  2881 |    706 |     5300.50 |     11303.14 |     50689.66 |            2.13 |            9.56 |
|                      |       |        |             |              |  **Average** |       **x2.11** |       **x9.71** |

### Tree Map

| Module               | Nodes | Leaves | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:---------------------|------:|-------:|------------:|-------------:|-------------:|----------------:|----------------:|
| TinyCustom           |    53 |     16 |       99.41 |       210.24 |       878.38 |            2.11 |            8.84 |
| AlexNet              |   188 |     32 |      347.83 |       798.17 |      3226.00 |            2.29 |            9.27 |
| ResNet18             |   698 |    244 |     1205.82 |      2554.11 |     11848.85 |            2.12 |            9.83 |
| ResNet34             |  1242 |    436 |     2213.00 |      4553.61 |     21232.94 |            2.06 |            9.59 |
| ResNet50             |  1702 |    640 |     2974.05 |      6171.12 |     28682.03 |            2.07 |            9.64 |
| ResNet101            |  3317 |   1252 |     5969.70 |     12273.57 |     57644.18 |            2.06 |            9.66 |
| ResNet152            |  4932 |   1864 |     8742.73 |     17892.96 |     83210.50 |            2.05 |            9.52 |
| VisionTransformerH14 |  3420 |    784 |     6541.53 |     13832.67 |     60161.13 |            2.11 |            9.20 |
| SwinTransformerB     |  2881 |    706 |     5529.90 |     11842.31 |     51890.98 |            2.14 |            9.38 |
|                      |       |        |             |              |  **Average** |       **x2.11** |       **x9.44** |

### Tree Map (nargs)

| Module               | Nodes | Leaves | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:---------------------|------:|-------:|------------:|-------------:|-------------:|----------------:|----------------:|
| TinyCustom           |    53 |     16 |      142.80 |       362.08 |          N/A |            2.54 |             N/A |
| AlexNet              |   188 |     32 |      490.24 |      1328.78 |          N/A |            2.71 |             N/A |
| ResNet18             |   698 |    244 |     1700.77 |      4255.26 |          N/A |            2.50 |             N/A |
| ResNet34             |  1242 |    436 |     3139.45 |      7650.95 |          N/A |            2.44 |             N/A |
| ResNet50             |  1702 |    640 |     4234.40 |     10229.45 |          N/A |            2.42 |             N/A |
| ResNet101            |  3317 |   1252 |     8219.68 |     19835.67 |          N/A |            2.41 |             N/A |
| ResNet152            |  4932 |   1864 |    12406.75 |     29422.20 |          N/A |            2.37 |             N/A |
| VisionTransformerH14 |  3420 |    784 |     9077.64 |     22913.44 |          N/A |            2.52 |             N/A |
| SwinTransformerB     |  2881 |    706 |     7958.56 |     19631.91 |          N/A |            2.47 |             N/A |
|                      |       |        |             |              |  **Average** |       **x2.49** |             N/A |

```text
TinyCustom(num_leaves=16, num_nodes=53, treespec=PyTreeSpec([OrderedDict([('tenso...), buffers=OrderedDict([])))])]))
| Subject          | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:-----------------|------------:|-------------:|-------------:|----------------:|----------------:|
| Tree Flatten     |       26.23 |        72.65 |       585.64 |            2.77 |           22.33 |
| Tree UnFlatten   |       61.01 |       121.27 |       226.04 |            1.99 |            3.70 |
| Tree Copy        |       95.68 |       202.91 |       824.76 |            2.12 |            8.62 |
| Tree Map         |       99.41 |       210.24 |       878.38 |            2.11 |            8.84 |
| Tree Map (nargs) |      142.80 |       362.08 |       nan    |            2.54 |          nan    |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
✔ OpTree :    26.23us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :    26.78us -- x1.02   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :    26.60us -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:    72.65us -- x2.77   <=  jax.tree_util.tree_leaves(x)
  PyTorch:   585.64us -- x22.33  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :    61.01us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :    61.15us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:   121.27us -- x1.99   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:   226.04us -- x3.70   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :    95.68us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
✔ OpTree :    94.52us -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :    95.03us -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:   202.91us -- x2.12   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch:   824.76us -- x8.62   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :    99.41us            <=  optree.tree_map(fn1, x)                      (None is Node)
✔ OpTree :    98.67us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :    99.19us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   210.24us -- x2.11   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch:   878.38us -- x8.84   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :   142.80us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :   147.90us -- x1.04   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :   149.76us -- x1.05   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   362.08us -- x2.54   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
AlexNet(num_leaves=32, num_nodes=188, treespec=PyTreeSpec(OrderedDict([('featur...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:-----------------|------------:|-------------:|-------------:|----------------:|----------------:|
| Tree Flatten     |       94.66 |       266.28 |      2210.98 |            2.81 |           23.36 |
| Tree UnFlatten   |      235.37 |       492.55 |       862.87 |            2.09 |            3.67 |
| Tree Copy        |      325.85 |       760.83 |      3163.96 |            2.33 |            9.71 |
| Tree Map         |      347.83 |       798.17 |      3226.00 |            2.29 |            9.27 |
| Tree Map (nargs) |      490.24 |      1328.78 |       nan    |            2.71 |          nan    |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
  OpTree :    94.66us            <=  optree.tree_leaves(x)                      (None is Node)
  OpTree :    95.13us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :    84.50us -- x0.89   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   266.28us -- x2.81   <=  jax.tree_util.tree_leaves(x)
  PyTorch:  2210.98us -- x23.36  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :   235.37us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :   247.28us -- x1.05   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:   492.55us -- x2.09   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:   862.87us -- x3.67   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :   325.85us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
  OpTree :   362.23us -- x1.11   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :   323.42us -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:   760.83us -- x2.33   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch:  3163.96us -- x9.71   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :   347.83us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :   346.01us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :   336.37us -- x0.97   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   798.17us -- x2.29   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch:  3226.00us -- x9.27   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :   490.24us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :   496.10us -- x1.01   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :   483.52us -- x0.99   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  1328.78us -- x2.71   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
ResNet18(num_leaves=244, num_nodes=698, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:-----------------|------------:|-------------:|-------------:|----------------:|----------------:|
| Tree Flatten     |      313.95 |       800.61 |      8413.09 |            2.55 |           26.80 |
| Tree UnFlatten   |      771.62 |      1514.16 |      2981.51 |            1.96 |            3.86 |
| Tree Copy        |     1131.25 |      2396.90 |     11425.73 |            2.12 |           10.10 |
| Tree Map         |     1205.82 |      2554.11 |     11848.85 |            2.12 |            9.83 |
| Tree Map (nargs) |     1700.77 |      4255.26 |       nan    |            2.50 |          nan    |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :   313.95us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :   319.86us -- x1.02   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :   293.63us -- x0.94   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   800.61us -- x2.55   <=  jax.tree_util.tree_leaves(x)
  PyTorch:  8413.09us -- x26.80  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :   771.62us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :   770.51us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  1514.16us -- x1.96   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  2981.51us -- x3.86   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  1131.25us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  1138.35us -- x1.01   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  1103.02us -- x0.98   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  2396.90us -- x2.12   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 11425.73us -- x10.10  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  1205.82us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  1198.64us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1184.54us -- x0.98   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  2554.11us -- x2.12   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 11848.85us -- x9.83   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  1700.77us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  1699.87us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :  1677.94us -- x0.99   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4255.26us -- x2.50   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
ResNet34(num_leaves=436, num_nodes=1242, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:-----------------|------------:|-------------:|-------------:|----------------:|----------------:|
| Tree Flatten     |      607.69 |      1475.81 |     14854.69 |            2.43 |           24.44 |
| Tree UnFlatten   |     1397.50 |      2768.87 |      5458.85 |            1.98 |            3.91 |
| Tree Copy        |     2099.06 |      4321.82 |     20640.78 |            2.06 |            9.83 |
| Tree Map         |     2213.00 |      4553.61 |     21232.94 |            2.06 |            9.59 |
| Tree Map (nargs) |     3139.45 |      7650.95 |       nan    |            2.44 |          nan    |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
  OpTree :   607.69us            <=  optree.tree_leaves(x)                      (None is Node)
  OpTree :   605.88us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :   541.75us -- x0.89   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  1475.81us -- x2.43   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 14854.69us -- x24.44  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :  1397.50us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :  1409.78us -- x1.01   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  2768.87us -- x1.98   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  5458.85us -- x3.91   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  2099.06us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  2080.91us -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  2037.15us -- x0.97   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  4321.82us -- x2.06   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 20640.78us -- x9.83   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  2213.00us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  2208.49us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  2167.27us -- x0.98   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4553.61us -- x2.06   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 21232.94us -- x9.59   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  3139.45us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  3140.90us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :  3118.84us -- x0.99   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  7650.95us -- x2.44   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
ResNet50(num_leaves=640, num_nodes=1702, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:-----------------|------------:|-------------:|-------------:|----------------:|----------------:|
| Tree Flatten     |      815.27 |      1949.11 |     20424.27 |            2.39 |           25.05 |
| Tree UnFlatten   |     1884.59 |      3682.84 |      7494.53 |            1.95 |            3.98 |
| Tree Copy        |     2828.82 |      5747.07 |     28492.64 |            2.03 |           10.07 |
| Tree Map         |     2974.05 |      6171.12 |     28682.03 |            2.07 |            9.64 |
| Tree Map (nargs) |     4234.40 |     10229.45 |       nan    |            2.42 |          nan    |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
  OpTree :   815.27us            <=  optree.tree_leaves(x)                      (None is Node)
  OpTree :   826.64us -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :   722.10us -- x0.89   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  1949.11us -- x2.39   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 20424.27us -- x25.05  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :  1884.59us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :  1887.39us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  3682.84us -- x1.95   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  7494.53us -- x3.98   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  2828.82us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  2840.35us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  2743.29us -- x0.97   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  5747.07us -- x2.03   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 28492.64us -- x10.07  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  2974.05us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  3007.56us -- x1.01   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  2941.31us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  6171.12us -- x2.07   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 28682.03us -- x9.64   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  4234.40us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  4258.74us -- x1.01   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :  4223.89us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 10229.45us -- x2.42   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
ResNet101(num_leaves=1252, num_nodes=3317, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:-----------------|------------:|-------------:|-------------:|----------------:|----------------:|
| Tree Flatten     |     1679.05 |      3880.55 |     39937.00 |            2.31 |           23.79 |
| Tree UnFlatten   |     3844.82 |      7237.28 |     14820.25 |            1.88 |            3.85 |
| Tree Copy        |     5658.26 |     11466.17 |     56537.20 |            2.03 |            9.99 |
| Tree Map         |     5969.70 |     12273.57 |     57644.18 |            2.06 |            9.66 |
| Tree Map (nargs) |     8219.68 |     19835.67 |       nan    |            2.41 |          nan    |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
  OpTree :  1679.05us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1641.53us -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1499.52us -- x0.89   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  3880.55us -- x2.31   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 39937.00us -- x23.79  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  3844.82us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  3834.13us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  7237.28us -- x1.88   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 14820.25us -- x3.85   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  5658.26us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  5694.40us -- x1.01   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  5492.35us -- x0.97   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 11466.17us -- x2.03   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 56537.20us -- x9.99   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  5969.70us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  5984.09us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  5882.16us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 12273.57us -- x2.06   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 57644.18us -- x9.66   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  8219.68us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  8240.13us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :  8180.94us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 19835.67us -- x2.41   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
ResNet152(num_leaves=1864, num_nodes=4932, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:-----------------|------------:|-------------:|-------------:|----------------:|----------------:|
| Tree Flatten     |     2546.81 |      5876.13 |     60198.69 |            2.31 |           23.64 |
| Tree UnFlatten   |     5682.82 |     10812.01 |     21647.86 |            1.90 |            3.81 |
| Tree Copy        |     8321.20 |     16697.24 |     82391.62 |            2.01 |            9.90 |
| Tree Map         |     8742.73 |     17892.96 |     83210.50 |            2.05 |            9.52 |
| Tree Map (nargs) |    12406.75 |     29422.20 |       nan    |            2.37 |          nan    |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
  OpTree :  2546.81us            <=  optree.tree_leaves(x)                      (None is Node)
  OpTree :  2547.52us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  2278.54us -- x0.89   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  5876.13us -- x2.31   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 60198.69us -- x23.64  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :  5682.82us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :  5730.33us -- x1.01   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA: 10812.01us -- x1.90   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 21647.86us -- x3.81   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  8321.20us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  8318.23us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  7986.33us -- x0.96   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 16697.24us -- x2.01   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 82391.62us -- x9.90   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  8742.73us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  8725.09us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  8586.67us -- x0.98   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 17892.96us -- x2.05   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 83210.50us -- x9.52   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree : 12406.75us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree : 12372.89us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree : 12194.97us -- x0.98   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 29422.20us -- x2.37   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
VisionTransformerH14(num_leaves=784, num_nodes=3420, treespec=PyTreeSpec(OrderedDict([('conv_p...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:-----------------|------------:|-------------:|-------------:|----------------:|----------------:|
| Tree Flatten     |     1823.95 |      4445.94 |     42621.66 |            2.44 |           23.37 |
| Tree UnFlatten   |     4328.64 |      8727.60 |     16262.43 |            2.02 |            3.76 |
| Tree Copy        |     6385.36 |     13552.72 |     61474.43 |            2.12 |            9.63 |
| Tree Map         |     6541.53 |     13832.67 |     60161.13 |            2.11 |            9.20 |
| Tree Map (nargs) |     9077.64 |     22913.44 |       nan    |            2.52 |          nan    |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1823.95us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1835.76us -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1708.92us -- x0.94   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4445.94us -- x2.44   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 42621.66us -- x23.37  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :  4328.64us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :  4390.96us -- x1.01   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  8727.60us -- x2.02   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 16262.43us -- x3.76   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  6385.36us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  6284.22us -- x0.98   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  6176.32us -- x0.97   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 13552.72us -- x2.12   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 61474.43us -- x9.63   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  6541.53us            <=  optree.tree_map(fn1, x)                      (None is Node)
✔ OpTree :  6305.01us -- x0.96   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  6387.82us -- x0.98   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 13832.67us -- x2.11   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 60161.13us -- x9.20   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :  9077.64us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  9088.96us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  9092.49us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 22913.44us -- x2.52   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
SwinTransformerB(num_leaves=706, num_nodes=2881, treespec=PyTreeSpec(OrderedDict([('featur...]), buffers=OrderedDict([])))])))
| Subject          | OpTree (us) | JAX XLA (us) | PyTorch (us) | Speedup (J / O) | Speedup (P / O) |
|:-----------------|------------:|-------------:|-------------:|----------------:|----------------:|
| Tree Flatten     |     1553.94 |      3958.88 |     36197.56 |            2.55 |           23.29 |
| Tree UnFlatten   |     3660.17 |      7357.22 |     13516.73 |            2.01 |            3.69 |
| Tree Copy        |     5300.50 |     11303.14 |     50689.66 |            2.13 |            9.56 |
| Tree Map         |     5529.90 |     11842.31 |     51890.98 |            2.14 |            9.38 |
| Tree Map (nargs) |     7958.56 |     19631.91 |       nan    |            2.47 |          nan    |

### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1553.94us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1555.72us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1456.35us -- x0.94   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  3958.88us -- x2.55   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 36197.56us -- x23.29  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  3660.17us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  3629.90us -- x0.99   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  7357.22us -- x2.01   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 13516.73us -- x3.69   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  5300.50us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  5291.56us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  5224.23us -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 11303.14us -- x2.13   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 50689.66us -- x9.56   <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  5529.90us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  5500.23us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  5423.18us -- x0.98   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 11842.31us -- x2.14   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 51890.98us -- x9.38   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  7958.56us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  7905.76us -- x0.99   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
✔ OpTree :  7894.04us -- x0.99   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 19631.91us -- x2.47   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

--------------------------------------------------------------------------------

## License

OpTree is released under the Apache License 2.0.

OpTree is heavily based on JAX's implementation of the PyTree utility, with deep refactoring and several improvements.
The original licenses can be found at [JAX's Apache License 2.0](https://github.com/google/jax/blob/HEAD/LICENSE) and [Tensorflow's Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/HEAD/LICENSE).
