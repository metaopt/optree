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
- [License](#license)

--------------------------------------------------------------------------------

## Installation

Install from PyPI ([![PyPI](https://img.shields.io/pypi/v/optree?logo=pypi)](https://pypi.org/project/optree) / ![Status](https://img.shields.io/pypi/status/optree)):

```bash
pip3 install --upgrade optree
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
4. **Inherited subclasses are not implicitly registered.** The registration lookup uses `type(obj) is registered_type` rather than `isinstance(obj, registered_type)`. Users need to explicitly register all custom classes explicitly.

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

- OpTree ([`@44f7410`](https://github.com/metaopt/optree/commit/44f74108e0ff3a392593a2201c1ce33abbd76cdc))
- JAX XLA ([`jax[cpu] == 0.3.23`](https://pypi.org/project/jax/0.3.23))
- PyTorch ([`torch == 1.12.1`](https://pypi.org/project/torch/1.12.1))

All results are reported on a workstation with an AMD Ryzen 9 5950X CPU @ 4.45GHz in an isolated virtual environment with Python 3.10.8.
Run with the following command:

```bash
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

and AlexNet, ResNet18, ResNet50, ResNet101, ResNet152, VisionTransformerH14 (ViT-H/14), and SwinTransformerB (Swin-B) from [`torchvsion`](https://github.com/pytorch/vision).
Please refer to [`benchmark.py`](https://github.com/metaopt/optree/blob/HEAD/benchmark.py) for more details.

```text
TinyCustom(num_leaves=16, num_nodes=53, treespec=PyTreeSpec([OrderedDict([('tenso...), buffers=OrderedDict([])))])]))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :    27.18us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :    27.38us -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :    27.09us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:    76.18us -- x2.80   <=  jax.tree_util.tree_leaves(x)
  PyTorch:   671.56us -- x24.71  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :    63.18us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :    63.30us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:   133.89us -- x2.12   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:   248.43us -- x3.93   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
✔ OpTree :    91.51us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :    94.17us -- x1.03   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :    94.23us -- x1.03   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:   216.89us -- x2.37   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch:   940.32us -- x10.28  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
✔ OpTree :    99.84us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :   102.82us -- x1.03   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :   102.21us -- x1.02   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   236.62us -- x2.37   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch:   984.33us -- x9.86   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :   139.50us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :   142.06us -- x1.02   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :   143.19us -- x1.03   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   377.20us -- x2.70   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
AlexNet(num_leaves=32, num_nodes=188, treespec=PyTreeSpec(OrderedDict([('featur...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
✔ OpTree :    80.37us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :    87.58us -- x1.09   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :    87.06us -- x1.08   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   297.95us -- x3.71   <=  jax.tree_util.tree_leaves(x)
  PyTorch:  2650.24us -- x32.98  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :   245.71us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :   247.18us -- x1.01   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:   545.17us -- x2.22   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:   984.27us -- x4.01   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
✔ OpTree :   332.03us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :   338.27us -- x1.02   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :   337.91us -- x1.02   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:   866.38us -- x2.61   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch:  3654.14us -- x11.01  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
✔ OpTree :   347.63us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :   353.01us -- x1.02   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :   353.95us -- x1.02   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   882.86us -- x2.54   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch:  3703.29us -- x10.65  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :   498.91us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :   499.80us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :   499.17us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  1447.79us -- x2.90   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
ResNet18(num_leaves=244, num_nodes=698, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
✔ OpTree :   283.47us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :   290.72us -- x1.03   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :   288.77us -- x1.02   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   928.19us -- x3.27   <=  jax.tree_util.tree_leaves(x)
  PyTorch:  9306.54us -- x32.83  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :   816.54us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :   814.80us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  1669.60us -- x2.04   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  3476.16us -- x4.26   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  1145.87us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  1149.91us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  1145.46us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  2676.30us -- x2.34   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 13040.00us -- x11.38  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  1246.56us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  1241.41us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1236.65us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  2842.26us -- x2.28   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 13090.70us -- x10.50  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :  1754.06us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  1758.19us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  1763.34us -- x1.01   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4581.00us -- x2.61   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
ResNet50(num_leaves=640, num_nodes=1702, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :   749.18us            <=  optree.tree_leaves(x)                      (None is Node)
✔ OpTree :   735.95us -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :   737.40us -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  2210.58us -- x2.95   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 21962.28us -- x29.32  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  1967.52us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  1944.32us -- x0.99   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  3909.87us -- x1.99   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  8222.80us -- x4.18   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  2782.21us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  2788.91us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  2771.00us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  6235.03us -- x2.24   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 30523.06us -- x10.97  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  2997.47us            <=  optree.tree_map(fn1, x)                      (None is Node)
✔ OpTree :  2993.21us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  3004.80us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  6582.85us -- x2.20   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 30674.16us -- x10.23  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  4190.19us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree :  4187.02us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  4200.12us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 10519.68us -- x2.51   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
ResNet101(num_leaves=1252, num_nodes=3317, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1443.28us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1462.98us -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1414.41us -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4301.47us -- x2.98   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 42706.19us -- x29.59  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  3889.41us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  3885.40us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  7656.81us -- x1.97   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 16058.19us -- x4.13   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  5442.14us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  5422.35us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  5407.01us -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 12184.79us -- x2.24   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 59239.08us -- x10.89  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  5857.83us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  5845.61us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  5819.69us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 12816.78us -- x2.19   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 59487.90us -- x10.16  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  8145.57us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree :  8138.75us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  8148.81us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 20070.87us -- x2.46   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
ResNet152(num_leaves=1864, num_nodes=4932, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  2180.13us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  2170.51us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  2140.18us -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  6225.77us -- x2.86   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 62329.75us -- x28.59  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  5734.21us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  5715.35us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA: 11297.46us -- x1.97   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 22897.60us -- x3.99   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  7997.82us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  8009.89us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  7960.10us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 17619.27us -- x2.20   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 85951.24us -- x10.75  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  8524.99us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  8522.01us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  8512.22us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 18695.07us -- x2.19   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 86562.20us -- x10.15  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree : 11886.65us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree : 11928.96us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree : 11902.22us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 29821.16us -- x2.51   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
VisionTransformerH14(num_leaves=784, num_nodes=3420, treespec=PyTreeSpec(OrderedDict([('conv_p...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1651.80us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1651.72us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1647.63us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4725.58us -- x2.86   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 44551.83us -- x26.97  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :  4321.63us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :  4335.10us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  9133.98us -- x2.11   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 17448.01us -- x4.04   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  6116.50us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  6100.51us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  6095.86us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 14116.93us -- x2.31   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 62494.90us -- x10.22  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
✔ OpTree :  6262.65us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  6272.06us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  6272.94us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 14489.26us -- x2.31   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 62355.37us -- x9.96   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :  8886.77us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  8893.93us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  8932.03us -- x1.01   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 23434.28us -- x2.64   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

```text
SwinTransformerB(num_leaves=706, num_nodes=2867, treespec=PyTreeSpec(OrderedDict([('featur...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1369.87us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1382.60us -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1368.06us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4066.15us -- x2.97   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 37490.24us -- x27.37  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  3707.21us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  3693.14us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  7749.16us -- x2.09   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 14828.41us -- x4.00   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  5154.76us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
✔ OpTree :  5127.40us -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :  5149.86us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 12031.65us -- x2.33   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 52536.88us -- x10.19  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  5359.65us            <=  optree.tree_map(fn1, x)                      (None is Node)
✔ OpTree :  5334.72us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  5335.08us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 12371.49us -- x2.31   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 52645.16us -- x9.82   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  7535.16us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree :  7520.10us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  7531.14us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 19884.10us -- x2.64   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

--------------------------------------------------------------------------------

## License

OpTree is released under the Apache License 2.0.

OpTree is heavily based on JAX's implementation of the PyTree utility, with deep refactoring and several improvements.
The original licenses can be found at [JAX's Apache License 2.0](https://github.com/google/jax/blob/HEAD/LICENSE) and [Tensorflow's Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/HEAD/LICENSE).
