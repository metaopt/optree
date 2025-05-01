<!-- markdownlint-disable html -->

# OpTree

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
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
conda install conda-forge::optree
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

The following options are available while building the Python C extension from the source:

```bash
export CMAKE_COMMAND="/path/to/custom/cmake"
export CMAKE_BUILD_TYPE="Debug"
export CMAKE_CXX_STANDARD="20"  # C++17 is tested on Linux/macOS (C++20 is required on Windows)
export OPTREE_CXX_WERROR="OFF"
export _GLIBCXX_USE_CXX11_ABI="1"
export pybind11_DIR="/path/to/custom/pybind11"
pip3 install .
```

Compiling from the source requires Python 3.9+, a compiler (`gcc` / `clang` / `icc` / `cl.exe`) that supports C++20 and a `cmake` installation.

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
>>> optree.tree_map(lambda x: x**2, tree)
{'b': (4, [9, 16]), 'a': 1, 'c': 25, 'd': 36}
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
>>> optree.tree_map(lambda x: x**2, {'a': [1, 2], 'b': [3]})
{'a': [1, 4], 'b': [9]}
>>> optree.tree_map(lambda x: x**2, {'b': [3], 'a': [1, 2]})
{'b': [9], 'a': [1, 4]}
```

> [!TIP]
>
> Since OpTree v0.14.1, a new namespace `optree.pytree` is introduced as aliases for `optree.tree_*` functions. The following examples are equivalent to the above:
>
> ```python
> >>> import optree.pytree as pt
> >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': 5, 'd': 6}
> >>> pt.flatten(tree)
> ([1, 2, 3, 4, 5, 6], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}))
> >>> pt.flatten(1)
> ([1], PyTreeSpec(*))
> >>> pt.flatten(None)
> ([], PyTreeSpec(None))
> >>> pt.map(lambda x: x**2, tree)
> {'b': (4, [9, 16]), 'a': 1, 'c': 25, 'd': 36}
> >>> pt.map(lambda x: x**2, {'a': [1, 2], 'b': [3]})
> {'a': [1, 4], 'b': [9]}
> >>> pt.map(lambda x: x**2, {'b': [3], 'a': [1, 2]})
> {'b': [9], 'a': [1, 4]}
> ```

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
The registry lookup uses the `is` operator to determine whether the type is matched.
So subclasses will need to explicitly register in the registry, otherwise, an object of that type will be considered a leaf.
The [`NoneType`](https://docs.python.org/3/library/constants.html#None) is a special case discussed in section [`None` is non-leaf Node vs. `None` is Leaf](#none-is-non-leaf-node-vs-none-is-leaf).

#### Registering a Container-like Custom Type as Non-leaf Nodes

A container-like Python type can be registered in the type registry with a pair of functions that specify:

- `flatten_func(container) -> (children, metadata, entries)`: convert an instance of the container type to a `(children, metadata, entries)` triple, where `children` is an iterable of subtrees and `entries` is an iterable of path entries of the container (e.g., indices or keys).
- `unflatten_func(metadata, children) -> container`: convert such a pair back to an instance of the container type.

The `metadata` is some necessary data apart from the children to reconstruct the container, e.g., the keys of the dictionary (the children are values).

The `entries` can be omitted (only returns a pair) or is optional to implement (returns `None`). If so, use `range(len(children))` (i.e., flat indices) as path entries of the current node. The signature for the flatten function can be one of the following:

- `flatten_func(container) -> (children, metadata, entries)`
- `flatten_func(container) -> (children, metadata, None)`
- `flatten_func(container) -> (children, metadata)`

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

class Torch2NumpyEntry(optree.PyTreeEntry):
    def __call__(self, obj):
        assert self.entry == 0
        return obj.cpu().detach().numpy()

    def codify(self, node=''):
        assert self.entry == 0
        return f'{node}.cpu().detach().numpy()'

optree.register_pytree_node(
    torch.Tensor,
    # (tensor) -> (children, metadata)
    flatten_func=lambda tensor: (
        (tensor.cpu().detach().numpy(),),
        {'dtype': tensor.dtype, 'device': tensor.device, 'requires_grad': tensor.requires_grad},
    ),
    # (metadata, children) -> tensor
    unflatten_func=lambda metadata, children: torch.tensor(children[0], **metadata),
    path_entry_type=Torch2NumpyEntry,
    namespace='torch2numpy',
)
```

```python
>>> tree = {'weight': torch.ones(size=(1, 2)).cuda(), 'bias': torch.zeros(size=(2,))}
>>> tree
{'weight': tensor([[1., 1.]], device='cuda:0'), 'bias': tensor([0., 0.])}

# Flatten without specifying the namespace
>>> optree.tree_flatten(tree)  # `torch.Tensor`s are leaf nodes
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

# Custom path entry type defines the pytree access behavior
>>> optree.tree_accessors(tree, namespace='torch2numpy')
[
    PyTreeAccessor(*['bias'].cpu().detach().numpy(), (MappingEntry(key='bias', type=<class 'dict'>), Torch2NumpyEntry(entry=0, type=<class 'torch.Tensor'>))),
    PyTreeAccessor(*['weight'].cpu().detach().numpy(), (MappingEntry(key='weight', type=<class 'dict'>), Torch2NumpyEntry(entry=0, type=<class 'torch.Tensor'>)))
]

# Unflatten back to a copy of the original object
>>> optree.tree_unflatten(treespec, leaves)
{'weight': tensor([[1., 1.]], device='cuda:0'), 'bias': tensor([0., 0.])}
```

Users can also extend the pytree registry by decorating the custom class and defining an instance method `tree_flatten` and a class method `tree_unflatten`.

```python
from collections import UserDict

@optree.register_pytree_node_class(namespace='mydict')
class MyDict(UserDict):
    TREE_PATH_ENTRY_TYPE = optree.MappingEntry  # used by accessor APIs

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
>>> optree.tree_flatten_with_accessor(tree, namespace='mydict')
(
    [
        PyTreeAccessor(*['c']['f'], (MappingEntry(key='c', type=<class 'MyDict'>), MappingEntry(key='f', type=<class 'MyDict'>))),
        PyTreeAccessor(*['c']['d'], (MappingEntry(key='c', type=<class 'MyDict'>), MappingEntry(key='d', type=<class 'MyDict'>))),
        PyTreeAccessor(*['b'], (MappingEntry(key='b', type=<class 'MyDict'>),)),
        PyTreeAccessor(*['a'][0], (MappingEntry(key='a', type=<class 'MyDict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
        PyTreeAccessor(*['a'][1], (MappingEntry(key='a', type=<class 'MyDict'>), SequenceEntry(index=1, type=<class 'tuple'>)))
    ],
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

> [!WARNING]
> For safety reasons, a `namespace` must be specified while registering a custom type. It is
> used to isolate the behavior of flattening and unflattening a pytree node type. This is to
> prevent accidental collisions between different libraries that may register the same type.

2. **The elements in the type registry are immutable.** Users can neither register the same type twice in the same namespace (i.e., update the type registry), nor remove a type from the type registry. To update the behavior of an already registered type, simply register it again with another `namespace`.

3. **Users cannot modify the behavior of already registered built-in types** listed in [Built-in PyTree Node Types](#built-in-pytree-node-types), such as key order sorting for `dict` and `collections.defaultdict`.

4. **Inherited subclasses are not implicitly registered.** The registry lookup uses `type(obj) is registered_type` rather than `isinstance(obj, registered_type)`. Users need to register the subclasses explicitly. To register all subclasses, it is easy to implement with [`metaclass`](https://docs.python.org/3/reference/datamodel.html#metaclasses) or [`__init_subclass__`](https://docs.python.org/3/reference/datamodel.html#customizing-class-creation), for example:

    ```python
    from collections import UserDict

    @optree.register_pytree_node_class(namespace='mydict')
    class MyDict(UserDict):
        TREE_PATH_ENTRY_TYPE = optree.MappingEntry  # used by accessor APIs

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
    >>> optree.tree_accessors(tree, namespace='mydict')
    [
        PyTreeAccessor(*['c']['f'], (MappingEntry(key='c', type=<class 'MyDict'>), MappingEntry(key='f', type=<class 'MyAnotherDict'>))),
        PyTreeAccessor(*['c']['d'], (MappingEntry(key='c', type=<class 'MyDict'>), MappingEntry(key='d', type=<class 'MyAnotherDict'>))),
        PyTreeAccessor(*['b'], (MappingEntry(key='b', type=<class 'MyDict'>),)),
        PyTreeAccessor(*['a'][0], (MappingEntry(key='a', type=<class 'MyDict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
        PyTreeAccessor(*['a'][1], (MappingEntry(key='a', type=<class 'MyDict'>), SequenceEntry(index=1, type=<class 'tuple'>)))
    ]
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
If `none_is_leaf=True`, the `None` object will be placed in the leaves list.
Otherwise, the `None` object will remain in the tree specification (structure).

```python
>>> import torch

>>> linear = torch.nn.Linear(in_features=3, out_features=2, bias=False)
>>> linear._parameters  # a container has None
OrderedDict({
    'weight': Parameter containing:
              tensor([[-0.6677,  0.5209,  0.3295],
                      [-0.4876, -0.3142,  0.1785]], requires_grad=True),
    'bias': None
})

>>> optree.tree_map(torch.zeros_like, linear._parameters)
OrderedDict({
    'weight': tensor([[0., 0., 0.],
                      [0., 0., 0.]]),
    'bias': None
})

>>> optree.tree_map(torch.zeros_like, linear._parameters, none_is_leaf=True)
Traceback (most recent call last):
    ...
TypeError: zeros_like(): argument 'input' (position 1) must be Tensor, not NoneType

>>> optree.tree_map(lambda t: torch.zeros_like(t) if t is not None else 0, linear._parameters, none_is_leaf=True)
OrderedDict({
    'weight': tensor([[0., 0., 0.],
                      [0., 0., 0.]]),
    'bias': 0
})
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
([1, 2, 3], PyTreeSpec(OrderedDict({'a': [*, *], 'b': [*]})))
>>> optree.tree_flatten(OrderedDict([('b', [3]), ('a', [1, 2])]))
([3, 1, 2], PyTreeSpec(OrderedDict({'b': [*], 'a': [*, *]})))
```

**Since OpTree v0.9.0, the key order of the reconstructed output dictionaries from `tree_unflatten` is guaranteed to be consistent with the key order of the input dictionaries in `tree_flatten`.**

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

> [!NOTE]
> Note that there are no restrictions on the `dict` to require the keys to be comparable (sortable).
> There can be multiple types of keys in the dictionary.
> The keys are sorted in ascending order by `key=lambda k: k` first if capable otherwise fallback to `key=lambda k: (f'{k.__class__.__module__}.{k.__class__.__qualname__}', k)`. This handles most cases.
>
> ```python
> >>> sorted({1: 2, 1.5: 1}.keys())
> [1, 1.5]
> >>> sorted({'a': 3, 1: 2, 1.5: 1}.keys())
> Traceback (most recent call last):
>     ...
> TypeError: '<' not supported between instances of 'int' and 'str'
> >>> sorted({'a': 3, 1: 2, 1.5: 1}.keys(), key=lambda k: (f'{k.__class__.__module__}.{k.__class__.__qualname__}', k))
> [1.5, 1, 'a']
> ```

--------------------------------------------------------------------------------

## Changelog

See [CHANGELOG.md](https://github.com/metaopt/optree/blob/HEAD/CHANGELOG.md).

--------------------------------------------------------------------------------

## License

OpTree is released under the Apache License 2.0.

OpTree is heavily based on JAX's implementation of the PyTree utility, with deep refactoring and several improvements.
The original licenses can be found at [JAX's Apache License 2.0](https://github.com/google/jax/blob/HEAD/LICENSE) and [Tensorflow's Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/HEAD/LICENSE).
