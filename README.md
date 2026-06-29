<!-- markdownlint-disable html -->

# OpTree

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
[![PyPI](https://img.shields.io/pypi/v/optree?logo=pypi)](https://pypi.org/project/optree)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/metaopt/optree/build.yml?label=build&logo=github)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/metaopt/optree/tests.yml?label=tests&logo=github)
[![Codecov](https://img.shields.io/codecov/c/github/metaopt/optree/main?logo=codecov)](https://codecov.io/gh/metaopt/optree)
[![Documentation Status](https://img.shields.io/readthedocs/optree?logo=readthedocs)](https://optree.readthedocs.io)
[![Downloads](https://static.pepy.tech/personalized-badge/optree?period=total&left_color=gray&right_color=blue&left_text=downloads)](https://pepy.tech/project/optree)
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
pip3 install git+https://github.com/metaopt/optree.git
```

Or, clone this repo and install manually:

```bash
git clone --depth=1 https://github.com/metaopt/optree.git && cd optree

pip3 install .
```

The following options are available while building the Python C extension from source:

```bash
export CMAKE_COMMAND="/path/to/custom/cmake"
export CMAKE_BUILD_TYPE="Debug"
export CMAKE_CXX_STANDARD="20"  # C++17 is tested on Linux/macOS (C++20 is required on Windows)
export OPTREE_CXX_WERROR="OFF"
export _GLIBCXX_USE_CXX11_ABI="1"  # set to 0 to use the old libstdc++ ABI
export _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR="1"  # set to "" to disable the workaround for MSVC mutex layout change in VS 2022 v17.10+
export pybind11_DIR="/path/to/custom/pybind11"

pip3 install .
```

Compiling from source requires Python 3.9+, a C++ compiler (`g++` / `clang++` / `icpx` / `cl.exe`) that supports C++20, and a `cmake` installation.

--------------------------------------------------------------------------------

## PyTrees

A PyTree is a recursive structure that can be an arbitrarily nested Python container (e.g., `tuple`, `list`, `dict`, `OrderedDict`, `namedtuple`, etc.) and/or an opaque Python object.
The key concepts of tree operations are tree flattening and its inverse (unflattening).
Additional tree operations can be built from these two primitives (e.g., `tree_map = tree_unflatten ∘ map ∘ tree_flatten`).

Tree flattening traverses the entire tree in a left-to-right depth-first manner and returns the leaves in a deterministic order.

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

This usually implies that equal pytrees produce equal lists of leaves and the same tree structure.
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
>
> Since OpTree v0.16.0, a re-export API `optree.pytree.reexport(...)` is available to create a new module that exports all the `optree.pytree` APIs with a given namespace.
> This is useful for downstream libraries to create their own pytree utilities without passing the `namespace` argument explicitly.
>
> ```python
> # foo/__init__.py
> import optree
> pytree = optree.pytree.reexport(namespace='foo')
> del optree
>
> # foo/bar.py
> from foo import pytree
>
> @pytree.dataclasses.dataclass
> class Bar:
>     a: int
>     b: float
>
> # User code
> >>> import foo
>
> >>> foo.pytree.flatten({'a': 1, 'b': 2, 'c': foo.bar.Bar(3, 4.0)})
> (
>     [1, 2, 3, 4.0],
>     PyTreeSpec({'a': *, 'b': *, 'c': CustomTreeNode(Bar[()], [*, *])}, namespace='foo')
> )
>
> >>> foo.pytree.functools.reduce(lambda x, y: x * y, {'a': 1, 'b': 2, 'c': foo.bar.Bar(3, 4.0)})
> 24.0
> ```

### Tree Nodes and Leaves

A tree is a collection of non-leaf nodes and leaf nodes, where the leaf nodes are opaque objects having no children to flatten.
`optree.tree_flatten(...)` will flatten the tree and return a list of leaf nodes while the non-leaf nodes will be stored in the tree structure specification.

#### Built-in PyTree Node Types

OpTree out-of-the-box supports the following Python container types in the global registry:

- [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple)
- [`list`](https://docs.python.org/3/library/stdtypes.html#list)
- [`dict`](https://docs.python.org/3/library/stdtypes.html#dict)
- [`collections.namedtuple`](https://docs.python.org/3/library/collections.html#collections.namedtuple) and its subclasses
- [`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict)
- [`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict)
- [`collections.deque`](https://docs.python.org/3/library/collections.html#collections.deque)
- [`PyStructSequence`](https://docs.python.org/3/c-api/tuple.html#struct-sequence-objects) types created by C API [`PyStructSequence_NewType`](https://docs.python.org/3/c-api/tuple.html#c.PyStructSequence_NewType)
- [`frozendict`](https://docs.python.org/3/library/stdtypes.html#frozendict) (Python 3.15+)

These types are considered non-leaf nodes in the tree.
Python objects whose type is not registered are treated as leaf nodes.
The registry lookup uses the `is` operator to determine whether the type matches, so subclasses need to be registered explicitly; otherwise, their instances will be treated as leaves.
The [`NoneType`](https://docs.python.org/3/library/constants.html#None) is a special case discussed in section [`None` is Non-leaf Node vs. `None` is Leaf](#none-is-non-leaf-node-vs-none-is-leaf).

#### Registering a Container-like Custom Type as Non-leaf Nodes

A container-like Python type can be registered in the type registry with a pair of functions that specify:

- `flatten_func(container) -> (children, metadata, entries)`: convert an instance of the container type to a `(children, metadata, entries)` triple, where `children` is an iterable of subtrees and `entries` is an iterable of path entries of the container (e.g., indices or keys).
- `unflatten_func(metadata, children) -> container`: convert such a pair back to an instance of the container type.

The `metadata` is some necessary data apart from the children to reconstruct the container, e.g., the keys of the dictionary (the children are values).

The `entries` can be omitted (only return a pair) or are optional to implement (return `None`). If so, use `range(len(children))` (i.e., flat indices) as path entries of the current node. The signature for the flatten function can be one of the following:

- `flatten_func(container) -> (children, metadata, entries)`
- `flatten_func(container) -> (children, metadata, None)`
- `flatten_func(container) -> (children, metadata)`

The following examples show how to register custom types and use them with `tree_flatten` and `tree_map`. Please refer to section [Notes about the PyTree Type Registry](#notes-about-the-pytree-type-registry) for more information.

```python
# Register a custom type with lambda functions
optree.register_pytree_node(
    set,
    lambda s: (sorted(s), None),        # flatten: (set) -> (children, metadata)
    lambda _, children: set(children),  # unflatten: (metadata, children) -> set
    namespace='set',
)

# Register a custom type into a namespace with accessor support
import types

# This can be whatever your container type is.
class MyContainer(types.SimpleNamespace):
    """A simple container type based on SimpleNamespace."""

# (Optional) Define a custom path entry type for your container for accessor support.
# Here we showcase how to define one. In practice, you can use the built-in `optree.GetAttrEntry`.
class MyContainerEntry(optree.PyTreeEntry):
    def __call__(self, obj):
        return getattr(obj, self.entry)

    def codify(self, node=''):
        return f'{node}.{self.entry}'

optree.register_pytree_node(
    MyContainer,
    flatten_func=lambda ct: (                 # flatten: (MyContainer) -> (children, metadata, entries)
        list(vars(ct).values()),
        list(vars(ct).keys()),
        list(vars(ct).keys()),
    ),
    unflatten_func=lambda keys, values: (     # unflatten: (metadata, children) -> MyContainer
        MyContainer(**dict(zip(keys, values)))
    ),
    path_entry_type=MyContainerEntry,
    namespace='mycontainer',
)
```

```python
>>> tree = {'config': MyContainer(lr=0.01, momentum=0.9), 'steps': 1000}

# Flatten without specifying the namespace
>>> optree.tree_flatten(tree)  # `MyContainer`s are leaf nodes
([MyContainer(lr=0.01, momentum=0.9), 1000], PyTreeSpec({'config': *, 'steps': *}))

# Flatten with the namespace
>>> leaves, treespec = optree.tree_flatten(tree, namespace='mycontainer')
>>> leaves, treespec
(
    [0.01, 0.9, 1000],
    PyTreeSpec(
        {
            'config': CustomTreeNode(MyContainer[['lr', 'momentum']], [*, *]),
            'steps': *
        },
        namespace='mycontainer'
    )
)

# Custom `entries` are defined as attribute names
>>> optree.tree_paths(tree, namespace='mycontainer')
[('config', 'lr'), ('config', 'momentum'), ('steps',)]

# Custom path entry type defines the pytree access behavior
>>> optree.tree_accessors(tree, namespace='mycontainer')
[
    PyTreeAccessor(*['config'].lr, (MappingEntry(key='config', type=<class 'dict'>), MyContainerEntry(entry='lr', type=<class 'MyContainer'>))),
    PyTreeAccessor(*['config'].momentum, (MappingEntry(key='config', type=<class 'dict'>), MyContainerEntry(entry='momentum', type=<class 'MyContainer'>))),
    PyTreeAccessor(*['steps'], (MappingEntry(key='steps', type=<class 'dict'>),))
]

# Unflatten back to a copy of the original object
>>> optree.tree_unflatten(treespec, leaves)
{'config': MyContainer(lr=0.01, momentum=0.9), 'steps': 1000}
```

Users can also extend the pytree registry by decorating the custom class and defining an instance method `__tree_flatten__` and a class method `__tree_unflatten__`.

```python
from collections import UserDict

@optree.register_pytree_node_class(namespace='mydict')
class MyDict(UserDict):
    TREE_PATH_ENTRY_TYPE = optree.MappingEntry  # used by accessor APIs

    def __tree_flatten__(self):  # -> (children, metadata, entries)
        reversed_keys = sorted(self.keys(), reverse=True)
        return (
            [self[key] for key in reversed_keys],  # children
            reversed_keys,  # metadata
            reversed_keys,  # entries
        )

    @classmethod
    def __tree_unflatten__(cls, metadata, children):
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

1. **The type registry is per-interpreter.** Registering a custom type affects all modules that use OpTree in the same interpreter. Each interpreter (including subinterpreters) maintains its own registry, while child processes forked via `multiprocessing` inherit a copy.

> [!WARNING]
> For safety reasons, a `namespace` must be specified while registering a custom type. It is
> used to isolate the behavior of flattening and unflattening a pytree node type. This is to
> prevent accidental collisions between different libraries that may register the same type.

2. **Duplicate registration is not allowed.** Registering the same type in the same namespace a second time raises an error. To update the behavior, first call `unregister_pytree_node(cls, namespace=...)` and then re-register. Alternatively, register the type under a different `namespace`.

    > [!WARNING]
    > Any `PyTreeSpec` objects created before the unregistration still hold a reference to the old registration. Unflattening such a `PyTreeSpec` will use the **old** `unflatten_func`, not the newly registered one.

3. **Built-in types cannot be re-registered.** The behavior of the types listed in [Built-in PyTree Node Types](#built-in-pytree-node-types) (e.g., key-sorted traversal for `dict`, `collections.defaultdict`, and `frozendict`) is fixed.

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

        def __tree_flatten__(self):  # -> (children, metadata, entries)
            reversed_keys = sorted(self.keys(), reverse=True)
            return (
                [self[key] for key in reversed_keys],  # children
                reversed_keys,  # metadata
                reversed_keys,  # entries
            )

        @classmethod
        def __tree_unflatten__(cls, metadata, children):
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

5. **Beware of infinite recursion in custom flatten functions.** The returned `children` are recursively flattened and may have the same type as the current node. Ensure your flatten function has a proper termination condition.

    ```python
    import numpy as np
    import torch

    optree.register_pytree_node(
        np.ndarray,
        # Children are nested lists of Python objects
        lambda array: (np.atleast_1d(array).tolist(), array.ndim == 0),
        lambda scalar, rows: np.asarray(rows) if not scalar else np.asarray(rows[0]),
        namespace='numpy1',
    )

    optree.register_pytree_node(
        np.ndarray,
        # Returns a list of `np.ndarray`s without termination condition -> RecursionError!
        lambda array: ([array.ravel()], array.shape),
        lambda shape, children: children[0].reshape(shape),
        namespace='numpy2',
    )
    ```

### `None` is Non-leaf Node vs. `None` is Leaf

The [`None`](https://docs.python.org/3/library/constants.html#None) object is Python's singleton for "no value" — analogous to `null` in other languages but also commonly used as a sentinel or implicit return value.

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
Otherwise, the `None` object will remain in the tree structure specification.

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

The built-in Python dictionary ([`builtins.dict`](https://docs.python.org/3/library/stdtypes.html#dict)) is a mapping whose leaves are its values.
Since [Python 3.7](https://docs.python.org/3/whatsnew/3.7.html), `dict` is guaranteed to be insertion ordered, but the equality operator (`==`) ignores key order.
To ensure [referential transparency](https://en.wikipedia.org/wiki/Referential_transparency) — "equal `dict`" implies "equal ordering of leaves" — the leaves (values) are returned in key-sorted order.
The same applies to [`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict) and [`frozendict`](https://docs.python.org/3/library/stdtypes.html#frozendict) (Python 3.15+).

```python
>>> optree.tree_flatten({'a': [1, 2], 'b': [3]})
([1, 2, 3], PyTreeSpec({'a': [*, *], 'b': [*]}))
>>> optree.tree_flatten({'b': [3], 'a': [1, 2]})
([1, 2, 3], PyTreeSpec({'a': [*, *], 'b': [*]}))
```

Sorting ensures that equal dictionaries always flatten to the same leaf sequence, regardless of insertion order. This is critical for operations that rely on positional correspondence between leaves. Consider two parameter `dict`s that are equal but constructed in different orders:

```python
>>> import numpy as np
>>> params1 = {'weight': np.array([[1.0, 2.0], [3.0, 4.0]]), 'bias': np.array([5.0, 6.0])}
>>> params2 = {'bias': np.array([5.0, 6.0]), 'weight': np.array([[1.0, 2.0], [3.0, 4.0]])}
>>> optree.tree_all(optree.tree_map(np.allclose, params1, params2))
True
```

Because `tree_map` zips leaves positionally, sorted keys guarantee correct element-wise operations:

```python
>>> optree.tree_map(lambda x, y: x - y, params1, params2)
{
    'weight': array([[0., 0.],
                     [0., 0.]]),
    'bias': array([0., 0.])
}
```

The same applies to `tree_ravel`, which concatenates all leaves into a single 1D array:

```python
>>> from optree.integrations.numpy import tree_ravel
>>> tree_ravel(params1)[0]
array([5., 6., 1., 2., 3., 4.])  # 'bias' before 'weight' (sorted)
>>> tree_ravel(params2)[0]
array([5., 6., 1., 2., 3., 4.])  # same order, despite different insertion order
```

Without sorting, insertion order would silently corrupt the results. Here is a counterexample using `dict_insertion_ordered`:

```python
>>> with optree.dict_insertion_ordered(True, namespace='demo'):
...     flat1, _ = tree_ravel(params1, namespace='demo')
...     flat2, _ = tree_ravel(params2, namespace='demo')
>>> flat1
array([1., 2., 3., 4., 5., 6.])  # weight, bias (insertion order of params1)
>>> flat2
array([5., 6., 1., 2., 3., 4.])  # bias, weight (insertion order of params2)
>>> flat1 - flat2                # WRONG! Should be all zeros for equal params
array([-4., -4.,  2.,  2.,  2.,  2.])
```

To preserve insertion order during pytree traversal, use [`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict), which considers key order in equality checks:

```python
>>> OrderedDict([('a', [1, 2]), ('b', [3])]) == OrderedDict([('b', [3]), ('a', [1, 2])])
False
>>> optree.tree_flatten(OrderedDict([('a', [1, 2]), ('b', [3])]))
([1, 2, 3], PyTreeSpec(OrderedDict({'a': [*, *], 'b': [*]})))
>>> optree.tree_flatten(OrderedDict([('b', [3]), ('a', [1, 2])]))
([3, 1, 2], PyTreeSpec(OrderedDict({'b': [*], 'a': [*, *]})))
```

To flatten [`builtins.dict`](https://docs.python.org/3/library/stdtypes.html#dict), [`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict), and [`frozendict`](https://docs.python.org/3/library/stdtypes.html#frozendict) (Python 3.15+) objects with the insertion order preserved, use the `dict_insertion_ordered` context manager:

```python
>>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
>>> optree.tree_flatten(tree)
(
    [1, 2, 3, 4, 5],
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
)
>>> with optree.dict_insertion_ordered(True, namespace='some-namespace'):
...     optree.tree_flatten(tree, namespace='some-namespace')
(
    [2, 3, 4, 1, 5],
    PyTreeSpec({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}, namespace='some-namespace')
)
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
> The `dict` keys are not required to be comparable (sortable) or of a single type.
> Keys are sorted by `key=lambda k: k` first if possible, otherwise falling back to `key=lambda k: (f'{k.__class__.__module__}.{k.__class__.__qualname__}', k)`. This handles most cases.
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

OpTree is based on JAX's implementation of the PyTree utility, with significant refactoring and several improvements.
The original licenses can be found at [JAX's Apache License 2.0](https://github.com/google/jax/blob/HEAD/LICENSE) and [Tensorflow's Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/HEAD/LICENSE).
