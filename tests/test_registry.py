# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
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

# pylint: disable=missing-function-docstring,invalid-name

import re
import sys
import weakref
from collections import UserDict, UserList, namedtuple
from dataclasses import dataclass

import pytest

import optree
import optree._C
from helpers import GLOBAL_NAMESPACE, PYPY, Py_GIL_DISABLED, gc_collect, skipif_pypy


def test_register_pytree_node_class_with_no_namespace():
    with pytest.raises(
        ValueError,
        match=re.escape('Must specify `namespace` when the first argument is a class.'),
    ):

        @optree.register_pytree_node_class
        class MyList(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)


def test_register_pytree_node_class_with_duplicate_namespace():
    with pytest.raises(
        ValueError,
        match=re.escape('Cannot specify `namespace` when the first argument is a string.'),
    ):

        @optree.register_pytree_node_class('mylist', namespace='mylist')
        class MyList(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)


def test_register_pytree_node_with_non_class():
    with pytest.raises(TypeError, match='Expected a class'):

        @optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
        def func1():
            pass

    with pytest.raises(TypeError, match='Expected a class'):
        optree.register_pytree_node(
            1,
            lambda s: (sorted(s), None, None),
            lambda _, s: set(s),
            namespace=GLOBAL_NAMESPACE,
        )

    with pytest.raises(TypeError, match='Expected a class'):

        @optree.register_pytree_node_class(namespace='func')
        def func2():
            pass

    with pytest.raises(TypeError, match='Expected a class'):
        optree.register_pytree_node(
            1,
            lambda s: (sorted(s), None, None),
            lambda _, s: set(s),
            namespace='non-class',
        )


def test_register_pytree_node_class_with_duplicate_registrations():
    @optree.register_pytree_node_class('mylist1')
    class MyList1(UserList):
        def tree_flatten(self):
            return self.data, None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    @optree.register_pytree_node_class(namespace='mylist2')
    class MyList2(UserList):
        def tree_flatten(self):
            return self.data, None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    with pytest.raises(
        ValueError,
        match=r"PyTree type.*is already registered in namespace 'mylist1'\.",
    ):
        optree.register_pytree_node_class(MyList1, namespace='mylist1')
    with pytest.raises(
        ValueError,
        match=r"PyTree type.*is already registered in namespace 'mylist2'\.",
    ):
        optree.register_pytree_node_class(MyList2, namespace='mylist2')

    optree.register_pytree_node_class(namespace='mylist3')(MyList1)


def test_register_pytree_node_with_invalid_namespace():
    with pytest.raises(TypeError, match='The namespace must be a string'):

        @optree.register_pytree_node_class(namespace=1)
        class MyList1(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(ValueError, match=re.escape('The namespace cannot be an empty string.')):

        @optree.register_pytree_node_class('')
        class MyList2(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(ValueError, match=re.escape('The namespace cannot be an empty string.')):

        @optree.register_pytree_node_class(namespace='')
        class MyList3(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(TypeError, match='The namespace must be a string'):
        optree.register_pytree_node(
            set,
            lambda s: (sorted(s), None, None),
            lambda _, s: set(s),
            namespace=1,
        )

    with pytest.raises(ValueError, match=re.escape('The namespace cannot be an empty string.')):
        optree.register_pytree_node(
            set,
            lambda s: (sorted(s), None, None),
            lambda _, s: set(s),
            namespace='',
        )


def test_register_pytree_node_with_invalid_path_entry_type():
    with pytest.raises(TypeError, match=r'Expected a subclass of PyTreeEntry, got .*\.'):

        @optree.register_pytree_node_class(namespace='error')
        class MyList1(UserList):
            TREE_PATH_ENTRY_TYPE = None

            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(TypeError, match=r'Expected a subclass of PyTreeEntry, got .*\.'):

        @optree.register_pytree_node_class(namespace='error')
        class MyList2(UserList):
            TREE_PATH_ENTRY_TYPE = int

            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(TypeError, match=r'Expected a subclass of PyTreeEntry, got .*\.'):

        @optree.register_pytree_node_class(path_entry_type=1, namespace='error')
        class MyList3(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(TypeError, match=r'Expected a subclass of PyTreeEntry, got .*\.'):

        @optree.register_pytree_node_class(path_entry_type=int, namespace='error')
        class MyList4(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(TypeError, match=r'Expected a subclass of PyTreeEntry, got .*\.'):
        optree.register_pytree_node(
            set,
            lambda s: (sorted(s), None, None),
            lambda _, s: set(s),
            path_entry_type=1,
            namespace='error',
        )

    with pytest.raises(TypeError, match=r'Expected a subclass of PyTreeEntry, got .*\.'):
        optree.register_pytree_node(
            set,
            lambda s: (sorted(s), None, None),
            lambda _, s: set(s),
            path_entry_type=int,
            namespace='error',
        )


def test_register_pytree_node_duplicate_builtins():
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'NoneType'> is a built-in type and cannot be re-registered.",
        ),
    ):
        optree.register_pytree_node(
            type(None),
            lambda n: ((), None, None),
            lambda _, n: None,
            namespace=GLOBAL_NAMESPACE,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'NoneType'> is a built-in type and cannot be re-registered.",
        ),
    ):
        optree.register_pytree_node(
            type(None),
            lambda n: ((), None, None),
            lambda _, n: None,
            namespace='none',
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'list'> is a built-in type and cannot be re-registered.",
        ),
    ):
        optree.register_pytree_node(
            list,
            lambda lst: (lst, None, None),
            lambda _, lst: lst,
            namespace=GLOBAL_NAMESPACE,
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'list'> is a built-in type and cannot be re-registered.",
        ),
    ):
        optree.register_pytree_node(
            list,
            lambda lst: (lst, None, None),
            lambda _, lst: lst,
            namespace='list',
        )


def test_register_pytree_node_namedtuple():
    mytuple1 = namedtuple('mytuple1', ['a', 'b', 'c'])  # noqa: PYI024
    with pytest.warns(
        UserWarning,
        match=re.escape(
            r"PyTree type <class 'test_registry.mytuple1'> is a subclass of `collections.namedtuple`, "
            r'which is already registered in the global namespace. '
            r'Override it with custom flatten/unflatten functions.',
        ),
    ):
        optree.register_pytree_node(
            mytuple1,
            lambda t: (reversed(t), None, None),
            lambda _, t: mytuple1(*reversed(t)),
            namespace=GLOBAL_NAMESPACE,
        )
    with pytest.warns(
        UserWarning,
        match=re.escape(
            r"PyTree type <class 'test_registry.mytuple1'> is a subclass of `collections.namedtuple`, "
            r'which is already registered in the global namespace. '
            r"Override it with custom flatten/unflatten functions in namespace 'mytuple'.",
        ),
    ):
        optree.register_pytree_node(
            mytuple1,
            lambda t: (list(t)[1::] + list(t)[:1], None, None),
            lambda _, t: mytuple1(*(list(t)[-1:] + list(t)[:-1])),
            namespace='mytuple',
        )

    tree = mytuple1(1, 2, 3)
    leaves1, treespec1 = optree.tree_flatten(tree)
    assert leaves1 == [3, 2, 1]
    assert str(treespec1) == 'PyTreeSpec(CustomTreeNode(mytuple1[None], [*, *, *]))'
    assert tree == optree.tree_unflatten(treespec1, leaves1)

    leaves2, treespec2 = optree.tree_flatten(tree, namespace='undefined')
    assert leaves2 == [3, 2, 1]
    assert (
        str(treespec2)
        == "PyTreeSpec(CustomTreeNode(mytuple1[None], [*, *, *]), namespace='undefined')"
    )
    assert tree == optree.tree_unflatten(treespec2, leaves2)
    assert treespec1 == treespec2

    leaves3, treespec3 = optree.tree_flatten(tree, namespace='mytuple')
    assert leaves3 == [2, 3, 1]
    assert (
        str(treespec3)
        == "PyTreeSpec(CustomTreeNode(mytuple1[None], [*, *, *]), namespace='mytuple')"
    )
    assert tree == optree.tree_unflatten(treespec3, leaves3)
    assert treespec1 != treespec3
    assert treespec2 != treespec3

    mytuple2 = namedtuple('mytuple2', ['a', 'b', 'c'])  # noqa: PYI024
    with pytest.warns(
        UserWarning,
        match=re.escape(
            r"PyTree type <class 'test_registry.mytuple2'> is a subclass of `collections.namedtuple`, "
            r'which is already registered in the global namespace. '
            r"Override it with custom flatten/unflatten functions in namespace 'mytuple'.",
        ),
    ):
        optree.register_pytree_node(
            mytuple2,
            lambda t: (reversed(t), None, None),
            lambda _, t: mytuple2(*reversed(t)),
            namespace='mytuple',
        )

    tree = mytuple2(1, 2, 3)
    leaves1, treespec1 = optree.tree_flatten(tree)
    assert leaves1 == [1, 2, 3]
    assert str(treespec1) == 'PyTreeSpec(mytuple2(a=*, b=*, c=*))'
    assert tree == optree.tree_unflatten(treespec1, leaves1)

    leaves2, treespec2 = optree.tree_flatten(tree, namespace='undefined')
    assert leaves2 == [1, 2, 3]
    assert str(treespec2) == 'PyTreeSpec(mytuple2(a=*, b=*, c=*))'
    assert tree == optree.tree_unflatten(treespec2, leaves2)
    assert treespec1 == treespec2

    leaves3, treespec3 = optree.tree_flatten(tree, namespace='mytuple')
    assert leaves3 == [3, 2, 1]
    assert (
        str(treespec3)
        == "PyTreeSpec(CustomTreeNode(mytuple2[None], [*, *, *]), namespace='mytuple')"
    )
    assert tree == optree.tree_unflatten(treespec3, leaves3)
    assert treespec1 != treespec3


def test_flatten_with_wrong_number_of_returns():
    @optree.register_pytree_node_class(namespace='error')
    class MyList1(UserList):
        def tree_flatten(self):
            return (self.data,)

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    with pytest.raises(
        RuntimeError,
        match=r"PyTree custom flatten function for type <class '.*'> should return a 2- or 3-tuple, got 1\.",
    ):
        optree.tree_flatten(MyList1([1, 2, 3]), namespace='error')

    with pytest.raises(
        RuntimeError,
        match=r"PyTree custom flatten function for type <class '.*'> should return a 2- or 3-tuple, got 1\.",
    ):
        optree.tree_flatten_one_level(MyList1([1, 2, 3]), namespace='error')

    @optree.register_pytree_node_class(namespace='error')
    class MyList4(UserList):
        def tree_flatten(self):
            return self.data, None, None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    with pytest.raises(
        RuntimeError,
        match=r"PyTree custom flatten function for type <class '.*'> should return a 2- or 3-tuple, got 4\.",
    ):
        optree.tree_flatten(MyList4([1, 2, 3]), namespace='error')

    with pytest.raises(
        RuntimeError,
        match=r"PyTree custom flatten function for type <class '.*'> should return a 2- or 3-tuple, got 4\.",
    ):
        optree.tree_flatten_one_level(MyList4([1, 2, 3]), namespace='error')

    @optree.register_pytree_node_class(namespace='error')
    class MyListEntryMismatch(UserList):
        def tree_flatten(self):
            return self.data, None, range(len(self) + 1)

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    with pytest.raises(
        RuntimeError,
        match=r"PyTree custom flatten function for type <class '.*'> returned inconsistent number of children \(3\) and number of entries \(4\)\.",
    ):
        optree.tree_flatten(MyListEntryMismatch([1, 2, 3]), namespace='error')

    with pytest.raises(
        RuntimeError,
        match=r"PyTree custom flatten function for type <class '.*'> returned inconsistent number of children \(3\) and number of entries \(4\)\.",
    ):
        optree.tree_flatten_one_level(MyListEntryMismatch([1, 2, 3]), namespace='error')


def test_pytree_node_registry_get():
    registry1 = optree.register_pytree_node.get()
    assert optree.register_pytree_node.get() == registry1
    assert optree.register_pytree_node.get(namespace=GLOBAL_NAMESPACE) == registry1
    assert optree.register_pytree_node.get(namespace='undefined') == registry1
    assert optree.register_pytree_node.get() is not registry1
    assert all(handler.namespace == '' for handler in registry1.values())
    with optree.dict_insertion_ordered(True, namespace=GLOBAL_NAMESPACE):
        registry2 = optree.register_pytree_node.get()
        assert registry2 != registry1
        assert list(registry2) == list(registry1)
        assert optree.register_pytree_node.get() == registry2
        assert optree.register_pytree_node.get(namespace='undefined') == registry2
        assert optree.register_pytree_node.get() is not registry2
        assert all(handler.namespace == '' for handler in registry2.values())

    handler = optree.register_pytree_node.get(list)
    assert handler is not None
    assert handler is registry1[list]
    assert handler is registry2[list]
    lst = [1, 2, 3]
    assert tuple(handler.flatten_func(lst))[:2] == (lst, None)

    handler = optree.register_pytree_node.get(list, namespace='any')
    assert handler is not None
    assert handler is registry1[list]
    assert handler is registry2[list]
    lst = [1, 2, 3]
    assert tuple(handler.flatten_func(lst))[:2] == (lst, None)

    handler = optree.register_pytree_node.get(dict)
    assert handler is not None
    assert handler is registry1[dict]
    assert handler is not registry2[dict]
    dct = {'b': 2, 'c': 3, 'a': 1}
    assert tuple(handler.flatten_func(dct))[:2] == ((1, 2, 3), ['a', 'b', 'c'])

    handler = optree.register_pytree_node.get(dict, namespace='any')
    assert handler is not None
    assert handler is registry1[dict]
    assert handler is not registry2[dict]
    dct = {'b': 2, 'c': 3, 'a': 1}
    assert tuple(handler.flatten_func(dct))[:2] == ((1, 2, 3), ['a', 'b', 'c'])

    with optree.dict_insertion_ordered(True, namespace=GLOBAL_NAMESPACE):
        handler = optree.register_pytree_node.get(dict)
        assert handler is not None
        assert handler is registry2[dict]
        assert handler is not registry1[dict]
        dct = {'b': 2, 'c': 3, 'a': 1}
        assert tuple(handler.flatten_func(dct))[:2] == ((2, 3, 1), ['b', 'c', 'a'])

        handler = optree.register_pytree_node.get(dict, namespace='any')
        assert handler is not None
        assert handler is registry2[dict]
        assert handler is not registry1[dict]
        dct = {'b': 2, 'c': 3, 'a': 1}
        assert tuple(handler.flatten_func(dct))[:2] == ((2, 3, 1), ['b', 'c', 'a'])

    handler = optree.register_pytree_node.get(set)
    assert handler is None
    assert set not in registry1
    assert set not in registry2

    optree.register_pytree_node(
        set,
        lambda s: (sorted(s), None, None),
        lambda _, s: set(s),
        namespace=GLOBAL_NAMESPACE,
    )
    registry3 = optree.register_pytree_node.get()
    handler = optree.register_pytree_node.get(set)
    assert handler is not None
    assert set in registry3
    assert handler is registry3[set]
    assert list(registry1) == list(registry3)[:-1]

    handler = optree.register_pytree_node.get(set, namespace='set')
    assert handler is not None
    assert optree.register_pytree_node.get(namespace='set') == registry3

    @optree.register_pytree_node_class(namespace='mylist')
    class MyList(UserList):
        def tree_flatten(self):
            return self.data, None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    handler = optree.register_pytree_node.get(MyList)
    assert handler is None
    assert optree.register_pytree_node.get() == registry3
    assert MyList not in registry3
    handler = optree.register_pytree_node.get(MyList, namespace='set')
    assert handler is None
    assert MyList not in registry3

    handler = optree.register_pytree_node.get(MyList, namespace='mylist')
    registry4 = optree.register_pytree_node.get(namespace='mylist')
    assert handler is not None
    assert MyList in registry4
    assert handler is registry4[MyList]
    assert list(registry3) == list(registry4)[: len(registry3)]
    for cls, handler in registry4.items():
        assert handler.type is cls
        if cls not in registry3:
            assert handler.namespace == 'mylist'
        else:
            assert handler.namespace == ''
            assert handler is registry3[cls]


def test_pytree_node_registry_get_with_invalid_arguments():
    registry = optree.register_pytree_node.get()
    assert optree.register_pytree_node.get(None) == registry
    assert optree.register_pytree_node.get(namespace=GLOBAL_NAMESPACE) == registry
    assert optree.register_pytree_node.get(namedtuple) is registry[namedtuple]  # noqa: PYI024
    if sys.version_info[:2] != (3, 9) or PYPY:
        with pytest.raises(TypeError, match='Expected a class or None'):
            optree.register_pytree_node.get(dataclass)
        with pytest.raises(TypeError, match='The namespace must be a string'):
            optree.register_pytree_node.get(list, namespace=None)


def test_pytree_node_registry_with_init_subclass():
    @optree.register_pytree_node_class(namespace='mydict')
    class MyDict(UserDict):
        def __init_subclass__(cls):
            super().__init_subclass__()
            optree.register_pytree_node_class(cls, namespace='mydict')

        def tree_flatten(self):
            reversed_keys = sorted(self.keys(), reverse=True)
            return [self[key] for key in reversed_keys], reversed_keys, reversed_keys

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(zip(metadata, children))

    class MyAnotherDict(MyDict):
        pass

    tree = MyDict(b=4, a=(2, 3), c=MyAnotherDict({'d': 5, 'f': 6}))
    leaves, treespec = optree.tree_flatten(tree, namespace='mydict')
    assert leaves == [6, 5, 4, 2, 3]
    assert (
        str(treespec)
        == "PyTreeSpec(CustomTreeNode(MyDict[['c', 'b', 'a']], [CustomTreeNode(MyAnotherDict[['f', 'd']], [*, *]), *, (*, *)]), namespace='mydict')"
    )
    paths, leaves, treespec = optree.tree_flatten_with_path(tree, namespace='mydict')
    assert paths == [('c', 'f'), ('c', 'd'), ('b',), ('a', 0), ('a', 1)]
    assert leaves == [6, 5, 4, 2, 3]
    assert paths == treespec.paths()
    assert (
        str(treespec)
        == "PyTreeSpec(CustomTreeNode(MyDict[['c', 'b', 'a']], [CustomTreeNode(MyAnotherDict[['f', 'd']], [*, *]), *, (*, *)]), namespace='mydict')"
    )
    accessors, leaves, treespec = optree.tree_flatten_with_accessor(tree, namespace='mydict')
    assert accessors == [
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('c', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('f', MyAnotherDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('c', MyDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('d', MyAnotherDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor((optree.MappingEntry('b', MyDict, optree.PyTreeKind.CUSTOM),)),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
                optree.SequenceEntry(0, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
                optree.SequenceEntry(1, tuple, optree.PyTreeKind.TUPLE),
            ),
        ),
    ]
    assert leaves == [6, 5, 4, 2, 3]
    assert accessors == treespec.accessors()
    assert (
        str(treespec)
        == "PyTreeSpec(CustomTreeNode(MyDict[['c', 'b', 'a']], [CustomTreeNode(MyAnotherDict[['f', 'd']], [*, *]), *, (*, *)]), namespace='mydict')"
    )


def test_unregister_pytree_node_with_non_class():
    with pytest.raises(TypeError, match='Expected a class'):
        optree.unregister_pytree_node(dataclass, namespace=GLOBAL_NAMESPACE)

    def func1():
        pass

    with pytest.raises(TypeError, match='Expected a class'):
        optree.unregister_pytree_node(func1, namespace=GLOBAL_NAMESPACE)

    with pytest.raises(TypeError, match='Expected a class'):
        optree.unregister_pytree_node(1, namespace=GLOBAL_NAMESPACE)

    def func2():
        pass

    with pytest.raises(TypeError, match='Expected a class'):
        optree.unregister_pytree_node(func2, namespace='func')

    with pytest.raises(TypeError, match='Expected a class'):
        optree.unregister_pytree_node(1, namespace='non-class')


def test_unregister_pytree_node_with_non_registered_class():
    class MyList(UserList):
        def tree_flatten(self):
            return self.data, None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    with pytest.raises(
        ValueError,
        match=r"PyTree type <class '.*'> is not registered in namespace 'undefined'\.",
    ):
        optree.unregister_pytree_node(MyList, namespace='undefined')

    with pytest.raises(
        ValueError,
        match=r"PyTree type <class '.*'> is not registered in the global namespace\.",
    ):
        optree.unregister_pytree_node(MyList, namespace=GLOBAL_NAMESPACE)

    optree.register_pytree_node_class(MyList, namespace='mylist')

    with pytest.raises(
        ValueError,
        match=r"PyTree type <class '.*'> is not registered in the global namespace\.",
    ):
        optree.unregister_pytree_node(MyList, namespace=GLOBAL_NAMESPACE)

    optree.unregister_pytree_node(MyList, namespace='mylist')

    with pytest.raises(
        ValueError,
        match=r"PyTree type <class '.*'> is not registered in namespace 'mylist'\.",
    ):
        optree.unregister_pytree_node(MyList, namespace='mylist')


def test_unregister_pytree_node_with_invalid_namespace():
    with pytest.raises(TypeError, match='The namespace must be a string'):
        optree.unregister_pytree_node(set, namespace=1)

    with pytest.raises(ValueError, match=re.escape('The namespace cannot be an empty string.')):
        optree.unregister_pytree_node(set, namespace='')


def test_unregister_pytree_node_with_builtins():
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'NoneType'> is a built-in type and cannot be unregistered.",
        ),
    ):
        optree.unregister_pytree_node(type(None), namespace=GLOBAL_NAMESPACE)

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'NoneType'> is a built-in type and cannot be unregistered.",
        ),
    ):
        optree.unregister_pytree_node(type(None), namespace='none')

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'list'> is a built-in type and cannot be unregistered.",
        ),
    ):
        optree.unregister_pytree_node(list, namespace=GLOBAL_NAMESPACE)

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'list'> is a built-in type and cannot be unregistered.",
        ),
    ):
        optree.unregister_pytree_node(list, namespace='list')


def test_unregister_pytree_node_namedtuple():
    mytuple1 = namedtuple('mytuple1', ['a', 'b', 'c'])  # noqa: PYI024
    with pytest.warns(
        UserWarning,
        match=re.escape(
            r"PyTree type <class 'test_registry.mytuple1'> is a subclass of `collections.namedtuple`, "
            r'which is already registered in the global namespace. '
            r'Override it with custom flatten/unflatten functions.',
        ),
    ):
        optree.register_pytree_node(
            mytuple1,
            lambda t: (reversed(t), None, None),
            lambda _, t: mytuple1(*reversed(t)),
            namespace=GLOBAL_NAMESPACE,
        )

    tree = mytuple1(1, 2, 3)
    leaves1, treespec1 = optree.tree_flatten(tree)
    assert leaves1 == [3, 2, 1]
    assert str(treespec1) == 'PyTreeSpec(CustomTreeNode(mytuple1[None], [*, *, *]))'
    assert tree == optree.tree_unflatten(treespec1, leaves1)

    optree.unregister_pytree_node(mytuple1, namespace=GLOBAL_NAMESPACE)
    assert str(treespec1) == 'PyTreeSpec(CustomTreeNode(mytuple1[None], [*, *, *]))'
    assert tree == optree.tree_unflatten(treespec1, leaves1)

    leaves2, treespec2 = optree.tree_flatten(tree)
    assert leaves2 == [1, 2, 3]
    assert str(treespec2) == 'PyTreeSpec(mytuple1(a=*, b=*, c=*))'
    assert tree == optree.tree_unflatten(treespec2, leaves2)
    assert treespec1 != treespec2

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'test_registry.mytuple1'> is a subclass of `collections.namedtuple`, "
            r"which is not explicitly registered in namespace 'undefined'.",
        ),
    ):
        optree.unregister_pytree_node(mytuple1, namespace='undefined')
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'test_registry.mytuple1'> is a subclass of `collections.namedtuple`, "
            r'which is not explicitly registered in the global namespace.',
        ),
    ):
        optree.unregister_pytree_node(mytuple1, namespace=GLOBAL_NAMESPACE)

    mytuple2 = namedtuple('mytuple2', ['a', 'b', 'c'])  # noqa: PYI024
    with pytest.warns(
        UserWarning,
        match=re.escape(
            r"PyTree type <class 'test_registry.mytuple2'> is a subclass of `collections.namedtuple`, "
            r'which is already registered in the global namespace. '
            r"Override it with custom flatten/unflatten functions in namespace 'mytuple'.",
        ),
    ):
        optree.register_pytree_node(
            mytuple2,
            lambda t: (reversed(t), None, None),
            lambda _, t: mytuple2(*reversed(t)),
            namespace='mytuple',
        )

    tree = mytuple2(1, 2, 3)
    leaves1, treespec1 = optree.tree_flatten(tree)
    assert leaves1 == [1, 2, 3]
    assert str(treespec1) == 'PyTreeSpec(mytuple2(a=*, b=*, c=*))'
    assert tree == optree.tree_unflatten(treespec1, leaves1)

    leaves2, treespec2 = optree.tree_flatten(tree, namespace='undefined')
    assert leaves2 == [1, 2, 3]
    assert str(treespec2) == 'PyTreeSpec(mytuple2(a=*, b=*, c=*))'
    assert tree == optree.tree_unflatten(treespec2, leaves2)
    assert treespec1 == treespec2

    leaves3, treespec3 = optree.tree_flatten(tree, namespace='mytuple')
    assert leaves3 == [3, 2, 1]
    assert (
        str(treespec3)
        == "PyTreeSpec(CustomTreeNode(mytuple2[None], [*, *, *]), namespace='mytuple')"
    )
    assert tree == optree.tree_unflatten(treespec3, leaves3)
    assert treespec1 != treespec3

    optree.unregister_pytree_node(mytuple2, namespace='mytuple')
    assert (
        str(treespec3)
        == "PyTreeSpec(CustomTreeNode(mytuple2[None], [*, *, *]), namespace='mytuple')"
    )
    assert tree == optree.tree_unflatten(treespec3, leaves3)

    leaves4, treespec4 = optree.tree_flatten(tree, namespace='mytuple')
    assert leaves4 == [1, 2, 3]
    assert str(treespec4) == 'PyTreeSpec(mytuple2(a=*, b=*, c=*))'
    assert tree == optree.tree_unflatten(treespec4, leaves4)
    assert treespec1 == treespec4
    assert treespec3 != treespec4


@skipif_pypy
def test_unregister_pytree_node_no_reference_leak():  # noqa: C901
    @optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
    class MyList1(UserList):
        def tree_flatten(self):
            return self.data, None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    wr = weakref.ref(MyList1)
    assert wr() is not None

    optree.unregister_pytree_node(MyList1, namespace=GLOBAL_NAMESPACE)
    del MyList1
    gc_collect()
    if not Py_GIL_DISABLED:
        assert wr() is None

    @optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
    class MyList2(UserList):
        def tree_flatten(self):
            return reversed(self.data), None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(reversed(children))

    wr = weakref.ref(MyList2)
    assert wr() is not None

    leaves, treespec = optree.tree_flatten(MyList2([1, 2, 3]))
    assert leaves == [3, 2, 1]
    assert str(treespec) == 'PyTreeSpec(CustomTreeNode(MyList2[None], [*, *, *]))'

    optree.unregister_pytree_node(MyList2, namespace=GLOBAL_NAMESPACE)
    del MyList2
    gc_collect()
    assert wr() is not None
    assert wr() is treespec.type
    assert optree.tree_unflatten(treespec, leaves) == wr()([1, 2, 3])

    del treespec
    gc_collect()
    if not Py_GIL_DISABLED:
        assert wr() is None

    @optree.register_pytree_node_class(namespace=GLOBAL_NAMESPACE)
    class MyList3(UserList):
        def tree_flatten(self):
            return reversed(self.data), None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(reversed(children))

    wr = weakref.ref(MyList3)
    assert wr() is not None

    leaves, treespec = optree.tree_flatten(MyList3([1, 2, 3]), namespace='undefined')
    assert leaves == [3, 2, 1]
    assert (
        str(treespec)
        == "PyTreeSpec(CustomTreeNode(MyList3[None], [*, *, *]), namespace='undefined')"
    )

    optree.unregister_pytree_node(MyList3, namespace=GLOBAL_NAMESPACE)
    del MyList3
    gc_collect()
    assert wr() is not None
    assert wr() is treespec.type
    assert optree.tree_unflatten(treespec, leaves) == wr()([1, 2, 3])

    del treespec
    gc_collect()
    if not Py_GIL_DISABLED:
        assert wr() is None

    @optree.register_pytree_node_class(namespace='mylist')
    class MyList4(UserList):
        def tree_flatten(self):
            return self.data, None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    wr = weakref.ref(MyList4)
    assert wr() is not None

    optree.unregister_pytree_node(MyList4, namespace='mylist')
    del MyList4
    gc_collect()
    if not Py_GIL_DISABLED:
        assert wr() is None

    @optree.register_pytree_node_class(namespace='mylist')
    class MyList5(UserList):
        def tree_flatten(self):
            return reversed(self.data), None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(reversed(children))

    wr = weakref.ref(MyList5)
    assert wr() is not None

    leaves, treespec = optree.tree_flatten(MyList5([1, 2, 3]), namespace='mylist')
    assert leaves == [3, 2, 1]
    assert (
        str(treespec) == "PyTreeSpec(CustomTreeNode(MyList5[None], [*, *, *]), namespace='mylist')"
    )

    optree.unregister_pytree_node(MyList5, namespace='mylist')
    del MyList5
    gc_collect()
    assert wr() is not None
    assert wr() is treespec.type
    assert optree.tree_unflatten(treespec, leaves) == wr()([1, 2, 3])

    del treespec
    gc_collect()
    if not Py_GIL_DISABLED:
        assert wr() is None


def test_dict_insertion_order_with_invalid_namespace():
    with pytest.raises(TypeError, match='The namespace must be a string'):
        with optree.dict_insertion_ordered(True, namespace=1):
            pass
    with pytest.raises(ValueError, match=re.escape('The namespace cannot be an empty string.')):
        with optree.dict_insertion_ordered(True, namespace=''):
            pass


def test_dict_insertion_order_with_nested_context():
    def is_dict_insertion_ordered(namespace):
        insertion_ordered = optree._C.is_dict_insertion_ordered(namespace)
        assert optree.tree_leaves({'b': 2, 'a': 1}, namespace=namespace) == (
            [2, 1] if insertion_ordered else [1, 2]
        )
        return insertion_ordered

    assert not is_dict_insertion_ordered('')
    assert not is_dict_insertion_ordered('namespace')
    with optree.dict_insertion_ordered(True, namespace=GLOBAL_NAMESPACE):
        assert is_dict_insertion_ordered('')
        assert is_dict_insertion_ordered('namespace')

    assert not is_dict_insertion_ordered('')
    assert not is_dict_insertion_ordered('namespace')
    assert not is_dict_insertion_ordered('other-namespace')
    with optree.dict_insertion_ordered(True, namespace=GLOBAL_NAMESPACE):
        assert is_dict_insertion_ordered('')
        assert is_dict_insertion_ordered('namespace')
        assert is_dict_insertion_ordered('other-namespace')
        with optree.dict_insertion_ordered(True, namespace='namespace'):
            assert is_dict_insertion_ordered('')
            assert is_dict_insertion_ordered('namespace')
            assert is_dict_insertion_ordered('other-namespace')
            with optree.dict_insertion_ordered(False, namespace=GLOBAL_NAMESPACE):
                assert not is_dict_insertion_ordered('')
                assert is_dict_insertion_ordered('namespace')
                assert not is_dict_insertion_ordered('other-namespace')

                with optree.dict_insertion_ordered(True, namespace='other-namespace'):
                    assert not is_dict_insertion_ordered('')
                    assert is_dict_insertion_ordered('namespace')
                    assert is_dict_insertion_ordered('other-namespace')

                assert not is_dict_insertion_ordered('')
                assert is_dict_insertion_ordered('namespace')
                assert not is_dict_insertion_ordered('other-namespace')

            assert is_dict_insertion_ordered('')
            assert is_dict_insertion_ordered('namespace')
            assert is_dict_insertion_ordered('other-namespace')

        assert is_dict_insertion_ordered('')
        assert is_dict_insertion_ordered('namespace')
        assert is_dict_insertion_ordered('other-namespace')
        with optree.dict_insertion_ordered(False, namespace='namespace'):
            assert is_dict_insertion_ordered('')
            assert is_dict_insertion_ordered('namespace')
            assert is_dict_insertion_ordered('other-namespace')
            with optree.dict_insertion_ordered(False, namespace=GLOBAL_NAMESPACE):
                assert not is_dict_insertion_ordered('')
                assert not is_dict_insertion_ordered('namespace')
                assert not is_dict_insertion_ordered('other-namespace')

            assert is_dict_insertion_ordered('')
            assert is_dict_insertion_ordered('namespace')
            assert is_dict_insertion_ordered('other-namespace')

    assert not is_dict_insertion_ordered('')
    assert not is_dict_insertion_ordered('namespace')
    assert not is_dict_insertion_ordered('other-namespace')

    with optree.dict_insertion_ordered(True, namespace='namespace'):
        assert not is_dict_insertion_ordered('')
        assert is_dict_insertion_ordered('namespace')
        assert not is_dict_insertion_ordered('other-namespace')
        with optree.dict_insertion_ordered(True, namespace='namespace'):
            assert not is_dict_insertion_ordered('')
            assert is_dict_insertion_ordered('namespace')
            assert not is_dict_insertion_ordered('other-namespace')

        assert not is_dict_insertion_ordered('')
        assert is_dict_insertion_ordered('namespace')
        assert not is_dict_insertion_ordered('other-namespace')
        with optree.dict_insertion_ordered(False, namespace='namespace'):
            assert not is_dict_insertion_ordered('')
            assert not is_dict_insertion_ordered('namespace')
            assert not is_dict_insertion_ordered('other-namespace')
            with optree.dict_insertion_ordered(True, namespace='namespace'):
                assert not is_dict_insertion_ordered('')
                assert is_dict_insertion_ordered('namespace')
                assert not is_dict_insertion_ordered('other-namespace')

            assert not is_dict_insertion_ordered('')
            assert not is_dict_insertion_ordered('namespace')
            assert not is_dict_insertion_ordered('other-namespace')

        assert not is_dict_insertion_ordered('')
        assert is_dict_insertion_ordered('namespace')
        assert not is_dict_insertion_ordered('other-namespace')
        with optree.dict_insertion_ordered(True, namespace=GLOBAL_NAMESPACE):
            assert is_dict_insertion_ordered('')
            assert is_dict_insertion_ordered('namespace')
            assert is_dict_insertion_ordered('other-namespace')

        assert not is_dict_insertion_ordered('')
        assert is_dict_insertion_ordered('namespace')
        assert not is_dict_insertion_ordered('other-namespace')
        with optree.dict_insertion_ordered(True, namespace='other-namespace'):
            assert not is_dict_insertion_ordered('')
            assert is_dict_insertion_ordered('namespace')
            assert is_dict_insertion_ordered('other-namespace')

            with optree.dict_insertion_ordered(False, namespace='other-namespace'):
                assert not is_dict_insertion_ordered('')
                assert is_dict_insertion_ordered('namespace')
                assert not is_dict_insertion_ordered('other-namespace')

            assert not is_dict_insertion_ordered('')
            assert is_dict_insertion_ordered('namespace')
            assert is_dict_insertion_ordered('other-namespace')

        assert not is_dict_insertion_ordered('')
        assert is_dict_insertion_ordered('namespace')
        assert not is_dict_insertion_ordered('other-namespace')

    assert not is_dict_insertion_ordered('')
    assert not is_dict_insertion_ordered('namespace')
    assert not is_dict_insertion_ordered('other-namespace')
