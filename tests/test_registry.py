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

# pylint: disable=missing-function-docstring,invalid-name

import re
from collections import UserDict, UserList, namedtuple

import pytest

import optree


def test_register_pytree_node_class_with_no_namespace():
    with pytest.raises(
        ValueError,
        match='Must specify `namespace` when the first argument is a class.',
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
        match='Cannot specify `namespace` when the first argument is a string.',
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

        @optree.register_pytree_node_class(namespace='func')
        def func():
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

    with pytest.raises(ValueError, match='The namespace cannot be an empty string.'):

        @optree.register_pytree_node_class('')
        class MyList2(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(ValueError, match='The namespace cannot be an empty string.'):

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

    with pytest.raises(ValueError, match='The namespace cannot be an empty string.'):
        optree.register_pytree_node(
            set,
            lambda s: (sorted(s), None, None),
            lambda _, s: set(s),
            namespace='',
        )


def test_register_pytree_node_duplicate_builtin_namespace():
    with pytest.raises(
        ValueError,
        match=r"PyTree type <class 'NoneType'> is already registered in the global namespace.",
    ):
        optree.register_pytree_node(
            type(None),
            lambda n: ((), None, None),
            lambda _, n: None,
            namespace=optree.registry.__GLOBAL_NAMESPACE,
        )

    with pytest.raises(
        ValueError,
        match=r"PyTree type <class 'NoneType'> is already registered in the global namespace.",
    ):
        optree.register_pytree_node(
            type(None),
            lambda n: ((), None, None),
            lambda _, n: None,
            namespace='none',
        )

    with pytest.raises(
        ValueError,
        match=r"PyTree type <class 'list'> is already registered in the global namespace.",
    ):
        optree.register_pytree_node(
            list,
            lambda lst: (lst, None, None),
            lambda _, lst: lst,
            namespace=optree.registry.__GLOBAL_NAMESPACE,
        )
    with pytest.raises(
        ValueError,
        match=r"PyTree type <class 'list'> is already registered in the global namespace.",
    ):
        optree.register_pytree_node(
            list,
            lambda lst: (lst, None, None),
            lambda _, lst: lst,
            namespace='list',
        )


def test_register_pytree_node_namedtuple():
    mytuple1 = namedtuple('mytuple1', ['a', 'b', 'c'])
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
            namespace=optree.registry.__GLOBAL_NAMESPACE,
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"PyTree type <class 'test_registry.mytuple1'> is already registered in the global namespace.",
        ),
    ):
        optree.register_pytree_node(
            mytuple1,
            lambda t: (reversed(t), None, None),
            lambda _, t: mytuple1(*reversed(t)),
            namespace='mytuple',
        )

    tree1 = mytuple1(1, 2, 3)
    leaves1, treespec1 = optree.tree_flatten(tree1)
    assert leaves1 == [3, 2, 1]
    assert str(treespec1) == 'PyTreeSpec(CustomTreeNode(mytuple1[None], [*, *, *]))'
    assert tree1 == optree.tree_unflatten(treespec1, leaves1)

    mytuple2 = namedtuple('mytuple2', ['a', 'b', 'c'])
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

    tree2 = mytuple2(1, 2, 3)
    leaves2, treespec2 = optree.tree_flatten(tree2)
    assert leaves2 == [1, 2, 3]
    assert str(treespec2) == 'PyTreeSpec(mytuple2(a=*, b=*, c=*))'
    assert tree2 == optree.tree_unflatten(treespec2, leaves2)

    leaves2, treespec2 = optree.tree_flatten(tree2, namespace='undefined')
    assert leaves2 == [1, 2, 3]
    assert str(treespec2) == 'PyTreeSpec(mytuple2(a=*, b=*, c=*))'
    assert tree2 == optree.tree_unflatten(treespec2, leaves2)

    leaves2, treespec2 = optree.tree_flatten(tree2, namespace='mytuple')
    assert leaves2 == [3, 2, 1]
    assert (
        str(treespec2)
        == "PyTreeSpec(CustomTreeNode(mytuple2[None], [*, *, *]), namespace='mytuple')"
    )
    assert tree2 == optree.tree_unflatten(treespec2, leaves2)


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
        optree.ops.flatten_one_level(MyList1([1, 2, 3]), namespace='error')

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
        optree.ops.flatten_one_level(MyList4([1, 2, 3]), namespace='error')

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
        optree.ops.flatten_one_level(MyListEntryMismatch([1, 2, 3]), namespace='error')


def test_pytree_node_registry_get():
    handler = optree.register_pytree_node.get(list)
    assert handler is not None
    lst = [1, 2, 3]
    assert handler.to_iterable(lst)[:2] == (lst, None)

    handler = optree.register_pytree_node.get(list, namespace='any')
    assert handler is not None
    lst = [1, 2, 3]
    assert handler.to_iterable(lst)[:2] == (lst, None)

    handler = optree.register_pytree_node.get(set)
    assert handler is None

    optree.register_pytree_node(
        set,
        lambda s: (sorted(s), None, None),
        lambda _, s: set(s),
        namespace=optree.registry.__GLOBAL_NAMESPACE,
    )
    handler = optree.register_pytree_node.get(set)
    assert handler is not None

    handler = optree.register_pytree_node.get(set, namespace='set')
    assert handler is not None

    @optree.register_pytree_node_class(namespace='mylist')
    class MyList(UserList):
        def tree_flatten(self):
            return self.data, None, None

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls(children)

    handler = optree.register_pytree_node.get(MyList)
    assert handler is None
    handler = optree.register_pytree_node.get(MyList, namespace='set')
    assert handler is None
    handler = optree.register_pytree_node.get(MyList, namespace='mylist')
    assert handler is not None


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
    paths, leaves, treespec = optree.tree_flatten_with_path(tree, namespace='mydict')
    assert paths == [('c', 'f'), ('c', 'd'), ('b',), ('a', 0), ('a', 1)]
    assert leaves == [6, 5, 4, 2, 3]
    assert paths == treespec.paths()
    assert (
        str(treespec)
        == "PyTreeSpec(CustomTreeNode(MyDict[['c', 'b', 'a']], [CustomTreeNode(MyAnotherDict[['f', 'd']], [*, *]), *, (*, *)]), namespace='mydict')"
    )
    leaves, treespec = optree.tree_flatten(tree, namespace='mydict')
    assert leaves == [6, 5, 4, 2, 3]
    assert (
        str(treespec)
        == "PyTreeSpec(CustomTreeNode(MyDict[['c', 'b', 'a']], [CustomTreeNode(MyAnotherDict[['f', 'd']], [*, *]), *, (*, *)]), namespace='mydict')"
    )
