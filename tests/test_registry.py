# Copyright 2022 MetaOPT Team. All Rights Reserved.
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

from collections import UserList

import pytest

import optree


def test_register_pytree_node_class_with_no_namespace():
    with pytest.raises(
        ValueError, match='Must specify `namespace` when the first argument is a class.'
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
        ValueError, match='Cannot specify `namespace` when the first argument is a string.'
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
            1, lambda s: (sorted(s), None, None), lambda _, s: set(s), namespace='non-class'
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
        ValueError, match=r"PyTree type.*is already registered in namespace 'mylist1'\."
    ):
        optree.register_pytree_node_class(MyList1, namespace='mylist1')
    with pytest.raises(
        ValueError, match=r"PyTree type.*is already registered in namespace 'mylist2'\."
    ):
        optree.register_pytree_node_class(MyList2, namespace='mylist2')

    optree.register_pytree_node_class(namespace='mylist3')(MyList1)


def test_register_pytree_node_with_invalid_namespace():
    with pytest.raises(TypeError, match='The namespace must be a string'):

        @optree.register_pytree_node_class(namespace=1)
        class MyList(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(ValueError, match='The namespace cannot be an empty string.'):

        @optree.register_pytree_node_class('')
        class MyList(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(ValueError, match='The namespace cannot be an empty string.'):

        @optree.register_pytree_node_class(namespace='')
        class MyList(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(children)

    with pytest.raises(TypeError, match='The namespace must be a string'):
        optree.register_pytree_node(
            set, lambda s: (sorted(s), None, None), lambda _, s: set(s), namespace=1
        )

    with pytest.raises(ValueError, match='The namespace cannot be an empty string.'):
        optree.register_pytree_node(
            set, lambda s: (sorted(s), None, None), lambda _, s: set(s), namespace=''
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
            lambda l: (l, None, None),
            lambda _, l: l,
            namespace=optree.registry.__GLOBAL_NAMESPACE,
        )

    with pytest.raises(
        ValueError,
        match=r"PyTree type <class 'list'> is already registered in the global namespace.",
    ):
        optree.register_pytree_node(
            list,
            lambda l: (l, None, None),
            lambda _, l: l,
            namespace='list',
        )


def test_pytree_node_registry_get():
    handler = optree.register_pytree_node.get(list)
    assert handler is not None
    l = [1, 2, 3]
    assert handler.to_iterable(l)[:2] == (l, None)

    handler = optree.register_pytree_node.get(list, namespace='any')
    assert handler is not None
    l = [1, 2, 3]
    assert handler.to_iterable(l)[:2] == (l, None)

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
