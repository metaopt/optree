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

# pylint: disable=missing-function-docstring,invalid-name,wrong-import-order

import functools

import optree
from helpers import GLOBAL_NAMESPACE, parametrize


def dummy_func(*args, **kwargs):  # pylint: disable=unused-argument
    return


dummy_partial_func = functools.partial(dummy_func, a=1)


@parametrize(
    tree=[
        optree.functools.partial(dummy_func),
        optree.functools.partial(dummy_func, 1, 2),
        optree.functools.partial(dummy_func, x='a'),
        optree.functools.partial(dummy_func, 1, 2, 3, x=4, y=5),
        optree.functools.partial(dummy_func, 1, None, x=4, y=5, z=None),
        optree.functools.partial(dummy_partial_func, 1, 2, 3, x=4, y=5),
    ],
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_partial_round_trip(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        leaves, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        actual = optree.tree_unflatten(treespec, leaves)
        assert actual.func == tree.func
        assert actual.args == tree.args
        assert actual.keywords == tree.keywords
        assert tuple(actual.keywords.items()) == tuple(tree.keywords.items())


def test_partial_does_not_merge_with_other_partials():
    def f(a=None, b=None, c=None):
        return a, b, c

    g = functools.partial(f, 2)
    h = optree.functools.partial(g, 3)
    assert h.args == (3,)
    assert g() == (2, None, None)
    assert h() == (2, 3, None)


def test_partial_func_attribute_has_stable_hash():
    fn = functools.partial(print, 1)
    p1 = optree.functools.partial(fn, 2)
    p2 = optree.functools.partial(fn, 2)
    assert p1.func == fn  # pylint: disable=comparison-with-callable
    assert fn == p1.func  # pylint: disable=comparison-with-callable
    assert p1.func == p2.func
    assert hash(p1.func) == hash(p2.func)
