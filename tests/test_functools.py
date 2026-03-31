# Copyright 2022-2026 MetaOPT Team. All Rights Reserved.
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

import functools

import pytest

import optree
from helpers import GLOBAL_NAMESPACE, parametrize


HAS_PLACEHOLDER = hasattr(functools, 'Placeholder')
needs_placeholder = pytest.mark.skipif(
    not HAS_PLACEHOLDER,
    reason='functools.Placeholder requires Python 3.14+',
)


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
def test_partial_roundtrip(
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


@needs_placeholder
def test_partial_placeholder_roundtrip():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    p1 = optree.functools.partial(f, ph, 42)
    leaves, treespec = optree.tree_flatten(p1)
    p2 = optree.tree_unflatten(treespec, leaves)
    assert p2.func == p1.func
    assert p2.args == p1.args
    assert p2.args[0] is ph
    assert p2.keywords == p1.keywords
    assert p2('x') == f('x', 42)


@needs_placeholder
def test_partial_placeholder_call_after_roundtrip():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    p1 = optree.functools.partial(f, ph, 42)
    leaves, treespec = optree.tree_flatten(p1)
    p2 = optree.tree_unflatten(treespec, leaves)

    # Fill placeholder
    assert p2('x') == (('x', 42), {})

    # Extra args beyond placeholder
    assert p2('x', 'y') == (('x', 42, 'y'), {})

    # Missing placeholder arg
    with pytest.raises(TypeError, match='missing positional arguments'):
        p2()


@needs_placeholder
def test_partial_multiple_placeholders_roundtrip():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    p1 = optree.functools.partial(f, ph, 42, ph, 99)
    leaves, treespec = optree.tree_flatten(p1)
    p2 = optree.tree_unflatten(treespec, leaves)
    assert p2.args == (ph, 42, ph, 99)
    assert p2.args[0] is ph
    assert p2.args[2] is ph
    assert p2('a', 'b') == (('a', 42, 'b', 99), {})


@needs_placeholder
def test_partial_placeholder_with_keywords():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    p1 = optree.functools.partial(f, ph, 42, key='value')
    leaves, treespec = optree.tree_flatten(p1)
    p2 = optree.tree_unflatten(treespec, leaves)
    assert p2.args == (ph, 42)
    assert p2.keywords == {'key': 'value'}
    assert p2('x') == (('x', 42), {'key': 'value'})


@needs_placeholder
def test_partial_placeholder_is_leaf():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    p = optree.functools.partial(f, ph, 42)
    leaves = optree.tree_leaves(p)
    assert ph in leaves
    assert 42 in leaves


@needs_placeholder
def test_partial_placeholder_tree_map():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    p1 = optree.functools.partial(f, ph, 42)

    # Identity tree_map preserves Placeholder
    p2 = optree.tree_map(lambda x: x, p1)
    assert p2.args[0] is ph
    assert p2.args[1] == 42
    assert p2('test') == (('test', 42), {})


@needs_placeholder
def test_partial_placeholder_in_larger_tree():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    p = optree.functools.partial(f, ph, 42)
    tree = {'fn': p, 'data': [1, 2, 3]}
    leaves, treespec = optree.tree_flatten(tree)
    tree2 = optree.tree_unflatten(treespec, leaves)
    assert tree2['fn'].args[0] is ph
    assert tree2['fn']('test') == (('test', 42), {})
    assert tree2['data'] == [1, 2, 3]


@needs_placeholder
def test_partial_wrapping_stdlib_partial_with_placeholder():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    stdlib_p = functools.partial(f, ph, 42)
    op1 = optree.functools.partial(stdlib_p, 'extra')

    # Anti-merge: outer args are separate
    assert op1.args == ('extra',)
    assert op1() == (('extra', 42), {})

    # Roundtrip
    leaves, treespec = optree.tree_flatten(op1)
    op2 = optree.tree_unflatten(treespec, leaves)
    assert op2.args == ('extra',)
    assert op2() == (('extra', 42), {})


@needs_placeholder
def test_partial_wrapping_stdlib_partial_with_placeholder_no_extra_args():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    stdlib_p = functools.partial(f, ph, 42)
    op1 = optree.functools.partial(stdlib_p)
    assert op1.args == ()
    assert op1('hello') == (('hello', 42), {})

    # Roundtrip
    leaves, treespec = optree.tree_flatten(op1)
    op2 = optree.tree_unflatten(treespec, leaves)
    assert op2('hello') == (('hello', 42), {})


@needs_placeholder
def test_partial_nested_optree_partial_with_placeholder():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    inner = optree.functools.partial(f, ph, 42)
    outer = optree.functools.partial(inner, 'extra')

    # Anti-merge behavior
    assert outer.args == ('extra',)
    assert outer() == (('extra', 42), {})

    # Roundtrip of outer
    leaves, treespec = optree.tree_flatten(outer)
    outer2 = optree.tree_unflatten(treespec, leaves)
    assert outer2() == (('extra', 42), {})


@needs_placeholder
def test_partial_trailing_placeholder_rejection():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    with pytest.raises(TypeError, match='trailing Placeholders are not allowed'):
        optree.functools.partial(f, 42, ph)

    with pytest.raises(TypeError, match='trailing Placeholders are not allowed'):
        optree.functools.partial(f, ph)

    with pytest.raises(TypeError, match='trailing Placeholders are not allowed'):
        optree.functools.partial(f, ph, 1, ph)


@needs_placeholder
def test_partial_keyword_placeholder_rejection():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    with pytest.raises(TypeError, match='Placeholder'):
        optree.functools.partial(f, kw=ph)


@needs_placeholder
def test_partial_repr_with_placeholder():
    ph = functools.Placeholder

    def f(*args, **kwargs):
        return args, kwargs

    p = optree.functools.partial(f, ph, 42)
    r = repr(p)
    assert 'Placeholder' in r
    assert '42' in r


@needs_placeholder
def test_partial_placeholder_reexport():
    assert hasattr(optree.functools, 'Placeholder')
    assert optree.functools.Placeholder is functools.Placeholder
