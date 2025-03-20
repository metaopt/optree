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

import random
import re
import textwrap
from collections import OrderedDict, defaultdict, deque

import pytest

import optree
from helpers import (
    GLOBAL_NAMESPACE,
    TREES,
    CustomTuple,
    FlatCache,
    TimeStructTimeType,
    Vector2D,
    parametrize,
)


def test_different_types():
    lhs, rhs = (1, 2), [1, 2]
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(ValueError, match=r'Expected an instance of tuple, got .*\.'):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = {'a': 1, 'b': 2}, [1, 2]
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=(
            r'Expected an instance of dict, collections.OrderedDict, or collections.defaultdict, '
            r'got .*\.'
        ),
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = {'a': 1, 'b': 2}, OrderedDict({'a': 1, 'b': [2, 3]})
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert lhs_treespec.is_prefix(rhs_treespec)
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = {'a': 1, 'b': 2}, defaultdict(int, {'a': 1, 'b': [2, 3]})
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert lhs_treespec.is_prefix(rhs_treespec)
    () = optree.prefix_errors(lhs, rhs)


def test_different_types_nested():
    lhs, rhs = ((1,), (2,)), ([3], (4,))
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(ValueError, match=r'Expected an instance of .*, got .*\.'):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = {'a': 1, 'b': None}, {'a': 1, 'b': 2}
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(ValueError, match=r'Expected None, got .*\.'):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes['b']
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = {'a': 1, 'b': None}, {'a': 1, 'b': 2}
    lhs_treespec, rhs_treespec = (
        optree.tree_structure(lhs, none_is_leaf=True),
        optree.tree_structure(rhs, none_is_leaf=True),
    )
    optree.tree_map_(lambda x, y: None, lhs, rhs, none_is_leaf=True)
    assert lhs_treespec.is_prefix(rhs_treespec)
    () = optree.prefix_errors(lhs, rhs, none_is_leaf=True)


def test_different_types_multiple():
    lhs, rhs = ((1,), (2,)), ([3], [4])
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(ValueError, match=r'Expected an instance of .*, got .*\.'):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    e1, e2 = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_different_num_children():
    lhs, rhs = (1,), (2, 3)
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'tuple arity mismatch; expected: \d+, got: \d+; tuple: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_num_children_nested():
    lhs, rhs = [[1]], [[2, 3]]
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'list arity mismatch; expected: \d+, got: \d+; list: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_num_children_multiple():
    lhs, rhs = [[1], [2]], [[3, 4], [5, 6]]
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'list arity mismatch; expected: \d+, got: \d+; list: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    e1, e2 = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


def test_different_metadata():
    lhs, rhs = {1: 2}, {3: 4}
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; dict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = OrderedDict({'a': 1, 'b': 2}), OrderedDict({'a': 3, 'c': 4})
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; OrderedDict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = OrderedDict({'a': 1, 'b': 2}), OrderedDict({'a': 3})
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; OrderedDict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = OrderedDict({'a': 1, 'b': 2}), OrderedDict({'a': 3, 'b': 4, 'c': 5})
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; OrderedDict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    lhs, rhs = OrderedDict({'a': 1, 'b': [2, 3]}), OrderedDict({'b': [5, [6]], 'a': 4})
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore key ordering
    assert lhs_treespec.is_prefix(rhs_treespec)
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = defaultdict(list, {'a': 1, 'b': [2, 3]}), defaultdict(set, {'b': [5, [6]], 'a': 4})
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore default factory
    assert lhs_treespec.is_prefix(rhs_treespec)
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = {'a': 1, 'b': [2, 3]}, defaultdict(list, {'b': [5, [6]], 'a': 4})
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore dictionary types
    assert lhs_treespec.is_prefix(rhs_treespec)
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = OrderedDict({'b': [2, 3], 'a': 1}), {'a': 4, 'b': [5, 6]}
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore dictionary types
    assert lhs_treespec.is_prefix(rhs_treespec)
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = OrderedDict({'b': [2, 3], 'a': 1}), {'a': 4, 'b': [5, [6]]}
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore dictionary types and key ordering
    assert lhs_treespec.is_prefix(rhs_treespec)
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = deque([1, 2], maxlen=None), deque([3, [4, 5]], maxlen=3)
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    optree.tree_map_(lambda x, y: None, lhs, rhs)  # ignore maxlen
    assert lhs_treespec.is_prefix(rhs_treespec)
    () = optree.prefix_errors(lhs, rhs)

    lhs, rhs = FlatCache([None, 1]), FlatCache(1)
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'Mismatch custom node data; expected: .*, got: .*; value: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree metadata at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_metadata_nested():
    lhs, rhs = [{1: 2}], [{3: 4}]
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; dict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_different_metadata_multiple():
    lhs, rhs = [{1: 2}, {3: 4}], [{3: 4}, {5: 6}]
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(
        ValueError,
        match=r'dictionary key mismatch; expected key\(s\): .*, got key\(s\): .*; dict: .*\.',
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    e1, e2 = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes[0]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e1('in_axes')
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e2('in_axes')


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_standard_dictionary(
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
        random.seed(0)

        def build_subtree(x):
            return random.choice([x, (x,), [x, x], (x, [x]), {'a': x, 'b': [x]}])

        suffix_tree = optree.tree_map(
            build_subtree,
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        treespec = optree.tree_structure(
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )

        if 'FlatCache' in str(treespec):
            return

        def shuffle_dictionary(x):
            if type(x) in {dict, OrderedDict, defaultdict}:
                items = list(x.items())
                random.shuffle(items)
                dict_type = random.choice([dict, OrderedDict, defaultdict])
                if dict_type is defaultdict:
                    return defaultdict(getattr(x, 'default_factory', int), items)
                return dict_type(items)
            return x

        shuffled_tree = optree.tree_map(
            shuffle_dictionary,
            tree,
            is_leaf=lambda x: type(x) in {dict, OrderedDict, defaultdict},
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        shuffled_treespec = optree.tree_structure(
            shuffled_tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        shuffled_suffix_tree = optree.tree_map(
            shuffle_dictionary,
            suffix_tree,
            is_leaf=lambda x: type(x) in {dict, OrderedDict, defaultdict},
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        shuffled_suffix_treespec = optree.tree_structure(
            shuffled_suffix_tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )

        # Ignore dictionary types and key ordering
        optree.tree_map_(
            lambda x, y: None,
            shuffled_tree,
            shuffled_suffix_tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert shuffled_treespec.is_prefix(shuffled_suffix_treespec)
        () == optree.prefix_errors(  # noqa: B015
            shuffled_tree,
            shuffled_suffix_tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )


def test_namedtuple():
    lhs, rhs = CustomTuple(1, [2, [3]]), CustomTuple(4, [5, 6])
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(ValueError, match=re.escape('Expected an instance of list, got 6.')):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes.bar[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_structseq():
    lhs, rhs = (
        TimeStructTimeType((1, [2, [3]], *range(7))),
        TimeStructTimeType((4, [5, 6], *range(7))),
    )
    lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
    with pytest.raises(ValueError, match=re.escape('Expected an instance of list, got 6.')):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
    assert not lhs_treespec.is_prefix(rhs_treespec)

    (e,) = optree.prefix_errors(lhs, rhs)
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes.tm_mon[1]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_fallback():
    (e,) = optree.prefix_errors(Vector2D(1, [2]), Vector2D(3, 4))
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different types at key path
                in_axes[<flat index 1>]
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')


def test_no_errors():
    for lhs, rhs in (
        ((1, 2), ((11, 12, 13), 2)),
        ({'a': 1, 'b': 2}, {'b': (11, 12, 13), 'a': 2}),
    ):
        optree.tree_map_(lambda x, y: None, lhs, rhs)
        lhs_treespec, rhs_treespec = optree.tree_structure(lhs), optree.tree_structure(rhs)
        assert lhs_treespec.is_prefix(rhs_treespec)
        () = optree.prefix_errors(lhs, rhs)


def test_different_structure_no_children():
    (e,) = optree.prefix_errors((), ([],))
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different numbers of pytree children at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')

    (e,) = optree.prefix_errors({}, {'a': []})
    expected = re.escape(
        textwrap.dedent(
            """
            pytree structure error: different pytree keys at key path
                in_axes tree root
            """,
        ).strip(),
    )
    with pytest.raises(ValueError, match=expected):
        raise e('in_axes')
