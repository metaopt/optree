import optree
import optree.pytree as pytree
import optree.treespec as treespec


def test_pytree_reexports():
    assert set(pytree.__all__) == {
        name[len('tree_') :] for name in optree.__all__ if name.startswith('tree_')
    }

    for name in pytree.__all__:
        assert getattr(pytree, name) is getattr(optree, f'tree_{name}')


def test_pytree_reexports():
    # Leave as templates for future tests.
    # Not all `treespec` functions are re-exported,
    # assert two sets make non sense.

    # assert set(treespec.__all__) == {
    #     'leaf',
    #     'none',
    #     'tuple',
    #     'list',
    #     'dict',
    #     'namedtuple',
    #     'ordereddict',
    #     'defaultdict',
    #     'deque',
    #     'structseq',
    #     'from_collection',
    # }

    for name in treespec.__all__:
        assert getattr(treespec, name) is getattr(optree, f'treespec_{name}')
