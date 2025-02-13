import optree
import optree.pytree as pytree


def test_pytree_namespace():
    assert set(pytree.__all__) == {
        name[len('tree_') :] for name in optree.__all__ if name.startswith('tree_')
    }

    for name in pytree.__all__:
        assert (
            getattr(pytree, name).__annotations__ == getattr(optree, f'tree_{name}').__annotations__
        )
