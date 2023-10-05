Tree Operations
===============

.. currentmodule:: optree

Constants
---------

.. autosummary::

    MAX_RECURSION_DEPTH
    NONE_IS_NODE
    NONE_IS_LEAF

.. autodata:: MAX_RECURSION_DEPTH
.. autodata:: NONE_IS_NODE
.. autodata:: NONE_IS_LEAF

------

Tree Manipulation Functions
---------------------------

.. autosummary::

    tree_flatten
    tree_flatten_with_path
    tree_unflatten
    tree_leaves
    tree_structure
    tree_paths
    all_leaves
    tree_map
    tree_map_
    tree_map_with_path
    tree_map_with_path_
    tree_replace_nones
    tree_transpose
    tree_broadcast_prefix
    broadcast_prefix
    prefix_errors

.. autofunction:: tree_flatten
.. autofunction:: tree_flatten_with_path
.. autofunction:: tree_unflatten
.. autofunction:: tree_leaves
.. autofunction:: tree_structure
.. autofunction:: tree_paths
.. autofunction:: all_leaves
.. autofunction:: tree_map
.. autofunction:: tree_map_
.. autofunction:: tree_map_with_path
.. autofunction:: tree_map_with_path_
.. autofunction:: tree_replace_nones
.. autofunction:: tree_transpose
.. autofunction:: tree_broadcast_prefix
.. autofunction:: broadcast_prefix
.. autofunction:: prefix_errors

------

Tree Reduce Functions
---------------------

.. autosummary::

    tree_reduce
    tree_sum
    tree_max
    tree_min
    tree_all
    tree_any

.. autofunction:: tree_reduce
.. autofunction:: tree_sum
.. autofunction:: tree_max
.. autofunction:: tree_min
.. autofunction:: tree_all
.. autofunction:: tree_any

------

PyTreeSpec Functions
--------------------

.. autosummary::

    treespec_is_prefix
    treespec_is_suffix
    treespec_paths
    treespec_entries
    treespec_entry
    treespec_children
    treespec_child
    treespec_is_leaf
    treespec_is_strict_leaf
    treespec_leaf
    treespec_none
    treespec_tuple

.. autofunction:: treespec_is_prefix
.. autofunction:: treespec_is_suffix
.. autofunction:: treespec_paths
.. autofunction:: treespec_entries
.. autofunction:: treespec_entry
.. autofunction:: treespec_children
.. autofunction:: treespec_child
.. autofunction:: treespec_is_leaf
.. autofunction:: treespec_is_strict_leaf
.. autofunction:: treespec_leaf
.. autofunction:: treespec_none
.. autofunction:: treespec_tuple
