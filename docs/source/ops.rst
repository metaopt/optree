.. _Tree Operations:

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

.. _Tree Manipulation Functions:

Tree Manipulation Functions
---------------------------

.. autosummary::

    dict_insertion_ordered
    tree_flatten
    tree_flatten_with_path
    tree_flatten_with_accessor
    tree_unflatten
    tree_iter
    tree_leaves
    tree_structure
    tree_paths
    tree_accessors
    tree_is_leaf
    all_leaves
    tree_map
    tree_map_
    tree_map_with_path
    tree_map_with_path_
    tree_map_with_accessor
    tree_map_with_accessor_
    tree_replace_nones
    tree_partition
    tree_transpose
    tree_transpose_map
    tree_transpose_map_with_path
    tree_transpose_map_with_accessor
    tree_broadcast_prefix
    broadcast_prefix
    tree_broadcast_common
    broadcast_common
    tree_broadcast_map
    tree_broadcast_map_with_path
    tree_broadcast_map_with_accessor
    tree_flatten_one_level
    prefix_errors

.. autofunction:: dict_insertion_ordered
.. autofunction:: tree_flatten
.. autofunction:: tree_flatten_with_path
.. autofunction:: tree_flatten_with_accessor
.. autofunction:: tree_unflatten
.. autofunction:: tree_iter
.. autofunction:: tree_leaves
.. autofunction:: tree_structure
.. autofunction:: tree_paths
.. autofunction:: tree_accessors
.. autofunction:: tree_is_leaf
.. autofunction:: all_leaves
.. autofunction:: tree_map
.. autofunction:: tree_map_
.. autofunction:: tree_map_with_path
.. autofunction:: tree_map_with_path_
.. autofunction:: tree_map_with_accessor
.. autofunction:: tree_map_with_accessor_
.. autofunction:: tree_replace_nones
.. autofunction:: tree_partition
.. autofunction:: tree_transpose
.. autofunction:: tree_transpose_map
.. autofunction:: tree_transpose_map_with_path
.. autofunction:: tree_transpose_map_with_accessor
.. autofunction:: tree_broadcast_prefix
.. autofunction:: broadcast_prefix
.. autofunction:: tree_broadcast_common
.. autofunction:: broadcast_common
.. autofunction:: tree_broadcast_map
.. autofunction:: tree_broadcast_map_with_path
.. autofunction:: tree_broadcast_map_with_accessor
.. autofunction:: tree_flatten_one_level
.. autofunction:: prefix_errors

------

.. _Tree Reduce Functions:

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

.. _PyTreeSpec Functions:

:class:`PyTreeSpec` Functions
-----------------------------

.. autosummary::

    treespec_paths
    treespec_accessors
    treespec_entries
    treespec_entry
    treespec_children
    treespec_child
    treespec_one_level
    treespec_transform
    treespec_is_leaf
    treespec_is_strict_leaf
    treespec_is_one_level
    treespec_is_prefix
    treespec_is_suffix
    treespec_leaf
    treespec_none
    treespec_tuple
    treespec_list
    treespec_dict
    treespec_namedtuple
    treespec_ordereddict
    treespec_defaultdict
    treespec_deque
    treespec_structseq
    treespec_from_collection

.. autofunction:: treespec_paths
.. autofunction:: treespec_accessors
.. autofunction:: treespec_entries
.. autofunction:: treespec_entry
.. autofunction:: treespec_children
.. autofunction:: treespec_child
.. autofunction:: treespec_one_level
.. autofunction:: treespec_transform
.. autofunction:: treespec_is_leaf
.. autofunction:: treespec_is_strict_leaf
.. autofunction:: treespec_is_one_level
.. autofunction:: treespec_is_prefix
.. autofunction:: treespec_is_suffix
.. autofunction:: treespec_leaf
.. autofunction:: treespec_none
.. autofunction:: treespec_tuple
.. autofunction:: treespec_list
.. autofunction:: treespec_dict
.. autofunction:: treespec_namedtuple
.. autofunction:: treespec_ordereddict
.. autofunction:: treespec_defaultdict
.. autofunction:: treespec_deque
.. autofunction:: treespec_structseq
.. autofunction:: treespec_from_collection
