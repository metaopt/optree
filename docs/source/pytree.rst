PyTree Operations
=================

.. currentmodule:: optree.pytree

.. automodule:: optree.pytree

Tree Operations
---------------

Check the :ref:`Tree Manipulation Functions` and :ref:`Tree Reduce Functions` sections for more
detailed documentation.

.. autosummary::

    dict_insertion_ordered
    flatten
    flatten_with_path
    flatten_with_accessor
    unflatten
    iter
    leaves
    structure
    paths
    accessors
    is_leaf
    map
    map_
    map_with_path
    map_with_path_
    map_with_accessor
    map_with_accessor_
    replace_nones
    transpose
    transpose_map
    transpose_map_with_path
    transpose_map_with_accessor
    broadcast_prefix
    broadcast_common
    broadcast_map
    broadcast_map_with_path
    broadcast_map_with_accessor
    reduce
    sum
    max
    min
    all
    any
    flatten_one_level

Node Registration
-----------------

Check the :ref:`PyTree Node Registration` section for more detailed documentation.

.. autosummary::

    register_node
    register_node_class
    unregister_node
