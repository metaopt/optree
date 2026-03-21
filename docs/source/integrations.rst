Integrations with Third-Party Libraries
=======================================

Integration for `attrs <https://github.com/python-attrs/attrs>`_
----------------------------------------------------------------

.. currentmodule:: optree.integrations.attrs

.. autosummary::

    field
    define
    frozen
    mutable
    make_class
    register_node
    AttrsEntry

.. autofunction:: field
.. autofunction:: define
.. autofunction:: frozen
.. data:: mutable

   Alias for :func:`define`.
.. autofunction:: make_class
.. autofunction:: register_node
.. autoclass:: AttrsEntry

------

Integration for `JAX <https://github.com/jax-ml/jax>`_
------------------------------------------------------

.. currentmodule:: optree.integrations.jax

.. autosummary::

    tree_ravel

.. autofunction:: tree_ravel

------

Integration for `NumPy <https://github.com/numpy/numpy>`_
---------------------------------------------------------

.. currentmodule:: optree.integrations.numpy

.. autosummary::

    tree_ravel

.. autofunction:: tree_ravel

------

Integration for `PyTorch <https://github.com/pytorch/pytorch>`_
---------------------------------------------------------------

.. currentmodule:: optree.integrations.torch

.. autosummary::

    tree_ravel

.. autofunction:: tree_ravel
