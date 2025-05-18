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

import importlib
import re
import sys
import types

import pytest

import optree


def test_pytree_reexports():
    tree_operations = [name[len('tree_') :] for name in optree.__all__ if name.startswith('tree_')]
    assert optree.pytree.__all__ == [
        'reexport',
        'PyTreeSpec',
        'PyTreeKind',
        'PyTreeEntry',
        *tree_operations,
        'register_node',
        'register_node_class',
        'unregister_node',
        'dict_insertion_ordered',
    ]

    assert optree.pytree.PyTreeSpec is optree.PyTreeSpec
    assert optree.pytree.PyTreeKind is optree.PyTreeKind
    assert optree.pytree.PyTreeEntry is optree.PyTreeEntry
    for name in tree_operations:
        assert getattr(optree.pytree, name) is getattr(optree, f'tree_{name}')
    assert optree.pytree.register_node is optree.register_pytree_node
    assert optree.pytree.register_node_class is optree.register_pytree_node_class
    assert optree.pytree.unregister_node is optree.unregister_pytree_node
    assert optree.pytree.dict_insertion_ordered is optree.dict_insertion_ordered
    assert optree.pytree.__version__ == optree.__version__


def test_treespec_reexports():
    treespec_operations = [
        name[len('treespec_') :] for name in optree.__all__ if name.startswith('treespec_')
    ]
    treespec_all_set = set(optree.treespec.__all__)
    assert treespec_all_set.issubset(treespec_operations)
    assert optree.treespec.__all__ == [
        name for name in treespec_operations if name in treespec_all_set
    ]

    for name in optree.treespec.__all__:
        assert getattr(optree.treespec, name) is getattr(optree, f'treespec_{name}')


def test_pytree_reexport_with_invalid_module():
    with pytest.raises(ValueError, match=r'invalid module name'):
        optree.pytree.reexport(namespace='some-namespace', module='123')
    with pytest.raises(ValueError, match=r'invalid module name'):
        optree.pytree.reexport(namespace='some-namespace', module='abc.123')
    with pytest.raises(ValueError, match=r'invalid module name'):
        optree.pytree.reexport(namespace='some-namespace', module='abc-def')
    with pytest.raises(ValueError, match=r'invalid module name'):
        optree.pytree.reexport(namespace='some-namespace', module=' ')
    with pytest.raises(ValueError, match=r'invalid module name'):
        optree.pytree.reexport(namespace='some-namespace', module=' abc')
    with pytest.raises(ValueError, match=r'invalid module name'):
        optree.pytree.reexport(namespace='some-namespace', module='')


def test_pytree_reexport_with_empty_module():
    assert f'{__name__}.pytree' not in sys.modules
    assert f'{__name__}.pytree.dataclasses' not in sys.modules
    assert f'{__name__}.pytree.functools' not in sys.modules

    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import test_shortcut.pytree
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import test_shortcut.pytree.dataclasses
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import test_shortcut.pytree.functools  # noqa: F401

    pytree1 = optree.pytree.reexport(namespace='pytree1')
    assert f'{__name__}.pytree' in sys.modules
    assert f'{__name__}.pytree.dataclasses' in sys.modules
    assert f'{__name__}.pytree.functools' in sys.modules
    assert pytree1 is sys.modules[f'{__name__}.pytree']
    assert pytree1.dataclasses is sys.modules[f'{__name__}.pytree.dataclasses']
    assert pytree1.functools is sys.modules[f'{__name__}.pytree.functools']
    assert pytree1.__name__ == f'{__name__}.pytree'
    assert pytree1.dataclasses.__name__ == f'{__name__}.pytree.dataclasses'
    assert pytree1.functools.__name__ == f'{__name__}.pytree.functools'
    assert type(pytree1) is optree.pytree.ReexportedModule
    assert type(pytree1.dataclasses) is optree.pytree.ReexportedModule
    assert type(pytree1.functools) is optree.pytree.ReexportedModule
    assert 'dataclasses' not in pytree1.__all__
    assert 'functools' not in pytree1.__all__
    assert '__version__' not in pytree1.__all__
    assert 'reexport' not in pytree1.__all__
    assert 'dataclasses' in dir(pytree1)
    assert 'functools' in dir(pytree1)
    assert '__version__' in dir(pytree1)
    assert 'reexport' not in dir(pytree1)
    assert pytree1.__version__ == optree.__version__

    assert importlib.import_module('test_shortcut.pytree') is pytree1
    assert importlib.import_module('test_shortcut.pytree.dataclasses') is pytree1.dataclasses
    assert importlib.import_module('test_shortcut.pytree.functools') is pytree1.functools

    with pytest.raises(ValueError, match=re.escape(f"module '{__name__}.pytree' already exists")):
        optree.pytree.reexport(namespace='pytree2')


def test_pytree_reexport_with_module():
    assert 'some_package' not in sys.modules
    assert 'some_package.pytree_mod' not in sys.modules
    assert 'some_package.pytree_mod.dataclasses' not in sys.modules
    assert 'some_package.pytree_mod.functools' not in sys.modules

    sys.modules['some_package'] = types.ModuleType('some_package')

    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import some_package.pytree_mod
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import some_package.pytree_mod.dataclasses
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import some_package.pytree_mod.functools  # noqa: F401

    pytree3 = optree.pytree.reexport(namespace='pytree3', module='some_package.pytree_mod')
    assert 'some_package.pytree_mod' in sys.modules
    assert 'some_package.pytree_mod.dataclasses' in sys.modules
    assert 'some_package.pytree_mod.functools' in sys.modules
    assert pytree3 is sys.modules['some_package.pytree_mod']
    assert pytree3.dataclasses is sys.modules['some_package.pytree_mod.dataclasses']
    assert pytree3.functools is sys.modules['some_package.pytree_mod.functools']
    assert pytree3.__name__ == 'some_package.pytree_mod'
    assert pytree3.dataclasses.__name__ == 'some_package.pytree_mod.dataclasses'
    assert pytree3.functools.__name__ == 'some_package.pytree_mod.functools'
    assert type(pytree3) is optree.pytree.ReexportedModule
    assert type(pytree3.dataclasses) is optree.pytree.ReexportedModule
    assert type(pytree3.functools) is optree.pytree.ReexportedModule
    assert 'dataclasses' not in pytree3.__all__
    assert 'functools' not in pytree3.__all__
    assert '__version__' not in pytree3.__all__
    assert 'reexport' not in pytree3.__all__
    assert 'dataclasses' in dir(pytree3)
    assert 'functools' in dir(pytree3)
    assert '__version__' in dir(pytree3)
    assert 'reexport' not in dir(pytree3)
    assert pytree3.__version__ == optree.__version__

    assert importlib.import_module('some_package.pytree_mod') is pytree3
    assert importlib.import_module('some_package.pytree_mod.dataclasses') is pytree3.dataclasses
    assert importlib.import_module('some_package.pytree_mod.functools') is pytree3.functools

    with pytest.raises(
        ValueError,
        match=re.escape("module 'some_package.pytree_mod' already exists"),
    ):
        optree.pytree.reexport(namespace='pytree4', module='some_package.pytree_mod')
