#!/usr/bin/env python3
#
# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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

# pylint: disable=missing-module-docstring,missing-function-docstring,invalid-name

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict, namedtuple
from itertools import count
from typing import Any, Iterable

import jax
import pandas as pd
import torch
import torch.nn as nn
import torch.utils._pytree as torch_utils_pytree
import tree as dm_tree  # noqa: F401 # pylint: disable=unused-import
from torchvision import models

import optree


try:
    from termcolor import colored
except ImportError:
    colored = None

if not sys.stdout.isatty() or colored is None:

    def colored(  # pylint: disable=function-redefined,unused-argument
        text: str,
        color: str | None = None,
        on_color: str | None = None,
        attrs: Iterable[str] | None = None,
    ) -> str:
        return text


torch_utils_pytree._register_pytree_node(  # pylint: disable=protected-access
    dict,
    lambda d: tuple(zip(*sorted(d.items())))[::-1] if d else ((), ()),
    lambda values, keys: dict(zip(keys, values)),
)


Tensors = namedtuple('Tensors', ['parameters', 'buffers'])
Containers = namedtuple('Containers', ['parameters', 'buffers'])


def get_none() -> None:
    return None


def tiny_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 1, bias=True),
        nn.BatchNorm1d(1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Linear(1, 1, bias=False),
        nn.Sigmoid(),
    )


def extract(module: nn.Module, unordered: bool) -> Any:
    if isinstance(module, (nn.Sequential, nn.ModuleList)):
        return [extract(submodule, unordered=unordered) for submodule in module]

    dict_factory = dict if unordered else OrderedDict
    extracted = dict_factory(
        [
            (name, extract(submodule, unordered=unordered))
            for name, submodule in module.named_children()
        ]
    )
    extracted.update(
        tensors=Tensors(
            parameters=tuple(t.data for t in module.parameters(recurse=False)),
            buffers=[t.data for t in module.buffers(recurse=False)],
        ),
        containers=Containers(
            parameters=dict_factory(module._parameters),  # pylint: disable=protected-access
            buffers=dict_factory(module._buffers),  # pylint: disable=protected-access
        ),
    )

    return extracted


cmark = '✔'  # pylint: disable=invalid-name
xmark = '✘'  # pylint: disable=invalid-name
tie = '~'  # pylint: disable=invalid-name
CMARK = colored(cmark, color='green', attrs=('bold',))
XMARK = colored(xmark, color='red', attrs=('bold',))
TIE = colored(tie, color='yellow', attrs=('bold',))

INIT = """
import jax
import optree
import torch.utils._pytree as torch_utils_pytree
import tree as dm_tree

def fn1(x):
    return None

def fn3(x, y, z):
    return None

def fnp1(p, x):
    return None

def fnp3(p, x, y, z):
    return None
"""

STMTS = OrderedDict(
    [
        (
            'Tree Flatten',
            OrderedDict(
                [
                    ('OpTree(default)', ('optree.tree_leaves(x)', '')),
                    ('OpTree(NoneIsNode)', ('optree.tree_leaves(x, none_is_leaf=False)', '')),
                    ('OpTree(NoneIsLeaf)', ('optree.tree_leaves(x, none_is_leaf=True)', '')),
                    ('JAX XLA', ('jax.tree_util.tree_leaves(x)', '')),
                    ('PyTorch', ('torch_utils_pytree.tree_flatten(x)[0]', '')),
                    ('DM-Tree', ('dm_tree.flatten(x)', '')),
                ]
            ),
        ),
        (
            'Tree UnFlatten',
            OrderedDict(
                [
                    (
                        'OpTree(NoneIsNode)',  # default
                        (
                            'optree.tree_unflatten(spec, flat)',
                            'flat, spec = optree.tree_flatten(x, none_is_leaf=False)',
                        ),
                    ),
                    (
                        'OpTree(NoneIsLeaf)',
                        (
                            'optree.tree_unflatten(spec, flat)',
                            'flat, spec = optree.tree_flatten(x, none_is_leaf=True)',
                        ),
                    ),
                    (
                        'JAX XLA',
                        (
                            'jax.tree_util.tree_unflatten(spec, flat)',
                            'flat, spec = jax.tree_util.tree_flatten(x)',
                        ),
                    ),
                    (
                        'PyTorch',
                        (
                            'torch_utils_pytree.tree_unflatten(flat, spec)',
                            'flat, spec = torch_utils_pytree.tree_flatten(x)',
                        ),
                    ),
                    (
                        'DM-Tree',
                        (
                            'dm_tree.unflatten_as(spec, flat)',
                            'flat, spec = dm_tree.flatten(x), x',
                        ),
                    ),
                ]
            ),
        ),
        (
            'Tree Flatten with Path',
            OrderedDict(
                [
                    ('OpTree(default)', ('optree.tree_flatten_with_path(x)', '')),
                    (
                        'OpTree(NoneIsNode)',
                        ('optree.tree_flatten_with_path(x, none_is_leaf=False)', ''),
                    ),
                    (
                        'OpTree(NoneIsLeaf)',
                        ('optree.tree_flatten_with_path(x, none_is_leaf=True)', ''),
                    ),
                    ('JAX XLA', ('jax.tree_util.tree_flatten_with_path(x)', '')),
                    ('DM-Tree', ('dm_tree.flatten_with_path(x)', '')),
                ]
            ),
        ),
        (
            'Tree Copy',
            OrderedDict(
                [
                    (
                        'OpTree(default)',
                        ('optree.tree_unflatten(*optree.tree_flatten(x)[::-1])', ''),
                    ),
                    (
                        'OpTree(NoneIsNode)',
                        (
                            'optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])',
                            '',
                        ),
                    ),
                    (
                        'OpTree(NoneIsLeaf)',
                        (
                            'optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])',
                            '',
                        ),
                    ),
                    (
                        'JAX XLA',
                        ('jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])', ''),
                    ),
                    (
                        'PyTorch',
                        (
                            'torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))',
                            '',
                        ),
                    ),
                    (
                        'DM-Tree',
                        ('dm_tree.unflatten_as(x, dm_tree.flatten(x))', ''),
                    ),
                ]
            ),
        ),
        (
            'Tree Map',
            OrderedDict(
                [
                    ('OpTree(default)', ('optree.tree_map(fn1, x)', '')),
                    ('OpTree(NoneIsNode)', ('optree.tree_map(fn1, x, none_is_leaf=False)', '')),
                    ('OpTree(NoneIsLeaf)', ('optree.tree_map(fn1, x, none_is_leaf=True)', '')),
                    ('JAX XLA', ('jax.tree_util.tree_map(fn1, x)', '')),
                    ('PyTorch', ('torch_utils_pytree.tree_map(fn1, x)', '')),
                    ('DM-Tree', ('dm_tree.map_structure(fn1, x)', '')),
                ]
            ),
        ),
        (
            'Tree Map (nargs)',
            OrderedDict(
                [
                    ('OpTree(default)', ('optree.tree_map(fn3, x, y, z)', '')),
                    (
                        'OpTree(NoneIsNode)',
                        ('optree.tree_map(fn3, x, y, z, none_is_leaf=False)', ''),
                    ),
                    (
                        'OpTree(NoneIsLeaf)',
                        ('optree.tree_map(fn3, x, y, z, none_is_leaf=True)', ''),
                    ),
                    ('JAX XLA', ('jax.tree_util.tree_map(fn3, x, y, z)', '')),
                    ('DM-Tree', ('dm_tree.map_structure_up_to(x, fn3, x, y, z)', '')),
                ]
            ),
        ),
        (
            'Tree Map with Path',
            OrderedDict(
                [
                    ('OpTree(default)', ('optree.tree_map_with_path(fnp1, x)', '')),
                    (
                        'OpTree(NoneIsNode)',
                        ('optree.tree_map_with_path(fnp1, x, none_is_leaf=False)', ''),
                    ),
                    (
                        'OpTree(NoneIsLeaf)',
                        ('optree.tree_map_with_path(fnp1, x, none_is_leaf=True)', ''),
                    ),
                    ('JAX XLA', ('jax.tree_util.tree_map_with_path(fnp1, x)', '')),
                    ('DM-Tree', ('dm_tree.map_structure_with_path(fnp1, x)', '')),
                ]
            ),
        ),
        (
            'Tree Map with Path (nargs)',
            OrderedDict(
                [
                    ('OpTree(default)', ('optree.tree_map_with_path(fnp3, x, y, z)', '')),
                    (
                        'OpTree(NoneIsNode)',
                        (
                            'optree.tree_map_with_path(fnp3, x, y, z, none_is_leaf=False)',
                            '',
                        ),
                    ),
                    (
                        'OpTree(NoneIsLeaf)',
                        (
                            'optree.tree_map_with_path(fnp3, x, y, z, none_is_leaf=True)',
                            '',
                        ),
                    ),
                    ('JAX XLA', ('jax.tree_util.tree_map_with_path(fnp3, x, y, z)', '')),
                    ('DM-Tree', ('dm_tree.map_structure_with_path(fnp3, x, y, z)', '')),
                ]
            ),
        ),
    ]
)


def cprint(text: str = '') -> None:
    text = (
        text.replace(
            ', none_is_leaf=False',
            colored(', none_is_leaf=False', attrs=('dark',)),
        )
        .replace(
            ', none_is_leaf=True',
            colored(', none_is_leaf=True', attrs=('dark',)),
        )
        .replace(
            'NoneIsNode',
            colored('NoneIsNode', attrs=('dark',)),
        )
        .replace(
            'NoneIsLeaf',
            colored('NoneIsLeaf', attrs=('dark',)),
        )
        .replace(
            'None is Node',
            colored('None is Node', attrs=('dark',)),
        )
        .replace(
            'None is Leaf',
            colored('None is Leaf', attrs=('dark',)),
        )
    )

    print(text)


def check(tree: Any) -> None:
    print('### Check ###')
    if optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree:
        cprint(
            f'{CMARK} COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree'
        )
    else:
        cprint(
            f'{XMARK} COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) != tree'
        )
    if optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree:
        cprint(
            f'{CMARK} COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree'
        )
    else:
        cprint(
            f'{XMARK} COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) != tree'
        )

    optree_flat = optree.tree_leaves(tree, none_is_leaf=False)
    jax_flat = jax.tree_util.tree_leaves(tree)
    if len(optree_flat) == len(jax_flat) and all(map(lambda a, b: a is b, optree_flat, jax_flat)):
        cprint(
            f'{CMARK} FLATTEN (OpTree vs. JAX XLA): '
            f'optree.tree_leaves(tree, none_is_leaf=False)'
            f' == jax.tree_util.tree_leaves(tree)'
        )
    else:
        cprint(
            f'{XMARK} FLATTEN (OpTree vs. JAX XLA): '
            f'optree.tree_leaves(tree, none_is_leaf=False)'
            f' != jax.tree_util.tree_leaves(tree)'
        )

    optree_flat = optree.tree_leaves(tree, none_is_leaf=True)
    torch_flat = torch_utils_pytree.tree_flatten(tree)[0]
    if len(optree_flat) == len(torch_flat) and all(
        map(lambda a, b: a is b, optree_flat, torch_flat)
    ):
        cprint(
            f'{CMARK} FLATTEN (OpTree vs. PyTorch): '
            f'optree.tree_leaves(tree, none_is_leaf=True)'
            f' == torch_utils_pytree.tree_flatten(tree)[0]'
        )
    else:
        cprint(
            f'{XMARK} FLATTEN (OpTree vs. PyTorch): '
            f'optree.tree_leaves(tree, none_is_leaf=True)'
            f' != torch_utils_pytree.tree_flatten(tree)[0]'
        )

    counter = count()
    optree_map_res = optree.tree_map(lambda t: next(counter), tree, none_is_leaf=False)
    counter = count()
    jax_map_res = jax.tree_util.tree_map(lambda t: next(counter), tree)
    if optree_map_res == jax_map_res:
        cprint(
            f'{CMARK} TREEMAP (OpTree vs. JAX XLA): '
            f'optree.tree_map(fn, tree, none_is_leaf=False)'
            f' == jax.tree_util.tree_map(fn, tree)'
        )
    else:
        cprint(
            f'{XMARK} TREEMAP (OpTree vs. JAX XLA): '
            f'optree.tree_map(fn, tree, none_is_leaf=False)'
            f' != jax.tree_util.tree_map(fn, tree)'
        )

    counter = count()
    optree_map_res = optree.tree_map(lambda t: next(counter), tree, none_is_leaf=True)
    counter = count()
    torch_map_res = torch_utils_pytree.tree_map(lambda t: next(counter), tree)
    if optree_map_res == torch_map_res:
        cprint(
            f'{CMARK} TREEMAP (OpTree vs. PyTorch): '
            f'optree.tree_map(fn, tree, none_is_leaf=True)'
            f' == torch_utils_pytree.tree_map(fn, tree)'
        )
    else:
        cprint(
            f'{XMARK} TREEMAP (OpTree vs. PyTorch): '
            f'optree.tree_map(fn, tree, none_is_leaf=True)'
            f' != torch_utils_pytree.tree_map(fn, tree)'
        )

    print(flush=True)


def timeit(
    stmt: str,
    init_stmt: str,
    number: int,
    repeat: int = 5,
    globals: dict[str, Any] | None = None,  # pylint: disable=redefined-builtin
) -> float:
    import timeit  # pylint: disable=redefined-outer-name,import-outside-toplevel

    globals = globals or {}
    init_warmup = f"""
{INIT.strip()}

{init_stmt.strip()}

for _ in range({number} // 10):
    {stmt.strip()}
"""

    times = timeit.repeat(
        stmt.strip(),
        setup=init_warmup,
        globals=globals,
        number=number,
        repeat=repeat,
    )
    return min(times) / number


def compare(  # pylint: disable=too-many-locals
    subject: str,
    stmts: dict[str, tuple[str, str]],
    number: int,
    repeat: int = 5,
    globals: dict[str, Any] | None = None,  # pylint: disable=redefined-builtin
) -> dict[str, float]:
    times_us = OrderedDict(
        [
            (lib, 10e6 * timeit(stmt, init_stmt, number, repeat=repeat, globals=globals))
            for lib, (stmt, init_stmt) in stmts.items()
        ]
    )
    base_time = next(iter(times_us.values()))
    best_time = min(times_us.values())
    speedups = {lib: time / base_time for lib, time in times_us.items()}
    best_speedups = {lib: time / best_time for lib, time in times_us.items()}
    labels = {
        lib: (cmark if speedup == 1.0 else tie if speedup < 1.1 else ' ')
        for lib, speedup in best_speedups.items()
    }
    colors = {
        lib: (
            'green'
            if speedup == 1.0
            else 'cyan'
            if speedup < 1.1
            else 'yellow'
            if speedup < 4.0
            else 'red'
        )
        for lib, speedup in best_speedups.items()
    }
    attrs = {lib: ('bold',) if color == 'green' else () for lib, color in colors.items()}

    cprint(f'### {subject} ###')
    for lib, (stmt, _) in stmts.items():
        label = labels[lib]
        color = colors[lib]
        attr = attrs[lib]
        label = colored(label + ' ' + lib, color=color, attrs=attr)
        time = colored(f'{times_us[lib]:8.2f}μs', attrs=attr)
        speedup = speedups[lib]
        if speedup != 1.0:
            speedup = ' -- ' + colored(f'x{speedup:.2f}'.ljust(6), color=color, attrs=attr)
        else:
            speedup = '          '
        if '(NoneIsLeaf)' in label:
            label = label.replace('(NoneIsLeaf)', ' ')
            if 'none_is_leaf=True' in stmt:
                stmt += ' '
            stmt += '  (None is Leaf)'
        elif '(NoneIsNode)' in label:
            label = label.replace('(NoneIsNode)', ' ').replace('(default)', ' ')
            stmt += '  (None is Node)'
        elif '(default)' in label:
            label = label.replace('(default)', ' ')
            stmt += '                      (None is Node)'
        cprint(f'{label}: {time}{speedup}  <=  {stmt}')

    print(flush=True)

    return times_us


def benchmark(  # pylint: disable=too-many-locals
    name: str,
    module: nn.Module,
    number: int = 10000,
    repeat: int = 5,
    unordered: bool = False,
) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            'Subject',
            'Module',
            'Nodes',
            'Leaves',
            'OpTree (μs)',
            'JAX XLA (μs)',
            'PyTorch (μs)',
            'DM-Tree (μs)',
            'Speedup (J / O)',
            'Speedup (P / O)',
            'Speedup (D / O)',
        ]
    )

    x = extract(module, unordered=unordered)
    treespec = optree.tree_structure(x)
    treespec_repr = repr(treespec)
    treespec_repr = (
        treespec_repr
        if len(treespec_repr) < 67
        else f'{treespec_repr[:32]}...{treespec_repr[-32:]}'
    )
    print(
        f'{colored(name, color="blue", attrs=("bold",))}'
        f'(num_nodes={treespec.num_nodes}, num_leaves={treespec.num_leaves}, treespec={treespec_repr})',
        flush=True,
    )
    y = optree.tree_map(torch.zeros_like, x)
    z = optree.tree_map(lambda t: (t, None), x)  # pylint: disable=invalid-name

    check(x)
    for subject, stmts in STMTS.items():
        times_us = compare(
            subject,
            stmts,
            number=number,
            repeat=repeat,
            globals={'x': x, 'y': y, 'z': z},
        )
        data = {
            'Subject': subject,
            'Module': name,
            'Nodes': treespec.num_nodes,
            'Leaves': treespec.num_leaves,
            'OpTree (μs)': next(iter(times_us.values())),
            'JAX XLA (μs)': times_us.get('JAX XLA', pd.NA),
            'PyTorch (μs)': times_us.get('PyTorch', pd.NA),
            'DM-Tree (μs)': times_us.get('DM-Tree', pd.NA),
        }
        data['Speedup (J / O)'] = data['JAX XLA (μs)'] / data['OpTree (μs)']
        data['Speedup (P / O)'] = data['PyTorch (μs)'] / data['OpTree (μs)']
        data['Speedup (D / O)'] = data['DM-Tree (μs)'] / data['OpTree (μs)']
        records = pd.DataFrame.from_records(data, index=[len(df)])
        df = pd.concat([df, records], ignore_index=True)

    print(
        df.drop(columns=['Module', 'Nodes', 'Leaves'])
        .to_markdown(floatfmt='8.2f', index=False)
        .replace('|  ', '|')
        .replace('|--', '|')
        .replace('nan', 'N/A')
        .replace('N/A    |', '   N/A |')
    )
    print(flush=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--unordered', action='store_true', help='whether to use `dict` rather than `OrderedDict`'
    )
    parser.add_argument(
        '--number',
        '-n',
        metavar='N',
        type=int,
        default=10_000,
        help='how many times to execute for each test (default: %(default)d)',
    )
    parser.add_argument(
        '--repeat',
        '-r',
        metavar='N',
        type=int,
        default=5,
        help='how many times to repeat the timer and report the best (default: %(default)d)',
    )

    args = parser.parse_args()
    unordered = args.unordered
    number = args.number
    repeat = args.repeat

    df = pd.DataFrame(
        columns=[
            'Subject',
            'Module',
            'Nodes',
            'Leaves',
            'OpTree (μs)',
            'JAX XLA (μs)',
            'PyTorch (μs)',
            'DM-Tree (μs)',
            'Speedup (J / O)',
            'Speedup (P / O)',
            'Speedup (D / O)',
        ]
    )
    for name, module_factory in (
        ('TinyMLP', tiny_mlp),
        ('AlexNet', models.alexnet),
        ('ResNet18', models.resnet18),
        ('ResNet34', models.resnet34),
        ('ResNet50', models.resnet50),
        ('ResNet101', models.resnet101),
        ('ResNet152', models.resnet152),
        ('ViT-H/14', models.vit_h_14),
        ('Swin-B', models.swin_b),
    ):
        module = module_factory()
        results = benchmark(name, module, number=number, repeat=repeat, unordered=unordered)
        df = pd.concat([df, results], ignore_index=True)

    print('#' * 143)
    print()

    for subject, records in df.groupby('Subject', sort=False):
        print(f'### {subject} ###')
        print(
            records.drop(columns=['Subject'])
            .to_markdown(floatfmt='8.2f', index=False)
            .replace('|  ', '|')
            .replace('|--', '|')
            .replace('nan', 'N/A')
            .replace('N/A    |', 'N/A |')
        )
        print(flush=True)

    df.to_csv('benchmark.csv', index=False)


if __name__ == '__main__':
    main()
