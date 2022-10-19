#!/usr/bin/env python3
#
# Copyright 2022 MetaOPT Team. All Rights Reserved.
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

# pylint: disable=missing-function-docstring

import argparse
import sys
import timeit
from collections import OrderedDict, namedtuple
from itertools import count
from typing import Any, Dict, Iterable, Optional, Tuple

import jax
import torch
import torch.nn as nn
import torch.utils._pytree as torch_utils_pytree
from torchvision import models

import optree


try:
    from termcolor import colored
except ImportError:
    colored = None

if not sys.stdout.isatty() or colored is None:

    def colored(  # pylint: disable=function-redefined,unused-argument
        text: str,
        color: Optional[str] = None,
        on_color: Optional[str] = None,
        attrs: Optional[Iterable[str]] = None,
    ) -> str:
        return text


torch_utils_pytree._register_pytree_node(  # pylint: disable=protected-access
    OrderedDict,
    lambda od: (tuple(od.values()), tuple(od.keys())),
    lambda values, keys: OrderedDict(zip(keys, values)),
)
torch_utils_pytree._register_pytree_node(  # pylint: disable=protected-access
    dict,
    lambda d: tuple(zip(*sorted(d.items())))[::-1] if d else ((), ()),
    lambda values, keys: dict(zip(keys, values)),
)


Tensors = namedtuple('Tensors', ['parameters', 'buffers'])
Containers = namedtuple('Containers', ['parameters', 'buffers'])


def get_none() -> None:
    return None


def tiny_custom_module() -> nn.Module:
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
            parameters=tuple(map(lambda t: t.data, module.parameters(recurse=False))),
            buffers=list(map(lambda t: t.data, module.buffers(recurse=False))),
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
TIE = colored(tie, color='yellow', attrs=('bold',))
XMARK = colored(xmark, color='red', attrs=('bold',))

INIT = """
import jax
import optree
import torch.utils._pytree as torch_utils_pytree

def fn1(x):
    return None

def fn3(x, y, z):
    return None
"""


def cprint(text=''):
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


def benchmark(
    stmt: str,
    init_stmt: str,
    number: int,
    repeat: int = 5,
    globals: Optional[Dict[str, Any]] = None,  # pylint: disable=redefined-builtin
) -> float:
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
    stmts: Dict[str, Tuple[str, str]],
    number: int,
    repeat: int = 5,
    globals: Optional[Dict[str, Any]] = None,  # pylint: disable=redefined-builtin
) -> str:
    times = OrderedDict(
        [
            (lib, benchmark(stmt, init_stmt, number, repeat=repeat, globals=globals))
            for lib, (stmt, init_stmt) in stmts.items()
        ]
    )
    base_time = next(iter(times.values()))
    best_time = min(times.values())
    speedups = {lib: time / base_time for lib, time in times.items()}
    best_speedups = {lib: time / best_time for lib, time in times.items()}
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
    for lib, (stmt, init_stmt) in stmts.items():
        label = labels[lib]
        color = colors[lib]
        attr = attrs[lib]
        label = colored(label + ' ' + lib, color=color, attrs=attr)
        time = colored(f'{times[lib] * 10e6:8.2f}us', attrs=attr)
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


def main():
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

    for name, module_factory in (
        ('TinyCustom', tiny_custom_module),
        ('AlexNet', models.alexnet),
        ('ResNet18', models.resnet18),
        ('ResNet50', models.resnet50),
        ('ResNet101', models.resnet101),
        ('ResNet152', models.resnet152),
        ('VisionTransformerH14', models.vit_h_14),
        ('SwinTransformerB', models.swin_b),
    ):

        module = module_factory()
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
            f'(num_leaves={treespec.num_leaves}, num_nodes={treespec.num_nodes}, treespec={treespec_repr})',
            flush=True,
        )
        y = optree.tree_map(torch.zeros_like, x)
        z = optree.tree_map(lambda t: (t, None), x)  # pylint: disable=invalid-name

        check(x)
        compare(
            subject='Tree Flatten',
            stmts=OrderedDict(
                [
                    ('OpTree(default)', ('optree.tree_leaves(x)', '')),
                    ('OpTree(NoneIsNode)', ('optree.tree_leaves(x, none_is_leaf=False)', '')),
                    ('OpTree(NoneIsLeaf)', ('optree.tree_leaves(x, none_is_leaf=True)', '')),
                    ('JAX XLA', ('jax.tree_util.tree_leaves(x)', '')),
                    ('PyTorch', ('torch_utils_pytree.tree_flatten(x)[0]', '')),
                ]
            ),
            number=number,
            repeat=repeat,
            globals={'x': x},
        )
        compare(
            subject='Tree UnFlatten',
            stmts=OrderedDict(
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
                ]
            ),
            number=number,
            repeat=repeat,
            globals={'x': x},
        )
        compare(
            subject='Tree Copy',
            stmts=OrderedDict(
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
                ]
            ),
            number=number,
            repeat=repeat,
            globals={'x': x},
        )
        compare(
            subject='Tree Map',
            stmts=OrderedDict(
                [
                    ('OpTree(default)', ('optree.tree_map(fn1, x)', '')),
                    ('OpTree(NoneIsNode)', ('optree.tree_map(fn1, x, none_is_leaf=False)', '')),
                    ('OpTree(NoneIsLeaf)', ('optree.tree_map(fn1, x, none_is_leaf=True)', '')),
                    ('JAX XLA', ('jax.tree_util.tree_map(fn1, x)', '')),
                    ('PyTorch', ('torch_utils_pytree.tree_map(fn1, x)', '')),
                ]
            ),
            number=number,
            repeat=repeat,
            globals={'x': x},
        )
        compare(
            subject='Tree Map (nargs)',
            stmts=OrderedDict(
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
                ]
            ),
            number=number,
            repeat=repeat,
            globals={'x': x, 'y': y, 'z': z},
        )


if __name__ == '__main__':
    main()
