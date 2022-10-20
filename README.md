# OpTree

Optimized PyTree.

------

## Benchmark

We benchmark the performance of:

- tree flatten
- tree unflatten
- tree copy (i.e., `unflatten(flatten(...))`)
- tree map

compared with the following libraries:

- OpTree ([`@44f7410`](https://github.com/metaopt/optree/commit/44f74108e0ff3a392593a2201c1ce33abbd76cdc))
- JAX XLA ([`jax[cpu] == 0.3.23`](https://pypi.org/project/jax/0.3.23))
- PyTorch ([`torch == 1.12.1`](https://pypi.org/project/torch/1.12.1))

All results are reported on a workstation with an AMD Ryzen 9 5950X CPU @ 4.45GHz in an isolated virtual environment with Python 3.10.8.

The test inputs are nested containers (i.e., PyTrees) extracted from `torch.nn.Module` objects. They are:

```python
tiny_custom = nn.Sequential(
    nn.Linear(1, 1, bias=True),
    nn.BatchNorm1d(1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Linear(1, 1, bias=False),
    nn.Sigmoid(),
)
```

and AlexNet, ResNet18, ResNet50, ResNet101, ResNet152, VisionTransformerH14 (ViT-H/14), and SwinTransformerB (Swin-B) from [`torchvsion`](https://github.com/pytorch/vision).
Please refer to [`benchmark.py`](https://github.com/metaopt/optree/blob/HEAD/benchmark.py) for more details.

```text
TinyCustom(num_leaves=16, num_nodes=53, treespec=PyTreeSpec([OrderedDict([('tenso...), buffers=OrderedDict([])))])]))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :    27.18us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :    27.38us -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :    27.09us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:    76.18us -- x2.80   <=  jax.tree_util.tree_leaves(x)
  PyTorch:   671.56us -- x24.71  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :    63.18us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :    63.30us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:   133.89us -- x2.12   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:   248.43us -- x3.93   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
✔ OpTree :    91.51us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :    94.17us -- x1.03   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :    94.23us -- x1.03   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:   216.89us -- x2.37   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch:   940.32us -- x10.28  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
✔ OpTree :    99.84us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :   102.82us -- x1.03   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :   102.21us -- x1.02   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   236.62us -- x2.37   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch:   984.33us -- x9.86   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :   139.50us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :   142.06us -- x1.02   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :   143.19us -- x1.03   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   377.20us -- x2.70   <=  jax.tree_util.tree_map(fn3, x, y, z)

AlexNet(num_leaves=32, num_nodes=188, treespec=PyTreeSpec(OrderedDict([('featur...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
✔ OpTree :    80.37us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :    87.58us -- x1.09   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :    87.06us -- x1.08   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   297.95us -- x3.71   <=  jax.tree_util.tree_leaves(x)
  PyTorch:  2650.24us -- x32.98  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :   245.71us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :   247.18us -- x1.01   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:   545.17us -- x2.22   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:   984.27us -- x4.01   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
✔ OpTree :   332.03us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :   338.27us -- x1.02   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :   337.91us -- x1.02   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:   866.38us -- x2.61   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch:  3654.14us -- x11.01  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
✔ OpTree :   347.63us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :   353.01us -- x1.02   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :   353.95us -- x1.02   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   882.86us -- x2.54   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch:  3703.29us -- x10.65  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :   498.91us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :   499.80us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :   499.17us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  1447.79us -- x2.90   <=  jax.tree_util.tree_map(fn3, x, y, z)

ResNet18(num_leaves=244, num_nodes=698, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
✔ OpTree :   283.47us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :   290.72us -- x1.03   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :   288.77us -- x1.02   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:   928.19us -- x3.27   <=  jax.tree_util.tree_leaves(x)
  PyTorch:  9306.54us -- x32.83  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :   816.54us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :   814.80us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  1669.60us -- x2.04   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  3476.16us -- x4.26   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  1145.87us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  1149.91us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  1145.46us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  2676.30us -- x2.34   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 13040.00us -- x11.38  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  1246.56us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  1241.41us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1236.65us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  2842.26us -- x2.28   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 13090.70us -- x10.50  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :  1754.06us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  1758.19us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  1763.34us -- x1.01   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4581.00us -- x2.61   <=  jax.tree_util.tree_map(fn3, x, y, z)

ResNet50(num_leaves=640, num_nodes=1702, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :   749.18us            <=  optree.tree_leaves(x)                      (None is Node)
✔ OpTree :   735.95us -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
~ OpTree :   737.40us -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  2210.58us -- x2.95   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 21962.28us -- x29.32  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  1967.52us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  1944.32us -- x0.99   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  3909.87us -- x1.99   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch:  8222.80us -- x4.18   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  2782.21us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  2788.91us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  2771.00us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA:  6235.03us -- x2.24   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 30523.06us -- x10.97  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  2997.47us            <=  optree.tree_map(fn1, x)                      (None is Node)
✔ OpTree :  2993.21us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  3004.80us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  6582.85us -- x2.20   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 30674.16us -- x10.23  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  4190.19us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree :  4187.02us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  4200.12us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 10519.68us -- x2.51   <=  jax.tree_util.tree_map(fn3, x, y, z)

ResNet101(num_leaves=1252, num_nodes=3317, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1443.28us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1462.98us -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1414.41us -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4301.47us -- x2.98   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 42706.19us -- x29.59  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  3889.41us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  3885.40us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  7656.81us -- x1.97   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 16058.19us -- x4.13   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  5442.14us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  5422.35us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  5407.01us -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 12184.79us -- x2.24   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 59239.08us -- x10.89  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  5857.83us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  5845.61us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  5819.69us -- x0.99   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 12816.78us -- x2.19   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 59487.90us -- x10.16  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  8145.57us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree :  8138.75us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  8148.81us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 20070.87us -- x2.46   <=  jax.tree_util.tree_map(fn3, x, y, z)

ResNet152(num_leaves=1864, num_nodes=4932, treespec=PyTreeSpec(OrderedDict([('conv1'...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  2180.13us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  2170.51us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  2140.18us -- x0.98   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  6225.77us -- x2.86   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 62329.75us -- x28.59  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  5734.21us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  5715.35us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA: 11297.46us -- x1.97   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 22897.60us -- x3.99   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  7997.82us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  8009.89us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  7960.10us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 17619.27us -- x2.20   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 85951.24us -- x10.75  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  8524.99us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  8522.01us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
✔ OpTree :  8512.22us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 18695.07us -- x2.19   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 86562.20us -- x10.15  <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree : 11886.65us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree : 11928.96us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree : 11902.22us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 29821.16us -- x2.51   <=  jax.tree_util.tree_map(fn3, x, y, z)

VisionTransformerH14(num_leaves=784, num_nodes=3420, treespec=PyTreeSpec(OrderedDict([('conv_p...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1651.80us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1651.72us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1647.63us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4725.58us -- x2.86   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 44551.83us -- x26.97  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
✔ OpTree :  4321.63us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
~ OpTree :  4335.10us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  9133.98us -- x2.11   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 17448.01us -- x4.04   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  6116.50us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
~ OpTree :  6100.51us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
✔ OpTree :  6095.86us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 14116.93us -- x2.31   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 62494.90us -- x10.22  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
✔ OpTree :  6262.65us            <=  optree.tree_map(fn1, x)                      (None is Node)
~ OpTree :  6272.06us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  6272.94us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 14489.26us -- x2.31   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 62355.37us -- x9.96   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
✔ OpTree :  8886.77us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
~ OpTree :  8893.93us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  8932.03us -- x1.01   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 23434.28us -- x2.64   <=  jax.tree_util.tree_map(fn3, x, y, z)

SwinTransformerB(num_leaves=706, num_nodes=2867, treespec=PyTreeSpec(OrderedDict([('featur...]), buffers=OrderedDict([])))])))
### Check ###
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=False)[::-1]) == tree
✔ COPY: optree.tree_unflatten(*optree.tree_flatten(tree, none_is_leaf=True)[::-1]) == tree
✔ FLATTEN (OpTree vs. JAX XLA): optree.tree_leaves(tree, none_is_leaf=False) == jax.tree_util.tree_leaves(tree)
✔ FLATTEN (OpTree vs. PyTorch): optree.tree_leaves(tree, none_is_leaf=True) == torch_utils_pytree.tree_flatten(tree)[0]
✔ TREEMAP (OpTree vs. JAX XLA): optree.tree_map(fn, tree, none_is_leaf=False) == jax.tree_util.tree_map(fn, tree)
✔ TREEMAP (OpTree vs. PyTorch): optree.tree_map(fn, tree, none_is_leaf=True) == torch_utils_pytree.tree_map(fn, tree)

### Tree Flatten ###
~ OpTree :  1369.87us            <=  optree.tree_leaves(x)                      (None is Node)
~ OpTree :  1382.60us -- x1.01   <=  optree.tree_leaves(x, none_is_leaf=False)  (None is Node)
✔ OpTree :  1368.06us -- x1.00   <=  optree.tree_leaves(x, none_is_leaf=True)   (None is Leaf)
  JAX XLA:  4066.15us -- x2.97   <=  jax.tree_util.tree_leaves(x)
  PyTorch: 37490.24us -- x27.37  <=  torch_utils_pytree.tree_flatten(x)[0]

### Tree UnFlatten ###
~ OpTree :  3707.21us            <=  optree.tree_unflatten(spec, flat)  (None is Node)
✔ OpTree :  3693.14us -- x1.00   <=  optree.tree_unflatten(spec, flat)  (None is Leaf)
  JAX XLA:  7749.16us -- x2.09   <=  jax.tree_util.tree_unflatten(spec, flat)
  PyTorch: 14828.41us -- x4.00   <=  torch_utils_pytree.tree_unflatten(flat, spec)

### Tree Copy ###
~ OpTree :  5154.76us            <=  optree.tree_unflatten(*optree.tree_flatten(x)[::-1])                      (None is Node)
✔ OpTree :  5127.40us -- x0.99   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=False)[::-1])  (None is Node)
~ OpTree :  5149.86us -- x1.00   <=  optree.tree_unflatten(*optree.tree_flatten(x, none_is_leaf=True)[::-1])   (None is Leaf)
  JAX XLA: 12031.65us -- x2.33   <=  jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(x)[::-1])
  PyTorch: 52536.88us -- x10.19  <=  torch_utils_pytree.tree_unflatten(*torch_utils_pytree.tree_flatten(x))

### Tree Map ###
~ OpTree :  5359.65us            <=  optree.tree_map(fn1, x)                      (None is Node)
✔ OpTree :  5334.72us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=False)  (None is Node)
~ OpTree :  5335.08us -- x1.00   <=  optree.tree_map(fn1, x, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 12371.49us -- x2.31   <=  jax.tree_util.tree_map(fn1, x)
  PyTorch: 52645.16us -- x9.82   <=  torch_utils_pytree.tree_map(fn1, x)

### Tree Map (nargs) ###
~ OpTree :  7535.16us            <=  optree.tree_map(fn3, x, y, z)                      (None is Node)
✔ OpTree :  7520.10us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=False)  (None is Node)
~ OpTree :  7531.14us -- x1.00   <=  optree.tree_map(fn3, x, y, z, none_is_leaf=True)   (None is Leaf)
  JAX XLA: 19884.10us -- x2.64   <=  jax.tree_util.tree_map(fn3, x, y, z)
```

------

## License

OpTree is released under the Apache License 2.0.

OpTree is heavily based on JAX's implementation of the PyTree utility, with deep refactoring and several improvements. The original licenses can be found at [JAX's Apache License 2.0](https://github.com/google/jax/blob/HEAD/LICENSE) and [Tensorflow's Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/HEAD/LICENSE) .
