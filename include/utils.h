/*
Copyright 2022 MetaOPT Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
================================================================================
*/

// See https://jax.readthedocs.io/en/latest/pytrees.html for the documentation
// about PyTrees.

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#pragma once

#include <pybind11/pybind11.h>

#include <string>

#define CHECK(condition)                                                              \
    if (!(condition))                                                                 \
    throw std::runtime_error(std::string(#condition) + " failed at " __FILE__ + ':' + \
                             std::to_string(__LINE__))

#define DCHECK(condition) CHECK(condition)

namespace pybind11 {
const module_ collections_module = module_::import("collections");
const object OrderedDict = collections_module.attr("OrderedDict");
const object DefaultDict = collections_module.attr("defaultdict");
const object Deque = collections_module.attr("deque");
}  // namespace pybind11
