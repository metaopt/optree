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

// Caution: this code uses exceptions. The exception use is local to the binding
// code and the idiomatic way to emit Python exceptions.

#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <pybind11/pybind11.h>

#include <memory>

#include "include/utils.h"

namespace optree {

enum class PyTreeKind {
    Leaf,         // An opaque leaf node
    None,         // None
    Tuple,        // A tuple
    List,         // A list
    Dict,         // A dict
    NamedTuple,   // A collections.namedtuple
    OrderedDict,  // A collections.OrderedDict
    DefaultDict,  // A collections.defaultdict
    Deque,        // A collections.deque
    Custom,       // A custom type
};

// Registry of custom node types.
class PyTreeTypeRegistry {
 public:
    struct Registration {
        PyTreeKind kind;

        // The following values are populated for custom types.
        // The Python type object, used to identify the type.
        py::object type;
        // A function with signature: object -> (iterable, metadata)
        py::function to_iterable;
        // A function with signature: (metadata, iterable) -> object
        py::function from_iterable;
    };

    // Registers a new custom type. Objects of `type` will be treated as container node types in
    // PyTrees.
    static void Register(const py::object &type,
                         const py::function &to_iterable,
                         const py::function &from_iterable);

    // Finds the custom type registration for `type`. Returns nullptr if none exists.
    template <bool NoneIsLeaf>
    static const Registration *Lookup(const py::handle &type);

 private:
    template <bool NoneIsLeaf>
    struct SingletonHelper {
        static PyTreeTypeRegistry *get();
    };

    template <bool NoneIsLeaf>
    static PyTreeTypeRegistry *Singleton();

    class TypeHash {
     public:
        using is_transparent = void;
        size_t operator()(const py::object &t) const;
        size_t operator()(const py::handle &t) const;
    };
    class TypeEq {
     public:
        using is_transparent = void;
        bool operator()(const py::object &a, const py::object &b) const;
        bool operator()(const py::object &a, const py::handle &b) const;
    };

    absl::flat_hash_map<py::object, std::unique_ptr<Registration>, TypeHash, TypeEq> registrations;
};

}  // namespace optree
