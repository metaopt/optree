/*
Copyright 2022-2024 MetaOPT Team. All Rights Reserved.

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

#pragma once

#include <pybind11/pybind11.h>

#include <memory>         // std::unique_ptr
#include <string>         // std::string
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::pair

#include "include/utils.h"

namespace optree {

namespace py = pybind11;
using size_t = py::size_t;
using ssize_t = py::ssize_t;

enum class PyTreeKind {
    Custom = 0,      // A custom type
    Leaf,            // An opaque leaf node
    None,            // None
    Tuple,           // A tuple
    List,            // A list
    Dict,            // A dict
    NamedTuple,      // A collections.namedtuple
    OrderedDict,     // A collections.OrderedDict
    DefaultDict,     // A collections.defaultdict
    Deque,           // A collections.deque
    StructSequence,  // A PyStructSequence
};

constexpr PyTreeKind kCustom = PyTreeKind::Custom;
constexpr PyTreeKind kLeaf = PyTreeKind::Leaf;
constexpr PyTreeKind kNone = PyTreeKind::None;
constexpr PyTreeKind kTuple = PyTreeKind::Tuple;
constexpr PyTreeKind kList = PyTreeKind::List;
constexpr PyTreeKind kDict = PyTreeKind::Dict;
constexpr PyTreeKind kNamedTuple = PyTreeKind::NamedTuple;
constexpr PyTreeKind kOrderedDict = PyTreeKind::OrderedDict;
constexpr PyTreeKind kDefaultDict = PyTreeKind::DefaultDict;
constexpr PyTreeKind kDeque = PyTreeKind::Deque;
constexpr PyTreeKind kStructSequence = PyTreeKind::StructSequence;

// Registry of custom node types.
class PyTreeTypeRegistry {
 public:
    struct Registration {
        PyTreeKind kind = PyTreeKind::Custom;

        // The following values are populated for custom types.
        // The Python type object, used to identify the type.
        py::object type{};
        // A function with signature: object -> (iterable, metadata, entries)
        py::function flatten_func{};
        // A function with signature: (metadata, iterable) -> object
        py::function unflatten_func{};
    };

    // Registers a new custom type. Objects of `cls` will be treated as container node types in
    // PyTrees.
    static void Register(const py::object &cls,
                         const py::function &flatten_func,
                         const py::function &unflatten_func,
                         const std::string &registry_namespace = "");

    // Finds the custom type registration for `type`. Returns nullptr if none exists.
    template <bool NoneIsLeaf>
    static const Registration *Lookup(const py::object &cls, const std::string &registry_namespace);

 private:
    template <bool NoneIsLeaf>
    static PyTreeTypeRegistry *Singleton();

    template <bool NoneIsLeaf>
    static void RegisterImpl(const py::object &cls,
                             const py::function &flatten_func,
                             const py::function &unflatten_func,
                             const std::string &registry_namespace);

    std::unordered_map<py::object, std::unique_ptr<Registration>, TypeHash, TypeEq>
        m_registrations{};
    std::unordered_map<std::pair<std::string, py::object>,
                       std::unique_ptr<Registration>,
                       NamedTypeHash,
                       NamedTypeEq>
        m_named_registrations{};
};

}  // namespace optree
