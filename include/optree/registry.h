/*
Copyright 2022-2025 MetaOPT Team. All Rights Reserved.

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

#include <cstdint>        // std::uint8_t
#include <memory>         // std::shared_ptr
#include <optional>       // std::optional, std::nullopt
#include <string>         // std::string
#include <tuple>          // std::tuple
#include <unordered_map>  // std::unordered_map
#include <unordered_set>  // std::unordered_set
#include <utility>        // std::pair

#include <pybind11/pybind11.h>

#include "optree/exceptions.h"
#include "optree/hashing.h"
#include "optree/pymacros.h"
#include "optree/synchronization.h"

namespace optree {

namespace py = pybind11;
using size_t = py::size_t;
using ssize_t = py::ssize_t;

constexpr bool NONE_IS_LEAF = true;
constexpr bool NONE_IS_NODE = false;

enum class PyTreeKind : std::uint8_t {
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
    NumKinds,        // Number of kinds (placed at the end)
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
constexpr PyTreeKind kNumPyTreeKinds = PyTreeKind::NumKinds;

// Registry of custom node types.
class PyTreeTypeRegistry {
public:
    PyTreeTypeRegistry() = default;
    ~PyTreeTypeRegistry() = default;

    PyTreeTypeRegistry(const PyTreeTypeRegistry &) = delete;
    PyTreeTypeRegistry &operator=(const PyTreeTypeRegistry &) = delete;
    PyTreeTypeRegistry(PyTreeTypeRegistry &&) = default;
    PyTreeTypeRegistry &operator=(PyTreeTypeRegistry &&) = default;

    struct Registration {
        PyTreeKind kind = PyTreeKind::Custom;

        // NOTE: the registration should use `py::object` instead of `py::handle`
        // to hold extra references to the Python objects. Otherwise, the Python
        // objects may be destroyed if the registration is unregistered from the
        // Python side while the shared pointer still holds the references.

        // The following values are populated for custom types.
        // The Python type object, used to identify the type.
        py::object type{};
        // A function with signature: object -> (iterable, metadata, entries)
        py::function flatten_func{};
        // A function with signature: (metadata, iterable) -> object
        py::function unflatten_func{};
        // The Python type object for the path entry class.
        py::object path_entry_type{};
    };

    using RegistrationPtr = std::shared_ptr<const Registration>;

    // Get the number of registered types.
    [[nodiscard]] ssize_t Size(
        const std::optional<std::string> &registry_namespace = std::nullopt) const;

    // Register a new custom type. Objects of type `cls` will be treated as container node types in
    // PyTrees.
    static void Register(const py::object &cls,
                         const py::function &flatten_func,
                         const py::function &unflatten_func,
                         const py::object &path_entry_type,
                         const std::string &registry_namespace = "");

    // Unregister a previously registered custom type.
    static void Unregister(const py::object &cls, const std::string &registry_namespace = "");

    // Find the custom type registration for `type`. Return nullptr if none exists.
    template <bool NoneIsLeaf>
    [[nodiscard]] static RegistrationPtr Lookup(const py::object &cls,
                                                const std::string &registry_namespace);

    // Compute the node kind of a given Python object.
    template <bool NoneIsLeaf>
    [[nodiscard]] static PyTreeKind GetKind(const py::handle &handle,
                                            RegistrationPtr &custom,  // NOLINT[runtime/references]
                                            const std::string &registry_namespace);

    // Get the number of registered types.
    [[nodiscard]] static inline Py_ALWAYS_INLINE ssize_t GetRegistrySize(
        const std::optional<std::string> &registry_namespace = std::nullopt) {
        const ssize_t count = GetSingleton<NONE_IS_NODE>().Size(registry_namespace);
        EXPECT_EQ(count,
                  GetSingleton<NONE_IS_LEAF>().Size(registry_namespace) + 1,
                  "The number of registered types in the two registries should match "
                  "up to the extra None type in the NoneIsNode registry.");
        return count;
    }

    // Get the number of alive interpreters that have seen the registry.
    [[nodiscard]] static inline Py_ALWAYS_INLINE ssize_t GetNumInterpretersAlive() {
        const scoped_read_lock lock{sm_mutex};
        return sm_num_interpreters_seen;
    }

    // Get the number of interpreters that have seen the registry.
    [[nodiscard]] static inline Py_ALWAYS_INLINE ssize_t GetNumInterpretersSeen() {
        const scoped_read_lock lock{sm_mutex};
        EXPECT_EQ(py::ssize_t_cast(sm_builtins_types.size()),
                  sm_num_interpreters_alive,
                  "The number of alive interpreters should match the size of the "
                  "interpreter-scoped registered types map.");
        return sm_num_interpreters_alive;
    }

    // Get the IDs of alive interpreters that have seen the registry.
    [[nodiscard]] static inline Py_ALWAYS_INLINE std::unordered_set<interpid_t>
    GetAliveInterpreterIDs() {
        const scoped_read_lock lock{sm_mutex};
        EXPECT_EQ(py::ssize_t_cast(sm_builtins_types.size()),
                  sm_num_interpreters_alive,
                  "The number of alive interpreters should match the size of the "
                  "interpreter-scoped registered types map.");
        std::unordered_set<interpid_t> interpids;
        for (const auto &[interpid, _] : sm_builtins_types) {
            interpids.insert(interpid);
        }
        return interpids;
    }

    friend void BuildModule(py::module_ &mod);  // NOLINT[runtime/references]

private:
    template <bool NoneIsLeaf>
    [[nodiscard]] static PyTreeTypeRegistry &GetSingleton();

    template <bool NoneIsLeaf>
    static void RegisterImpl(const py::object &cls,
                             const py::function &flatten_func,
                             const py::function &unflatten_func,
                             const py::object &path_entry_type,
                             const std::string &registry_namespace);

    template <bool NoneIsLeaf>
    [[nodiscard]] static RegistrationPtr UnregisterImpl(const py::object &cls,
                                                        const std::string &registry_namespace);

    // Initialize the registry for the current interpreter.
    void Init();

    // Clear the registry on cleanup for the current interpreter.
    static void Clear();

    using RegistrationsMap = std::unordered_map<py::handle, RegistrationPtr>;
    using NamedRegistrationsMap =
        std::unordered_map<std::pair<std::string, py::handle>, RegistrationPtr>;
    using BuiltinsTypesSet = std::unordered_set<py::handle>;

    // Get the registrations for the current Python interpreter.
    [[nodiscard]] inline Py_ALWAYS_INLINE std::
        tuple<RegistrationsMap &, NamedRegistrationsMap &, BuiltinsTypesSet &>
        GetRegistrationsForCurrentPyInterpreterLocked() const;

    bool m_none_is_leaf = false;
    std::unordered_map<interpid_t, RegistrationsMap> m_registrations{};
    std::unordered_map<interpid_t, NamedRegistrationsMap> m_named_registrations{};

    static inline std::unordered_map<interpid_t, BuiltinsTypesSet> sm_builtins_types{};
    static inline read_write_mutex sm_mutex{};
    static inline ssize_t sm_num_interpreters_alive = 0;
    static inline ssize_t sm_num_interpreters_seen = 0;
};

}  // namespace optree
