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

#include <memory>       // std::make_shared
#include <optional>     // std::optional
#include <sstream>      // std::ostringstream
#include <string>       // std::string
#include <tuple>        // std::tuple
#include <type_traits>  // std::remove_const_t
#include <utility>      // std::move, std::make_pair

#include <Python.h>

#include "optree/optree.h"

namespace optree {

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry &PyTreeTypeRegistry::GetSingleton() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<PyTreeTypeRegistry> storage;
    return storage
        .call_once_and_store_result([]() -> PyTreeTypeRegistry {
            PyTreeTypeRegistry registry{};
            registry.m_none_is_leaf = NoneIsLeaf;
            return registry;
        })
        .get_stored();
}

template PyTreeTypeRegistry &PyTreeTypeRegistry::GetSingleton<NONE_IS_NODE>();
template PyTreeTypeRegistry &PyTreeTypeRegistry::GetSingleton<NONE_IS_LEAF>();

std::tuple<PyTreeTypeRegistry::RegistrationsMap &,
           PyTreeTypeRegistry::NamedRegistrationsMap &,
           PyTreeTypeRegistry::BuiltinsTypesSet &>
PyTreeTypeRegistry::GetRegistrationsForCurrentPyInterpreterLocked() const {
    const auto interpid = GetCurrentPyInterpreterID();

    EXPECT_NE(m_registrations.find(interpid),
              m_registrations.end(),
              "Interpreter ID " + std::to_string(interpid) + " not found in `m_registrations`.");
    EXPECT_NE(
        m_named_registrations.find(interpid),
        m_named_registrations.end(),
        "Interpreter ID " + std::to_string(interpid) + " not found in `m_named_registrations`.");
    EXPECT_NE(sm_builtins_types.find(interpid),
              sm_builtins_types.end(),
              "Interpreter ID " + std::to_string(interpid) + " not found in `sm_builtins_types`.");

    // NOLINTBEGIN[cppcoreguidelines-pro-type-const-cast]
    return {const_cast<RegistrationsMap &>(m_registrations.at(interpid)),
            const_cast<NamedRegistrationsMap &>(m_named_registrations.at(interpid)),
            const_cast<BuiltinsTypesSet &>(sm_builtins_types.at(interpid))};
    // NOLINTEND[cppcoreguidelines-pro-type-const-cast]
}

void PyTreeTypeRegistry::Init() {
    const scoped_write_lock lock{sm_mutex};

    const auto interpid = GetCurrentPyInterpreterID();

    EXPECT_EQ(m_registrations.find(interpid),
              m_registrations.end(),
              "Interpreter ID " + std::to_string(interpid) +
                  " is already initialized in `m_registrations`.");
    EXPECT_EQ(m_named_registrations.find(interpid),
              m_named_registrations.end(),
              "Interpreter ID " + std::to_string(interpid) +
                  " is already initialized in `m_named_registrations`.");
    if (!m_none_is_leaf) [[likely]] {
        EXPECT_EQ(sm_builtins_types.find(interpid),
                  sm_builtins_types.end(),
                  "Interpreter ID " + std::to_string(interpid) +
                      " is already initialized in `sm_builtins_types`.");
    } else {
        EXPECT_NE(sm_builtins_types.find(interpid),
                  sm_builtins_types.end(),
                  "Interpreter ID " + std::to_string(interpid) +
                      " is not initialized in `sm_builtins_types`.");
    }

    auto &registrations = m_registrations.try_emplace(interpid).first->second;
    auto &named_registrations = m_named_registrations.try_emplace(interpid).first->second;
    auto &builtins_types = sm_builtins_types.try_emplace(interpid).first->second;

    (void)named_registrations;  // silence unused variable warning

    const auto add_builtin_type =
        [&registrations, &builtins_types](const py::object &cls, const PyTreeKind &kind) -> void {
        auto registration = std::make_shared<std::remove_const_t<RegistrationPtr::element_type>>();
        registration->kind = kind;
        registration->type = py::reinterpret_borrow<py::object>(cls);
        EXPECT_TRUE(
            registrations.emplace(cls, std::move(registration)).second,
            "PyTree type " + PyRepr(cls) + " is already registered in the global namespace.");
        if (builtins_types.emplace(cls).second) [[likely]] {
            cls.inc_ref();
        }
    };
    if (!m_none_is_leaf) [[likely]] {
        add_builtin_type(PyNoneTypeObject, PyTreeKind::None);
    }
    add_builtin_type(PyTupleTypeObject, PyTreeKind::Tuple);
    add_builtin_type(PyListTypeObject, PyTreeKind::List);
    add_builtin_type(PyDictTypeObject, PyTreeKind::Dict);
    add_builtin_type(PyOrderedDictTypeObject, PyTreeKind::OrderedDict);
    add_builtin_type(PyDefaultDictTypeObject, PyTreeKind::DefaultDict);
    add_builtin_type(PyDequeTypeObject, PyTreeKind::Deque);
}

ssize_t PyTreeTypeRegistry::Size(const std::optional<std::string> &registry_namespace) const {
    const scoped_read_lock lock{sm_mutex};

    const auto &[registrations, named_registrations, builtins_types] =
        GetRegistrationsForCurrentPyInterpreterLocked();

    (void)builtins_types;  // silence unused variable warning

    ssize_t count = py::ssize_t_cast(registrations.size());
    for (const auto &[named_type, _] : named_registrations) {
        if (!registry_namespace || named_type.first == *registry_namespace) [[likely]] {
            ++count;
        }
    }
    return count;
}

template <bool NoneIsLeaf>
/*static*/ void PyTreeTypeRegistry::RegisterImpl(const py::object &cls,
                                                 const py::function &flatten_func,
                                                 const py::function &unflatten_func,
                                                 const py::object &path_entry_type,
                                                 const std::string &registry_namespace) {
    const auto &registry = GetSingleton<NoneIsLeaf>();

    auto [registrations, named_registrations, builtins_types] =
        registry.GetRegistrationsForCurrentPyInterpreterLocked();

    if (builtins_types.find(cls) != builtins_types.end()) [[unlikely]] {
        throw py::value_error("PyTree type " + PyRepr(cls) +
                              " is a built-in type and cannot be re-registered.");
    }

    auto registration = std::make_shared<std::remove_const_t<RegistrationPtr::element_type>>();
    registration->kind = PyTreeKind::Custom;
    registration->type = py::reinterpret_borrow<py::object>(cls);
    registration->flatten_func = py::reinterpret_borrow<py::function>(flatten_func);
    registration->unflatten_func = py::reinterpret_borrow<py::function>(unflatten_func);
    registration->path_entry_type = py::reinterpret_borrow<py::object>(path_entry_type);
    if (registry_namespace.empty()) [[unlikely]] {
        if (!registrations.emplace(cls, std::move(registration)).second) [[unlikely]] {
            throw py::value_error("PyTree type " + PyRepr(cls) +
                                  " is already registered in the global namespace.");
        }
        if (IsStructSequenceClass(cls)) [[unlikely]] {
            PyErr_WarnEx(PyExc_UserWarning,
                         ("PyTree type " + PyRepr(cls) +
                          " is a class of `PyStructSequence`, "
                          "which is already registered in the global namespace. "
                          "Override it with custom flatten/unflatten functions.")
                             .c_str(),
                         /*stack_level=*/2);
        } else if (IsNamedTupleClass(cls)) [[unlikely]] {
            PyErr_WarnEx(PyExc_UserWarning,
                         ("PyTree type " + PyRepr(cls) +
                          " is a subclass of `collections.namedtuple`, "
                          "which is already registered in the global namespace. "
                          "Override it with custom flatten/unflatten functions.")
                             .c_str(),
                         /*stack_level=*/2);
        }
    } else [[likely]] {
        if (!named_registrations
                 .emplace(std::make_pair(registry_namespace, cls), std::move(registration))
                 .second) [[unlikely]] {
            std::ostringstream oss{};
            oss << "PyTree type " << PyRepr(cls) << " is already registered in namespace "
                << PyRepr(registry_namespace) << ".";
            throw py::value_error(oss.str());
        }
        if (IsStructSequenceClass(cls)) [[unlikely]] {
            std::ostringstream oss{};
            oss << "PyTree type " << PyRepr(cls)
                << " is a class of `PyStructSequence`, "
                   "which is already registered in the global namespace. "
                   "Override it with custom flatten/unflatten functions in namespace "
                << PyRepr(registry_namespace) << ".";
            PyErr_WarnEx(PyExc_UserWarning,
                         oss.str().c_str(),
                         /*stack_level=*/2);
        } else if (IsNamedTupleClass(cls)) [[unlikely]] {
            std::ostringstream oss{};
            oss << "PyTree type " << PyRepr(cls)
                << " is a subclass of `collections.namedtuple`, "
                   "which is already registered in the global namespace. "
                   "Override it with custom flatten/unflatten functions in namespace "
                << PyRepr(registry_namespace) << ".";
            PyErr_WarnEx(PyExc_UserWarning,
                         oss.str().c_str(),
                         /*stack_level=*/2);
        }
    }
}

/*static*/ void PyTreeTypeRegistry::Register(const py::object &cls,
                                             const py::function &flatten_func,
                                             const py::function &unflatten_func,
                                             const py::object &path_entry_type,
                                             const std::string &registry_namespace) {
    const scoped_write_lock lock{sm_mutex};

    RegisterImpl<NONE_IS_NODE>(cls,
                               flatten_func,
                               unflatten_func,
                               path_entry_type,
                               registry_namespace);
    RegisterImpl<NONE_IS_LEAF>(cls,
                               flatten_func,
                               unflatten_func,
                               path_entry_type,
                               registry_namespace);
    cls.inc_ref();
    flatten_func.inc_ref();
    unflatten_func.inc_ref();
    path_entry_type.inc_ref();
}

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry::RegistrationPtr PyTreeTypeRegistry::UnregisterImpl(
    const py::object &cls,
    const std::string &registry_namespace) {
    const auto &registry = GetSingleton<NoneIsLeaf>();

    auto [registrations, named_registrations, builtins_types] =
        registry.GetRegistrationsForCurrentPyInterpreterLocked();

    if (builtins_types.find(cls) != builtins_types.end()) [[unlikely]] {
        throw py::value_error("PyTree type " + PyRepr(cls) +
                              " is a built-in type and cannot be unregistered.");
    }

    if (registry_namespace.empty()) [[unlikely]] {
        const auto it = registrations.find(cls);
        if (it == registrations.end()) [[unlikely]] {
            std::ostringstream oss{};
            oss << "PyTree type " << PyRepr(cls) << " ";
            if (IsStructSequenceClass(cls)) [[unlikely]] {
                oss << "is a class of `PyStructSequence`, "
                    << "which is not explicitly registered in the global namespace.";
            } else if (IsNamedTupleClass(cls)) [[unlikely]] {
                oss << "is a subclass of `collections.namedtuple`, "
                    << "which is not explicitly registered in the global namespace.";
            } else [[likely]] {
                oss << "is not registered in the global namespace.";
            }
            throw py::value_error(oss.str());
        }
        RegistrationPtr registration = it->second;
        registrations.erase(it);
        return registration;
    } else [[likely]] {
        const auto named_it = named_registrations.find(std::make_pair(registry_namespace, cls));
        if (named_it == named_registrations.end()) [[unlikely]] {
            std::ostringstream oss{};
            oss << "PyTree type " << PyRepr(cls) << " ";
            if (IsStructSequenceClass(cls)) [[unlikely]] {
                oss << "is a class of `PyStructSequence`, "
                    << "which is not explicitly registered ";
            } else if (IsNamedTupleClass(cls)) [[unlikely]] {
                oss << "is a subclass of `collections.namedtuple`, "
                    << "which is not explicitly registered ";
            } else [[likely]] {
                oss << "is not registered ";
            }
            oss << "in namespace " << PyRepr(registry_namespace) << ".";
            throw py::value_error(oss.str());
        }
        RegistrationPtr registration = named_it->second;
        named_registrations.erase(named_it);
        return registration;
    }
}

/*static*/ void PyTreeTypeRegistry::Unregister(const py::object &cls,
                                               const std::string &registry_namespace) {
    const scoped_write_lock lock{sm_mutex};

    const auto registration1 = UnregisterImpl<NONE_IS_NODE>(cls, registry_namespace);
    const auto registration2 = UnregisterImpl<NONE_IS_LEAF>(cls, registry_namespace);
    EXPECT_TRUE(registration1->type.is(registration2->type));
    EXPECT_TRUE(registration1->flatten_func.is(registration2->flatten_func));
    EXPECT_TRUE(registration1->unflatten_func.is(registration2->unflatten_func));
    EXPECT_TRUE(registration1->path_entry_type.is(registration2->path_entry_type));
    registration1->type.dec_ref();
    registration1->flatten_func.dec_ref();
    registration1->unflatten_func.dec_ref();
    registration1->path_entry_type.dec_ref();
}

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry::RegistrationPtr PyTreeTypeRegistry::Lookup(
    const py::object &cls,
    const std::string &registry_namespace) {
    const scoped_read_lock lock{sm_mutex};

    const auto &registry = GetSingleton<NoneIsLeaf>();

    const auto &[registrations, named_registrations, _] =
        registry.GetRegistrationsForCurrentPyInterpreterLocked();

    if (!registry_namespace.empty()) [[unlikely]] {
        const auto named_it = named_registrations.find(std::make_pair(registry_namespace, cls));
        if (named_it != named_registrations.end()) [[likely]] {
            return named_it->second;
        }
    }
    const auto it = registrations.find(cls);
    return it != registrations.end() ? it->second : nullptr;
}

template PyTreeTypeRegistry::RegistrationPtr PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(
    const py::object &,
    const std::string &);
template PyTreeTypeRegistry::RegistrationPtr PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(
    const py::object &,
    const std::string &);

template <bool NoneIsLeaf>
/*static*/ PyTreeKind PyTreeTypeRegistry::GetKind(
    const py::handle &handle,
    PyTreeTypeRegistry::RegistrationPtr &custom,  // NOLINT[runtime/references]
    const std::string &registry_namespace) {
    const RegistrationPtr registration =
        Lookup<NoneIsLeaf>(py::type::of(handle), registry_namespace);
    if (registration) [[likely]] {
        if (registration->kind == PyTreeKind::Custom) [[unlikely]] {
            custom = registration;
        } else [[likely]] {
            custom = nullptr;
        }
        return registration->kind;
    }
    custom = nullptr;
    if (IsStructSequenceInstance(handle)) [[unlikely]] {
        return PyTreeKind::StructSequence;
    }
    if (IsNamedTupleInstance(handle)) [[unlikely]] {
        return PyTreeKind::NamedTuple;
    }
    return PyTreeKind::Leaf;
}

template PyTreeKind PyTreeTypeRegistry::GetKind<NONE_IS_NODE>(
    const py::handle &,
    PyTreeTypeRegistry::RegistrationPtr &,  // NOLINT[runtime/references]
    const std::string &);
template PyTreeKind PyTreeTypeRegistry::GetKind<NONE_IS_LEAF>(
    const py::handle &,
    PyTreeTypeRegistry::RegistrationPtr &,  // NOLINT[runtime/references]
    const std::string &);

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ void PyTreeTypeRegistry::Clear() {
    const scoped_write_lock lock{sm_mutex};

    const auto interpid = GetCurrentPyInterpreterID();

    auto &registry1 = GetSingleton<NONE_IS_NODE>();
    auto &registry2 = GetSingleton<NONE_IS_LEAF>();

    auto [registrations1, named_registrations1, builtins_types] =
        registry1.GetRegistrationsForCurrentPyInterpreterLocked();
    auto [registrations2, named_registrations2, builtins_types_] =
        registry2.GetRegistrationsForCurrentPyInterpreterLocked();

    EXPECT_LE(builtins_types.size(), registrations1.size());
    EXPECT_EQ(registrations1.size(), registrations2.size() + 1);
    EXPECT_EQ(named_registrations1.size(), named_registrations2.size());
    EXPECT_EQ(&builtins_types, &builtins_types_);

#if defined(Py_DEBUG)
    for (const auto &cls : builtins_types) {
        EXPECT_NE(registrations1.find(cls), registrations1.end());
    }
    for (const auto &[cls2, registration2] : registrations2) {
        const auto it1 = registrations1.find(cls2);
        EXPECT_NE(it1, registrations1.end());

        const auto &registration1 = it1->second;
        EXPECT_TRUE(registration1->type.is(registration2->type));
        EXPECT_TRUE(registration1->flatten_func.is(registration2->flatten_func));
        EXPECT_TRUE(registration1->unflatten_func.is(registration2->unflatten_func));
        EXPECT_TRUE(registration1->path_entry_type.is(registration2->path_entry_type));
    }
    for (const auto &[named_cls2, registration2] : named_registrations2) {
        const auto it1 = named_registrations1.find(named_cls2);
        EXPECT_NE(it1, named_registrations1.end());

        const auto &registration1 = it1->second;
        EXPECT_TRUE(registration1->type.is(registration2->type));
        EXPECT_TRUE(registration1->flatten_func.is(registration2->flatten_func));
        EXPECT_TRUE(registration1->unflatten_func.is(registration2->unflatten_func));
        EXPECT_TRUE(registration1->path_entry_type.is(registration2->path_entry_type));
    }
#endif

    EXPECT_EQ(py::ssize_t_cast(sm_builtins_types.size()), sm_num_interpreters_alive);

    for (const auto &[_, registration] : registrations1) {
        registration->type.dec_ref();
        registration->flatten_func.dec_ref();
        registration->unflatten_func.dec_ref();
        registration->path_entry_type.dec_ref();
    }
    for (const auto &[_, registration] : named_registrations1) {
        registration->type.dec_ref();
        registration->flatten_func.dec_ref();
        registration->unflatten_func.dec_ref();
        registration->path_entry_type.dec_ref();
    }

    builtins_types.clear();
    registrations1.clear();
    named_registrations1.clear();
    registrations2.clear();
    named_registrations2.clear();

    sm_builtins_types.erase(interpid);
    registry1.m_registrations.erase(interpid);
    registry1.m_named_registrations.erase(interpid);
    registry2.m_registrations.erase(interpid);
    registry2.m_named_registrations.erase(interpid);

    --sm_num_interpreters_alive;
}

}  // namespace optree
