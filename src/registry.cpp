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
#include <sstream>      // std::ostringstream
#include <string>       // std::string
#include <type_traits>  // std::remove_const_t
#include <utility>      // std::move, std::make_pair

#include <Python.h>

#include "optree/optree.h"

namespace optree {

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::Singleton() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<PyTreeTypeRegistry> storage;
    return &(storage
                 .call_once_and_store_result([]() -> PyTreeTypeRegistry {
                     PyTreeTypeRegistry registry{};

                     const auto add_builtin_type = [&registry](const py::object& cls,
                                                               const PyTreeKind& kind) -> void {
                         auto registration =
                             std::make_shared<std::remove_const_t<RegistrationPtr::element_type>>();
                         registration->kind = kind;
                         registration->type = py::reinterpret_borrow<py::object>(cls);
                         EXPECT_TRUE(
                             registry.m_registrations.emplace(cls, std::move(registration)).second,
                             "PyTree type " + PyRepr(cls) +
                                 " is already registered in the global namespace.");
                         if (sm_builtins_types.emplace(cls).second) [[likely]] {
                             cls.inc_ref();
                         }
                     };
                     if constexpr (!NoneIsLeaf) {
                         add_builtin_type(PyNoneTypeObject, PyTreeKind::None);
                     }
                     add_builtin_type(PyTupleTypeObject, PyTreeKind::Tuple);
                     add_builtin_type(PyListTypeObject, PyTreeKind::List);
                     add_builtin_type(PyDictTypeObject, PyTreeKind::Dict);
                     add_builtin_type(PyOrderedDictTypeObject, PyTreeKind::OrderedDict);
                     add_builtin_type(PyDefaultDictTypeObject, PyTreeKind::DefaultDict);
                     add_builtin_type(PyDequeTypeObject, PyTreeKind::Deque);
                     return registry;
                 })
                 .get_stored());
}

template PyTreeTypeRegistry* PyTreeTypeRegistry::Singleton<NONE_IS_NODE>();
template PyTreeTypeRegistry* PyTreeTypeRegistry::Singleton<NONE_IS_LEAF>();

template <bool NoneIsLeaf>
/*static*/ void PyTreeTypeRegistry::RegisterImpl(const py::object& cls,
                                                 const py::function& flatten_func,
                                                 const py::function& unflatten_func,
                                                 const py::object& path_entry_type,
                                                 const std::string& registry_namespace) {
    if (sm_builtins_types.find(cls) != sm_builtins_types.end()) [[unlikely]] {
        throw py::value_error("PyTree type " + PyRepr(cls) +
                              " is a built-in type and cannot be re-registered.");
    }

    PyTreeTypeRegistry* const registry = Singleton<NoneIsLeaf>();
    auto registration = std::make_shared<std::remove_const_t<RegistrationPtr::element_type>>();
    registration->kind = PyTreeKind::Custom;
    registration->type = py::reinterpret_borrow<py::object>(cls);
    registration->flatten_func = py::reinterpret_borrow<py::function>(flatten_func);
    registration->unflatten_func = py::reinterpret_borrow<py::function>(unflatten_func);
    registration->path_entry_type = py::reinterpret_borrow<py::object>(path_entry_type);
    if (registry_namespace.empty()) [[unlikely]] {
        if (!registry->m_registrations.emplace(cls, std::move(registration)).second) [[unlikely]] {
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
        if (!registry->m_named_registrations
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

/*static*/ void PyTreeTypeRegistry::Register(const py::object& cls,
                                             const py::function& flatten_func,
                                             const py::function& unflatten_func,
                                             const py::object& path_entry_type,
                                             const std::string& registry_namespace) {
    const scoped_write_lock_guard lock{sm_mutex};

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
    const py::object& cls,
    const std::string& registry_namespace) {
    if (sm_builtins_types.find(cls) != sm_builtins_types.end()) [[unlikely]] {
        throw py::value_error("PyTree type " + PyRepr(cls) +
                              " is a built-in type and cannot be unregistered.");
    }

    PyTreeTypeRegistry* const registry = Singleton<NoneIsLeaf>();
    if (registry_namespace.empty()) [[unlikely]] {
        const auto it = registry->m_registrations.find(cls);
        if (it == registry->m_registrations.end()) [[unlikely]] {
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
        registry->m_registrations.erase(it);
        return registration;
    } else [[likely]] {
        const auto named_it =
            registry->m_named_registrations.find(std::make_pair(registry_namespace, cls));
        if (named_it == registry->m_named_registrations.end()) [[unlikely]] {
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
        registry->m_named_registrations.erase(named_it);
        return registration;
    }
}

/*static*/ void PyTreeTypeRegistry::Unregister(const py::object& cls,
                                               const std::string& registry_namespace) {
    const scoped_write_lock_guard lock{sm_mutex};

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
    const py::object& cls,
    const std::string& registry_namespace) {
    const scoped_read_lock_guard lock{sm_mutex};

    PyTreeTypeRegistry* const registry = Singleton<NoneIsLeaf>();
    if (!registry_namespace.empty()) [[unlikely]] {
        const auto named_it =
            registry->m_named_registrations.find(std::make_pair(registry_namespace, cls));
        if (named_it != registry->m_named_registrations.end()) [[likely]] {
            return named_it->second;
        }
    }
    const auto it = registry->m_registrations.find(cls);
    return it != registry->m_registrations.end() ? it->second : nullptr;
}

template PyTreeTypeRegistry::RegistrationPtr PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(
    const py::object&,
    const std::string&);
template PyTreeTypeRegistry::RegistrationPtr PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(
    const py::object&,
    const std::string&);

template <bool NoneIsLeaf>
/*static*/ PyTreeKind PyTreeTypeRegistry::GetKind(
    const py::handle& handle,
    PyTreeTypeRegistry::RegistrationPtr& custom,  // NOLINT[runtime/references]
    const std::string& registry_namespace) {
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
    const py::handle&,
    PyTreeTypeRegistry::RegistrationPtr&,  // NOLINT[runtime/references]
    const std::string&);
template PyTreeKind PyTreeTypeRegistry::GetKind<NONE_IS_LEAF>(
    const py::handle&,
    PyTreeTypeRegistry::RegistrationPtr&,  // NOLINT[runtime/references]
    const std::string&);

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ void PyTreeTypeRegistry::Clear() {
    const scoped_write_lock_guard lock{sm_mutex};

    PyTreeTypeRegistry* const registry1 = PyTreeTypeRegistry::Singleton<NONE_IS_NODE>();
    PyTreeTypeRegistry* const registry2 = PyTreeTypeRegistry::Singleton<NONE_IS_LEAF>();

    EXPECT_LE(sm_builtins_types.size(), registry1->m_registrations.size());
    EXPECT_EQ(registry1->m_registrations.size(), registry2->m_registrations.size() + 1);
    EXPECT_EQ(registry1->m_named_registrations.size(), registry2->m_named_registrations.size());

#ifdef Py_DEBUG
    for (const auto& cls : sm_builtins_types) {
        EXPECT_NE(registry1->m_registrations.find(cls), registry1->m_registrations.end());
    }
    for (const auto& entry : registry2->m_registrations) {
        const auto it = registry1->m_registrations.find(entry.first);
        EXPECT_NE(it, registry1->m_registrations.end());

        const auto& registration1 = it->second;
        const auto& registration2 = entry.second;
        EXPECT_TRUE(registration1->type.is(registration2->type));
        EXPECT_TRUE(registration1->flatten_func.is(registration2->flatten_func));
        EXPECT_TRUE(registration1->unflatten_func.is(registration2->unflatten_func));
        EXPECT_TRUE(registration1->path_entry_type.is(registration2->path_entry_type));
    }
    for (const auto& entry : registry2->m_named_registrations) {
        const auto it = registry1->m_named_registrations.find(entry.first);
        EXPECT_NE(it, registry1->m_named_registrations.end());

        const auto& registration1 = it->second;
        const auto& registration2 = entry.second;
        EXPECT_TRUE(registration1->type.is(registration2->type));
        EXPECT_TRUE(registration1->flatten_func.is(registration2->flatten_func));
        EXPECT_TRUE(registration1->unflatten_func.is(registration2->unflatten_func));
        EXPECT_TRUE(registration1->path_entry_type.is(registration2->path_entry_type));
    }
#endif

    for (const auto& entry : registry1->m_registrations) {
        entry.second->type.dec_ref();
        entry.second->flatten_func.dec_ref();
        entry.second->unflatten_func.dec_ref();
        entry.second->path_entry_type.dec_ref();
    }
    for (const auto& entry : registry1->m_named_registrations) {
        entry.second->type.dec_ref();
        entry.second->flatten_func.dec_ref();
        entry.second->unflatten_func.dec_ref();
        entry.second->path_entry_type.dec_ref();
    }

    sm_builtins_types.clear();
    registry1->m_registrations.clear();
    registry1->m_named_registrations.clear();
    registry2->m_registrations.clear();
    registry2->m_named_registrations.clear();
}

}  // namespace optree
