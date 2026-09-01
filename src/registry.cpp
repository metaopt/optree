/*
Copyright 2022-2026 MetaOPT Team. All Rights Reserved.

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
#include <type_traits>  // std::remove_const_t
#include <utility>      // std::move, std::make_pair

#include "optree/optree.h"

namespace optree {

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry &PyTreeTypeRegistry::GetSingleton() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<PyTreeTypeRegistry> storage;
    return storage
        .call_once_and_store_result([]() -> PyTreeTypeRegistry {
            PyTreeTypeRegistry registry{};

            const auto add_builtin_type = [&registry](const py::object &cls,
                                                      const PyTreeKind &kind) -> void {
                EXPECT_TRUE(registry.m_builtins_types.emplace(cls).second,
                            "PyTree type " + PyRepr(cls) +
                                " is already registered in the built-in types set.");
                if (!NoneIsLeaf || kind != PyTreeKind::None) [[likely]] {
                    auto registration =
                        std::make_shared<std::remove_const_t<RegistrationPtr::element_type>>();
                    registration->kind = kind;
                    registration->type = py::reinterpret_borrow<py::object>(cls);
                    EXPECT_TRUE(
                        registry.m_registrations.emplace(cls, std::move(registration)).second,
                        "PyTree type " + PyRepr(cls) +
                            " is already registered in the global namespace.");
                }
                if constexpr (!NoneIsLeaf) {
                    cls.inc_ref();
                }
            };
            add_builtin_type(PyNoneTypeObject, PyTreeKind::None);
            add_builtin_type(PyTupleTypeObject, PyTreeKind::Tuple);
            add_builtin_type(PyListTypeObject, PyTreeKind::List);
            add_builtin_type(PyDictTypeObject, PyTreeKind::Dict);
            add_builtin_type(PyOrderedDictTypeObject, PyTreeKind::OrderedDict);
            add_builtin_type(PyDefaultDictTypeObject, PyTreeKind::DefaultDict);
            add_builtin_type(PyDequeTypeObject, PyTreeKind::Deque);
#if defined(OPTREE_HAS_FROZENDICT)
            add_builtin_type(PyFrozenDictTypeObject, PyTreeKind::FrozenDict);
#endif
            return registry;
        })
        .get_stored();
}

template PyTreeTypeRegistry &PyTreeTypeRegistry::GetSingleton<NONE_IS_NODE>();
template PyTreeTypeRegistry &PyTreeTypeRegistry::GetSingleton<NONE_IS_LEAF>();

ssize_t PyTreeTypeRegistry::SizeImpl(const std::optional<std::string> &registry_namespace) const {
    // The caller must hold `sm_mutex`.
    ssize_t count = py::ssize_t_cast(m_registrations.size());
    for (const auto &[named_type, _] : m_named_registrations) {
        if (!registry_namespace || named_type.first == *registry_namespace) [[likely]] {
            ++count;
        }
    }
    return count;
}

ssize_t PyTreeTypeRegistry::Size(const std::optional<std::string> &registry_namespace) const {
    const scoped_read_lock lock{sm_mutex};
    return SizeImpl(registry_namespace);
}

// The caller must hold `sm_mutex` in write mode. No Python may run here; see `RegistryStatus`.
PyTreeTypeRegistry::RegistryStatus PyTreeTypeRegistry::RegisterImpl(
    const py::object &cls,
    const py::function &flatten_func,
    const py::function &unflatten_func,
    const py::object &path_entry_type,
    const std::string &registry_namespace) {
    if (m_builtins_types.find(cls) != m_builtins_types.end()) [[unlikely]] {
        return RegistryStatus::BuiltinType;
    }

    auto registration = std::make_shared<std::remove_const_t<RegistrationPtr::element_type>>();
    registration->kind = PyTreeKind::Custom;
    registration->type = py::reinterpret_borrow<py::object>(cls);
    registration->flatten_func = py::reinterpret_borrow<py::function>(flatten_func);
    registration->unflatten_func = py::reinterpret_borrow<py::function>(unflatten_func);
    registration->path_entry_type = py::reinterpret_borrow<py::object>(path_entry_type);
    // The registration only ever borrows objects the caller keeps alive, so the drop below when the
    // insert fails cannot reach zero and cannot run Python.
    if (registry_namespace.empty()) [[unlikely]] {
        if (!m_registrations.emplace(cls, std::move(registration)).second) [[unlikely]] {
            return RegistryStatus::AlreadyRegistered;
        }
    } else [[likely]] {
        if (!m_named_registrations
                 .emplace(std::make_pair(registry_namespace, cls), std::move(registration))
                 .second) [[unlikely]] {
            return RegistryStatus::AlreadyRegistered;
        }
    }
    return RegistryStatus::Ok;
}

/*static*/ void PyTreeTypeRegistry::Register(const py::object &cls,
                                             const py::function &flatten_func,
                                             const py::function &unflatten_func,
                                             const py::object &path_entry_type,
                                             const std::string &registry_namespace) {
    // Classify the type BEFORE taking `sm_mutex`: `IsStructSequenceClass` / `IsNamedTupleClass` run
    // Python and release the GIL, and doing that under the write lock inverts the GIL <->
    // `sm_mutex` lock order against a concurrent flatten (mirrors `Unregister`).
    const char *overridden_kind = nullptr;
    if (IsStructSequenceClass(cls)) [[unlikely]] {
        overridden_kind = " is a class of `PyStructSequence`, ";
    } else if (IsNamedTupleClass(cls)) [[unlikely]] {
        overridden_kind = " is a subclass of `collections.namedtuple`, ";
    }

    // Acquire both singletons BEFORE `sm_mutex`, mirroring `Init`/`Clear`. Under
    // `per_interpreter_gil`, `GetSingleton()` releases the GIL on every call once a subinterpreter
    // has existed; doing that while holding `sm_mutex` inverts the GIL <-> `sm_mutex` lock order
    // against a concurrent flatten (read lock) and deadlocks.
    auto &registry1 = GetSingleton<NONE_IS_NODE>();
    auto &registry2 = GetSingleton<NONE_IS_LEAF>();

    RegistryStatus status = RegistryStatus::Ok;
    {
        const scoped_write_lock lock{sm_mutex};

        status = registry1.RegisterImpl(cls,
                                        flatten_func,
                                        unflatten_func,
                                        path_entry_type,
                                        registry_namespace);
        if (status == RegistryStatus::Ok) [[likely]] {
            status = registry2.RegisterImpl(cls,
                                            flatten_func,
                                            unflatten_func,
                                            path_entry_type,
                                            registry_namespace);
        }
        if (status == RegistryStatus::Ok) [[likely]] {
            cls.inc_ref();
            flatten_func.inc_ref();
            unflatten_func.inc_ref();
            path_entry_type.inc_ref();
        }
    }

    // Format the error only after the lock is released: `PyRepr` runs the (meta)class `__repr__` as
    // Python bytecode, which can hand off the GIL to a thread blocking on `sm_mutex` in read mode.
    if (status != RegistryStatus::Ok) [[unlikely]] {
        if (status == RegistryStatus::BuiltinType) [[unlikely]] {
            throw py::value_error("PyTree type " + PyRepr(cls) +
                                  " is a built-in type and cannot be re-registered.");
        }
        std::ostringstream oss{};
        oss << "PyTree type " << PyRepr(cls) << " is already registered in ";
        if (registry_namespace.empty()) [[unlikely]] {
            oss << "the global namespace.";
        } else [[likely]] {
            oss << "namespace " << PyRepr(registry_namespace) << ".";
        }
        throw py::value_error(oss.str());
    }

    // Warn only once the registration succeeded: a rejected one overrides nothing. `PyErr_WarnEx`
    // runs Python, so it must not run under `sm_mutex` either. Under warnings-as-errors it raises,
    // so undo the registration to keep `Register` atomic.
    if (overridden_kind != nullptr) [[unlikely]] {
        std::ostringstream oss{};
        oss << "PyTree type " << PyRepr(cls) << overridden_kind
            << "which is already registered in the global namespace. "
               "Override it with custom flatten/unflatten functions";
        if (!registry_namespace.empty()) [[likely]] {
            oss << " in namespace " << PyRepr(registry_namespace);
        }
        oss << ".";
        try {
            if (PyErr_WarnEx(PyExc_UserWarning, oss.str().c_str(), /*stack_level=*/2) < 0)
                [[unlikely]] {
                throw py::error_already_set();
            }
        } catch (...) {
            Unregister(cls, registry_namespace);
            throw;
        }
    }
}

// The caller must hold `sm_mutex` in write mode. No Python may run here; see `RegistryStatus`.
PyTreeTypeRegistry::RegistryStatus PyTreeTypeRegistry::UnregisterImpl(
    const py::object &cls,
    const std::string &registry_namespace,
    RegistrationPtr &registration) {
    if (m_builtins_types.find(cls) != m_builtins_types.end()) [[unlikely]] {
        return RegistryStatus::BuiltinType;
    }

    if (registry_namespace.empty()) [[unlikely]] {
        const auto it = m_registrations.find(cls);
        if (it == m_registrations.end()) [[unlikely]] {
            return RegistryStatus::NotRegistered;
        }
        registration = it->second;
        m_registrations.erase(it);
    } else [[likely]] {
        const auto named_it = m_named_registrations.find(std::make_pair(registry_namespace, cls));
        if (named_it == m_named_registrations.end()) [[unlikely]] {
            return RegistryStatus::NotRegistered;
        }
        registration = named_it->second;
        m_named_registrations.erase(named_it);
    }
    return RegistryStatus::Ok;
}

/*static*/ void PyTreeTypeRegistry::Unregister(const py::object &cls,
                                               const std::string &registry_namespace) {
    // Classify the type BEFORE taking `sm_mutex`. On the not-found path `UnregisterImpl` builds its
    // error message from `IsStructSequenceClass` / `IsNamedTupleClass`, which run Python and
    // release the GIL; calling them while holding `sm_mutex` in write mode inverts the GIL <->
    // `sm_mutex` lock order and deadlocks a concurrent flatten that holds the GIL while waiting on
    // `sm_mutex` in read mode (mirrors `Register`).
    const bool is_structsequence_class = IsStructSequenceClass(cls);
    const bool is_namedtuple_class = IsNamedTupleClass(cls);

    // Acquire both singletons BEFORE `sm_mutex`, mirroring `Init`/`Clear` (see `Lookup`/`Register`
    // for the lock-order rationale).
    auto &registry1 = GetSingleton<NONE_IS_NODE>();
    auto &registry2 = GetSingleton<NONE_IS_LEAF>();

    // These outlive the locked scope: dropping the last reference to a member runs arbitrary Python
    // (`__del__`, weakref callbacks) that can re-enter optree and deadlock on `sm_mutex`.
    RegistrationPtr registration1{nullptr};
    RegistrationPtr registration2{nullptr};
    RegistryStatus status = RegistryStatus::Ok;
    {
        const scoped_write_lock lock{sm_mutex};

        status = registry1.UnregisterImpl(cls, registry_namespace, registration1);
        if (status == RegistryStatus::Ok) [[likely]] {
            status = registry2.UnregisterImpl(cls, registry_namespace, registration2);
        }
        if (status == RegistryStatus::Ok) [[likely]] {
            EXPECT_TRUE(registration1->type.is(registration2->type));
            EXPECT_TRUE(registration1->flatten_func.is(registration2->flatten_func));
            EXPECT_TRUE(registration1->unflatten_func.is(registration2->unflatten_func));
            EXPECT_TRUE(registration1->path_entry_type.is(registration2->path_entry_type));
        }
    }

    // Format the error only after the lock is released (mirrors `Register`).
    if (status != RegistryStatus::Ok) [[unlikely]] {
        if (status == RegistryStatus::BuiltinType) [[unlikely]] {
            throw py::value_error("PyTree type " + PyRepr(cls) +
                                  " is a built-in type and cannot be unregistered.");
        }
        std::ostringstream oss{};
        oss << "PyTree type " << PyRepr(cls) << " ";
        if (is_structsequence_class) [[unlikely]] {
            oss << "is a class of `PyStructSequence`, which is not explicitly registered ";
        } else if (is_namedtuple_class) [[unlikely]] {
            oss << "is a subclass of `collections.namedtuple`, which is not explicitly registered ";
        } else [[likely]] {
            oss << "is not registered ";
        }
        if (registry_namespace.empty()) [[unlikely]] {
            oss << "in the global namespace.";
        } else [[likely]] {
            oss << "in namespace " << PyRepr(registry_namespace) << ".";
        }
        throw py::value_error(oss.str());
    }

    // Drop the registry's references with the lock released; the registrations die at scope exit.
    registration1->type.dec_ref();
    registration1->flatten_func.dec_ref();
    registration1->unflatten_func.dec_ref();
    registration1->path_entry_type.dec_ref();
}

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry::RegistrationPtr PyTreeTypeRegistry::Lookup(
    const py::object &cls,
    const std::string &registry_namespace) {
    // Acquire the singleton BEFORE `sm_mutex`, mirroring `Init`/`Clear`. Under
    // `per_interpreter_gil`, `GetSingleton()` releases the GIL on every call once a subinterpreter
    // has existed; doing that while holding `sm_mutex` inverts the GIL <-> `sm_mutex` lock order
    // against a concurrent registration (write lock) and deadlocks.
    const auto &registry = GetSingleton<NoneIsLeaf>();

    {
        const scoped_read_lock lock{sm_mutex};
        if (!registry_namespace.empty()) [[unlikely]] {
            const auto named_it =
                registry.m_named_registrations.find(std::make_pair(registry_namespace, cls));
            if (named_it != registry.m_named_registrations.end()) [[likely]] {
                return named_it->second;
            }
        }
        const auto it = registry.m_registrations.find(cls);
        return it != registry.m_registrations.end() ? it->second : nullptr;
    }
}

template PyTreeTypeRegistry::RegistrationPtr PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(
    const py::object &,
    const std::string &);
template PyTreeTypeRegistry::RegistrationPtr PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(
    const py::object &,
    const std::string &);

template <bool NoneIsLeaf>
/*static*/ PyTreeKind PyTreeTypeRegistry::GetKind(const py::handle &handle,
                                                  PyTreeTypeRegistry::RegistrationPtr &custom,
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

template PyTreeKind PyTreeTypeRegistry::GetKind<NONE_IS_NODE>(const py::handle &,
                                                              PyTreeTypeRegistry::RegistrationPtr &,
                                                              const std::string &);
template PyTreeKind PyTreeTypeRegistry::GetKind<NONE_IS_LEAF>(const py::handle &,
                                                              PyTreeTypeRegistry::RegistrationPtr &,
                                                              const std::string &);

/*static*/ void PyTreeTypeRegistry::Init() {
    const auto &registry1 = GetSingleton<NONE_IS_NODE>();
    const auto &registry2 = GetSingleton<NONE_IS_LEAF>();
    const auto interpid = GetCurrentPyInterpreterID();

    {
        const scoped_write_lock lock{sm_mutex};

        ++sm_num_interpreters_seen;
        EXPECT_TRUE(
            sm_alive_interpids.insert(interpid).second,
            "The current interpreter ID should not be already present in the alive interpreters "
            "set.");

        EXPECT_EQ(registry1.m_builtins_types.size(), registry2.m_builtins_types.size());
        EXPECT_LE(registry1.m_builtins_types.size(), registry1.m_registrations.size());
        EXPECT_EQ(registry1.m_registrations.size(), registry2.m_registrations.size() + 1);
        EXPECT_EQ(registry1.m_named_registrations.size(), registry2.m_named_registrations.size());
    }

    // `atexit.register` runs Python, so it must not run under `sm_mutex`, and it can raise: without
    // the rollback a failed import would leave an ID that no callback can ever remove (mirrors
    // `WeakKeyCache::LookupOrInsert`). The rollback locks with the GIL held, as `Clear` does.
    try {
        const auto atexit_register = py::getattr(py::module_::import("atexit"), "register");
        atexit_register(py::cpp_function(&Clear));
    } catch (...) {
        const scoped_write_lock lock{sm_mutex};

        sm_alive_interpids.erase(interpid);
        --sm_num_interpreters_seen;
        throw;
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
/*static*/ void PyTreeTypeRegistry::Clear() {
    auto &registry1 = GetSingleton<NONE_IS_NODE>();
    auto &registry2 = GetSingleton<NONE_IS_LEAF>();
    const auto interpid = GetCurrentPyInterpreterID();

    // Detached under the lock and destroyed after it, for the reason given in `Unregister`.
    RegistrationsMap registrations1{};
    NamedRegistrationsMap named_registrations1{};
    RegistrationsMap registrations2{};
    NamedRegistrationsMap named_registrations2{};
    {
        const scoped_write_lock lock{sm_mutex};

        EXPECT_NE(sm_alive_interpids.find(interpid),
                  sm_alive_interpids.end(),
                  "The current interpreter ID should be present in the alive interpreters set.");
        sm_alive_interpids.erase(interpid);

        {
            const scoped_write_lock namespace_lock{sm_dict_order_mutex};
            auto entries =
                reserved_vector<decltype(sm_dict_insertion_ordered_namespaces)::key_type>(4);
            for (const auto &entry : sm_dict_insertion_ordered_namespaces) {
                if (entry.first == interpid) [[likely]] {
                    entries.emplace_back(entry);
                }
            }
            for (const auto &entry : entries) {
                sm_dict_insertion_ordered_namespaces.erase(entry);
            }
            if (sm_alive_interpids.empty()) [[likely]] {
                EXPECT_TRUE(
                    sm_dict_insertion_ordered_namespaces.empty(),
                    "The dict insertion ordered namespaces map should be empty when there is no "
                    "alive Python interpreter.");
            }
        }

        EXPECT_EQ(registry1.m_builtins_types.size(), registry2.m_builtins_types.size());
        EXPECT_LE(registry1.m_builtins_types.size(), registry1.m_registrations.size());
        EXPECT_EQ(registry1.m_registrations.size(), registry2.m_registrations.size() + 1);
        EXPECT_EQ(registry1.m_named_registrations.size(), registry2.m_named_registrations.size());

#if defined(Py_DEBUG)
        for (const auto &cls : registry1.m_builtins_types) {
            EXPECT_NE(registry1.m_registrations.find(cls), registry1.m_registrations.end());
            EXPECT_NE(registry2.m_builtins_types.find(cls), registry2.m_builtins_types.end());
        }
        for (const auto &cls : registry2.m_builtins_types) {
            if (cls.is(PyNoneTypeObject)) [[unlikely]] {
                EXPECT_EQ(registry2.m_registrations.find(cls), registry2.m_registrations.end());
            } else [[likely]] {
                EXPECT_NE(registry2.m_registrations.find(cls), registry2.m_registrations.end());
            }
        }
        for (const auto &[cls2, registration2] : registry2.m_registrations) {
            const auto it1 = registry1.m_registrations.find(cls2);
            EXPECT_NE(it1, registry1.m_registrations.end());

            const auto &registration1 = it1->second;
            EXPECT_TRUE(registration1->type.is(registration2->type));
            EXPECT_TRUE(registration1->flatten_func.is(registration2->flatten_func));
            EXPECT_TRUE(registration1->unflatten_func.is(registration2->unflatten_func));
            EXPECT_TRUE(registration1->path_entry_type.is(registration2->path_entry_type));
        }
        for (const auto &[named_cls2, registration2] : registry2.m_named_registrations) {
            const auto it1 = registry1.m_named_registrations.find(named_cls2);
            EXPECT_NE(it1, registry1.m_named_registrations.end());

            const auto &registration1 = it1->second;
            EXPECT_TRUE(registration1->type.is(registration2->type));
            EXPECT_TRUE(registration1->flatten_func.is(registration2->flatten_func));
            EXPECT_TRUE(registration1->unflatten_func.is(registration2->unflatten_func));
            EXPECT_TRUE(registration1->path_entry_type.is(registration2->path_entry_type));
        }
#endif

        registry1.m_builtins_types.clear();
        registry1.m_registrations.swap(registrations1);
        registry1.m_named_registrations.swap(named_registrations1);
        registry2.m_builtins_types.clear();
        registry2.m_registrations.swap(registrations2);
        registry2.m_named_registrations.swap(named_registrations2);
    }

    for (const auto &[_, registration1] : registrations1) {
        registration1->type.dec_ref();
        registration1->flatten_func.dec_ref();
        registration1->unflatten_func.dec_ref();
        registration1->path_entry_type.dec_ref();
    }
    for (const auto &[_, registration1] : named_registrations1) {
        registration1->type.dec_ref();
        registration1->flatten_func.dec_ref();
        registration1->unflatten_func.dec_ref();
        registration1->path_entry_type.dec_ref();
    }
}

}  // namespace optree
