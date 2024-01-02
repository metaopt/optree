/*
Copyright 2022-2023 MetaOPT Team. All Rights Reserved.

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

#include "include/registry.h"

#include <Python.h>

#include <memory>   // std::make_unique
#include <sstream>  // std::ostringstream
#include <string>   // std::string
#include <utility>  // std::move, std::make_pair

#include "include/exceptions.h"
#include "include/utils.h"

namespace optree {

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::Singleton() {
    static PyTreeTypeRegistry singleton{[]() {
        PyTreeTypeRegistry registry;

        auto add_builtin_type = [&registry](const py::object& cls, const PyTreeKind& kind) -> void {
            auto registration = std::make_unique<Registration>();
            registration->kind = kind;
            registration->type = py::reinterpret_borrow<py::object>(cls);
            EXPECT(
                registry.m_registrations.emplace(cls, std::move(registration)).second,
                "PyTree type " + PyRepr(cls) + " is already registered in the global namespace.");
            cls.inc_ref();
        };
        if (!NoneIsLeaf) {
            add_builtin_type(py::type::of(py::none()), PyTreeKind::None);
        }
        add_builtin_type(
            py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyTuple_Type)),
            PyTreeKind::Tuple);
        add_builtin_type(
            py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyList_Type)),
            PyTreeKind::List);
        add_builtin_type(
            py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyDict_Type)),
            PyTreeKind::Dict);
        add_builtin_type(PyOrderedDictTypeObject, PyTreeKind::OrderedDict);
        add_builtin_type(PyDefaultDictTypeObject, PyTreeKind::DefaultDict);
        add_builtin_type(PyDequeTypeObject, PyTreeKind::Deque);
        return registry;
    }()};
    return &singleton;
}

template <bool NoneIsLeaf>
/*static*/ void PyTreeTypeRegistry::RegisterImpl(const py::object& cls,
                                                 const py::function& flatten_func,
                                                 const py::function& unflatten_func,
                                                 const std::string& registry_namespace) {
    PyTreeTypeRegistry* registry = Singleton<NoneIsLeaf>();
    auto registration = std::make_unique<Registration>();
    registration->kind = PyTreeKind::Custom;
    registration->type = py::reinterpret_borrow<py::object>(cls);
    registration->flatten_func = py::reinterpret_borrow<py::function>(flatten_func);
    registration->unflatten_func = py::reinterpret_borrow<py::function>(unflatten_func);
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
        if (registry->m_registrations.find(cls) != registry->m_registrations.end()) [[unlikely]] {
            throw py::value_error("PyTree type " + PyRepr(cls) +
                                  " is already registered in the global namespace.");
        }
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
                                             const std::string& registry_namespace) {
    RegisterImpl<NONE_IS_NODE>(cls, flatten_func, unflatten_func, registry_namespace);
    RegisterImpl<NONE_IS_LEAF>(cls, flatten_func, unflatten_func, registry_namespace);
    cls.inc_ref();
    flatten_func.inc_ref();
    unflatten_func.inc_ref();
}

template <bool NoneIsLeaf>
/*static*/ const PyTreeTypeRegistry::Registration* PyTreeTypeRegistry::Lookup(
    const py::object& cls, const std::string& registry_namespace) {
    PyTreeTypeRegistry* registry = Singleton<NoneIsLeaf>();
    auto it = registry->m_registrations.find(cls);
    if (it != registry->m_registrations.end()) [[likely]] {
        return it->second.get();
    }
    if (registry_namespace.empty()) [[likely]] {
        return nullptr;
    } else [[unlikely]] {
        auto named_it =
            registry->m_named_registrations.find(std::make_pair(registry_namespace, cls));
        return named_it != registry->m_named_registrations.end() ? named_it->second.get() : nullptr;
    }
}

template const PyTreeTypeRegistry::Registration* PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(
    const py::object&, const std::string&);
template const PyTreeTypeRegistry::Registration* PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(
    const py::object&, const std::string&);

}  // namespace optree
