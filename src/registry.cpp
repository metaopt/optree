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

#include "include/registry.h"

namespace optree {

template <>
/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::SingletonHelper<NONE_IS_NODE>::get() {
    static PyTreeTypeRegistry registry([]() {
        PyTreeTypeRegistry registry;

        auto add_builtin_type = [&](PyTypeObject* type_obj, PyTreeKind kind) {
            auto type = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(type_obj));
            auto registration = std::make_unique<Registration>();
            registration->kind = kind;
            registration->type = type;
            CHECK(registry.registrations.emplace(type, std::move(registration)).second);
        };
        add_builtin_type(Py_TYPE(Py_None), PyTreeKind::None);
        add_builtin_type(&PyTuple_Type, PyTreeKind::Tuple);
        add_builtin_type(&PyList_Type, PyTreeKind::List);
        add_builtin_type(&PyDict_Type, PyTreeKind::Dict);
        add_builtin_type(reinterpret_cast<PyTypeObject*>(PyOrderedDictTypeObject.ptr()),
                         PyTreeKind::OrderedDict);
        add_builtin_type(reinterpret_cast<PyTypeObject*>(PyDefaultDictTypeObject.ptr()),
                         PyTreeKind::DefaultDict);
        add_builtin_type(reinterpret_cast<PyTypeObject*>(PyDequeTypeObject.ptr()),
                         PyTreeKind::Deque);
        return registry;
    }());
    return &registry;
}

template <>
/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::SingletonHelper<NONE_IS_LEAF>::get() {
    static PyTreeTypeRegistry registry([]() {
        PyTreeTypeRegistry registry;

        auto add_builtin_type = [&](PyTypeObject* type_obj, PyTreeKind kind) {
            auto type = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(type_obj));
            auto registration = std::make_unique<Registration>();
            registration->kind = kind;
            registration->type = type;
            CHECK(registry.registrations.emplace(type, std::move(registration)).second);
        };
        add_builtin_type(&PyTuple_Type, PyTreeKind::Tuple);
        add_builtin_type(&PyList_Type, PyTreeKind::List);
        add_builtin_type(&PyDict_Type, PyTreeKind::Dict);
        add_builtin_type(reinterpret_cast<PyTypeObject*>(PyOrderedDictTypeObject.ptr()),
                         PyTreeKind::OrderedDict);
        add_builtin_type(reinterpret_cast<PyTypeObject*>(PyDefaultDictTypeObject.ptr()),
                         PyTreeKind::DefaultDict);
        add_builtin_type(reinterpret_cast<PyTypeObject*>(PyDequeTypeObject.ptr()),
                         PyTreeKind::Deque);
        return registry;
    }());
    return &registry;
}

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::Singleton() {
    return SingletonHelper<NoneIsLeaf>::get();
}

/*static*/ void PyTreeTypeRegistry::Register(const py::object& cls,
                                             const py::function& to_iterable,
                                             const py::function& from_iterable,
                                             const std::string& regnamespace) {
    {
        PyTreeTypeRegistry* registry = Singleton<NONE_IS_NODE>();
        auto registration = std::make_unique<Registration>();
        registration->kind = PyTreeKind::Custom;
        registration->type = cls;
        registration->to_iterable = to_iterable;
        registration->from_iterable = from_iterable;
        if (regnamespace.empty()) [[unlikely]] {  // NOLINT
            if (!registry->registrations.emplace(cls, std::move(registration)).second)
                [[unlikely]] {
                throw std::invalid_argument(
                    absl::StrFormat("PyTree type %s is already registered in the global namespace.",
                                    py::repr(cls)));
            }
        } else [[likely]] {  // NOLINT
            if (registry->registrations.find(cls) != registry->registrations.end()) [[unlikely]] {
                throw std::invalid_argument(
                    absl::StrFormat("PyTree type %s is already registered in the global namespace.",
                                    py::repr(cls)));
            }
            if (!registry->namespaced_registrations
                     .emplace(std::make_pair(regnamespace, cls), std::move(registration))
                     .second) [[unlikely]] {
                throw std::invalid_argument(
                    absl::StrFormat("PyTree type %s is already registered in namespace %s.",
                                    py::repr(cls),
                                    py::repr(py::str(regnamespace))));
            }
        }
    }
    {
        PyTreeTypeRegistry* registry = Singleton<NONE_IS_LEAF>();
        auto registration = std::make_unique<Registration>();
        registration->kind = PyTreeKind::Custom;
        registration->type = cls;
        registration->to_iterable = to_iterable;
        registration->from_iterable = from_iterable;
        if (regnamespace.empty()) [[unlikely]] {  // NOLINT
            if (!registry->registrations.emplace(cls, std::move(registration)).second)
                [[unlikely]] {
                throw std::invalid_argument(
                    absl::StrFormat("PyTree type %s is already registered in the global namespace.",
                                    py::repr(cls)));
            }
        } else [[likely]] {  // NOLINT
            if (registry->registrations.find(cls) != registry->registrations.end()) [[unlikely]] {
                throw std::invalid_argument(
                    absl::StrFormat("PyTree type %s is already registered in the global namespace.",
                                    py::repr(cls)));
            }
            if (!registry->namespaced_registrations
                     .emplace(std::make_pair(regnamespace, cls), std::move(registration))
                     .second) [[unlikely]] {
                throw std::invalid_argument(
                    absl::StrFormat("PyTree type %s is already registered in namespace %s.",
                                    py::repr(cls),
                                    py::repr(py::str(regnamespace))));
            }
        }
    }
}

template <bool NoneIsLeaf>
/*static*/ const PyTreeTypeRegistry::Registration* PyTreeTypeRegistry::Lookup(
    const py::handle& type, const std::string& regnamespace) {
    PyTreeTypeRegistry* registry = Singleton<NoneIsLeaf>();
    auto it = registry->registrations.find(type);
    if (it != registry->registrations.end()) [[likely]] {
        return it->second.get();
    }
    if (regnamespace.empty()) [[likely]] {
        return nullptr;
    } else [[unlikely]] {  // NOLINT
        auto it = registry->namespaced_registrations.find(std::make_pair(regnamespace, type));
        return it != registry->namespaced_registrations.end() ? it->second.get() : nullptr;
    }
}

template const PyTreeTypeRegistry::Registration* PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(
    const py::handle&, const std::string&);
template const PyTreeTypeRegistry::Registration* PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(
    const py::handle&, const std::string&);

size_t PyTreeTypeRegistry::TypeHash::operator()(const py::object& t) const {
    return absl::HashOf(t.ptr());
}
size_t PyTreeTypeRegistry::TypeHash::operator()(const py::handle& t) const {
    return absl::HashOf(t.ptr());
}

bool PyTreeTypeRegistry::TypeEq::operator()(const py::object& a, const py::object& b) const {
    return a.ptr() == b.ptr();
}
bool PyTreeTypeRegistry::TypeEq::operator()(const py::object& a, const py::handle& b) const {
    return a.ptr() == b.ptr();
}

size_t PyTreeTypeRegistry::NamedTypeHash::operator()(
    const std::pair<std::string, py::object>& p) const {
    return absl::HashOf(std::make_pair(p.first, p.second.ptr()));
}
size_t PyTreeTypeRegistry::NamedTypeHash::operator()(
    const std::pair<std::string, py::handle>& p) const {
    return absl::HashOf(std::make_pair(p.first, p.second.ptr()));
}

bool PyTreeTypeRegistry::NamedTypeEq::operator()(
    const std::pair<std::string, py::object>& a,
    const std::pair<std::string, py::object>& b) const {
    return a.first == b.first && a.second.ptr() == b.second.ptr();
}
bool PyTreeTypeRegistry::NamedTypeEq::operator()(
    const std::pair<std::string, py::object>& a,
    const std::pair<std::string, py::handle>& b) const {
    return a.first == b.first && a.second.ptr() == b.second.ptr();
}

}  // namespace optree
