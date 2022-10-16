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

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#include "include/registry.h"

namespace optree {

/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::SingletonHelper::get() {
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
        return registry;
    }());
    return &registry;
}

/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::Singleton() { return SingletonHelper::get(); }

/*static*/ void PyTreeTypeRegistry::Register(const py::object& type,
                                             const py::function& to_iterable,
                                             const py::function& from_iterable) {
    PyTreeTypeRegistry* registry = Singleton();
    auto registration = std::make_unique<Registration>();
    registration->kind = PyTreeKind::Custom;
    registration->type = type;
    registration->to_iterable = std::move(to_iterable);
    registration->from_iterable = std::move(from_iterable);
    if (!registry->registrations.emplace(type, std::move(registration)).second) {
        throw std::invalid_argument(
            absl::StrFormat("Duplicate custom PyTree type registration for %s.", py::repr(type)));
    }
}

/*static*/ const PyTreeTypeRegistry::Registration* PyTreeTypeRegistry::Lookup(
    const py::handle& type) {
    PyTreeTypeRegistry* registry = Singleton();
    auto it = registry->registrations.find(type);
    return it == registry->registrations.end() ? nullptr : it->second.get();
}

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

}  // namespace optree
