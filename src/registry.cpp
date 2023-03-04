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

namespace optree {

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::SingletonHelper<NoneIsLeaf>::get() {
    static PyTreeTypeRegistry singleton{[]() {
        PyTreeTypeRegistry registry;

        auto add_builtin_type = [&](PyTypeObject* type_obj, PyTreeKind kind) {
            auto type = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(type_obj));
            auto registration = std::make_unique<Registration>();
            registration->kind = kind;
            registration->type = type;
            EXPECT(registry.m_registrations.emplace(type, std::move(registration)).second,
                   absl::StrFormat("PyTree type %s is already registered in the global namespace.",
                                   py::repr(type)));
        };
        if (!NoneIsLeaf) {
            add_builtin_type(Py_TYPE(Py_None), PyTreeKind::None);
        }
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
    }()};
    return &singleton;
}

template <bool NoneIsLeaf>
/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::Singleton() {
    return SingletonHelper<NoneIsLeaf>::get();
}

template <bool NoneIsLeaf>
/*static*/ void PyTreeTypeRegistry::RegisterImpl(const py::object& cls,
                                                 const py::function& to_iterable,
                                                 const py::function& from_iterable,
                                                 const std::string& registry_namespace) {
    PyTreeTypeRegistry* registry = Singleton<NoneIsLeaf>();
    auto registration = std::make_unique<Registration>();
    registration->kind = PyTreeKind::Custom;
    registration->type = py::reinterpret_borrow<py::object>(cls);
    registration->to_iterable = py::reinterpret_borrow<py::function>(to_iterable);
    registration->from_iterable = py::reinterpret_borrow<py::function>(from_iterable);
    if (registry_namespace.empty()) [[unlikely]] {
        if (!registry->m_registrations.emplace(cls, std::move(registration)).second) [[unlikely]] {
            throw py::value_error(absl::StrFormat(
                "PyTree type %s is already registered in the global namespace.", py::repr(cls)));
        }
        if (IsNamedTupleClass(cls)) [[unlikely]] {
            PyErr_WarnEx(
                PyExc_UserWarning,
                absl::StrFormat("PyTree type %s is a subclass of `collections.namedtuple`, "
                                "which is already registered in the global namespace. "
                                "Override it with custom flatten/unflatten functions.",
                                py::repr(cls))
                    .c_str(),
                /*stack_level=*/2);
        }
        if (IsStructSequenceClass(cls)) [[unlikely]] {
            PyErr_WarnEx(PyExc_UserWarning,
                         absl::StrFormat("PyTree type %s is a class of `PyStructSequence`, "
                                         "which is already registered in the global namespace. "
                                         "Override it with custom flatten/unflatten functions.",
                                         py::repr(cls))
                             .c_str(),
                         /*stack_level=*/2);
        }
    } else [[likely]] {
        if (registry->m_registrations.find(cls) != registry->m_registrations.end()) [[unlikely]] {
            throw py::value_error(absl::StrFormat(
                "PyTree type %s is already registered in the global namespace.", py::repr(cls)));
        }
        if (!registry->m_named_registrations
                 .emplace(std::make_pair(registry_namespace, cls), std::move(registration))
                 .second) [[unlikely]] {
            throw py::value_error(
                absl::StrFormat("PyTree type %s is already registered in namespace %s.",
                                py::repr(cls),
                                py::repr(py::str(registry_namespace))));
        }
        if (IsNamedTupleClass(cls)) [[unlikely]] {
            PyErr_WarnEx(PyExc_UserWarning,
                         absl::StrFormat(
                             "PyTree type %s is a subclass of `collections.namedtuple`, "
                             "which is already registered in the global namespace. "
                             "Override it with custom flatten/unflatten functions in namespace %s.",
                             py::repr(cls),
                             py::repr(py::str(registry_namespace)))
                             .c_str(),
                         /*stack_level=*/2);
        }
        if (IsStructSequenceClass(cls)) [[unlikely]] {
            PyErr_WarnEx(PyExc_UserWarning,
                         absl::StrFormat(
                             "PyTree type %s is a class of `PyStructSequence`, "
                             "which is already registered in the global namespace. "
                             "Override it with custom flatten/unflatten functions in namespace %s.",
                             py::repr(cls),
                             py::repr(py::str(registry_namespace)))
                             .c_str(),
                         /*stack_level=*/2);
        }
    }
}

/*static*/ void PyTreeTypeRegistry::Register(const py::object& cls,
                                             const py::function& to_iterable,
                                             const py::function& from_iterable,
                                             const std::string& registry_namespace) {
    RegisterImpl<NONE_IS_NODE>(cls, to_iterable, from_iterable, registry_namespace);
    RegisterImpl<NONE_IS_LEAF>(cls, to_iterable, from_iterable, registry_namespace);
    cls.inc_ref();
    to_iterable.inc_ref();
    from_iterable.inc_ref();
}

template <bool NoneIsLeaf>
/*static*/ const PyTreeTypeRegistry::Registration* PyTreeTypeRegistry::Lookup(
    const py::handle& type, const std::string& registry_namespace) {
    PyTreeTypeRegistry* registry = Singleton<NoneIsLeaf>();
    auto it = registry->m_registrations.find(type);
    if (it != registry->m_registrations.end()) [[likely]] {
        return it->second.get();
    }
    if (registry_namespace.empty()) [[likely]] {
        return nullptr;
    } else [[unlikely]] {
        auto named_it =
            registry->m_named_registrations.find(std::make_pair(registry_namespace, type));
        return named_it != registry->m_named_registrations.end() ? named_it->second.get() : nullptr;
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
