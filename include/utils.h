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

// See https://jax.readthedocs.io/en/latest/pytrees.html for the documentation
// about PyTrees.

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#pragma once

#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CHECK(condition)                                                              \
    if (!(condition))                                                                 \
    throw std::runtime_error(std::string(#condition) + " failed at " __FILE__ + ':' + \
                             std::to_string(__LINE__))

#define DCHECK(condition) CHECK(condition)

namespace py = pybind11;
using size_t = py::size_t;
using ssize_t = py::ssize_t;

namespace pybind11 {
const module_ collections_module = module_::import("collections");
const object OrderedDict = collections_module.attr("OrderedDict");
const object DefaultDict = collections_module.attr("defaultdict");
const object Deque = collections_module.attr("deque");
}  // namespace pybind11

template <typename T>
inline std::vector<T> reserved_vector(const size_t& size) {
    std::vector<T> v;
    v.reserve(size);
    return v;
}

inline py::list SortedDictKeys(const py::dict& dict) {
    py::list keys = py::reinterpret_borrow<py::list>(PyDict_Keys(dict.ptr()));

    try {
        // Sort directly if possible.
        if (PyList_Sort(keys.ptr())) {
            throw py::error_already_set();
        }
    } catch (py::error_already_set& ex1) {
        if (ex1.matches(PyExc_TypeError)) {
            // Found incomparable keys (e.g. `int` vs. `str`, or user-defined types).
            try {
                // Sort with `keys.sort(key=lambda o: (o.__class__.__qualname__, o))`.
                auto sort_key_fn = py::cpp_function([](const py::object& o) {
                    return py::make_tuple(o.get_type().attr("__qualname__"), o);
                });
                keys.attr("sort")(py::arg("key") = sort_key_fn);
            } catch (py::error_already_set& ex2) {
                if (ex2.matches(PyExc_TypeError)) {
                    // Found incomparable user-defined key types.
                    // The keys remain in the insertion order.
                    PyErr_Clear();
                } else {
                    throw;
                }
            }
        } else {
            throw;
        }
    }

    return keys;
}

template <typename Sized = py::object>
inline size_t GetSize(const py::handle& sized) {
    return py::len(sized);
}
template <>
inline size_t GetSize<py::tuple>(const py::handle& tuple) {
    return PyTuple_Size(tuple.ptr());
}
template <>
inline size_t GetSize<py::list>(const py::handle& list) {
    return PyList_Size(list.ptr());
}
template <>
inline size_t GetSize<py::dict>(const py::handle& dict) {
    return PyDict_Size(dict.ptr());
}

template <typename Sized = py::object>
inline size_t GET_SIZE(const py::handle& sized) {
    return py::len(sized);
}
template <>
inline size_t GET_SIZE<py::tuple>(const py::handle& tuple) {
    return PyTuple_GET_SIZE(tuple.ptr());
}
template <>
inline size_t GET_SIZE<py::list>(const py::handle& list) {
    return PyList_GET_SIZE(list.ptr());
}
template <>
inline size_t GET_SIZE<py::dict>(const py::handle& dict) {
    return PyDict_GET_SIZE(dict.ptr());
}

template <typename Container, typename Item>
inline py::handle GET_ITEM_HANDLE(const py::handle& container, const Item& item) {
    return container[item];
}
template <>
inline py::handle GET_ITEM_HANDLE<py::tuple>(const py::handle& tuple, const ssize_t& index) {
    return PyTuple_GET_ITEM(tuple.ptr(), index);
}
template <>
inline py::handle GET_ITEM_HANDLE<py::tuple>(const py::handle& tuple, const size_t& index) {
    return PyTuple_GET_ITEM(tuple.ptr(), py::ssize_t_cast(index));
}
template <>
inline py::handle GET_ITEM_HANDLE<py::tuple>(const py::handle& tuple, const int& index) {
    return PyTuple_GET_ITEM(tuple.ptr(), py::ssize_t_cast(index));
}
template <>
inline py::handle GET_ITEM_HANDLE<py::list>(const py::handle& list, const ssize_t& index) {
    return PyList_GET_ITEM(list.ptr(), index);
}
template <>
inline py::handle GET_ITEM_HANDLE<py::list>(const py::handle& list, const size_t& index) {
    return PyList_GET_ITEM(list.ptr(), py::ssize_t_cast(index));
}
template <>
inline py::handle GET_ITEM_HANDLE<py::list>(const py::handle& list, const int& index) {
    return PyList_GET_ITEM(list.ptr(), py::ssize_t_cast(index));
}

template <typename Container, typename Item>
inline py::object GET_ITEM_BORROW(const py::handle& container, const Item& item) {
    return container[item];
}
template <>
inline py::object GET_ITEM_BORROW<py::tuple>(const py::handle& tuple, const ssize_t& index) {
    return py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(tuple.ptr(), index));
}
template <>
inline py::object GET_ITEM_BORROW<py::tuple>(const py::handle& tuple, const size_t& index) {
    return py::reinterpret_borrow<py::object>(
        PyTuple_GET_ITEM(tuple.ptr(), py::ssize_t_cast(index)));
}
template <>
inline py::object GET_ITEM_BORROW<py::tuple>(const py::handle& tuple, const int& index) {
    return py::reinterpret_borrow<py::object>(
        PyTuple_GET_ITEM(tuple.ptr(), py::ssize_t_cast(index)));
}
template <>
inline py::object GET_ITEM_BORROW<py::list>(const py::handle& list, const ssize_t& index) {
    return py::reinterpret_borrow<py::object>(PyList_GET_ITEM(list.ptr(), index));
}
template <>
inline py::object GET_ITEM_BORROW<py::list>(const py::handle& list, const size_t& index) {
    return py::reinterpret_borrow<py::object>(PyList_GET_ITEM(list.ptr(), py::ssize_t_cast(index)));
}
template <>
inline py::object GET_ITEM_BORROW<py::list>(const py::handle& list, const int& index) {
    return py::reinterpret_borrow<py::object>(PyList_GET_ITEM(list.ptr(), py::ssize_t_cast(index)));
}

template <typename Container, typename Item>
inline py::object GET_ITEM_STEAL(const py::handle& container, const Item& item) {
    return container[item];
}
template <>
inline py::object GET_ITEM_STEAL<py::tuple>(const py::handle& tuple, const ssize_t& index) {
    return py::reinterpret_steal<py::object>(PyTuple_GET_ITEM(tuple.ptr(), index));
}
template <>
inline py::object GET_ITEM_STEAL<py::tuple>(const py::handle& tuple, const size_t& index) {
    return py::reinterpret_steal<py::object>(
        PyTuple_GET_ITEM(tuple.ptr(), py::ssize_t_cast(index)));
}
template <>
inline py::object GET_ITEM_STEAL<py::tuple>(const py::handle& tuple, const int& index) {
    return py::reinterpret_steal<py::object>(
        PyTuple_GET_ITEM(tuple.ptr(), py::ssize_t_cast(index)));
}
template <>
inline py::object GET_ITEM_STEAL<py::list>(const py::handle& list, const ssize_t& index) {
    return py::reinterpret_steal<py::object>(PyList_GET_ITEM(list.ptr(), index));
}
template <>
inline py::object GET_ITEM_STEAL<py::list>(const py::handle& list, const size_t& index) {
    return py::reinterpret_steal<py::object>(PyList_GET_ITEM(list.ptr(), py::ssize_t_cast(index)));
}
template <>
inline py::object GET_ITEM_STEAL<py::list>(const py::handle& list, const int& index) {
    return py::reinterpret_steal<py::object>(PyList_GET_ITEM(list.ptr(), py::ssize_t_cast(index)));
}

template <typename Container, typename Item>
inline void SET_ITEM(const py::handle& container, const Item& item, const py::handle& value) {
    container[item] = value;
}
template <>
inline void SET_ITEM<py::tuple>(const py::handle& tuple,
                                const ssize_t& index,
                                const py::handle& value) {
    PyTuple_SET_ITEM(tuple.ptr(), index, value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::tuple>(const py::handle& tuple,
                                const size_t& index,
                                const py::handle& value) {
    PyTuple_SET_ITEM(tuple.ptr(), py::ssize_t_cast(index), value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::tuple>(const py::handle& tuple,
                                const int& index,
                                const py::handle& value) {
    PyTuple_SET_ITEM(tuple.ptr(), py::ssize_t_cast(index), value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::list>(const py::handle& list,
                               const ssize_t& index,
                               const py::handle& value) {
    PyList_SET_ITEM(list.ptr(), index, value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::list>(const py::handle& list,
                               const size_t& index,
                               const py::handle& value) {
    PyList_SET_ITEM(list.ptr(), py::ssize_t_cast(index), value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::list>(const py::handle& list, const int& index, const py::handle& value) {
    PyList_SET_ITEM(list.ptr(), py::ssize_t_cast(index), value.inc_ref().ptr());
}

template <typename PyType>
inline void AssertExact(const py::handle& object) {
    if (!py::isinstance<PyType>(object)) {
        throw std::runtime_error(absl::StrFormat(
            "Expected an instance of %s, got %s.", typeid(PyType).name(), py::repr(object)));
    }
}
template <>
inline void AssertExact<py::list>(const py::handle& object) {
    if (!PyList_CheckExact(object.ptr())) {
        throw std::invalid_argument(absl::StrFormat("Expected list, got %s.", py::repr(object)));
    }
}
template <>
inline void AssertExact<py::tuple>(const py::handle& object) {
    if (!PyTuple_CheckExact(object.ptr())) {
        throw std::invalid_argument(absl::StrFormat("Expected tuple, got %s.", py::repr(object)));
    }
}
template <>
inline void AssertExact<py::dict>(const py::handle& object) {
    if (!PyDict_CheckExact(object.ptr())) {
        throw std::invalid_argument(absl::StrFormat("Expected dict, got %s.", py::repr(object)));
    }
}

inline bool IsNamedTuple(const py::handle& object) {
    // We can only identify namedtuples heuristically, here by the presence of a _fields attribute.
    return PyTuple_Check(object.ptr()) && PyObject_HasAttrString(object.ptr(), "_fields") == 1;
}

inline void AssertExactNamedTuple(const py::handle& object) {
    if (!IsNamedTuple(object)) {
        throw std::invalid_argument(
            absl::StrFormat("Expected named tuple, got %s.", py::repr(object)));
    }
}
