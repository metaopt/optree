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

#pragma once

#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef SOURCE_PATH_PREFIX_SIZE
#define SOURCE_PATH_PREFIX_SIZE 0
#endif
#ifndef __FILENAME__
#define __FILENAME__ (&(__FILE__[SOURCE_PATH_PREFIX_SIZE]))
#endif

#define VFUNC2(__0, __1, NAME, ...) NAME
#define VFUNC3(__0, __1, __2, NAME, ...) NAME

#define INTERNAL_ERROR1(message) \
    throw std::logic_error(absl::StrFormat("%s (at file %s:%lu)", message, __FILENAME__, __LINE__))
#define INTERNAL_ERROR0() INTERNAL_ERROR1("Unreachable code.")
#define INTERNAL_ERROR(...) /* NOLINTNEXTLINE[whitespace/parens] */ \
    VFUNC2(__0 __VA_OPT__(, ) __VA_ARGS__, INTERNAL_ERROR1, INTERNAL_ERROR0)(__VA_ARGS__)

#define EXPECT2(condition, message)  \
    if (!(condition)) [[unlikely]] { \
        INTERNAL_ERROR1(message);    \
    }
#define EXPECT0() INTERNAL_ERROR0()
#define EXPECT1(condition) EXPECT2(condition, "`" #condition "` failed.")
#define EXPECT(...) /* NOLINTNEXTLINE[whitespace/parens] */ \
    VFUNC3(__0 __VA_OPT__(, ) __VA_ARGS__, EXPECT2, EXPECT1, EXPECT0)(__VA_ARGS__)

#define EXPECT_EQ(a, b, ...) \
    EXPECT((a) == (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_NE(a, b, ...) \
    EXPECT((a) != (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_LT(a, b, ...) \
    EXPECT((a) < (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_LE(a, b, ...) \
    EXPECT((a) <= (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_GT(a, b, ...) \
    EXPECT((a) > (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_GE(a, b, ...) \
    EXPECT((a) >= (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]

#define NONE_IS_LEAF true
#define NONE_IS_NODE false

namespace py = pybind11;
using size_t = py::size_t;
using ssize_t = py::ssize_t;

#define PyCollectionsModule (ImportCollections())
#define PyOrderedDictTypeObject (ImportOrderedDict())
#define PyDefaultDictTypeObject (ImportDefaultDict())
#define PyDequeTypeObject (ImportDeque())

inline const py::module_& ImportCollections() {
    // NOTE: Use raw pointers to leak the memory intentionally to avoid py::object deallocation and
    // garbage collection
    static const py::module_* ptr = new py::module_{py::module_::import("collections")};
    return *ptr;
}
inline const py::object& ImportOrderedDict() {
    // NOTE: Use raw pointers to leak the memory intentionally to avoid py::object deallocation and
    // garbage collection
    static const py::object* ptr = new py::object{py::getattr(PyCollectionsModule, "OrderedDict")};
    return *ptr;
}
inline const py::object& ImportDefaultDict() {
    // NOTE: Use raw pointers to leak the memory intentionally to avoid py::object deallocation and
    // garbage collection
    static const py::object* ptr = new py::object{py::getattr(PyCollectionsModule, "defaultdict")};
    return *ptr;
}
inline const py::object& ImportDeque() {
    // NOTE: Use raw pointers to leak the memory intentionally to avoid py::object deallocation and
    // garbage collection
    static const py::object* ptr = new py::object{py::getattr(PyCollectionsModule, "deque")};
    return *ptr;
}

template <typename T>
inline std::vector<T> reserved_vector(const size_t& size) {
    std::vector<T> v;
    v.reserve(size);
    return v;
}

inline py::list DictKeys(const py::dict& dict) {
    return py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
}

inline py::list SortedDictKeys(const py::dict& dict) {
    py::list keys = DictKeys(dict);

    try {
        // Sort directly if possible.
        if (PyList_Sort(keys.ptr())) [[unlikely]] {
            throw py::error_already_set();
        }
    } catch (py::error_already_set& ex1) {
        if (ex1.matches(PyExc_TypeError)) [[likely]] {  // NOLINT
            // Found incomparable keys (e.g. `int` vs. `str`, or user-defined types).
            try {
                // Sort with `(f'{o.__class__.__module__}.{o.__class__.__qualname__}', o)`
                auto sort_key_fn = py::cpp_function([](const py::object& o) {
                    py::handle t = o.get_type();
                    py::str qualname{absl::StrFormat(
                        "%s.%s",
                        static_cast<std::string>(py::getattr(t, "__module__").cast<py::str>()),
                        static_cast<std::string>(py::getattr(t, "__qualname__").cast<py::str>()))};
                    return py::make_tuple(qualname, o);
                });
                keys.attr("sort")(py::arg("key") = sort_key_fn);
            } catch (py::error_already_set& ex2) {
                if (ex2.matches(PyExc_TypeError)) [[likely]] {
                    // Found incomparable user-defined key types.
                    // The keys remain in the insertion order.
                    PyErr_Clear();
                } else [[unlikely]] {  // NOLINT
                    throw;
                }
            }
        } else [[unlikely]] {  // NOLINT
            throw;
        }
    }

    return keys;
}

template <typename Sized = py::object>
inline ssize_t GetSize(const py::handle& sized) {
    return (ssize_t)py::len(sized);
}
template <>
inline ssize_t GetSize<py::tuple>(const py::handle& sized) {
    return PyTuple_Size(sized.ptr());
}
template <>
inline ssize_t GetSize<py::list>(const py::handle& sized) {
    return PyList_Size(sized.ptr());
}
template <>
inline ssize_t GetSize<py::dict>(const py::handle& sized) {
    return PyDict_Size(sized.ptr());
}

template <typename Sized = py::object>
inline ssize_t GET_SIZE(const py::handle& sized) {
    return (ssize_t)py::len(sized);
}
template <>
inline ssize_t GET_SIZE<py::tuple>(const py::handle& sized) {
    return PyTuple_GET_SIZE(sized.ptr());
}
template <>
inline ssize_t GET_SIZE<py::list>(const py::handle& sized) {
    return PyList_GET_SIZE(sized.ptr());
}
#ifdef PyDict_GET_SIZE
template <>
inline ssize_t GET_SIZE<py::dict>(const py::handle& sized) {
    return PyDict_GET_SIZE(sized.ptr());
}
#else
template <>
inline ssize_t GET_SIZE<py::dict>(const py::handle& sized) {
    return PyDict_Size(sized.ptr());
}
#endif

template <typename Container, typename Item>
inline py::handle GET_ITEM_HANDLE(const py::handle& container, const Item& item) {
    return container[item];
}
template <>
inline py::handle GET_ITEM_HANDLE<py::tuple>(const py::handle& container, const ssize_t& item) {
    return PyTuple_GET_ITEM(container.ptr(), item);
}
template <>
inline py::handle GET_ITEM_HANDLE<py::tuple>(const py::handle& container, const size_t& item) {
    return PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item));
}
template <>
inline py::handle GET_ITEM_HANDLE<py::tuple>(const py::handle& container, const int& item) {
    return PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item));
}
template <>
inline py::handle GET_ITEM_HANDLE<py::list>(const py::handle& container, const ssize_t& item) {
    return PyList_GET_ITEM(container.ptr(), item);
}
template <>
inline py::handle GET_ITEM_HANDLE<py::list>(const py::handle& container, const size_t& item) {
    return PyList_GET_ITEM(container.ptr(), py::ssize_t_cast(item));
}
template <>
inline py::handle GET_ITEM_HANDLE<py::list>(const py::handle& container, const int& item) {
    return PyList_GET_ITEM(container.ptr(), py::ssize_t_cast(item));
}

template <typename Container, typename Item>
inline py::object GET_ITEM_BORROW(const py::handle& container, const Item& item) {
    return container[item];
}
template <>
inline py::object GET_ITEM_BORROW<py::tuple>(const py::handle& container, const ssize_t& item) {
    return py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(container.ptr(), item));
}
template <>
inline py::object GET_ITEM_BORROW<py::tuple>(const py::handle& container, const size_t& item) {
    return py::reinterpret_borrow<py::object>(
        PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_BORROW<py::tuple>(const py::handle& container, const int& item) {
    return py::reinterpret_borrow<py::object>(
        PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_BORROW<py::list>(const py::handle& container, const ssize_t& item) {
    return py::reinterpret_borrow<py::object>(PyList_GET_ITEM(container.ptr(), item));
}
template <>
inline py::object GET_ITEM_BORROW<py::list>(const py::handle& container, const size_t& item) {
    return py::reinterpret_borrow<py::object>(
        PyList_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_BORROW<py::list>(const py::handle& container, const int& item) {
    return py::reinterpret_borrow<py::object>(
        PyList_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}

template <typename Container, typename Item>
inline py::object GET_ITEM_STEAL(const py::handle& container, const Item& item) {
    return container[item];
}
template <>
inline py::object GET_ITEM_STEAL<py::tuple>(const py::handle& container, const ssize_t& item) {
    return py::reinterpret_steal<py::object>(PyTuple_GET_ITEM(container.ptr(), item));
}
template <>
inline py::object GET_ITEM_STEAL<py::tuple>(const py::handle& container, const size_t& item) {
    return py::reinterpret_steal<py::object>(
        PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_STEAL<py::tuple>(const py::handle& container, const int& item) {
    return py::reinterpret_steal<py::object>(
        PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_STEAL<py::list>(const py::handle& container, const ssize_t& item) {
    return py::reinterpret_steal<py::object>(PyList_GET_ITEM(container.ptr(), item));
}
template <>
inline py::object GET_ITEM_STEAL<py::list>(const py::handle& container, const size_t& item) {
    return py::reinterpret_steal<py::object>(
        PyList_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_STEAL<py::list>(const py::handle& container, const int& item) {
    return py::reinterpret_steal<py::object>(
        PyList_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}

template <typename Container, typename Item>
inline void SET_ITEM(const py::handle& container, const Item& item, const py::handle& value) {
    container[item] = value;
}
template <>
inline void SET_ITEM<py::tuple>(const py::handle& container,
                                const ssize_t& item,
                                const py::handle& value) {
    PyTuple_SET_ITEM(container.ptr(), item, value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::tuple>(const py::handle& container,
                                const size_t& item,
                                const py::handle& value) {
    PyTuple_SET_ITEM(container.ptr(), py::ssize_t_cast(item), value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::tuple>(const py::handle& container,
                                const int& item,
                                const py::handle& value) {
    PyTuple_SET_ITEM(container.ptr(), py::ssize_t_cast(item), value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::list>(const py::handle& container,
                               const ssize_t& item,
                               const py::handle& value) {
    PyList_SET_ITEM(container.ptr(), item, value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::list>(const py::handle& container,
                               const size_t& item,
                               const py::handle& value) {
    PyList_SET_ITEM(container.ptr(), py::ssize_t_cast(item), value.inc_ref().ptr());
}
template <>
inline void SET_ITEM<py::list>(const py::handle& container,
                               const int& item,
                               const py::handle& value) {
    PyList_SET_ITEM(container.ptr(), py::ssize_t_cast(item), value.inc_ref().ptr());
}

template <typename PyType>
inline void AssertExact(const py::handle& object) {
    if (!py::isinstance<PyType>(object)) [[unlikely]] {
        throw std::invalid_argument(absl::StrFormat(
            "Expected an instance of %s, got %s.", typeid(PyType).name(), py::repr(object)));
    }
}
template <>
inline void AssertExact<py::list>(const py::handle& object) {
    if (!PyList_CheckExact(object.ptr())) [[unlikely]] {
        throw std::invalid_argument(absl::StrFormat("Expected list, got %s.", py::repr(object)));
    }
}
template <>
inline void AssertExact<py::tuple>(const py::handle& object) {
    if (!PyTuple_CheckExact(object.ptr())) [[unlikely]] {
        throw std::invalid_argument(absl::StrFormat("Expected tuple, got %s.", py::repr(object)));
    }
}
template <>
inline void AssertExact<py::dict>(const py::handle& object) {
    if (!PyDict_CheckExact(object.ptr())) [[unlikely]] {
        throw std::invalid_argument(absl::StrFormat("Expected dict, got %s.", py::repr(object)));
    }
}

inline bool IsNamedTuple(const py::handle& object) {
    // We can only identify namedtuples heuristically, here by the presence of a _fields attribute.
    return PyTuple_Check(object.ptr()) && PyObject_HasAttrString(object.ptr(), "_fields") == 1;
}

inline void AssertExactNamedTuple(const py::handle& object) {
    if (!IsNamedTuple(object)) [[unlikely]] {
        throw std::invalid_argument(
            absl::StrFormat("Expected collections.namedtuple, got %s.", py::repr(object)));
    }
}

inline void AssertExactOrderedDict(const py::handle& object) {
    if (!object.get_type().is(PyOrderedDictTypeObject)) [[unlikely]] {
        throw std::invalid_argument(
            absl::StrFormat("Expected collections.OrderedDict, got %s.", py::repr(object)));
    }
}

inline void AssertExactDefaultDict(const py::handle& object) {
    if (!object.get_type().is(PyDefaultDictTypeObject)) [[unlikely]] {
        throw std::invalid_argument(
            absl::StrFormat("Expected collections.defaultdict, got %s.", py::repr(object)));
    }
}

inline void AssertExactDeque(const py::handle& object) {
    if (!object.get_type().is(PyDequeTypeObject)) [[unlikely]] {
        throw std::invalid_argument(
            absl::StrFormat("Expected collections.deque, got %s.", py::repr(object)));
    }
}
