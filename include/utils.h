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

#pragma once

#include <Python.h>

#if PY_VERSION_HEX < 0x30C00F0  // Python 3.12.0
#include <structmember.h>       // PyMemberDef
#endif

#include <pybind11/pybind11.h>

#include <exception>   // std::rethrow_exception, std::current_exception
#include <functional>  // std::hash
#include <sstream>     // std::ostringstream
#include <string>      // std::string
#include <utility>     // std::move, std::pair, std::make_pair
#include <vector>      // std::vector

namespace py = pybind11;
using size_t = py::size_t;
using ssize_t = py::ssize_t;

// boost::hash_combine
template <class T>
inline void HashCombine(py::size_t& seed, const T& v) {  // NOLINT[runtime/references]
    std::hash<T> hasher{};
    // NOLINTNEXTLINE[cppcoreguidelines-avoid-magic-numbers]
    seed ^= (hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
template <class T>
inline void HashCombine(py::ssize_t& seed, const T& v) {  // NOLINT[runtime/references]
    std::hash<T> hasher{};
    // NOLINTNEXTLINE[cppcoreguidelines-avoid-magic-numbers]
    seed ^= (hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

constexpr bool NONE_IS_LEAF = true;
constexpr bool NONE_IS_NODE = false;

// NOTE: Use raw pointers to leak the memory intentionally to avoid py::object deallocation and
//       garbage collection.
#define Py_Declare_ID(name)                                 \
    inline PyObject* Py_ID_##name() {                       \
        static PyObject* ptr = (new py::str{#name})->ptr(); \
        return ptr;                                         \
    }

#define Py_Get_ID(name) (Py_ID_##name())

Py_Declare_ID(__module__);         // type.__module__
Py_Declare_ID(__qualname__);       // type.__qualname__
Py_Declare_ID(__name__);           // type.__name__
Py_Declare_ID(sort);               // list.sort
Py_Declare_ID(copy);               // dict.copy
Py_Declare_ID(default_factory);    // defaultdict.default_factory
Py_Declare_ID(maxlen);             // deque.maxlen
Py_Declare_ID(_fields);            // namedtuple._fields
Py_Declare_ID(_make);              // namedtuple._make
Py_Declare_ID(_asdict);            // namedtuple._asdict
Py_Declare_ID(n_fields);           // structseq.n_fields
Py_Declare_ID(n_sequence_fields);  // structseq.n_sequence_fields
Py_Declare_ID(n_unnamed_fields);   // structseq.n_unnamed_fields

#define PyCollectionsModule (ImportCollections())
#define PyOrderedDictTypeObject (ImportOrderedDict())
#define PyDefaultDictTypeObject (ImportDefaultDict())
#define PyDequeTypeObject (ImportDeque())
#define PyOrderedDict_Type (reinterpret_cast<PyTypeObject*>(PyOrderedDictTypeObject.ptr()))
#define PyDefaultDict_Type (reinterpret_cast<PyTypeObject*>(PyDefaultDictTypeObject.ptr()))
#define PyDeque_Type (reinterpret_cast<PyTypeObject*>(PyDequeTypeObject.ptr()))

inline const py::module_& ImportCollections() {
    // NOTE: Use raw pointers to leak the memory intentionally to avoid py::object deallocation and
    //       garbage collection.
    static const py::module_* ptr = new py::module_{py::module_::import("collections")};
    return *ptr;
}
inline const py::object& ImportOrderedDict() {
    // NOTE: Use raw pointers to leak the memory intentionally to avoid py::object deallocation and
    //       garbage collection.
    static const py::object* ptr = new py::object{py::getattr(PyCollectionsModule, "OrderedDict")};
    return *ptr;
}
inline const py::object& ImportDefaultDict() {
    // NOTE: Use raw pointers to leak the memory intentionally to avoid py::object deallocation and
    //       garbage collection.
    static const py::object* ptr = new py::object{py::getattr(PyCollectionsModule, "defaultdict")};
    return *ptr;
}
inline const py::object& ImportDeque() {
    // NOTE: Use raw pointers to leak the memory intentionally to avoid py::object deallocation and
    //       garbage collection.
    static const py::object* ptr = new py::object{py::getattr(PyCollectionsModule, "deque")};
    return *ptr;
}

template <typename T>
inline std::vector<T> reserved_vector(const py::size_t& size) {
    std::vector<T> v{};
    v.reserve(size);
    return v;
}

template <typename Sized = py::object>
inline ssize_t GetSize(const py::handle& sized) {
    return py::ssize_t_cast(py::len(sized));
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
    return py::ssize_t_cast(py::len(sized));
}
template <>
inline ssize_t GET_SIZE<py::tuple>(const py::handle& sized) {
    return PyTuple_GET_SIZE(sized.ptr());
}
template <>
inline ssize_t GET_SIZE<py::list>(const py::handle& sized) {
    return PyList_GET_SIZE(sized.ptr());
}
template <>
inline ssize_t GET_SIZE<py::dict>(const py::handle& sized) {
    return PyDict_GET_SIZE(sized.ptr());
}

template <typename Container, typename Item>
inline py::handle GET_ITEM_HANDLE(const py::handle& container, const Item& item) {
    return container[item];
}
template <>
inline py::handle GET_ITEM_HANDLE<py::tuple>(const py::handle& container, const py::ssize_t& item) {
    return PyTuple_GET_ITEM(container.ptr(), item);
}
template <>
inline py::handle GET_ITEM_HANDLE<py::tuple>(const py::handle& container, const py::size_t& item) {
    return PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item));
}
template <>
inline py::handle GET_ITEM_HANDLE<py::tuple>(const py::handle& container, const int& item) {
    return PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item));
}
template <>
inline py::handle GET_ITEM_HANDLE<py::list>(const py::handle& container, const py::ssize_t& item) {
    return PyList_GET_ITEM(container.ptr(), item);
}
template <>
inline py::handle GET_ITEM_HANDLE<py::list>(const py::handle& container, const py::size_t& item) {
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
inline py::object GET_ITEM_BORROW<py::tuple>(const py::handle& container, const py::ssize_t& item) {
    return py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(container.ptr(), item));
}
template <>
inline py::object GET_ITEM_BORROW<py::tuple>(const py::handle& container, const py::size_t& item) {
    return py::reinterpret_borrow<py::object>(
        PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_BORROW<py::tuple>(const py::handle& container, const int& item) {
    return py::reinterpret_borrow<py::object>(
        PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_BORROW<py::list>(const py::handle& container, const py::ssize_t& item) {
    return py::reinterpret_borrow<py::object>(PyList_GET_ITEM(container.ptr(), item));
}
template <>
inline py::object GET_ITEM_BORROW<py::list>(const py::handle& container, const py::size_t& item) {
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
inline py::object GET_ITEM_STEAL<py::tuple>(const py::handle& container, const py::ssize_t& item) {
    return py::reinterpret_steal<py::object>(PyTuple_GET_ITEM(container.ptr(), item));
}
template <>
inline py::object GET_ITEM_STEAL<py::tuple>(const py::handle& container, const py::size_t& item) {
    return py::reinterpret_steal<py::object>(
        PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_STEAL<py::tuple>(const py::handle& container, const int& item) {
    return py::reinterpret_steal<py::object>(
        PyTuple_GET_ITEM(container.ptr(), py::ssize_t_cast(item)));
}
template <>
inline py::object GET_ITEM_STEAL<py::list>(const py::handle& container, const py::ssize_t& item) {
    return py::reinterpret_steal<py::object>(PyList_GET_ITEM(container.ptr(), item));
}
template <>
inline py::object GET_ITEM_STEAL<py::list>(const py::handle& container, const py::size_t& item) {
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
                                const py::size_t& item,
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
                               const py::size_t& item,
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
        std::ostringstream oss{};
        oss << "Expected an instance of " << typeid(PyType).name() << ", got "
            << static_cast<std::string>(py::repr(object)) << ".";
        throw py::value_error(oss.str());
    }
}
template <>
inline void AssertExact<py::list>(const py::handle& object) {
    if (!PyList_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of list, got " +
                              static_cast<std::string>(py::repr(object)) + ".");
    }
}
template <>
inline void AssertExact<py::tuple>(const py::handle& object) {
    if (!PyTuple_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of tuple, got " +
                              static_cast<std::string>(py::repr(object)) + ".");
    }
}
template <>
inline void AssertExact<py::dict>(const py::handle& object) {
    if (!PyDict_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of dict, got " +
                              static_cast<std::string>(py::repr(object)) + ".");
    }
}

inline void AssertExactOrderedDict(const py::handle& object) {
    if (!py::type::handle_of(object).is(PyOrderedDictTypeObject)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.OrderedDict, got " +
                              static_cast<std::string>(py::repr(object)) + ".");
    }
}

inline void AssertExactDefaultDict(const py::handle& object) {
    if (!py::type::handle_of(object).is(PyDefaultDictTypeObject)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.defaultdict, got " +
                              static_cast<std::string>(py::repr(object)) + ".");
    }
}

inline void AssertExactStandardDict(const py::handle& object) {
    if (!(PyDict_CheckExact(object.ptr()) ||
          py::type::handle_of(object).is(PyOrderedDictTypeObject) ||
          py::type::handle_of(object).is(PyDefaultDictTypeObject))) [[unlikely]] {
        throw py::value_error(
            "Expected an instance of "
            "dict, collections.OrderedDict, or collections.defaultdict, "
            "got " +
            static_cast<std::string>(py::repr(object)) + ".");
    }
}

inline void AssertExactDeque(const py::handle& object) {
    if (!py::type::handle_of(object).is(PyDequeTypeObject)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.deque, got " +
                              static_cast<std::string>(py::repr(object)) + ".");
    }
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
inline bool IsNamedTupleClassImpl(const py::handle& type) {
    // We can only identify namedtuples heuristically, here by the presence of a _fields attribute.
    if (PyType_FastSubclass(reinterpret_cast<PyTypeObject*>(type.ptr()), Py_TPFLAGS_TUPLE_SUBCLASS))
        [[unlikely]] {
        if (PyObject* _fields = PyObject_GetAttr(type.ptr(), Py_Get_ID(_fields))) [[unlikely]] {
            bool fields_ok = static_cast<bool>(PyTuple_CheckExact(_fields));
            if (fields_ok) [[likely]] {
                for (const auto& field : py::reinterpret_borrow<py::tuple>(_fields)) {
                    if (!static_cast<bool>(PyUnicode_CheckExact(field.ptr()))) [[unlikely]] {
                        fields_ok = false;
                        break;
                    }
                }
            }
            Py_DECREF(_fields);
            if (fields_ok) [[likely]] {
                // NOLINTNEXTLINE[readability-use-anyofallof]
                for (PyObject* name : {Py_Get_ID(_make), Py_Get_ID(_asdict)}) {
                    if (PyObject* attr = PyObject_GetAttr(type.ptr(), name)) [[likely]] {
                        bool result = static_cast<bool>(PyCallable_Check(attr));
                        Py_DECREF(attr);
                        if (!result) [[unlikely]] {
                            return false;
                        }
                    } else [[unlikely]] {
                        PyErr_Clear();
                        return false;
                    }
                }
                return true;
            }
        } else [[likely]] {
            PyErr_Clear();
        }
    }
    return false;
}
inline bool IsNamedTupleClass(const py::handle& type) {
    return PyType_Check(type.ptr()) && IsNamedTupleClassImpl(type);
}
inline bool IsNamedTupleInstance(const py::handle& object) {
    return IsNamedTupleClass(py::type::handle_of(object));
}
inline bool IsNamedTuple(const py::handle& object) {
    py::handle type = (PyType_Check(object.ptr()) ? object : py::type::handle_of(object));
    return IsNamedTupleClass(type);
}
inline void AssertExactNamedTuple(const py::handle& object) {
    if (!IsNamedTupleInstance(object)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.namedtuple, got " +
                              static_cast<std::string>(py::repr(object)) + ".");
    }
}
inline py::tuple NamedTupleGetFields(const py::handle& object) {
    py::handle type;
    if (PyType_Check(object.ptr())) [[unlikely]] {
        type = object;
        if (!IsNamedTupleClass(type)) [[unlikely]] {
            throw py::type_error("Expected a collections.namedtuple type, got " +
                                 static_cast<std::string>(py::repr(object)) + ".");
        }
    } else [[likely]] {
        type = py::type::handle_of(object);
        if (!IsNamedTupleClass(type)) [[unlikely]] {
            throw py::type_error("Expected an instance of collections.namedtuple type, got " +
                                 static_cast<std::string>(py::repr(object)) + ".");
        }
    }
    return py::getattr(type, Py_Get_ID(_fields));
}

inline bool IsStructSequenceClassImpl(const py::handle& type) {
    // We can only identify PyStructSequences heuristically, here by the presence of
    // n_fields, n_sequence_fields, n_unnamed_fields attributes.
    auto* type_object = reinterpret_cast<PyTypeObject*>(type.ptr());
    if (type_object->tp_bases != nullptr &&
        static_cast<bool>(PyTuple_CheckExact(type_object->tp_bases)) &&
        PyTuple_GET_SIZE(type_object->tp_bases) == 1 &&
        PyTuple_GET_ITEM(type_object->tp_bases, 0) == reinterpret_cast<PyObject*>(&PyTuple_Type) &&
        !static_cast<bool>(PyType_HasFeature(type_object, Py_TPFLAGS_BASETYPE))) [[unlikely]] {
        // NOLINTNEXTLINE[readability-use-anyofallof]
        for (PyObject* name :
             {Py_Get_ID(n_fields), Py_Get_ID(n_sequence_fields), Py_Get_ID(n_unnamed_fields)}) {
            if (PyObject* attr = PyObject_GetAttr(type.ptr(), name)) [[unlikely]] {
                bool result = static_cast<bool>(PyLong_CheckExact(attr));
                Py_DECREF(attr);
                if (!result) [[unlikely]] {
                    return false;
                }
            } else [[likely]] {
                PyErr_Clear();
                return false;
            }
        }
        return true;
    }
    return false;
}
inline bool IsStructSequenceClass(const py::handle& type) {
    return PyType_Check(type.ptr()) && IsStructSequenceClassImpl(type);
}
inline bool IsStructSequenceInstance(const py::handle& object) {
    return IsStructSequenceClass(py::type::handle_of(object));
}
inline bool IsStructSequence(const py::handle& object) {
    py::handle type = (PyType_Check(object.ptr()) ? object : py::type::handle_of(object));
    return IsStructSequenceClass(type);
}
inline void AssertExactStructSequence(const py::handle& object) {
    if (!IsStructSequenceInstance(object)) [[unlikely]] {
        throw py::value_error("Expected an instance of PyStructSequence type, got " +
                              static_cast<std::string>(py::repr(object)) + ".");
    }
}
inline py::tuple StructSequenceGetFields(const py::handle& object) {
    py::handle type;
    if (PyType_Check(object.ptr())) [[unlikely]] {
        type = object;
        if (!IsStructSequenceClass(type)) [[unlikely]] {
            throw py::type_error("Expected a PyStructSequence type, got " +
                                 static_cast<std::string>(py::repr(object)) + ".");
        }
    } else [[likely]] {
        type = py::type::handle_of(object);
        if (!IsStructSequenceClass(type)) [[unlikely]] {
            throw py::type_error("Expected an instance of PyStructSequence type, got " +
                                 static_cast<std::string>(py::repr(object)) + ".");
        }
    }

    const auto n_sequence_fields = getattr(type, Py_Get_ID(n_sequence_fields)).cast<ssize_t>();
    auto* members = reinterpret_cast<PyTypeObject*>(type.ptr())->tp_members;
    py::tuple fields{n_sequence_fields};
    for (ssize_t i = 0; i < n_sequence_fields; ++i) {
        // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
        SET_ITEM<py::tuple>(fields, i, py::str(members[i].name));
    }
    return fields;
}

inline void TotalOrderSort(py::list& list) {  // NOLINT[runtime/references]
    try {
        // Sort directly if possible.
        // NOLINTNEXTLINE[readability-implicit-bool-conversion]
        if (PyList_Sort(list.ptr())) [[unlikely]] {
            throw py::error_already_set();
        }
    } catch (py::error_already_set& ex1) {
        if (ex1.matches(PyExc_TypeError)) [[likely]] {
            // Found incomparable keys (e.g. `int` vs. `str`, or user-defined types).
            try {
                // Sort with `(f'{o.__class__.__module__}.{o.__class__.__qualname__}', o)`
                auto sort_key_fn = py::cpp_function([](const py::object& o) {
                    py::handle t = py::type::handle_of(o);
                    py::str qualname{static_cast<std::string>(
                                         py::getattr(t, Py_Get_ID(__module__)).cast<py::str>()) +
                                     "." +
                                     static_cast<std::string>(
                                         py::getattr(t, Py_Get_ID(__qualname__)).cast<py::str>())};
                    return py::make_tuple(qualname, o);
                });
                py::getattr(list, Py_Get_ID(sort))(py::arg("key") = sort_key_fn);
            } catch (py::error_already_set& ex2) {
                if (ex2.matches(PyExc_TypeError)) [[likely]] {
                    // Found incomparable user-defined key types.
                    // The keys remain in the insertion order.
                    PyErr_Clear();
                } else [[unlikely]] {
                    std::rethrow_exception(std::current_exception());
                }
            }
        } else [[unlikely]] {
            std::rethrow_exception(std::current_exception());
        }
    }
}

inline py::list DictKeys(const py::dict& dict) {
    return py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
}

inline py::list SortedDictKeys(const py::dict& dict) {
    py::list keys = DictKeys(dict);
    TotalOrderSort(keys);
    return keys;
}

inline bool DictKeysEqual(const py::list& /* unique */ keys, const py::dict& dict) {
    ssize_t list_len = GET_SIZE<py::list>(keys);
    ssize_t dict_len = GET_SIZE<py::dict>(dict);
    if (list_len != dict_len) [[likely]] {  // assumes keys are unique
        return false;
    }
    for (ssize_t i = 0; i < list_len; ++i) {
        py::object key = GET_ITEM_BORROW<py::list>(keys, i);
        int result = PyDict_Contains(dict.ptr(), key.ptr());
        if (result == -1) [[unlikely]] {
            throw py::error_already_set();
        }
        if (result == 0) [[likely]] {
            return false;
        }
    }
    return true;
}

inline std::pair<py::list, py::list> DictKeysDifference(const py::list& /* unique */ keys,
                                                        const py::dict& dict) {
    py::set expected_keys{keys};
    py::set got_keys{DictKeys(dict)};
    py::list missing_keys{expected_keys - got_keys};
    py::list extra_keys{got_keys - expected_keys};
    TotalOrderSort(missing_keys);
    TotalOrderSort(extra_keys);
    return std::make_pair(std::move(missing_keys), std::move(extra_keys));
}
