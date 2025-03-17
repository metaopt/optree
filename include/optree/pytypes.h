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

#pragma once

#include <exception>      // std::rethrow_exception, std::current_exception
#include <string>         // std::string
#include <type_traits>    // std::enable_if_t, std::is_base_of_v
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::move, std::pair, std::make_pair

#include <Python.h>

#if PY_VERSION_HEX < 0x030C00F0  // Python 3.12.0
#include <structmember.h>        // PyMemberDef
#endif

#include <pybind11/eval.h>  // pybind11::exec
#include <pybind11/pybind11.h>

#include "optree/hashing.h"
#include "optree/pymacros.h"
#include "optree/synchronization.h"

namespace py = pybind11;

inline Py_ALWAYS_INLINE std::string PyStr(const py::handle& object) {
    return EVALUATE_WITH_LOCK_HELD(static_cast<std::string>(py::str(object)), object);
}
inline Py_ALWAYS_INLINE std::string PyStr(const std::string& string) { return string; }
inline Py_ALWAYS_INLINE std::string PyRepr(const py::handle& object) {
    return EVALUATE_WITH_LOCK_HELD(static_cast<std::string>(py::repr(object)), object);
}
inline Py_ALWAYS_INLINE std::string PyRepr(const std::string& string) {
    return static_cast<std::string>(py::repr(py::str(string)));
}

// The maximum size of the type cache.
constexpr py::ssize_t MAX_TYPE_CACHE_SIZE = 4096;

#define PyNoneTypeObject                                                                           \
    (py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(Py_TYPE(Py_None))))
#define PyTupleTypeObject                                                                          \
    (py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyTuple_Type)))
#define PyListTypeObject                                                                           \
    (py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyList_Type)))
#define PyDictTypeObject                                                                           \
    (py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyDict_Type)))
#define PyOrderedDictTypeObject (ImportOrderedDict())
#define PyDefaultDictTypeObject (ImportDefaultDict())
#define PyDequeTypeObject (ImportDeque())
#define PyOrderedDict_Type (reinterpret_cast<PyTypeObject*>(PyOrderedDictTypeObject.ptr()))
#define PyDefaultDict_Type (reinterpret_cast<PyTypeObject*>(PyDefaultDictTypeObject.ptr()))
#define PyDeque_Type (reinterpret_cast<PyTypeObject*>(PyDequeTypeObject.ptr()))

inline const py::object& ImportOrderedDict() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
    return storage
        .call_once_and_store_result([]() -> py::object {
            return py::getattr(py::module_::import("collections"), "OrderedDict");
        })
        .get_stored();
}
inline const py::object& ImportDefaultDict() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
    return storage
        .call_once_and_store_result([]() -> py::object {
            return py::getattr(py::module_::import("collections"), "defaultdict");
        })
        .get_stored();
}
inline const py::object& ImportDeque() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
    return storage
        .call_once_and_store_result(
            []() -> py::object { return py::getattr(py::module_::import("collections"), "deque"); })
        .get_stored();
}

inline Py_ALWAYS_INLINE py::ssize_t TupleGetSize(const py::handle& tuple) {
    return PyTuple_GET_SIZE(tuple.ptr());
}
inline Py_ALWAYS_INLINE py::ssize_t ListGetSize(const py::handle& list) {
    return PyList_GET_SIZE(list.ptr());
}
inline Py_ALWAYS_INLINE py::ssize_t DictGetSize(const py::handle& dict) {
#ifdef PyDict_GET_SIZE
    return PyDict_GET_SIZE(dict.ptr());
#else
    return PyDict_Size(dict.ptr());
#endif
}

template <typename T, typename = std::enable_if_t<std::is_base_of_v<py::object, T>>>
inline Py_ALWAYS_INLINE T TupleGetItemAs(const py::handle& tuple, const py::ssize_t& index) {
    return py::reinterpret_borrow<T>(PyTuple_GET_ITEM(tuple.ptr(), index));
}
inline Py_ALWAYS_INLINE py::object TupleGetItem(const py::handle& tuple, const py::ssize_t& index) {
    return TupleGetItemAs<py::object>(tuple, index);
}
template <typename T, typename = std::enable_if_t<std::is_base_of_v<py::object, T>>>
inline Py_ALWAYS_INLINE T ListGetItemAs(const py::handle& list, const py::ssize_t& index) {
#if PY_VERSION_HEX >= 0x030D00A4  // Python 3.13.0a4
    PyObject* const item = PyList_GetItemRef(list.ptr(), index);
    if (item == nullptr) [[unlikely]] {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<T>(item);
#else
    return py::reinterpret_borrow<T>(PyList_GET_ITEM(list.ptr(), index));
#endif
}
inline Py_ALWAYS_INLINE py::object ListGetItem(const py::handle& list, const py::ssize_t& index) {
    return ListGetItemAs<py::object>(list, index);
}
template <typename T, typename = std::enable_if_t<std::is_base_of_v<py::object, T>>>
inline Py_ALWAYS_INLINE T DictGetItemAs(const py::handle& dict, const py::handle& key) {
#if PY_VERSION_HEX >= 0x030D00A1  // Python 3.13.0a1
    PyObject* value = nullptr;
    if (PyDict_GetItemRef(dict.ptr(), key.ptr(), &value) < 0) [[unlikely]] {
        throw py::error_already_set();
    }
    if (value == nullptr) [[unlikely]] {
        py::set_error(PyExc_KeyError, py::make_tuple(key));
        throw py::error_already_set();
    }
    return py::reinterpret_steal<T>(value);
#else
    return py::reinterpret_borrow<T>(PyDict_GetItem(dict.ptr(), key.ptr()));
#endif
}
inline Py_ALWAYS_INLINE py::object DictGetItem(const py::handle& dict, const py::handle& key) {
    return DictGetItemAs<py::object>(dict, key);
}

inline Py_ALWAYS_INLINE void TupleSetItem(const py::handle& tuple,
                                          const py::ssize_t& index,
                                          const py::handle& value) {
    PyTuple_SET_ITEM(tuple.ptr(), index, value.inc_ref().ptr());
}
inline Py_ALWAYS_INLINE void ListSetItem(const py::handle& list,
                                         const py::ssize_t& index,
                                         const py::handle& value) {
    PyList_SET_ITEM(list.ptr(), index, value.inc_ref().ptr());
}
inline Py_ALWAYS_INLINE void DictSetItem(const py::handle& dict,
                                         const py::handle& key,
                                         const py::handle& value) {
    if (PyDict_SetItem(dict.ptr(), key.ptr(), value.ptr()) < 0) [[unlikely]] {
        throw py::error_already_set();
    }
}

inline Py_ALWAYS_INLINE void AssertExactList(const py::handle& object) {
    if (!PyList_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of list, got " + PyRepr(object) + ".");
    }
}
inline Py_ALWAYS_INLINE void AssertExactTuple(const py::handle& object) {
    if (!PyTuple_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of tuple, got " + PyRepr(object) + ".");
    }
}
inline Py_ALWAYS_INLINE void AssertExactDict(const py::handle& object) {
    if (!PyDict_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of dict, got " + PyRepr(object) + ".");
    }
}

inline Py_ALWAYS_INLINE void AssertExactOrderedDict(const py::handle& object) {
    if (!py::type::handle_of(object).is(PyOrderedDictTypeObject)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.OrderedDict, got " +
                              PyRepr(object) + ".");
    }
}

inline Py_ALWAYS_INLINE void AssertExactDefaultDict(const py::handle& object) {
    if (!py::type::handle_of(object).is(PyDefaultDictTypeObject)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.defaultdict, got " +
                              PyRepr(object) + ".");
    }
}

inline Py_ALWAYS_INLINE void AssertExactStandardDict(const py::handle& object) {
    if (!(PyDict_CheckExact(object.ptr()) ||
          py::type::handle_of(object).is(PyOrderedDictTypeObject) ||
          py::type::handle_of(object).is(PyDefaultDictTypeObject))) [[unlikely]] {
        throw py::value_error(
            "Expected an instance of dict, collections.OrderedDict, or collections.defaultdict, "
            "got " +
            PyRepr(object) + ".");
    }
}

inline Py_ALWAYS_INLINE void AssertExactDeque(const py::handle& object) {
    if (!py::type::handle_of(object).is(PyDequeTypeObject)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.deque, got " + PyRepr(object) +
                              ".");
    }
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
inline bool IsNamedTupleClassImpl(const py::handle& type) {
    // We can only identify namedtuples heuristically, here by the presence of a _fields attribute.
    if (PyType_FastSubclass(reinterpret_cast<PyTypeObject*>(type.ptr()), Py_TPFLAGS_TUPLE_SUBCLASS))
        [[unlikely]] {
        if (PyObject* const _fields = PyObject_GetAttr(type.ptr(), Py_Get_ID(_fields)))
            [[unlikely]] {
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
                for (PyObject* const name : {Py_Get_ID(_make), Py_Get_ID(_asdict)}) {
                    if (PyObject* const attr = PyObject_GetAttr(type.ptr(), name)) [[likely]] {
                        const bool result = static_cast<bool>(PyCallable_Check(attr));
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
    if (!PyType_Check(type.ptr())) [[unlikely]] {
        return false;
    }

    static auto cache = std::unordered_map<py::handle, bool>{};
    static read_write_mutex mutex{};

    {
        const scoped_read_lock_guard lock{mutex};
        const auto it = cache.find(type);
        if (it != cache.end()) [[likely]] {
            return it->second;
        }
    }

    const bool result = EVALUATE_WITH_LOCK_HELD(IsNamedTupleClassImpl(type), type);
    {
        const scoped_write_lock_guard lock{mutex};
        if (cache.size() < MAX_TYPE_CACHE_SIZE) [[likely]] {
            cache.emplace(type, result);
            (void)py::weakref(type, py::cpp_function([type](py::handle weakref) -> void {
                                  const scoped_write_lock_guard lock{mutex};
                                  cache.erase(type);
                                  weakref.dec_ref();
                              }))
                .release();
        }
    }
    return result;
}
inline Py_ALWAYS_INLINE bool IsNamedTupleInstance(const py::handle& object) {
    return IsNamedTupleClass(py::type::handle_of(object));
}
inline Py_ALWAYS_INLINE bool IsNamedTuple(const py::handle& object) {
    const py::handle type = (PyType_Check(object.ptr()) ? object : py::type::handle_of(object));
    return IsNamedTupleClass(type);
}
inline Py_ALWAYS_INLINE void AssertExactNamedTuple(const py::handle& object) {
    if (!IsNamedTupleInstance(object)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.namedtuple, got " +
                              PyRepr(object) + ".");
    }
}
inline py::tuple NamedTupleGetFields(const py::handle& object) {
    py::handle type;
    if (PyType_Check(object.ptr())) [[unlikely]] {
        type = object;
        if (!IsNamedTupleClass(type)) [[unlikely]] {
            throw py::type_error("Expected a collections.namedtuple type, got " + PyRepr(object) +
                                 ".");
        }
    } else [[likely]] {
        type = py::type::handle_of(object);
        if (!IsNamedTupleClass(type)) [[unlikely]] {
            throw py::type_error("Expected an instance of collections.namedtuple type, got " +
                                 PyRepr(object) + ".");
        }
    }
    return EVALUATE_WITH_LOCK_HELD(py::getattr(type, Py_Get_ID(_fields)), type);
}

inline bool IsStructSequenceClassImpl(const py::handle& type) {
    // We can only identify PyStructSequences heuristically, here by the presence of
    // n_fields, n_sequence_fields, n_unnamed_fields attributes.
    auto* const type_object = reinterpret_cast<PyTypeObject*>(type.ptr());
    if (PyType_FastSubclass(type_object, Py_TPFLAGS_TUPLE_SUBCLASS) &&
        type_object->tp_bases != nullptr &&
        static_cast<bool>(PyTuple_CheckExact(type_object->tp_bases)) &&
        PyTuple_GET_SIZE(type_object->tp_bases) == 1 &&
        PyTuple_GET_ITEM(type_object->tp_bases, 0) == reinterpret_cast<PyObject*>(&PyTuple_Type))
        [[unlikely]] {
        // NOLINTNEXTLINE[readability-use-anyofallof]
        for (PyObject* const name :
             {Py_Get_ID(n_fields), Py_Get_ID(n_sequence_fields), Py_Get_ID(n_unnamed_fields)}) {
            if (PyObject* const attr = PyObject_GetAttr(type.ptr(), name)) [[unlikely]] {
                const bool result = static_cast<bool>(PyLong_CheckExact(attr));
                Py_DECREF(attr);
                if (!result) [[unlikely]] {
                    return false;
                }
            } else [[likely]] {
                PyErr_Clear();
                return false;
            }
        }
#ifdef PYPY_VERSION
        try {
            py::exec("class _(cls): pass", py::dict(py::arg("cls") = type));
        } catch (py::error_already_set& ex) {
            if (ex.matches(PyExc_AssertionError) || ex.matches(PyExc_TypeError)) [[likely]] {
                PyErr_Clear();
                return true;
            }
            std::rethrow_exception(std::current_exception());
        }
        return false;
#else
        return !static_cast<bool>(PyType_HasFeature(type_object, Py_TPFLAGS_BASETYPE));
#endif
    }
    return false;
}
inline bool IsStructSequenceClass(const py::handle& type) {
    if (!PyType_Check(type.ptr())) [[unlikely]] {
        return false;
    }

    static auto cache = std::unordered_map<py::handle, bool>{};
    static read_write_mutex mutex{};

    {
        const scoped_read_lock_guard lock{mutex};
        const auto it = cache.find(type);
        if (it != cache.end()) [[likely]] {
            return it->second;
        }
    }

    const bool result = EVALUATE_WITH_LOCK_HELD(IsStructSequenceClassImpl(type), type);
    {
        const scoped_write_lock_guard lock{mutex};
        if (cache.size() < MAX_TYPE_CACHE_SIZE) [[likely]] {
            cache.emplace(type, result);
            (void)py::weakref(type, py::cpp_function([type](py::handle weakref) -> void {
                                  const scoped_write_lock_guard lock{mutex};
                                  cache.erase(type);
                                  weakref.dec_ref();
                              }))
                .release();
        }
    }
    return result;
}
inline Py_ALWAYS_INLINE bool IsStructSequenceInstance(const py::handle& object) {
    return IsStructSequenceClass(py::type::handle_of(object));
}
inline Py_ALWAYS_INLINE bool IsStructSequence(const py::handle& object) {
    const py::handle type = (PyType_Check(object.ptr()) ? object : py::type::handle_of(object));
    return IsStructSequenceClass(type);
}
inline Py_ALWAYS_INLINE void AssertExactStructSequence(const py::handle& object) {
    if (!IsStructSequenceInstance(object)) [[unlikely]] {
        throw py::value_error("Expected an instance of PyStructSequence type, got " +
                              PyRepr(object) + ".");
    }
}
inline py::tuple StructSequenceGetFieldsImpl(const py::handle& type) {
#ifdef PYPY_VERSION
    py::list fields{};
    py::exec(
        R"py(
        import sys

        StructSequenceFieldType = type(type(sys.version_info).major)
        indices_by_name = {
            name: member.index
            for name, member in vars(cls).items()
            if isinstance(member, StructSequenceFieldType)
        }
        fields.extend(sorted(indices_by_name, key=indices_by_name.get)[:cls.n_sequence_fields])
        )py",
        py::dict(py::arg("cls") = type, py::arg("fields") = fields));
    return py::tuple{fields};
#else
    const auto n_sequence_fields = thread_safe_cast<py::ssize_t>(
        EVALUATE_WITH_LOCK_HELD(py::getattr(type, Py_Get_ID(n_sequence_fields)), type));
    const auto* const members = reinterpret_cast<PyTypeObject*>(type.ptr())->tp_members;
    py::tuple fields{n_sequence_fields};
    for (py::ssize_t i = 0; i < n_sequence_fields; ++i) {
        // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
        TupleSetItem(fields, i, py::str(members[i].name));
    }
    return fields;
#endif
}
inline py::tuple StructSequenceGetFields(const py::handle& object) {
    py::handle type;
    if (PyType_Check(object.ptr())) [[unlikely]] {
        type = object;
        if (!IsStructSequenceClass(type)) [[unlikely]] {
            throw py::type_error("Expected a PyStructSequence type, got " + PyRepr(object) + ".");
        }
    } else [[likely]] {
        type = py::type::handle_of(object);
        if (!IsStructSequenceClass(type)) [[unlikely]] {
            throw py::type_error("Expected an instance of PyStructSequence type, got " +
                                 PyRepr(object) + ".");
        }
    }

    static auto cache = std::unordered_map<py::handle, py::handle>{};
    static read_write_mutex mutex{};

    {
        const scoped_read_lock_guard lock{mutex};
        const auto it = cache.find(type);
        if (it != cache.end()) [[likely]] {
            return py::reinterpret_borrow<py::tuple>(it->second);
        }
    }

    const py::tuple fields = EVALUATE_WITH_LOCK_HELD(StructSequenceGetFieldsImpl(type), type);
    {
        const scoped_write_lock_guard lock{mutex};
        if (cache.size() < MAX_TYPE_CACHE_SIZE) [[likely]] {
            cache.emplace(type, fields);
            fields.inc_ref();
            (void)py::weakref(type, py::cpp_function([type](py::handle weakref) -> void {
                                  const scoped_write_lock_guard lock{mutex};
                                  const auto it = cache.find(type);
                                  if (it != cache.end()) [[likely]] {
                                      it->second.dec_ref();
                                      cache.erase(it);
                                  }
                                  weakref.dec_ref();
                              }))
                .release();
        }
    }
    return fields;
}

inline void TotalOrderSort(py::list& list) {  // NOLINT[runtime/references]
    try {
        // Sort directly if possible.
        if (static_cast<bool>(EVALUATE_WITH_LOCK_HELD(PyList_Sort(list.ptr()), list)))
            [[unlikely]] {
            throw py::error_already_set();
        }
    } catch (py::error_already_set& ex1) {
        if (ex1.matches(PyExc_TypeError)) [[likely]] {
            // Found incomparable keys (e.g. `int` vs. `str`, or user-defined types).
            try {
                // Sort with `(f'{obj.__class__.__module__}.{obj.__class__.__qualname__}', obj)`
                const auto sort_key_fn = py::cpp_function([](const py::object& obj) -> py::tuple {
                    const py::handle cls = py::type::handle_of(obj);
                    const py::str qualname{EVALUATE_WITH_LOCK_HELD(
                        PyStr(py::getattr(cls, Py_Get_ID(__module__))) + "." +
                            PyStr(py::getattr(cls, Py_Get_ID(__qualname__))),
                        cls)};
                    return py::make_tuple(qualname, obj);
                });
                {
                    const scoped_critical_section cs{list};
                    py::getattr(list, Py_Get_ID(sort))(py::arg("key") = sort_key_fn);
                }
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

inline Py_ALWAYS_INLINE py::list DictKeys(const py::dict& dict) {
    const scoped_critical_section cs{dict};
    return py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
}

inline py::list SortedDictKeys(const py::dict& dict) {
    py::list keys = DictKeys(dict);
    TotalOrderSort(keys);
    return keys;
}

inline bool DictKeysEqual(const py::list& /*unique*/ keys, const py::dict& dict) {
    const scoped_critical_section2 cs{keys, dict};
    const py::ssize_t list_len = ListGetSize(keys);
    const py::ssize_t dict_len = DictGetSize(dict);
    if (list_len != dict_len) [[likely]] {  // assumes keys are unique
        return false;
    }
    for (py::ssize_t i = 0; i < list_len; ++i) {
        const py::object key = ListGetItem(keys, i);
        const int result = PyDict_Contains(dict.ptr(), key.ptr());
        if (result == -1) [[unlikely]] {
            throw py::error_already_set();
        }
        if (result == 0) [[likely]] {
            return false;
        }
    }
    return true;
}

inline std::pair<py::list, py::list> DictKeysDifference(const py::list& /*unique*/ keys,
                                                        const py::dict& dict) {
    const py::set expected_keys = EVALUATE_WITH_LOCK_HELD(py::set{keys}, keys);
    const py::set got_keys = EVALUATE_WITH_LOCK_HELD(py::set{dict}, dict);
    py::list missing_keys{expected_keys - got_keys};
    py::list extra_keys{got_keys - expected_keys};
    TotalOrderSort(missing_keys);
    TotalOrderSort(extra_keys);
    return std::make_pair(std::move(missing_keys), std::move(extra_keys));
}
