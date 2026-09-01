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

#pragma once

#include <cstddef>        // std::size_t, offsetof
#include <exception>      // std::rethrow_exception, std::current_exception
#include <optional>       // std::optional
#include <string>         // std::string
#include <type_traits>    // std::enable_if_t, std::is_same_v, std::is_base_of_v, std::conditional_t
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::forward, std::pair, std::make_pair, std::move
#include <vector>         // std::vector

#include <Python.h>

#if PY_VERSION_HEX < 0x030C00F0  // Python 3.12.0
#    include <structmember.h>    // PyMemberDef
#endif

#include <pybind11/eval.h>  // pybind11::exec
#include <pybind11/pybind11.h>

#include "optree/hashing.h"
#include "optree/pymacros.h"
#include "optree/synchronization.h"

namespace py = pybind11;

[[nodiscard]] inline Py_ALWAYS_INLINE std::string PyStr(const py::handle &object) {
    return EVALUATE_WITH_LOCK_HELD(static_cast<std::string>(py::str(object)), object);
}
[[nodiscard]] inline Py_ALWAYS_INLINE std::string PyStr(const std::string &string) {
    return string;
}
[[nodiscard]] inline Py_ALWAYS_INLINE std::string PyRepr(const py::handle &object) {
    return EVALUATE_WITH_LOCK_HELD(static_cast<std::string>(py::repr(object)), object);
}
[[nodiscard]] inline Py_ALWAYS_INLINE std::string PyRepr(const std::string &string) {
    return static_cast<std::string>(py::repr(py::str(string)));
}

#define PyNoneTypeObject                                                                           \
    (py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(Py_TYPE(Py_None))))
#define PyTupleTypeObject                                                                          \
    (py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(&PyTuple_Type)))
#define PyListTypeObject                                                                           \
    (py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(&PyList_Type)))
#define PyDictTypeObject                                                                           \
    (py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(&PyDict_Type)))
#define PyOrderedDictTypeObject (ImportOrderedDict())
#define PyDefaultDictTypeObject (ImportDefaultDict())
#define PyDequeTypeObject (ImportDeque())
#define PyOrderedDict_Type (reinterpret_cast<PyTypeObject *>(PyOrderedDictTypeObject.ptr()))
#define PyDefaultDict_Type (reinterpret_cast<PyTypeObject *>(PyDefaultDictTypeObject.ptr()))
#define PyDeque_Type (reinterpret_cast<PyTypeObject *>(PyDequeTypeObject.ptr()))
#if defined(OPTREE_HAS_FROZENDICT)
#    define PyFrozenDictTypeObject                                                                 \
        (py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(&PyFrozenDict_Type)))
#endif

[[nodiscard]] inline const py::object &ImportOrderedDict() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
    return storage
        .call_once_and_store_result([]() -> py::object {
            return py::getattr(py::module_::import("collections"), "OrderedDict");
        })
        .get_stored();
}
[[nodiscard]] inline const py::object &ImportDefaultDict() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
    return storage
        .call_once_and_store_result([]() -> py::object {
            return py::getattr(py::module_::import("collections"), "defaultdict");
        })
        .get_stored();
}
[[nodiscard]] inline const py::object &ImportDeque() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
    return storage
        .call_once_and_store_result(
            []() -> py::object { return py::getattr(py::module_::import("collections"), "deque"); })
        .get_stored();
}

[[nodiscard]] inline Py_ALWAYS_INLINE py::ssize_t TupleGetSize(const py::handle &tuple) {
    return PyTuple_GET_SIZE(tuple.ptr());
}
[[nodiscard]] inline Py_ALWAYS_INLINE py::ssize_t ListGetSize(const py::handle &list) {
    return PyList_GET_SIZE(list.ptr());
}
[[nodiscard]] inline Py_ALWAYS_INLINE py::ssize_t DictGetSize(const py::handle &dict) {
#if defined(PyDict_GET_SIZE)
    return PyDict_GET_SIZE(dict.ptr());
#else
    return PyDict_Size(dict.ptr());
#endif
}

template <typename T, typename = std::enable_if_t<std::is_base_of_v<py::object, T>>>
[[nodiscard]] inline Py_ALWAYS_INLINE T TupleGetItemAs(const py::handle &tuple,
                                                       const py::ssize_t &index) {
    return py::reinterpret_borrow<T>(PyTuple_GET_ITEM(tuple.ptr(), index));
}
[[nodiscard]] inline Py_ALWAYS_INLINE py::object TupleGetItem(const py::handle &tuple,
                                                              const py::ssize_t &index) {
    return TupleGetItemAs<py::object>(tuple, index);
}
template <typename T, typename = std::enable_if_t<std::is_base_of_v<py::object, T>>>
[[nodiscard]] inline Py_ALWAYS_INLINE T ListGetItemAs(const py::handle &list,
                                                      const py::ssize_t &index) {
#if PY_VERSION_HEX >= 0x030D00A4  // Python 3.13.0a4
    PyObject * const item = PyList_GetItemRef(list.ptr(), index);
    if (item == nullptr) [[unlikely]] {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<T>(item);
#else
    // Bounds-check like `PyList_GetItemRef` does: callers read the length once and then run user
    // code, so the list can shrink mid-loop and the unchecked macro would read out of bounds.
    if (index < 0 || index >= PyList_GET_SIZE(list.ptr())) [[unlikely]] {
        py::set_error(PyExc_IndexError, "list index out of range");
        throw py::error_already_set();
    }
    return py::reinterpret_borrow<T>(PyList_GET_ITEM(list.ptr(), index));
#endif
}
[[nodiscard]] inline Py_ALWAYS_INLINE py::object ListGetItem(const py::handle &list,
                                                             const py::ssize_t &index) {
    return ListGetItemAs<py::object>(list, index);
}
template <typename T, typename = std::enable_if_t<std::is_base_of_v<py::object, T>>>
[[nodiscard]] inline Py_ALWAYS_INLINE T DictGetItemAs(const py::handle &dict,
                                                      const py::handle &key) {
#if PY_VERSION_HEX >= 0x030D00A1  // Python 3.13.0a1
    PyObject *value = nullptr;
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
[[nodiscard]] inline Py_ALWAYS_INLINE py::object DictGetItem(const py::handle &dict,
                                                             const py::handle &key) {
    return DictGetItemAs<py::object>(dict, key);
}

inline Py_ALWAYS_INLINE void TupleSetItem(const py::handle &tuple,
                                          const py::ssize_t &index,
                                          const py::handle &value) {
    PyTuple_SET_ITEM(tuple.ptr(), index, value.inc_ref().ptr());
}
inline Py_ALWAYS_INLINE void ListSetItem(const py::handle &list,
                                         const py::ssize_t &index,
                                         const py::handle &value) {
    PyList_SET_ITEM(list.ptr(), index, value.inc_ref().ptr());
}
inline Py_ALWAYS_INLINE void DictSetItem(const py::handle &dict,
                                         const py::handle &key,
                                         const py::handle &value) {
    if (PyDict_SetItem(dict.ptr(), key.ptr(), value.ptr()) < 0) [[unlikely]] {
        throw py::error_already_set();
    }
}

// Shallow copies through the C API. A Python-level `.copy()` would cost an attribute lookup, a
// bound-method allocation and a vectorcall per call.
[[nodiscard]] inline py::list ListCopy(const py::handle &list) {
    const scoped_critical_section cs{list};
    auto copy = py::reinterpret_steal<py::list>(PyList_GetSlice(list.ptr(), 0, ListGetSize(list)));
    if (!copy) [[unlikely]] {
        throw py::error_already_set();
    }
    return copy;
}
[[nodiscard]] inline py::dict DictCopy(const py::handle &dict) {
    const scoped_critical_section cs{dict};
    auto copy = py::reinterpret_steal<py::dict>(PyDict_Copy(dict.ptr()));
    if (!copy) [[unlikely]] {
        throw py::error_already_set();
    }
    return copy;
}

inline Py_ALWAYS_INLINE void AssertExactList(const py::handle &object) {
    if (!PyList_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of list, got " + PyRepr(object) + ".");
    }
}
inline Py_ALWAYS_INLINE void AssertExactTuple(const py::handle &object) {
    if (!PyTuple_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of tuple, got " + PyRepr(object) + ".");
    }
}
inline Py_ALWAYS_INLINE void AssertExactDict(const py::handle &object) {
    if (!PyDict_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of dict, got " + PyRepr(object) + ".");
    }
}

#if defined(OPTREE_HAS_FROZENDICT)
inline Py_ALWAYS_INLINE void AssertExactFrozenDict(const py::handle &object) {
    if (!PyFrozenDict_CheckExact(object.ptr())) [[unlikely]] {
        throw py::value_error("Expected an instance of frozendict, got " + PyRepr(object) + ".");
    }
}
#endif

inline Py_ALWAYS_INLINE void AssertExactOrderedDict(const py::handle &object) {
    if (!py::type::handle_of(object).is(PyOrderedDictTypeObject)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.OrderedDict, got " +
                              PyRepr(object) + ".");
    }
}

inline Py_ALWAYS_INLINE void AssertExactDefaultDict(const py::handle &object) {
    if (!py::type::handle_of(object).is(PyDefaultDictTypeObject)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.defaultdict, got " +
                              PyRepr(object) + ".");
    }
}

inline Py_ALWAYS_INLINE void AssertExactStandardDict(const py::handle &object) {
    if (!(PyDict_CheckExact(object.ptr()) ||
#if defined(OPTREE_HAS_FROZENDICT)
          PyFrozenDict_CheckExact(object.ptr()) ||
#endif
          py::type::handle_of(object).is(PyOrderedDictTypeObject) ||
          py::type::handle_of(object).is(PyDefaultDictTypeObject))) [[unlikely]] {
        throw py::value_error(
            "Expected an instance of dict, "
#if defined(OPTREE_HAS_FROZENDICT)
            "frozendict, "
#endif
            "collections.OrderedDict, or collections.defaultdict, got " +
            PyRepr(object) + ".");
    }
}

inline Py_ALWAYS_INLINE void AssertExactDeque(const py::handle &object) {
    if (!py::type::handle_of(object).is(PyDequeTypeObject)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.deque, got " + PyRepr(object) +
                              ".");
    }
}

// A process-global cache mapping a Python object (in practice a type) to a value computed from it,
// e.g. whether a type is a namedtuple. It is a function-local static shared by every interpreter
// and outlives the Python runtime.
//
// Entries are keyed by `(interpreter_id, object address)` rather than the address alone, because a
// shared key can map to a per-interpreter-owned result: `int` is immortal and lives at the same
// address in every interpreter, while a computed `py::tuple` belongs to the interpreter that made
// it. An address-only key would hand that value to another interpreter to use after the owner frees
// it on finalization.
//
// A per-entry weakref evicts an entry when its key is collected, so a later key reusing that
// address cannot read a stale value. A per-interpreter `atexit` callback clears that interpreter's
// entries, covering what the weakref cannot: an immortal key is never collected, and interpreter
// ids restart from 0 after a `Py_Finalize`/`Py_Initialize` cycle, so a fresh interpreter must not
// inherit a finalized one's entries.
//
// `ValueType` may be a value such as `bool`, or a pybind11 reference whose entry owns one reference
// that is dropped on eviction.
template <typename ValueType>
class WeakKeyCache {
public:
    explicit WeakKeyCache(const std::size_t &max_size) : m_max_size{max_size} {}
    ~WeakKeyCache() = default;

    WeakKeyCache(const WeakKeyCache &) = delete;
    WeakKeyCache(WeakKeyCache &&) = delete;
    WeakKeyCache &operator=(const WeakKeyCache &) = delete;
    WeakKeyCache &operator=(WeakKeyCache &&) = delete;

    // Return the value cached for `key`, computing and inserting it via `compute` on a miss.
    // `compute` is a nullary callable returning `ValueType`, invoked with the GIL held and the
    // cache lock NOT held.
    template <typename Compute>
    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    [[nodiscard]] ValueType LookupOrInsert(const py::handle &key, Compute &&compute) {
        // Read the interpreter id (part of the cache key) while the GIL is still held, before the
        // read lock below releases it: `GetCurrentPyInterpreterID()` needs a valid thread state.
        const interpid_t interpreter_id = GetCurrentPyInterpreterID();
        const CacheKey cache_key{interpreter_id, key};
        std::optional<StoredType> cached_value{};
        {
#if !defined(Py_GIL_DISABLED)
            const py::gil_scoped_release_simple gil_release{};
#endif
            const scoped_read_lock lock{m_mutex};
            const auto it = m_cache.find(cache_key);
            if (it != m_cache.end()) [[likely]] {
                cached_value = it->second;
            }
        }
        // The read lock is released and the GIL re-acquired (in that destruction order) BEFORE the
        // borrowed object is touched, so the GIL is never (re-)acquired while the lock is held.
        // Doing so would invert the lock order against the weakref eviction callback (which holds
        // the GIL, then takes the write lock) and could deadlock. `key` stays alive for the whole
        // call, so its entry cannot be evicted and `cached_value` stays valid.
        if (cached_value.has_value()) [[likely]] {
            if constexpr (std::is_same_v<StoredType, ValueType>) {
                // A value or a `py::handle`: the stored type is the value type.
                return *cached_value;
            } else {
                // An owning object stored as a borrowed `py::handle`: return a fresh owning borrow.
                return py::reinterpret_borrow<ValueType>(*cached_value);
            }
        }

        ValueType value = std::forward<Compute>(compute)();

        // Register the per-interpreter cleanup BEFORE publishing an entry. It runs Python and can
        // raise, so the interpreter is claimed first and only marked done once it succeeds: marking
        // done first would leave the interpreter believing a callback exists and never retry it.
        bool claimed = false;
        bool registered = false;
        {
#if !defined(Py_GIL_DISABLED)
            const py::gil_scoped_release_simple gil_release{};
#endif
            const scoped_write_lock lock{m_mutex};
            const auto [it, inserted] = m_cleanup_registered.try_emplace(interpreter_id, false);
            claimed = inserted;
            registered = it->second;
        }
        if (claimed) [[unlikely]] {
            try {
                RegisterInterpreterCleanup(interpreter_id);
            } catch (...) {
                // Take the lock with the GIL held, the same order the eviction callback uses.
                const scoped_write_lock lock{m_mutex};
                m_cleanup_registered.erase(interpreter_id);
                throw;
            }

            {
                const scoped_write_lock lock{m_mutex};
                m_cleanup_registered[interpreter_id] = true;
                registered = true;
            }
        }

        // Publish only against a completed registration: publishing while another thread is still
        // registering would outlive the owner if that registration raises. Skipping is safe.
        bool inserted = false;
        if (registered) [[likely]] {
#if !defined(Py_GIL_DISABLED)
            const py::gil_scoped_release_simple gil_release{};
#endif
            const scoped_write_lock lock{m_mutex};
            if (m_cache.size() < m_max_size) [[likely]] {
                // The GIL is released here, so store the value without touching any refcount (a
                // reference value is stored as a borrowed `py::handle` and owned by the `inc_ref()`
                // below).
                inserted = m_cache.emplace(cache_key, StoredType{value}).second;
            }
        }
        if (inserted) [[likely]] {
            // The GIL is held here, so we can safely increment the reference count and create the
            // weakref. If the weakref cannot be created, drop the entry again: a published entry
            // whose value is unowned and whose key has no eviction callback would be read back
            // after the value is freed.
            if constexpr (kValueIsPyReference) {
                value.inc_ref();
            }
            try {
                (void)py::weakref(key,
                                  py::cpp_function([this, cache_key](py::handle weakref) -> void {
                                      const scoped_write_lock lock{m_mutex};
                                      const auto it = m_cache.find(cache_key);
                                      if (it != m_cache.end()) [[likely]] {
                                          if constexpr (kValueIsPyReference) {
                                              it->second.dec_ref();
                                          }
                                          m_cache.erase(it);
                                      }
                                      weakref.dec_ref();
                                  }))
                    .release();
            } catch (...) {
                {
                    const scoped_write_lock lock{m_mutex};
                    m_cache.erase(cache_key);
                }
                if constexpr (kValueIsPyReference) {
                    value.dec_ref();
                }
                throw;
            }
        }
        return value;
    }

private:
    static constexpr bool kValueIsPyReference = std::is_base_of_v<py::handle, ValueType>;
    using StoredType = std::conditional_t<kValueIsPyReference, py::handle, ValueType>;
    using CacheKey = std::pair<interpid_t, py::handle>;

    // Register (once per interpreter, with the GIL held and WITHOUT the cache lock, mirroring
    // `PyTreeTypeRegistry::Init`) an `atexit` callback that evicts this interpreter's entries on
    // shutdown.
    void RegisterInterpreterCleanup(const interpid_t &interpreter_id) {
        const auto atexit_register = py::getattr(py::module_::import("atexit"), "register");
        atexit_register(py::cpp_function([this, interpreter_id]() -> void {
            const scoped_write_lock lock{m_mutex};
            for (auto it = m_cache.begin(); it != m_cache.end();) {
                if (it->first.first == interpreter_id) [[likely]] {
                    if constexpr (kValueIsPyReference) {
                        it->second.dec_ref();
                    }
                    it = m_cache.erase(it);
                } else [[unlikely]] {
                    ++it;
                }
            }
            m_cleanup_registered.erase(interpreter_id);
        }));
    }

    std::unordered_map<CacheKey, StoredType> m_cache{};
    // Interpreter id -> whether its `atexit` callback is registered. An id maps to `false` while
    // the claiming thread is still registering it.
    std::unordered_map<interpid_t, bool> m_cleanup_registered{};
    const std::size_t m_max_size{};
    mutable read_write_mutex m_mutex{};
};

// The maximum size of a type cache.
constexpr std::size_t MAX_TYPE_CACHE_SIZE = 4096;

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] inline bool IsNamedTupleClassImpl(const py::handle &type) {
    // We can only identify namedtuples heuristically, here by the presence of a _fields attribute.
    if (PyType_FastSubclass(reinterpret_cast<PyTypeObject *>(type.ptr()),
                            Py_TPFLAGS_TUPLE_SUBCLASS)) [[unlikely]] {
        if (PyObject * const _fields = PyObject_GetAttrString(type.ptr(), "_fields")) [[unlikely]] {
            bool fields_ok = static_cast<bool>(PyTuple_CheckExact(_fields));
            if (fields_ok) [[likely]] {
                for (const auto &field : py::reinterpret_borrow<py::tuple>(_fields)) {
                    if (!static_cast<bool>(PyUnicode_CheckExact(field.ptr()))) [[unlikely]] {
                        fields_ok = false;
                        break;
                    }
                }
            }
            Py_DECREF(_fields);
            if (fields_ok) [[likely]] {
                // NOLINTNEXTLINE(readability-use-anyofallof)
                for (const char * const name : {"_make", "_asdict"}) {
                    if (PyObject * const attr = PyObject_GetAttrString(type.ptr(), name))
                        [[likely]] {
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
[[nodiscard]] inline bool IsNamedTupleClass(const py::handle &type) {
    if (!PyType_Check(type.ptr())) [[unlikely]] {
        return false;
    }

    static WeakKeyCache<bool> cache{MAX_TYPE_CACHE_SIZE};
    return cache.LookupOrInsert(type, [&type]() -> bool {
        return EVALUATE_WITH_LOCK_HELD(IsNamedTupleClassImpl(type), type);
    });
}
[[nodiscard]] inline Py_ALWAYS_INLINE bool IsNamedTupleInstance(const py::handle &object) {
    return IsNamedTupleClass(py::type::handle_of(object));
}
[[nodiscard]] inline Py_ALWAYS_INLINE bool IsNamedTuple(const py::handle &object) {
    const py::handle type = (PyType_Check(object.ptr()) ? object : py::type::handle_of(object));
    return IsNamedTupleClass(type);
}
inline Py_ALWAYS_INLINE void AssertExactNamedTuple(const py::handle &object) {
    if (!IsNamedTupleInstance(object)) [[unlikely]] {
        throw py::value_error("Expected an instance of collections.namedtuple, got " +
                              PyRepr(object) + ".");
    }
}
[[nodiscard]] inline py::tuple NamedTupleGetFields(const py::handle &object) {
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
    return EVALUATE_WITH_LOCK_HELD(py::getattr(type, "_fields"), type);
}

[[nodiscard]] inline bool IsStructSequenceClassImpl(const py::handle &type) {
    // We can only identify PyStructSequences heuristically, here by the presence of
    // n_fields, n_sequence_fields, n_unnamed_fields attributes.
    auto * const type_object = reinterpret_cast<PyTypeObject *>(type.ptr());
    if (PyType_FastSubclass(type_object, Py_TPFLAGS_TUPLE_SUBCLASS) &&
        type_object->tp_bases != nullptr &&
        static_cast<bool>(PyTuple_CheckExact(type_object->tp_bases)) &&
        PyTuple_GET_SIZE(type_object->tp_bases) == 1 &&
        PyTuple_GET_ITEM(type_object->tp_bases, 0) == reinterpret_cast<PyObject *>(&PyTuple_Type))
        [[unlikely]] {
        for (const char * const name : {"n_fields", "n_sequence_fields", "n_unnamed_fields"}) {
            if (PyObject * const attr = PyObject_GetAttrString(type.ptr(), name)) [[unlikely]] {
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
#if defined(PYPY_VERSION)
        try {
            py::exec("class _(cls): pass", py::dict(py::arg("cls") = type));
        } catch (py::error_already_set &ex) {
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
[[nodiscard]] inline bool IsStructSequenceClass(const py::handle &type) {
    if (!PyType_Check(type.ptr())) [[unlikely]] {
        return false;
    }

    static WeakKeyCache<bool> cache{MAX_TYPE_CACHE_SIZE};
    return cache.LookupOrInsert(type, [&type]() -> bool {
        return EVALUATE_WITH_LOCK_HELD(IsStructSequenceClassImpl(type), type);
    });
}
[[nodiscard]] inline Py_ALWAYS_INLINE bool IsStructSequenceInstance(const py::handle &object) {
    return IsStructSequenceClass(py::type::handle_of(object));
}
[[nodiscard]] inline Py_ALWAYS_INLINE bool IsStructSequence(const py::handle &object) {
    const py::handle type = (PyType_Check(object.ptr()) ? object : py::type::handle_of(object));
    return IsStructSequenceClass(type);
}
inline Py_ALWAYS_INLINE void AssertExactStructSequence(const py::handle &object) {
    if (!IsStructSequenceInstance(object)) [[unlikely]] {
        throw py::value_error("Expected an instance of PyStructSequence type, got " +
                              PyRepr(object) + ".");
    }
}
[[nodiscard]] inline py::tuple StructSequenceGetFieldsImpl(const py::handle &type) {
#if defined(PYPY_VERSION)
    py::list fields{};
    py::exec(
        R"py(
        import sys

        StructSequenceFieldType = type(type(sys.version_info).major)
        # PyPy has no unnamed fields; a descriptor's `.index` is its sequence position.
        # Map index -> name and defensively fill any missing (unnamed) slot with the marker.
        names_by_index = {
            member.index: name
            for name, member in vars(cls).items()
            if isinstance(member, StructSequenceFieldType)
        }
        fields.extend(
            names_by_index.get(index, unnamed_field) for index in range(cls.n_sequence_fields)
        )
        )py",
        py::dict(py::arg("cls") = type,
                 py::arg("fields") = fields,
                 py::arg("unnamed_field") = py::str(PyStructSequenceUnnamedField())));
    return py::tuple{fields};
#else
    const auto n_sequence_fields = thread_safe_cast<py::ssize_t>(
        EVALUATE_WITH_LOCK_HELD(py::getattr(type, "n_sequence_fields"), type));
    const auto * const members = reinterpret_cast<PyTypeObject *>(type.ptr())->tp_members;
    // `tp_members` lists only the NAMED fields, but each carries a byte `offset` encoding its
    // sequence index, relative to `offsetof(PyTupleObject, ob_item)`. Map each member back to its
    // slot by offset: indexing `members[i]` by position mislabels every slot after the first
    // unnamed one (e.g. `os.stat_result` slots 7/8/9 reported as `st_atime`/`st_mtime`/`st_ctime`).
    py::tuple fields{n_sequence_fields};
    // Fill the named slots first, then default the remaining (unnamed) slots to the marker.
    // Pre-filling every slot with the marker and overwriting the named ones would leak each
    // overwritten marker: `TupleSetItem` uses `PyTuple_SET_ITEM`, which does not decref what it
    // replaces.
    std::vector<bool> named(n_sequence_fields, false);
    for (const PyMemberDef *member = members; member != nullptr && member->name != nullptr;
         // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
         ++member) {
        const py::ssize_t index =
            (member->offset - py::ssize_t_cast(offsetof(PyTupleObject, ob_item))) /
            py::ssize_t_cast(sizeof(PyObject *));
        if (index >= 0 && index < n_sequence_fields) [[likely]] {
            TupleSetItem(fields, index, py::str(member->name));
            named[index] = true;
        }
    }
    for (py::ssize_t i = 0; i < n_sequence_fields; ++i) {
        if (!named[i]) [[unlikely]] {
            TupleSetItem(fields, i, py::str(PyStructSequenceUnnamedField()));
        }
    }
    return fields;
#endif
}
[[nodiscard]] inline py::tuple StructSequenceGetFields(const py::handle &object) {
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

    static WeakKeyCache<py::tuple> cache{MAX_TYPE_CACHE_SIZE};
    return cache.LookupOrInsert(type, [&type]() -> py::tuple {
        return EVALUATE_WITH_LOCK_HELD(StructSequenceGetFieldsImpl(type), type);
    });
}

// `list.sort()` leaves the list partially reordered when a comparison raises, so each attempt sorts
// a copy and only a fully sorted one is committed (mirrors `optree.utils.total_order_sorted`).
inline void TotalOrderSort(py::list &list) {
    py::list sorted = ListCopy(list);
    try {
        // Sort directly if possible.
        if (static_cast<bool>(EVALUATE_WITH_LOCK_HELD(PyList_Sort(sorted.ptr()), sorted)))
            [[unlikely]] {
            throw py::error_already_set();
        }
        list = std::move(sorted);
        return;
    } catch (py::error_already_set &ex1) {
        if (!ex1.matches(PyExc_TypeError)) [[unlikely]] {
            std::rethrow_exception(std::current_exception());
        }
        // Found incomparable keys (e.g. `int` vs. `str`, or user-defined types).
    }

    sorted = ListCopy(list);
    try {
        // Sort with `(f'{obj.__class__.__module__}.{obj.__class__.__qualname__}', obj)`
        const auto sort_key_fn = py::cpp_function([](const py::object &obj) -> py::tuple {
            const py::handle cls = py::type::handle_of(obj);
            const py::str qualname{
                EVALUATE_WITH_LOCK_HELD(PyStr(py::getattr(cls, "__module__")) + "." +
                                            PyStr(py::getattr(cls, "__qualname__")),
                                        cls)};
            return py::make_tuple(qualname, obj);
        });
        {
            const scoped_critical_section cs{sorted};
            py::getattr(sorted, "sort")(py::arg("key") = sort_key_fn);
        }
        list = std::move(sorted);
    } catch (py::error_already_set &ex2) {
        if (ex2.matches(PyExc_TypeError)) [[likely]] {
            // Found incomparable user-defined key types.
            // The keys remain in the insertion order.
            PyErr_Clear();
        } else [[unlikely]] {
            std::rethrow_exception(std::current_exception());
        }
    }
}

[[nodiscard]] inline Py_ALWAYS_INLINE py::list DictKeys(const py::dict &dict) {
    const scoped_critical_section cs{dict};
    return py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
}

// Equivalent to Python's `dict.fromkeys(iterable)`: returns a new `dict[Key, None]` with the
// keys taken from `iterable` in order. When `iterable` is itself a dict, this hits CPython's
// `dict_dict_fromkeys` fast path (contiguous bucket copy, no per-key rehash).
[[nodiscard]] inline Py_ALWAYS_INLINE py::dict DictFromKeys(const py::handle &iterable) {
    const scoped_critical_section cs{iterable};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    PyObject *result = PyObject_CallMethod(reinterpret_cast<PyObject *>(&PyDict_Type),
                                           "fromkeys",
                                           "O",
                                           iterable.ptr());
    if (result == nullptr) [[unlikely]] {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::dict>(result);
}

[[nodiscard]] inline py::list SortedDictKeys(const py::dict &dict) {
    py::list keys = DictKeys(dict);
    TotalOrderSort(keys);
    return keys;
}

// Test whether `keys` and the keys of `dict` are the same set.
// Precondition: `keys` holds no duplicates; the length shortcut below relies on that.
[[nodiscard]] inline bool DictKeysEqual(const py::list &keys, const py::dict &dict) {
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

// Return the keys of `keys` missing from `dict` and the keys of `dict` not in `keys`.
// Precondition: `keys` holds no duplicates.
[[nodiscard]] inline std::pair<py::list, py::list> DictKeysDifference(const py::list &keys,
                                                                      const py::dict &dict) {
    const py::set expected_keys = EVALUATE_WITH_LOCK_HELD(py::set{keys}, keys);
    const py::set got_keys = EVALUATE_WITH_LOCK_HELD(py::set{dict}, dict);
    py::list missing_keys{expected_keys - got_keys};
    py::list extra_keys{got_keys - expected_keys};
    TotalOrderSort(missing_keys);
    TotalOrderSort(extra_keys);
    return std::make_pair(std::move(missing_keys), std::move(extra_keys));
}

[[nodiscard]] inline py::ssize_t DistinctCount(const py::handle &iterable) {
    const scoped_critical_section cs{iterable};
    const auto set = py::reinterpret_steal<py::object>(PySet_New(iterable.ptr()));
    if (!set) [[unlikely]] {  // a non-iterable or an unhashable element raised here
        throw py::error_already_set();
    }
    return PySet_GET_SIZE(set.ptr());
}
