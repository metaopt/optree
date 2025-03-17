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

#include <Python.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#if PY_VERSION_HEX < 0x03080000  // Python 3.8
#error "Python 3.8 or newer is required."
#endif

#ifndef Py_ALWAYS_INLINE
#define Py_ALWAYS_INLINE
#endif

#ifndef Py_NO_INLINE
#define Py_NO_INLINE
#endif

#ifndef Py_Is
#define Py_Is(x, y) ((x) == (y))
#endif
#ifndef Py_IsNone
#define Py_IsNone(x) Py_Is((x), Py_None)
#endif
#ifndef Py_IsTrue
#define Py_IsTrue(x) Py_Is((x), Py_True)
#endif
#ifndef Py_IsFalse
#define Py_IsFalse(x) Py_Is((x), Py_False)
#endif

inline constexpr Py_ALWAYS_INLINE bool Py_IsConstant(PyObject* x) noexcept {
    return Py_IsNone(x) || Py_IsTrue(x) || Py_IsFalse(x);
}
#define Py_IsConstant(x) Py_IsConstant(x)

#define Py_Declare_ID(name)                                                                        \
    namespace {                                                                                    \
    inline PyObject* Py_ID_##name() {                                                              \
        PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<PyObject*> storage;             \
        return storage                                                                             \
            .call_once_and_store_result([]() -> PyObject* {                                        \
                PyObject* const ptr = PyUnicode_InternFromString(#name);                           \
                if (ptr == nullptr) [[unlikely]] {                                                 \
                    throw py::error_already_set();                                                 \
                }                                                                                  \
                Py_INCREF(ptr); /* leak a reference on purpose */                                  \
                return ptr;                                                                        \
            })                                                                                     \
            .get_stored();                                                                         \
    }                                                                                              \
    }  // namespace

#define Py_Get_ID(name) (::Py_ID_##name())

Py_Declare_ID(optree);
Py_Declare_ID(__main__);           // __main__
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
