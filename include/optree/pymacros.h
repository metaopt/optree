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

#include <stdexcept>  // std::runtime_error

#include <Python.h>

#include <pybind11/pybind11.h>

#if !(defined(PY_VERSION_HEX) && PY_VERSION_HEX >= 0x03090000)  // Python 3.9
#    error "Python 3.9 or newer is required."
#endif

#if !(defined(PYBIND11_VERSION_HEX) && PYBIND11_VERSION_HEX >= 0x020C00F0)  // pybind11 2.12.0
#    error "pybind11 2.12.0 or newer is required."
#endif

// NOLINTNEXTLINE[bugprone-macro-parentheses]
#define NONZERO_OR_EMPTY(MACRO) ((MACRO + 0 != 0) || (0 - MACRO - 1 >= 0))

#if !defined(PYPY_VERSION) && (PY_VERSION_HEX >= 0x030E0000 /* Python 3.14 */) &&                  \
    (PYBIND11_VERSION_HEX >= 0x030002A0 /* pybind11 3.0.2.a0 */) &&                                \
    (defined(PYBIND11_HAS_SUBINTERPRETER_SUPPORT) &&                                               \
     NONZERO_OR_EMPTY(PYBIND11_HAS_SUBINTERPRETER_SUPPORT))
#    define OPTREE_HAS_SUBINTERPRETER_SUPPORT 1
#else
#    undef OPTREE_HAS_SUBINTERPRETER_SUPPORT
#endif

namespace py = pybind11;

#if !defined(Py_ALWAYS_INLINE)
#    define Py_ALWAYS_INLINE
#endif

#if !defined(Py_NO_INLINE)
#    define Py_NO_INLINE
#endif

#if !defined(Py_Is)
#    define Py_Is(x, y) ((x) == (y))
#endif
#if !defined(Py_IsNone)
#    define Py_IsNone(x) Py_Is((x), Py_None)
#endif
#if !defined(Py_IsTrue)
#    define Py_IsTrue(x) Py_Is((x), Py_True)
#endif
#if !defined(Py_IsFalse)
#    define Py_IsFalse(x) Py_Is((x), Py_False)
#endif

inline constexpr Py_ALWAYS_INLINE bool Py_IsConstant(PyObject *x) noexcept {
    return Py_IsNone(x) || Py_IsTrue(x) || Py_IsFalse(x);
}
#define Py_IsConstant(x) Py_IsConstant(x)

using interpid_t = decltype(PyInterpreterState_GetID(nullptr));

#if defined(PYBIND11_HAS_SUBINTERPRETER_SUPPORT) &&                                                \
    NONZERO_OR_EMPTY(PYBIND11_HAS_SUBINTERPRETER_SUPPORT)

[[nodiscard]] inline bool IsCurrentPyInterpreterMain() {
    return PyInterpreterState_Get() == PyInterpreterState_Main();
}

[[nodiscard]] inline interpid_t GetCurrentPyInterpreterID() {
    PyInterpreterState *interp = PyInterpreterState_Get();
    if (PyErr_Occurred() != nullptr) [[unlikely]] {
        throw py::error_already_set();
    }
    if (interp == nullptr) [[unlikely]] {
        throw std::runtime_error("Failed to get the current Python interpreter state.");
    }
    const interpid_t interpid = PyInterpreterState_GetID(interp);
    if (PyErr_Occurred() != nullptr) [[unlikely]] {
        throw py::error_already_set();
    }
    return interpid;
}

[[nodiscard]] inline interpid_t GetMainPyInterpreterID() {
    PyInterpreterState *interp = PyInterpreterState_Main();
    if (PyErr_Occurred() != nullptr) [[unlikely]] {
        throw py::error_already_set();
    }
    if (interp == nullptr) [[unlikely]] {
        throw std::runtime_error("Failed to get the main Python interpreter state.");
    }
    const interpid_t interpid = PyInterpreterState_GetID(interp);
    if (PyErr_Occurred() != nullptr) [[unlikely]] {
        throw py::error_already_set();
    }
    return interpid;
}

#else

[[nodiscard]] inline bool IsCurrentPyInterpreterMain() noexcept { return true; }
[[nodiscard]] inline interpid_t GetCurrentPyInterpreterID() noexcept { return 0; }
[[nodiscard]] inline interpid_t GetMainPyInterpreterID() noexcept { return 0; }

#endif
