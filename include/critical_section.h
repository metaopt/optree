/*
Copyright 2022-2024 MetaOPT Team. All Rights Reserved.

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

inline bool Py_IsConstant(PyObject* x) { return Py_IsNone(x) || Py_IsTrue(x) || Py_IsFalse(x); }
#define Py_IsConstant(x) Py_IsConstant(x)

class scoped_critical_section {
 public:
    scoped_critical_section() = delete;

#ifdef Py_GIL_DISABLED
    explicit scoped_critical_section(const py::handle& handle) : m_ptr{handle.ptr()} {
        if (m_ptr != nullptr && !Py_IsConstant(m_ptr)) [[likely]] {
            PyCriticalSection_Begin(&m_critical_section, m_ptr);
        }
    }

    ~scoped_critical_section() {
        if (m_ptr != nullptr && !Py_IsConstant(m_ptr)) [[likely]] {
            PyCriticalSection_End(&m_critical_section);
        }
    }
#else
    explicit scoped_critical_section(const py::handle& /*unused*/) {}
    ~scoped_critical_section() = default;
#endif

    scoped_critical_section(const scoped_critical_section&) = delete;
    scoped_critical_section& operator=(const scoped_critical_section&) = delete;
    scoped_critical_section(scoped_critical_section&&) = delete;
    scoped_critical_section& operator=(scoped_critical_section&&) = delete;

 private:
#ifdef Py_GIL_DISABLED
    PyObject* m_ptr{nullptr};
    PyCriticalSection m_critical_section{};
#endif
};

class scoped_critical_section2 {
 public:
    scoped_critical_section2() = delete;

#ifdef Py_GIL_DISABLED
    explicit scoped_critical_section2(const py::handle& handle1, const py::handle& handle2)
        : m_ptr1{handle1.ptr()}, m_ptr2{handle2.ptr()} {
        if (m_ptr1 != nullptr && !Py_IsConstant(m_ptr1)) [[likely]] {
            if (m_ptr2 != nullptr && !Py_IsConstant(m_ptr2)) [[likely]] {
                PyCriticalSection2_Begin(&m_critical_section2, m_ptr1, m_ptr2);
            } else [[unlikely]] {
                PyCriticalSection_Begin(&m_critical_section, m_ptr1);
            }
        } else if (m_ptr2 != nullptr && !Py_IsConstant(m_ptr2)) [[likely]] {
            PyCriticalSection_Begin(&m_critical_section, m_ptr2);
        }
    }

    ~scoped_critical_section2() {
        if (m_ptr1 != nullptr && !Py_IsConstant(m_ptr1)) [[likely]] {
            if (m_ptr2 != nullptr && !Py_IsConstant(m_ptr2)) [[likely]] {
                PyCriticalSection2_End(&m_critical_section2);
            } else [[unlikely]] {
                PyCriticalSection_End(&m_critical_section);
            }
        } else if (m_ptr2 != nullptr && !Py_IsConstant(m_ptr2)) [[likely]] {
            PyCriticalSection_End(&m_critical_section);
        }
    }
#else
    explicit scoped_critical_section2(const py::handle& /*unused*/, const py::handle& /*unused*/) {}
    ~scoped_critical_section2() = default;
#endif

    scoped_critical_section2(const scoped_critical_section2&) = delete;
    scoped_critical_section2& operator=(const scoped_critical_section2&) = delete;
    scoped_critical_section2(scoped_critical_section2&&) = delete;
    scoped_critical_section2& operator=(scoped_critical_section2&&) = delete;

 private:
#ifdef Py_GIL_DISABLED
    PyObject* m_ptr1{nullptr};
    PyObject* m_ptr2{nullptr};
    PyCriticalSection m_critical_section{};
    PyCriticalSection2 m_critical_section2{};
#endif
};

#ifdef Py_GIL_DISABLED

#define EVALUATE_WITH_LOCK_HELD(expression, handle)                                                \
    (((void)scoped_critical_section{(handle)}), (expression))

#define EVALUATE_WITH_LOCK_HELD2(expression, handle1, handle2)                                     \
    (((void)scoped_critical_section2{(handle1), (handle2)}), (expression))

#else

#define EVALUATE_WITH_LOCK_HELD(expression, handle) (expression)
#define EVALUATE_WITH_LOCK_HELD2(expression, handle1, handle2) (expression)

#endif
