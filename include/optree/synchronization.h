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

#include <mutex>  // std::mutex, std::recursive_mutex, std::lock_guard, std::unique_lock

#include <Python.h>

#include <pybind11/pybind11.h>

#include "optree/pymacros.h"  // Py_ALWAYS_INLINE, Py_IsConstant

namespace py = pybind11;

#ifdef Py_GIL_DISABLED

class pymutex {
public:
    pymutex() noexcept = default;
    ~pymutex() noexcept = default;

    pymutex(const pymutex&) = delete;
    pymutex& operator=(const pymutex&) = delete;
    pymutex(pymutex&&) = delete;
    pymutex& operator=(pymutex&&) = delete;

    void lock() { PyMutex_Lock(&mutex); }
    void unlock() { PyMutex_Unlock(&mutex); }

private:
    PyMutex mutex{0};
};

using mutex = pymutex;
using recursive_mutex = std::recursive_mutex;

#else

using mutex = std::mutex;
using recursive_mutex = std::recursive_mutex;

#endif

using scoped_lock_guard = std::lock_guard<mutex>;
using scoped_recursive_lock_guard = std::lock_guard<recursive_mutex>;

#if (defined(__APPLE__) /* header <shared_mutex> is not available on macOS build target */ &&      \
     PY_VERSION_HEX < /* Python 3.12.0 */ 0x030C00F0)

#undef HAVE_READ_WRITE_LOCK

using read_write_mutex = mutex;
using scoped_read_lock_guard = scoped_lock_guard;
using scoped_write_lock_guard = scoped_lock_guard;

#else

#define HAVE_READ_WRITE_LOCK

#include <shared_mutex>  // std::shared_mutex, std::shared_lock

using read_write_mutex = std::shared_mutex;
using scoped_read_lock_guard = std::shared_lock<read_write_mutex>;
using scoped_write_lock_guard = std::unique_lock<read_write_mutex>;

#endif

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
    explicit scoped_critical_section(const py::handle& /*unused*/) noexcept {}
    ~scoped_critical_section() noexcept = default;
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
    explicit scoped_critical_section2(const py::handle& /*unused*/,
                                      const py::handle& /*unused*/) noexcept {}
    ~scoped_critical_section2() noexcept = default;
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

template <typename T>
inline Py_ALWAYS_INLINE T thread_safe_cast(const py::handle& handle) {
    return EVALUATE_WITH_LOCK_HELD(py::cast<T>(handle), handle);
}
