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

#include <mutex>  // std::mutex, std::recursive_mutex, std::lock_guard, std::unique_lock

#include <Python.h>

#ifdef Py_GIL_DISABLED

class pymutex {
 public:
    pymutex() = default;
    ~pymutex() = default;

    pymutex(const pymutex &) = delete;
    pymutex &operator=(const pymutex &) = delete;
    pymutex(pymutex &&) = delete;
    pymutex &operator=(pymutex &&) = delete;

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
