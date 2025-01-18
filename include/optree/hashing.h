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

#include <cstddef>     // std::size_t
#include <functional>  // std::hash, std::{not_,}equal_to
#include <string>      // std::string
#include <utility>     // std::pair

#include <pybind11/pybind11.h>

#include "optree/pymacros.h"  // Py_ALWAYS_INLINE

namespace py = pybind11;

// boost::hash_combine
template <class T>
inline constexpr Py_ALWAYS_INLINE void HashCombine(
    py::size_t& seed,  // NOLINT[runtime/references]
    const T& v) noexcept(noexcept(std::hash<T>{}(v))) {
    // NOLINTNEXTLINE[cppcoreguidelines-avoid-magic-numbers]
    seed ^= (std::hash<T>{}(v) + 0x9E3779B9 + (seed << 6) + (seed >> 2));
}
template <class T>
inline constexpr Py_ALWAYS_INLINE void HashCombine(
    py::ssize_t& seed,  // NOLINT[runtime/references]
    const T& v) noexcept(noexcept(std::hash<T>{}(v))) {
    // NOLINTNEXTLINE[cppcoreguidelines-avoid-magic-numbers]
    seed ^= (std::hash<T>{}(v) + 0x9E3779B9 + (seed << 6) + (seed >> 2));
}

template <>
struct std::equal_to<py::handle> {
    using is_transparent = void;
    inline Py_ALWAYS_INLINE bool operator()(const py::handle& lhs,
                                            const py::handle& rhs) const noexcept {
        return lhs.is(rhs);
    }
};
template <>
struct std::not_equal_to<py::handle> {
    using is_transparent = void;
    inline Py_ALWAYS_INLINE bool operator()(const py::handle& lhs,
                                            const py::handle& rhs) const noexcept {
        return !lhs.is(rhs);
    }
};
template <>
struct std::hash<py::handle> {
    using is_transparent = void;
    inline Py_ALWAYS_INLINE std::size_t operator()(const py::handle& obj) const noexcept {
        return std::hash<PyObject*>{}(obj.ptr());
    }
};

template <>
struct std::equal_to<std::pair<std::string, py::handle>> {
    using is_transparent = void;
    inline constexpr Py_ALWAYS_INLINE bool operator()(const std::pair<std::string, py::handle>& lhs,
                                                      const std::pair<std::string, py::handle>& rhs)
        const noexcept(noexcept(std::equal_to<std::string>{}(lhs.first, rhs.first))) {
        return std::equal_to<std::string>{}(lhs.first, rhs.first) &&
               std::equal_to<py::handle>{}(lhs.second, rhs.second);
    }
};
template <>
struct std::not_equal_to<std::pair<std::string, py::handle>> {
    using is_transparent = void;
    inline constexpr Py_ALWAYS_INLINE bool operator()(const std::pair<std::string, py::handle>& lhs,
                                                      const std::pair<std::string, py::handle>& rhs)
        const noexcept(noexcept(std::not_equal_to<std::string>{}(lhs.first, rhs.first))) {
        return std::not_equal_to<std::string>{}(lhs.first, rhs.first) ||
               std::not_equal_to<py::handle>{}(lhs.second, rhs.second);
    }
};
template <class T, class U>
struct std::hash<std::pair<T, U>> {
    using is_transparent = void;
    inline constexpr Py_ALWAYS_INLINE std::size_t operator()(const std::pair<T, U>& p) const
        noexcept(noexcept(std::hash<T>{}(p.first)) && noexcept(std::hash<U>{}(p.second))) {
        std::size_t seed = 0;
        HashCombine(seed, p.first);
        HashCombine(seed, p.second);
        return seed;
    }
};
