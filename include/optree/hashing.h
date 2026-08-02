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

#include <cstddef>      // std::size_t
#include <functional>   // std::hash, std::{not_,}equal_to
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::pair

#include <Python.h>

#include <pybind11/pybind11.h>

#include "optree/pymacros.h"  // Py_ALWAYS_INLINE, interpid_t

namespace py = pybind11;

// boost::hash_combine
template <class T>
inline constexpr Py_ALWAYS_INLINE void HashCombine(
    py::size_t &seed,  // NOLINT[runtime/references]
    const T &v) noexcept(noexcept(std::hash<T>{}(v))) {
    // NOLINTNEXTLINE[cppcoreguidelines-avoid-magic-numbers]
    seed ^= (std::hash<T>{}(v) + 0x9E3779B9 + (seed << 6) + (seed >> 2));
}
template <class T>
inline constexpr Py_ALWAYS_INLINE void HashCombine(
    py::ssize_t &seed,  // NOLINT[runtime/references]
    const T &v) noexcept(noexcept(std::hash<T>{}(v))) {
    // NOLINTNEXTLINE[cppcoreguidelines-avoid-magic-numbers]
    seed ^= (std::hash<T>{}(v) + 0x9E3779B9 + (seed << 6) + (seed >> 2));
}

// NOLINTBEGIN[bugprone-std-namespace-modification]
template <>
struct std::equal_to<py::handle> {
    using is_transparent = void;
    inline Py_ALWAYS_INLINE bool operator()(const py::handle &lhs,
                                            const py::handle &rhs) const noexcept {
        return lhs.is(rhs);
    }
};
template <>
struct std::not_equal_to<py::handle> {
    using is_transparent = void;
    inline Py_ALWAYS_INLINE bool operator()(const py::handle &lhs,
                                            const py::handle &rhs) const noexcept {
        return !lhs.is(rhs);
    }
};
template <>
struct std::hash<py::handle> {
    using is_transparent = void;
    inline Py_ALWAYS_INLINE std::size_t operator()(const py::handle &obj) const noexcept {
        return std::hash<PyObject *>{}(obj.ptr());
    }
};

template <>
struct std::equal_to<std::pair<std::string, py::handle>> {
    using is_transparent = void;
    inline constexpr Py_ALWAYS_INLINE bool operator()(const std::pair<std::string, py::handle> &lhs,
                                                      const std::pair<std::string, py::handle> &rhs)
        const noexcept(noexcept(std::equal_to<std::string>{}(lhs.first, rhs.first))) {
        return std::equal_to<std::string>{}(lhs.first, rhs.first) &&
               std::equal_to<py::handle>{}(lhs.second, rhs.second);
    }
};
template <>
struct std::not_equal_to<std::pair<std::string, py::handle>> {
    using is_transparent = void;
    inline constexpr Py_ALWAYS_INLINE bool operator()(const std::pair<std::string, py::handle> &lhs,
                                                      const std::pair<std::string, py::handle> &rhs)
        const noexcept(noexcept(std::not_equal_to<std::string>{}(lhs.first, rhs.first))) {
        return std::not_equal_to<std::string>{}(lhs.first, rhs.first) ||
               std::not_equal_to<py::handle>{}(lhs.second, rhs.second);
    }
};
template <>
struct std::equal_to<std::pair<interpid_t, py::handle>> {
    using is_transparent = void;
    inline constexpr Py_ALWAYS_INLINE bool operator()(const std::pair<interpid_t, py::handle> &lhs,
                                                      const std::pair<interpid_t, py::handle> &rhs)
        const noexcept(noexcept(std::equal_to<interpid_t>{}(lhs.first, rhs.first))) {
        return std::equal_to<interpid_t>{}(lhs.first, rhs.first) &&
               std::equal_to<py::handle>{}(lhs.second, rhs.second);
    }
};
template <>
struct std::not_equal_to<std::pair<interpid_t, py::handle>> {
    using is_transparent = void;
    inline constexpr Py_ALWAYS_INLINE bool operator()(const std::pair<interpid_t, py::handle> &lhs,
                                                      const std::pair<interpid_t, py::handle> &rhs)
        const noexcept(noexcept(std::not_equal_to<interpid_t>{}(lhs.first, rhs.first))) {
        return std::not_equal_to<interpid_t>{}(lhs.first, rhs.first) ||
               std::not_equal_to<py::handle>{}(lhs.second, rhs.second);
    }
};
template <class T, class U>
struct std::hash<std::pair<T, U>> {
    using is_transparent = void;
    inline constexpr Py_ALWAYS_INLINE std::size_t operator()(const std::pair<T, U> &p) const
        noexcept(noexcept(std::hash<T>{}(p.first)) && noexcept(std::hash<U>{}(p.second))) {
        std::size_t seed = 0;
        HashCombine(seed, p.first);
        HashCombine(seed, p.second);
        return seed;
    }
};
// NOLINTEND[bugprone-std-namespace-modification]

namespace optree {

// Transparent hashers and comparators for the registry's pair keys. Their `operator()` MUST be
// templates: `is_transparent` only tells the container it may forward a foreign key type, and a
// non-template call operator then converts it back to the exact `key_type` — the very temporary
// heterogeneous lookup exists to avoid. The marker was inert for unordered containers before
// P0919R3, so this began costing a namespace-string copy per `Lookup` only at C++20.
// `std::hash<std::string_view>` is guaranteed to agree with `std::hash<std::string>`, so probing
// with a view finds entries inserted with a string.

// Key of `PyTreeTypeRegistry::m_named_registrations`: (namespace, type).
struct NamespacedTypeHash {
    using is_transparent = void;
    template <class S>
    inline Py_ALWAYS_INLINE std::size_t operator()(
        const std::pair<S, py::handle> &key) const noexcept {
        std::size_t seed = 0;
        HashCombine(seed, std::string_view{key.first});
        HashCombine(seed, key.second);
        return seed;
    }
};
struct NamespacedTypeEqual {
    using is_transparent = void;
    template <class S1, class S2>
    inline Py_ALWAYS_INLINE bool operator()(const std::pair<S1, py::handle> &lhs,
                                            const std::pair<S2, py::handle> &rhs) const noexcept {
        // Compare the type first: it is a pointer identity test, and it discriminates far more
        // often than the namespace does.
        return lhs.second.is(rhs.second) &&
               std::string_view{lhs.first} == std::string_view{rhs.first};
    }
};

// Key of `PyTreeTypeRegistry::sm_dict_insertion_ordered_namespaces`: (interpreter, namespace).
struct InterpreterNamespaceHash {
    using is_transparent = void;
    template <class S>
    inline Py_ALWAYS_INLINE std::size_t operator()(
        const std::pair<interpid_t, S> &key) const noexcept {
        std::size_t seed = 0;
        HashCombine(seed, key.first);
        HashCombine(seed, std::string_view{key.second});
        return seed;
    }
};
struct InterpreterNamespaceEqual {
    using is_transparent = void;
    template <class S1, class S2>
    inline Py_ALWAYS_INLINE bool operator()(const std::pair<interpid_t, S1> &lhs,
                                            const std::pair<interpid_t, S2> &rhs) const noexcept {
        return lhs.first == rhs.first &&
               std::string_view{lhs.second} == std::string_view{rhs.second};
    }
};

}  // namespace optree
