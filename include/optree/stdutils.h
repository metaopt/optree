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

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

#include "optree/pymacros.h"  // Py_ALWAYS_INLINE

template <typename T>
inline Py_ALWAYS_INLINE std::vector<T> reserved_vector(std::size_t size) {
    std::vector<T> v{};
    v.reserve(size);
    return v;
}
