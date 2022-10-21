/*
Copyright 2022 MetaOPT Team. All Rights Reserved.

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

// Caution: this code uses exceptions. The exception use is local to the binding
// code and the idiomatic way to emit Python exceptions.

#include "include/utils.h"

#ifdef _WIN32  // Windows
const py::object &ImportOrderedDict() {
    static const py::module_ collections = py::module_::import("collections");
    static const py::object object = py::getattr(collections, "OrderedDict");
    return object;
}

const py::object &ImportDefaultDict() {
    static const py::module_ collections = py::module_::import("collections");
    static const py::object object = py::getattr(collections, "defaultdict");
    return object;
}

const py::object &ImportDeque() {
    static const py::module_ collections = py::module_::import("collections");
    static const py::object object = py::getattr(collections, "deque");
    return object;
}
#endif
