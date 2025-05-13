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

#include "optree/optree.h"

namespace optree {

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ int PyTreeSpec::PyTpTraverse(PyObject* self_base, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self_base));
    auto* const instance = reinterpret_cast<py::detail::instance*>(self_base);
    if (!instance->get_value_and_holder().holder_constructed()) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    auto& self = thread_safe_cast<PyTreeSpec&>(py::handle{self_base});
    PYTREESPEC_SANITY_CHECK(self);
    for (const auto& node : self.m_traversal) {
        Py_VISIT(node.node_data.ptr());
        Py_VISIT(node.node_entries.ptr());
        Py_VISIT(node.original_keys.ptr());
    }
    return 0;
}

/*static*/ int PyTreeSpec::PyTpClear(PyObject* self_base) {
    auto* const instance = reinterpret_cast<py::detail::instance*>(self_base);
    if (!instance->get_value_and_holder().holder_constructed()) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    auto& self = thread_safe_cast<PyTreeSpec&>(py::handle{self_base});
    PYTREESPEC_SANITY_CHECK(self);
    for (auto& node : self.m_traversal) {
        Py_CLEAR(node.node_data.ptr());
        Py_CLEAR(node.node_entries.ptr());
        Py_CLEAR(node.original_keys.ptr());
    }
    self.m_traversal.clear();
    return 0;
}

/*static*/ int PyTreeIter::PyTpTraverse(PyObject* self_base, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self_base));
    auto* const instance = reinterpret_cast<py::detail::instance*>(self_base);
    if (!instance->get_value_and_holder().holder_constructed()) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    auto& self = thread_safe_cast<PyTreeIter&>(py::handle{self_base});
    for (const auto& pair : self.m_agenda) {
        Py_VISIT(pair.first.ptr());
    }
    Py_VISIT(self.m_root.ptr());
    return 0;
}

/*static*/ int PyTreeIter::PyTpClear(PyObject* self_base) {
    auto* const instance = reinterpret_cast<py::detail::instance*>(self_base);
    if (!instance->get_value_and_holder().holder_constructed()) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    auto& self = thread_safe_cast<PyTreeIter&>(py::handle{self_base});
    for (auto& pair : self.m_agenda) {
        Py_CLEAR(pair.first.ptr());
    }
    self.m_agenda.clear();
    Py_CLEAR(self.m_root.ptr());
    return 0;
}

}  // namespace optree
