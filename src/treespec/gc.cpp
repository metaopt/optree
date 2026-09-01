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

#include "optree/optree.h"

inline namespace {
// Avoid `py::cast`, which consults interpreter-local type metadata that may already be unavailable
// while GC slots run during interpreter finalization.
template <typename T>
[[nodiscard]] inline T *get_value_if_holder_constructed(PyObject *obj) {
    auto * const instance = reinterpret_cast<pybind11::detail::instance *>(obj);
    const auto value_and_holder = instance->get_value_and_holder();
    if (!value_and_holder.holder_constructed()) [[unlikely]] {
        return nullptr;
    }
    return value_and_holder.value_ptr<T>();
}
}  // namespace

namespace optree {

// No exception may escape a `tp_traverse` / `tp_clear` slot: unwinding across the `extern "C"`
// boundary calls `std::terminate`. In particular, no `PYTREESPEC_SANITY_CHECK` below: `PyTpClear`
// empties the traversal, so a cleared but still-alive treespec would abort on the next collection.

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
/*static*/ int PyTreeSpec::PyTpTraverse(PyObject *self_base, visitproc visit, void *arg) {
    Py_VISIT(Py_TYPE(self_base));
    auto * const self = ::get_value_if_holder_constructed<PyTreeSpec>(self_base);
    if (self == nullptr) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }

    // Report a registration's members once, and only when this treespec owns every reference to it.
    // The registration holds one reference to each member however many nodes point at it, so
    // reporting per node would decrement the same object once per node and underflow its shadow
    // refcount. While the registry still holds the registration it also keeps the members alive, so
    // skipping then leaks nothing.
    //
    // The holders are counted by rescanning the traversal rather than through a map: a treespec
    // references very few distinct registrations, and every container that could hold them
    // allocates, which `tp_traverse` cannot afford (see above).
    //
    // Known limitation: a treespec can only count its own nodes, so when several treespecs each
    // hold part of the references none of them reports the members and a cycle through them
    // survives. Fixing that needs the registration to be a garbage-collected object with its own
    // `tp_traverse`, so each edge is reported by its owner and no counting is needed.
    const ssize_t num_nodes = py::ssize_t_cast(self->m_traversal.size());
    for (ssize_t i = 0; i < num_nodes; ++i) {
        const auto &node = self->m_traversal[i];
        Py_VISIT(node.node_data.ptr());
        Py_VISIT(node.node_entries.ptr());
        Py_VISIT(node.original_keys.ptr());
        if (node.custom == nullptr) [[likely]] {
            continue;
        }
        // Scanning from the start, the first match decides: before `i` an earlier node already
        // reported this registration, at `i` this node is the first holder and counts the rest.
        ssize_t num_holders = 0;
        for (ssize_t j = 0; j < num_nodes; ++j) {
            if (self->m_traversal[j].custom != node.custom) [[likely]] {
                continue;
            }
            if (j < i) [[likely]] {
                num_holders = 0;  // not the first holder, skip reporting
                break;
            }
            ++num_holders;
        }
        if (num_holders > 0 && node.custom.use_count() == num_holders) [[unlikely]] {
            Py_VISIT(node.custom->type.ptr());
            Py_VISIT(node.custom->flatten_func.ptr());
            Py_VISIT(node.custom->unflatten_func.ptr());
            Py_VISIT(node.custom->path_entry_type.ptr());
        }
    }
    return 0;
}

/*static*/ int PyTreeSpec::PyTpClear(PyObject *self_base) {
    auto * const self = ::get_value_if_holder_constructed<PyTreeSpec>(self_base);
    if (self == nullptr) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    for (auto &node : self->m_traversal) {
        Py_CLEAR(node.node_data.ptr());
        Py_CLEAR(node.node_entries.ptr());
        Py_CLEAR(node.original_keys.ptr());
        node.custom.reset();
    }
    self->m_traversal.clear();
    return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
/*static*/ int PyTreeIter::PyTpTraverse(PyObject *self_base, visitproc visit, void *arg) {
    Py_VISIT(Py_TYPE(self_base));
    auto * const self = ::get_value_if_holder_constructed<PyTreeIter>(self_base);
    if (self == nullptr) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    for (const auto &[obj, _] : self->m_agenda) {
        Py_VISIT(obj.ptr());
    }
    Py_VISIT(self->m_root.ptr());
    if (self->m_leaf_predicate) [[likely]] {
        // The leaf predicate is an owned Python callback; it must be visited so the cyclic GC can
        // see reference cycles that pass through it (otherwise such cycles leak).
        Py_VISIT(self->m_leaf_predicate->ptr());
    }
    return 0;
}

/*static*/ int PyTreeIter::PyTpClear(PyObject *self_base) {
    auto * const self = ::get_value_if_holder_constructed<PyTreeIter>(self_base);
    if (self == nullptr) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    for (auto &[obj, _] : self->m_agenda) {
        Py_CLEAR(obj.ptr());
    }
    self->m_agenda.clear();
    Py_CLEAR(self->m_root.ptr());
    // Reset the optional rather than clearing the held function: `NextImpl` tests the optional for
    // presence, so leaving it engaged around a null callable would call through it after a
    // collection.
    self->m_leaf_predicate.reset();
    return 0;
}

}  // namespace optree
