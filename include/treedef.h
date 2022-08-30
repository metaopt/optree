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

// See https://jax.readthedocs.io/en/latest/pytrees.html for the documentation
// about PyTrees.

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#pragma once

#include <absl/container/inlined_vector.h>
#include <absl/hash/hash.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "include/registry.h"

namespace optree {

namespace py = pybind11;

// A PyTreeDef describes the tree structure of a PyTree. A PyTree is a tree of Python values, where
// the interior nodes are tuples, lists, dictionaries, or user-defined containers, and the leaves
// are other objects.
class PyTreeDef {
 private:
    struct Node;

 public:
    PyTreeDef() = default;

    // Flattens a PyTree into a list of leaves and a PyTreeDef.
    // Returns references to the flattened objects, which might be temporary objects in the case of
    // custom PyType handlers.
    static std::pair<std::vector<py::object>, std::unique_ptr<PyTreeDef>> Flatten(
        py::handle x, std::optional<py::function> leaf_predicate = std::nullopt);

    // Recursive helper used to implement Flatten().
    void FlattenInto(py::handle handle,
                     std::vector<py::object> &leaves,  // NOLINT
                     std::optional<py::function> leaf_predicate = std::nullopt);
    void FlattenInto(py::handle handle,
                     absl::InlinedVector<py::object, 2> &leaves,  // NOLINT
                     std::optional<py::function> leaf_predicate = std::nullopt);

    // Tests whether the given list is a flat list of leaves.
    static bool AllLeaves(const py::iterable &x);

    // Flattens a PyTree up to this PyTreeDef. 'this' must be a tree prefix of the tree-structure of
    // 'x'.
    // For example, if we flatten a value [(1, (2, 3)), {"foo": 4}] with a PyTreeDef [(*, *), *],
    // the result is the list of leaves [1, (2, 3), {"foo": 4}].
    py::list FlattenUpTo(py::handle x) const;

    // Returns an unflattened PyTree given an iterable of leaves and a PyTreeDef.
    py::object Unflatten(py::iterable leaves) const;
    py::object Unflatten(absl::Span<const py::object> leaves) const;

    // Composes two PyTreeDefs, replacing the leaves of this tree with copies of `inner`.
    std::unique_ptr<PyTreeDef> Compose(const PyTreeDef &inner) const;

    // Makes a Tuple PyTreeDef out of a vector of PyTreeDefs.
    static std::unique_ptr<PyTreeDef> Tuple(const std::vector<PyTreeDef> &defs);

    std::vector<std::unique_ptr<PyTreeDef>> Children() const;

    // Maps a function over a PyTree structure, applying f_leaf to each leaf, and
    // f_node(node, node_data) to each container node.
    py::object Walk(const py::function &f_node, py::handle f_leaf, py::iterable leaves) const;

    // Given a tree of iterables with the same node/leaf structure as this PyTree, build the
    // corresponding PyTree.
    // TODO(phawkins): use flattening everywhere instead and delete this method.
    py::object FromIterableTree(py::handle xs) const;

    Py_ssize_t num_leaves() const;

    Py_ssize_t num_nodes() const;

    bool operator==(const PyTreeDef &other) const;
    bool operator!=(const PyTreeDef &other) const;

    size_t Hash() const;

    template <typename H>
    friend H AbslHashValue(H h, const Node &n) {
        h = H::combine(std::move(h), n.kind, n.arity, n.custom);
        return h;
    }

    template <typename H>
    friend H AbslHashValue(H h, const PyTreeDef &t) {
        h = H::combine(std::move(h), t.traversal);
        return h;
    }

    std::string ToString() const;

    // Transforms the PyTreeDef into a pickleable object.
    // Used to implement `PyTreeDef.__getstate__`.
    py::object ToPickleable() const;

    // Transforms the object returned by `ToPickleable()` back to PyTreeDef.
    // Used to implement `PyTreeDef.__setstate__`.
    static PyTreeDef FromPickleable(py::object pickleable);

 private:
    struct Node {
        PyTreeKind kind = PyTreeKind::Leaf;

        // Arity for non-Leaf types.
        Py_ssize_t arity = 0;

        // Kind-specific auxiliary data.
        // For a NamedTuple, contains the tuple type object.
        // For a Dict, contains a sorted list of keys.
        // For a Custom type, contains the auxiliary data returned by the `to_iterable` function.
        py::object node_data;

        // Custom type registration. Must be null for non-custom types.
        const PyTreeTypeRegistry::Registration *custom = nullptr;

        // Number of leaf nodes in the subtree rooted at this node.
        Py_ssize_t num_leaves = 0;

        // Number of leaf and interior nodes in the subtree rooted at this node.
        Py_ssize_t num_nodes = 0;
    };

    // Helper that manufactures an instance of a node given its children.
    static py::object MakeNode(const Node &node, absl::Span<py::object> children);

    // Recursive helper used to implement FromIterableTree().
    py::object FromIterableTreeHelper(
        py::handle xs, absl::InlinedVector<PyTreeDef::Node, 1>::const_reverse_iterator *it) const;

    // Computes the node kind of a given Python object.
    static PyTreeKind GetKind(const py::handle &obj,
                              PyTreeTypeRegistry::Registration const **custom);

    template <typename T>
    void FlattenIntoImpl(py::handle handle,
                         T &leaves,  // NOLINT
                         const std::optional<py::function> &leaf_predicate);

    template <typename T>
    py::object UnflattenImpl(T leaves) const;

    // Nodes, in a post-order traversal. We use an ordered traversal to minimize allocations, and
    // post-order corresponds to the order we need to rebuild the tree structure.
    absl::InlinedVector<Node, 1> traversal;
};

}  // namespace optree
