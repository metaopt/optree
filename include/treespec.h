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

#pragma once

#include <absl/container/inlined_vector.h>
#include <absl/hash/hash.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "include/registry.h"
#include "include/utils.h"

namespace optree {

// The maximum depth of a pytree.
#ifdef _WIN32
constexpr ssize_t MAX_RECURSION_DEPTH = 4000;
#else
constexpr ssize_t MAX_RECURSION_DEPTH = 10000;
#endif

// A PyTreeSpec describes the tree structure of a PyTree. A PyTree is a tree of Python values, where
// the interior nodes are tuples, lists, dictionaries, or user-defined containers, and the leaves
// are other objects.
class PyTreeSpec {
 private:
    struct Node;

 public:
    PyTreeSpec() = default;

    // Flattens a PyTree into a list of leaves and a PyTreeSpec.
    // Returns references to the flattened objects, which might be temporary objects in the case of
    // custom PyType handlers.
    static std::pair<std::vector<py::object>, std::unique_ptr<PyTreeSpec>> Flatten(
        const py::handle &tree,
        const std::optional<py::function> &leaf_predicate = std::nullopt,
        const bool &none_is_leaf = false,
        const std::string &registry_namespace = "");

    // Recursive helper used to implement Flatten().
    bool FlattenInto(const py::handle &handle,
                     std::vector<py::object> &leaves,  // NOLINT
                     const std::optional<py::function> &leaf_predicate,
                     const bool &none_is_leaf,
                     const std::string &registry_namespace);

    // Flattens a PyTree into a list of leaves with a list of paths and a PyTreeSpec.
    // Returns references to the flattened objects, which might be temporary objects in the case of
    // custom PyType handlers.
    static std::tuple<std::vector<py::object>, std::vector<py::object>, std::unique_ptr<PyTreeSpec>>
    FlattenWithPath(const py::handle &tree,
                    const std::optional<py::function> &leaf_predicate = std::nullopt,
                    const bool &none_is_leaf = false,
                    const std::string &registry_namespace = "");

    // Recursive helper used to implement FlattenWithPath().
    bool FlattenIntoWithPath(const py::handle &handle,
                             std::vector<py::object> &leaves,  // NOLINT
                             std::vector<py::object> &paths,   // NOLINT
                             const std::optional<py::function> &leaf_predicate,
                             const bool &none_is_leaf,
                             const std::string &registry_namespace);

    // Flattens a PyTree up to this PyTreeSpec. 'this' must be a tree prefix of the tree-structure
    // of 'x'. For example, if we flatten a value [(1, (2, 3)), {"foo": 4}] with a PyTreeSpec [(*,
    // *), *], the result is the list of leaves [1, (2, 3), {"foo": 4}].
    py::list FlattenUpTo(const py::handle &full_tree) const;

    // Tests whether the given list is a flat list of leaves.
    static bool AllLeaves(const py::iterable &iterable,
                          const bool &none_is_leaf = false,
                          const std::string &registry_namespace = "");

    // Returns an unflattened PyTree given an iterable of leaves and a PyTreeSpec.
    py::object Unflatten(const py::iterable &leaves) const;

    // Composes two PyTreeSpecs, replacing the leaves of this tree with copies of `inner`.
    std::unique_ptr<PyTreeSpec> Compose(const PyTreeSpec &inner_treespec) const;

    // Makes a Tuple PyTreeSpec out of a vector of PyTreeSpecs.
    static std::unique_ptr<PyTreeSpec> Tuple(const std::vector<PyTreeSpec> &treespecs,
                                             const bool &none_is_leaf);

    // Makes a PyTreeSpec representing a leaf node.
    static std::unique_ptr<PyTreeSpec> Leaf(const bool &none_is_leaf);

    // Makes a PyTreeSpec representing a `None` node.
    static std::unique_ptr<PyTreeSpec> None(const bool &none_is_leaf);

    std::vector<std::unique_ptr<PyTreeSpec>> Children() const;

    // Maps a function over a PyTree structure, applying f_leaf to each leaf, and
    // f_node(node, node_data) to each container node.
    py::object Walk(const py::function &f_node,
                    const py::handle &f_leaf,
                    const py::iterable &leaves) const;

    ssize_t num_leaves() const;

    ssize_t num_nodes() const;

    bool get_none_is_leaf() const;

    std::string get_namespace() const;

    bool operator==(const PyTreeSpec &other) const;
    bool operator!=(const PyTreeSpec &other) const;

    size_t Hash() const;

    template <typename H>
    friend H AbslHashValue(H h, const Node &n) {
        h = H::combine(std::move(h), n.kind, n.arity, n.custom);
        return h;
    }

    template <typename H>
    friend H AbslHashValue(H h, const PyTreeSpec &t) {
        h = H::combine(std::move(h), t.m_traversal);
        return h;
    }

    std::string ToString() const;

    // Transforms the PyTreeSpec into a picklable object.
    // Used to implement `PyTreeSpec.__getstate__`.
    py::object ToPicklable() const;

    // Transforms the object returned by `ToPicklable()` back to PyTreeSpec.
    // Used to implement `PyTreeSpec.__setstate__`.
    static PyTreeSpec FromPicklable(const py::object &picklable);

 private:
    struct Node {
        PyTreeKind kind = PyTreeKind::Leaf;

        // Arity for non-Leaf types.
        ssize_t arity = 0;

        // Kind-specific auxiliary data.
        // For a NamedTuple, contains the tuple type object.
        // For a Dict, contains a sorted list of keys.
        // For a OrderedDict, contains a list of keys.
        // For a DefaultDict, contains a tuple of (default_factory, sorted list of keys).
        // For a Deque, contains the `maxlen` attribute.
        // For a Custom type, contains the auxiliary data returned by the `to_iterable` function.
        py::object node_data;

        // The tuple of path entries.
        // This is optional, if not specified, `range(arity)` is used.
        // For a sequence, contains the index of the element.
        // For a mapping, contains the key of the element.
        // For a Custom type, contains the path entries returned by the `to_iterable` function.
        py::object node_entries;

        // Custom type registration. Must be null for non-custom types.
        const PyTreeTypeRegistry::Registration *custom = nullptr;

        // Number of leaf nodes in the subtree rooted at this node.
        ssize_t num_leaves = 0;

        // Number of leaf and interior nodes in the subtree rooted at this node.
        ssize_t num_nodes = 0;
    };

    // Nodes, in a post-order traversal. We use an ordered traversal to minimize allocations, and
    // post-order corresponds to the order we need to rebuild the tree structure.
    absl::InlinedVector<Node, 1> m_traversal;

    // Whether to treat `None` as a leaf. If false, `None` is a non-leaf node with arity 0.
    bool m_none_is_leaf = false;

    // The registry namespace used to resolve the custom pytree node types
    std::string m_namespace;

    // Helper that manufactures an instance of a node given its children.
    static py::object MakeNode(const Node &node, const absl::Span<py::object> &children);

    // Computes the node kind of a given Python object.
    template <bool NoneIsLeaf>
    static PyTreeKind GetKind(const py::handle &handle,
                              PyTreeTypeRegistry::Registration const **custom,
                              const std::string &registry_namespace);

    template <bool NoneIsLeaf, typename Span>
    bool FlattenIntoImpl(const py::handle &handle,
                         Span &leaves,  // NOLINT
                         const ssize_t &depth,
                         const std::optional<py::function> &leaf_predicate,
                         const std::string &registry_namespace);

    template <bool NoneIsLeaf, typename Span, typename Stack>
    bool FlattenIntoWithPathImpl(const py::handle &handle,
                                 Span &leaves,  // NOLINT
                                 Span &paths,   // NOLINT
                                 Stack &stack,  // NOLINT
                                 const ssize_t &depth,
                                 const std::optional<py::function> &leaf_predicate,
                                 const std::string &registry_namespace);

    py::list FlattenUpToImpl(const py::handle &full_tree) const;

    template <bool NoneIsLeaf>
    static bool AllLeavesImpl(const py::iterable &iterable, const std::string &registry_namespace);

    template <typename Span>
    py::object UnflattenImpl(const Span &leaves) const;

    static PyTreeSpec FromPicklableImpl(const py::object &picklable);
};

}  // namespace optree
