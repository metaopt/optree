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

#include <pybind11/pybind11.h>

#include <memory>         // std::unique_ptr
#include <optional>       // std::optional, std::nullopt
#include <string>         // std::string
#include <thread>         // std::thread::id // NOLINT[build/c++11]
#include <tuple>          // std::tuple
#include <unordered_set>  // std::unordered_set
#include <utility>        // std::pair, std::make_pair
#include <vector>         // std::vector

#include "include/registry.h"
#include "include/utils.h"

namespace optree {

namespace py = pybind11;
using size_t = py::size_t;
using ssize_t = py::ssize_t;

// The maximum depth of a pytree.
constexpr ssize_t MAX_RECURSION_DEPTH = 1000;

// Test whether the given object is a leaf node.
bool IsLeaf(const py::object &object,
            const std::optional<py::function> &leaf_predicate,
            const bool &none_is_leaf = false,
            const std::string &registry_namespace = "");

// Test whether all elements in the given iterable are all leaves.
bool AllLeaves(const py::iterable &iterable,
               const std::optional<py::function> &leaf_predicate,
               const bool &none_is_leaf = false,
               const std::string &registry_namespace = "");

template <bool NoneIsLeaf>
bool IsLeafImpl(const py::handle &handle,
                const std::optional<py::function> &leaf_predicate,
                const std::string &registry_namespace);

template <bool NoneIsLeaf>
bool AllLeavesImpl(const py::iterable &iterable,
                   const std::optional<py::function> &leaf_predicate,
                   const std::string &registry_namespace);

// A PyTreeSpec describes the tree structure of a PyTree. A PyTree is a tree of Python values, where
// the interior nodes are tuples, lists, dictionaries, or user-defined containers, and the leaves
// are other objects.
class PyTreeSpec {
 private:
    struct Node;

 public:
    PyTreeSpec() = default;

    // Flatten a PyTree into a list of leaves and a PyTreeSpec.
    // Return references to the flattened objects, which might be temporary objects in the case of
    // custom PyType handlers.
    static std::pair<std::vector<py::object>, std::unique_ptr<PyTreeSpec>> Flatten(
        const py::object &tree,
        const std::optional<py::function> &leaf_predicate = std::nullopt,
        const bool &none_is_leaf = false,
        const std::string &registry_namespace = "");

    // Flatten a PyTree into a list of leaves with a list of paths and a PyTreeSpec.
    // Return references to the flattened objects, which might be temporary objects in the case of
    // custom PyType handlers.
    static std::tuple<std::vector<py::tuple>, std::vector<py::object>, std::unique_ptr<PyTreeSpec>>
    FlattenWithPath(const py::object &tree,
                    const std::optional<py::function> &leaf_predicate = std::nullopt,
                    const bool &none_is_leaf = false,
                    const std::string &registry_namespace = "");

    // Return an unflattened PyTree given an iterable of leaves and a PyTreeSpec.
    [[nodiscard]] py::object Unflatten(const py::iterable &leaves) const;

    // Flatten a PyTree up to this PyTreeSpec. 'this' must be a tree prefix of the tree-structure
    // of 'x'. For example, if we flatten a value [(1, (2, 3)), {"foo": 4}] with a PyTreeSpec [(*,
    // *), *], the result is the list of leaves [1, (2, 3), {"foo": 4}].
    [[nodiscard]] py::list FlattenUpTo(const py::object &full_tree) const;

    // Broadcast to a common suffix of this PyTreeSpec and other PyTreeSpec.
    [[nodiscard]] std::unique_ptr<PyTreeSpec> BroadcastToCommonSuffix(
        const PyTreeSpec &other) const;

    // Compose two PyTreeSpecs, replacing the leaves of this tree with copies of `inner`.
    [[nodiscard]] std::unique_ptr<PyTreeSpec> Compose(const PyTreeSpec &inner_treespec) const;

    // Map a function over a PyTree structure, applying f_leaf to each leaf, and
    // f_node(children, node_data) to each container node.
    [[nodiscard]] py::object Walk(const py::function &f_node,
                                  const py::object &f_leaf,
                                  const py::iterable &leaves) const;

    // Return paths to all leaves in the PyTreeSpec.
    [[nodiscard]] std::vector<py::tuple> Paths() const;

    // Return one-level entries of the PyTreeSpec to its children.
    [[nodiscard]] py::list Entries() const;

    // Return the one-level entry at the given index of the PyTreeSpec.
    [[nodiscard]] py::object Entry(ssize_t index) const;

    // Return the children of the PyTreeSpec.
    [[nodiscard]] std::vector<std::unique_ptr<PyTreeSpec>> Children() const;

    // Return the child at the given index of the PyTreeSpec.
    [[nodiscard]] std::unique_ptr<PyTreeSpec> Child(ssize_t index) const;

    [[nodiscard]] ssize_t GetNumLeaves() const;

    [[nodiscard]] ssize_t GetNumNodes() const;

    [[nodiscard]] ssize_t GetNumChildren() const;

    [[nodiscard]] bool GetNoneIsLeaf() const;

    [[nodiscard]] std::string GetNamespace() const;

    [[nodiscard]] py::object GetType() const;

    [[nodiscard]] PyTreeKind GetPyTreeKind() const;

    // Test whether this PyTreeSpec represents a leaf.
    [[nodiscard]] bool IsLeaf(const bool &strict = true) const;

    // Return true if this PyTreeSpec is a prefix of `other`.
    [[nodiscard]] bool IsPrefix(const PyTreeSpec &other, const bool &strict = false) const;

    // Return true if this PyTreeSpec is a suffix of `other`.
    [[nodiscard]] inline bool IsSuffix(const PyTreeSpec &other, const bool &strict = false) const {
        return other.IsPrefix(*this, strict);
    }

    bool operator==(const PyTreeSpec &other) const;
    inline bool operator!=(const PyTreeSpec &other) const { return !(*this == other); }
    inline bool operator<(const PyTreeSpec &other) const { return IsPrefix(other, true); }
    inline bool operator<=(const PyTreeSpec &other) const { return IsPrefix(other, false); }
    inline bool operator>(const PyTreeSpec &other) const { return IsSuffix(other, true); }
    inline bool operator>=(const PyTreeSpec &other) const { return IsSuffix(other, false); }

    // Return a string representation of the PyTreeSpec.
    [[nodiscard]] std::string ToString() const;

    // Return the hash value of the PyTreeSpec.
    [[nodiscard]] ssize_t HashValue() const;

    // Transform the PyTreeSpec into a picklable object.
    // Used to implement `PyTreeSpec.__getstate__`.
    [[nodiscard]] py::object ToPicklable() const;

    // Transform the object returned by `ToPicklable()` back to PyTreeSpec.
    // Used to implement `PyTreeSpec.__setstate__`.
    static std::unique_ptr<PyTreeSpec> FromPicklable(const py::object &picklable);

    // Make a PyTreeSpec representing a leaf node.
    static std::unique_ptr<PyTreeSpec> MakeLeaf(const bool &none_is_leaf = false,
                                                const std::string &registry_namespace = "");

    // Make a PyTreeSpec representing a `None` node.
    static std::unique_ptr<PyTreeSpec> MakeNone(const bool &none_is_leaf = false,
                                                const std::string &registry_namespace = "");

    // Make a PyTreeSpec out of a collection of PyTreeSpecs.
    static std::unique_ptr<PyTreeSpec> MakeFromCollection(
        const py::object &object,
        const bool &none_is_leaf = false,
        const std::string &registry_namespace = "");

 private:
    using RegistrationPtr = PyTreeTypeRegistry::RegistrationPtr;

    struct Node {
        PyTreeKind kind = PyTreeKind::Leaf;

        // Arity for non-Leaf types.
        ssize_t arity = 0;

        // Kind-specific auxiliary data.
        // For a NamedTuple/PyStructSequence, contains the tuple type object.
        // For a Dict, contains a sorted list of keys.
        // For a OrderedDict, contains a list of keys.
        // For a DefaultDict, contains a tuple of (default_factory, sorted list of keys).
        // For a Deque, contains the `maxlen` attribute.
        // For a Custom type, contains the auxiliary data returned by the `flatten_func` function.
        py::object node_data{};

        // The tuple of path entries.
        // This is optional, if not specified, `range(arity)` is used.
        // For a sequence, contains the index of the element.
        // For a mapping, contains the key of the element.
        // For a Custom type, contains the path entries returned by the `flatten_func` function.
        py::object node_entries{};

        // Custom type registration. Must be null for non-custom types.
        RegistrationPtr custom{nullptr};

        // Number of leaf nodes in the subtree rooted at this node.
        ssize_t num_leaves = 0;

        // Number of leaf and interior nodes in the subtree rooted at this node.
        ssize_t num_nodes = 0;

        // For a Dict or DefaultDict, contains the keys in insertion order.
        py::object original_keys{};
    };

    // Nodes, in a post-order traversal. We use an ordered traversal to minimize allocations, and
    // post-order corresponds to the order we need to rebuild the tree structure.
    std::vector<Node> m_traversal = reserved_vector<Node>(1);

    // Whether to treat `None` as a leaf. If false, `None` is a non-leaf node with arity 0.
    bool m_none_is_leaf = false;

    // The registry namespace used to resolve the custom pytree node types.
    std::string m_namespace{};

    // Helper that returns the string representation of a node kind.
    static std::string NodeKindToString(const Node &node);

    // Helper that manufactures an instance of a node given its children.
    static py::object MakeNode(const Node &node,
                               const py::object *children,
                               const size_t &num_children);

    // Recursive helper used to implement Flatten().
    bool FlattenInto(const py::handle &handle,
                     std::vector<py::object> &leaves,  // NOLINT[runtime/references]
                     const std::optional<py::function> &leaf_predicate,
                     const bool &none_is_leaf,
                     const std::string &registry_namespace);

    template <bool NoneIsLeaf, typename Span>
    bool FlattenIntoImpl(const py::handle &handle,
                         Span &leaves,  // NOLINT[runtime/references]
                         const ssize_t &depth,
                         const std::optional<py::function> &leaf_predicate,
                         const std::string &registry_namespace);

    // Recursive helper used to implement FlattenWithPath().
    bool FlattenIntoWithPath(const py::handle &handle,
                             std::vector<py::object> &leaves,  // NOLINT[runtime/references]
                             std::vector<py::tuple> &paths,    // NOLINT[runtime/references]
                             const std::optional<py::function> &leaf_predicate,
                             const bool &none_is_leaf,
                             const std::string &registry_namespace);

    template <bool NoneIsLeaf, typename LeafSpan, typename PathSpan, typename Stack>
    bool FlattenIntoWithPathImpl(const py::handle &handle,
                                 LeafSpan &leaves,  // NOLINT[runtime/references]
                                 PathSpan &paths,   // NOLINT[runtime/references]
                                 Stack &stack,      // NOLINT[runtime/references]
                                 const ssize_t &depth,
                                 const std::optional<py::function> &leaf_predicate,
                                 const std::string &registry_namespace);

    template <typename Span>
    py::object UnflattenImpl(const Span &leaves) const;

    static std::tuple<ssize_t, ssize_t, ssize_t, ssize_t> BroadcastToCommonSuffixImpl(
        std::vector<Node> &nodes,  // NOLINT[runtime/references]
        const std::vector<Node> &traversal,
        const ssize_t &pos,
        const std::vector<Node> &other_traversal,
        const ssize_t &other_pos);

    template <typename Span, typename Stack>
    [[nodiscard]] ssize_t PathsImpl(Span &paths,   // NOLINT[runtime/references]
                                    Stack &stack,  // NOLINT[runtime/references]
                                    const ssize_t &pos,
                                    const ssize_t &depth) const;

    [[nodiscard]] std::string ToStringImpl() const;

    // Get the hash value of the node.
    static void HashCombineNode(ssize_t &seed, const Node &node);  // NOLINT[runtime/references]

    [[nodiscard]] ssize_t HashValueImpl() const;

    template <bool NoneIsLeaf>
    static std::unique_ptr<PyTreeSpec> MakeFromCollectionImpl(const py::handle &handle,
                                                              std::string registry_namespace);

    class ThreadIndentTypeHash {
     public:
        using is_transparent = void;
        size_t operator()(const std::pair<const PyTreeSpec *, std::thread::id> &p) const;
    };

    // A set of (treespec, thread_id) pairs that are currently being represented as strings.
    inline static std::unordered_set<std::pair<const PyTreeSpec *, std::thread::id>,
                                     ThreadIndentTypeHash>
        sm_repr_running{};

    // A set of (treespec, thread_id) pairs that are currently being hashed.
    inline static std::unordered_set<std::pair<const PyTreeSpec *, std::thread::id>,
                                     ThreadIndentTypeHash>
        sm_hash_running{};
};

class PyTreeIter {
 public:
    PyTreeIter(const py::object &tree,
               const std::optional<py::function> &leaf_predicate,
               bool none_is_leaf,
               std::string registry_namespace)
        : m_agenda({std::make_pair(tree, 0)}),
          m_leaf_predicate(leaf_predicate),
          m_none_is_leaf(none_is_leaf),
          m_namespace(std::move(registry_namespace)){};

    PyTreeIter() = delete;

    ~PyTreeIter() = default;

    PyTreeIter(const PyTreeIter &) = delete;

    PyTreeIter operator=(const PyTreeIter &) = delete;

    PyTreeIter(PyTreeIter &&) = default;

    PyTreeIter &operator=(PyTreeIter &&) = default;

    [[nodiscard]] PyTreeIter &Iter() { return *this; }

    [[nodiscard]] py::object Next();

 private:
    std::vector<std::pair<py::object, ssize_t>> m_agenda;
    std::optional<py::function> m_leaf_predicate;
    bool m_none_is_leaf;
    std::string m_namespace;

    template <bool NoneIsLeaf>
    [[nodiscard]] py::object NextImpl();
};

}  // namespace optree
