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

#include <algorithm>      // std::min
#include <memory>         // std::unique_ptr
#include <optional>       // std::optional, std::nullopt
#include <string>         // std::string
#include <thread>         // std::thread::id
#include <tuple>          // std::tuple
#include <unordered_set>  // std::unordered_set
#include <utility>        // std::pair
#include <vector>         // std::vector

#include <pybind11/pybind11.h>

#include "optree/exceptions.h"
#include "optree/hashing.h"
#include "optree/registry.h"
#include "optree/synchronization.h"

namespace optree {

namespace py = pybind11;
using size_t = py::size_t;
using ssize_t = py::ssize_t;

// The maximum depth of a pytree.
#ifndef Py_C_RECURSION_LIMIT
#define Py_C_RECURSION_LIMIT 1000
#endif
#if !defined(PYPY_VERSION) && !(defined(MS_WINDOWS) && defined(Py_DEBUG))
constexpr ssize_t MAX_RECURSION_DEPTH = std::min(1000, Py_C_RECURSION_LIMIT);
#else
constexpr ssize_t MAX_RECURSION_DEPTH = std::min(500, Py_C_RECURSION_LIMIT);
#endif

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

py::module_ GetCxxModule(const std::optional<py::module_> &module = std::nullopt);

#define PYTREESPEC_SANITY_CHECK(treespec)                                                          \
    {                                                                                              \
        EXPECT_FALSE((treespec).m_traversal.empty(), "The tree node traversal is empty.");         \
        EXPECT_EQ((treespec).m_traversal.back().num_nodes,                                         \
                  py::ssize_t_cast((treespec).m_traversal.size()),                                 \
                  "The number of nodes does not match the traversal size.");                       \
    }

// A PyTreeSpec describes the tree structure of a PyTree. A PyTree is a tree of Python values, where
// the interior nodes are tuples, lists, dictionaries, or user-defined containers, and the leaves
// are other objects.
class PyTreeSpec {
private:
    struct Node;

public:
    PyTreeSpec() = default;
    ~PyTreeSpec() = default;

    PyTreeSpec(const PyTreeSpec &) = default;
    PyTreeSpec &operator=(const PyTreeSpec &) = default;
    PyTreeSpec(PyTreeSpec &&) = default;
    PyTreeSpec &operator=(PyTreeSpec &&) = default;

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

    // Flatten a PyTree up to this PyTreeSpec. 'this' must be a tree prefix of the tree-structure of
    // 'full_tree'.
    // For example, if we flatten a value [(1, (2, 3)), {"foo": 4}] with a PyTreeSpec([(*, *), *]),
    // the result is the list of leaves [1, (2, 3), {"foo": 4}].
    [[nodiscard]] py::list FlattenUpTo(const py::object &full_tree) const;

    // Broadcast to a common suffix of this PyTreeSpec and other PyTreeSpec.
    [[nodiscard]] std::unique_ptr<PyTreeSpec> BroadcastToCommonSuffix(
        const PyTreeSpec &other) const;

    // Transform a PyTreeSpec by applying `f_node(nodespec)` to nodes and `f_leaf(leafspec)` to
    // leaves.
    [[nodiscard]] std::unique_ptr<PyTreeSpec> Transform(
        const std::optional<py::function> &f_node = std::nullopt,
        const std::optional<py::function> &f_leaf = std::nullopt) const;

    // Compose two PyTreeSpecs, replacing the leaves of this tree with copies of `inner`.
    [[nodiscard]] std::unique_ptr<PyTreeSpec> Compose(const PyTreeSpec &inner_treespec) const;

    // Map a function over a PyTree structure, applying `f_leaf(leaf)` to each leaf,
    // and `f_node(node)` to each reconstructed non-leaf node.
    [[nodiscard]] py::object Traverse(
        const py::iterable &leaves,
        const std::optional<py::function> &f_node = std::nullopt,
        const std::optional<py::function> &f_leaf = std::nullopt) const;

    // Map a function over a PyTree structure, applying `f_leaf(leaf)` to each leaf,
    // and `f_node(node_type, node_data, children)` to each non-leaf node.
    [[nodiscard]] py::object Walk(const py::iterable &leaves,
                                  const std::optional<py::function> &f_node = std::nullopt,
                                  const std::optional<py::function> &f_leaf = std::nullopt) const;

    // Return paths to all leaves in the PyTreeSpec.
    [[nodiscard]] std::vector<py::tuple> Paths() const;

    // Return a list of accessors to all leaves in the PyTreeSpec.
    [[nodiscard]] std::vector<py::object> Accessors() const;

    // Return one-level entries of the PyTreeSpec to its children.
    [[nodiscard]] py::list Entries() const;

    // Return the one-level entry at the given index of the PyTreeSpec.
    [[nodiscard]] py::object Entry(ssize_t index) const;

    // Return the children of the PyTreeSpec.
    [[nodiscard]] std::vector<std::unique_ptr<PyTreeSpec>> Children() const;

    // Return the child at the given index of the PyTreeSpec.
    [[nodiscard]] std::unique_ptr<PyTreeSpec> Child(ssize_t index) const;

    // Return the one-level structure of the PyTreeSpec.
    [[nodiscard]] std::unique_ptr<PyTreeSpec> GetOneLevel(
        const std::optional<Node> &node = std::nullopt) const;

    [[nodiscard]] inline Py_ALWAYS_INLINE ssize_t GetNumLeaves() const {
        PYTREESPEC_SANITY_CHECK(*this);
        return m_traversal.back().num_leaves;
    }

    [[nodiscard]] inline Py_ALWAYS_INLINE ssize_t GetNumNodes() const {
        PYTREESPEC_SANITY_CHECK(*this);
        return py::ssize_t_cast(m_traversal.size());
    }

    [[nodiscard]] inline Py_ALWAYS_INLINE ssize_t GetNumChildren() const {
        PYTREESPEC_SANITY_CHECK(*this);
        return m_traversal.back().arity;
    }

    [[nodiscard]] inline Py_ALWAYS_INLINE bool GetNoneIsLeaf() const { return m_none_is_leaf; }

    [[nodiscard]] inline Py_ALWAYS_INLINE std::string GetNamespace() const { return m_namespace; }

    [[nodiscard]] py::object GetType(const std::optional<Node> &node = std::nullopt) const;

    [[nodiscard]] inline Py_ALWAYS_INLINE PyTreeKind GetPyTreeKind() const {
        PYTREESPEC_SANITY_CHECK(*this);
        return m_traversal.back().kind;
    }

    // Test whether this PyTreeSpec represents a leaf.
    [[nodiscard]] inline Py_ALWAYS_INLINE bool IsLeaf(const bool &strict = true) const {
        if (strict) [[likely]] {
            return GetNumNodes() == 1 && GetNumLeaves() == 1;
        }
        return GetNumNodes() == 1;
    }

    // Test whether this PyTreeSpec represents a one-level tree.
    [[nodiscard]] inline Py_ALWAYS_INLINE bool IsOneLevel() const {
        return GetNumNodes() == GetNumChildren() + 1 && GetNumLeaves() == GetNumChildren();
    }

    // Return true if this PyTreeSpec is a prefix of `other`.
    [[nodiscard]] bool IsPrefix(const PyTreeSpec &other, const bool &strict = false) const;

    // Return true if this PyTreeSpec is a suffix of `other`.
    [[nodiscard]] inline Py_ALWAYS_INLINE bool IsSuffix(const PyTreeSpec &other,
                                                        const bool &strict = false) const {
        return other.IsPrefix(*this, strict);
    }

    [[nodiscard]] bool EqualTo(const PyTreeSpec &other) const;
    inline Py_ALWAYS_INLINE bool operator==(const PyTreeSpec &other) const {
        return EqualTo(other);
    }
    inline Py_ALWAYS_INLINE bool operator!=(const PyTreeSpec &other) const {
        return !EqualTo(other);
    }
    inline Py_ALWAYS_INLINE bool operator<(const PyTreeSpec &other) const {
        return IsPrefix(other, /*strict=*/true);
    }
    inline Py_ALWAYS_INLINE bool operator<=(const PyTreeSpec &other) const {
        return IsPrefix(other, /*strict=*/false);
    }
    inline Py_ALWAYS_INLINE bool operator>(const PyTreeSpec &other) const {
        return IsSuffix(other, /*strict=*/true);
    }
    inline Py_ALWAYS_INLINE bool operator>=(const PyTreeSpec &other) const {
        return IsSuffix(other, /*strict=*/false);
    }

    // Return a string representation of the PyTreeSpec.
    [[nodiscard]] std::string ToString() const;

    // Return the hash value of the PyTreeSpec.
    [[nodiscard]] ssize_t HashValue() const;

    // Transform the PyTreeSpec into a pickleable object.
    // Used to implement `PyTreeSpec.__getstate__`.
    [[nodiscard]] py::object ToPickleable() const;

    // Transform the object returned by `ToPickleable()` back to PyTreeSpec.
    // Used to implement `PyTreeSpec.__setstate__`.
    static std::unique_ptr<PyTreeSpec> FromPickleable(const py::object &pickleable);

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

    // Check if should preserve the insertion order of the dictionary keys during flattening.
    static inline Py_ALWAYS_INLINE bool IsDictInsertionOrdered(
        const std::string &registry_namespace,
        const bool &inherit_global_namespace = true) {
        const scoped_read_lock_guard lock{sm_is_dict_insertion_ordered_mutex};

        return (sm_is_dict_insertion_ordered.find(registry_namespace) !=
                sm_is_dict_insertion_ordered.end()) ||
               (inherit_global_namespace &&
                sm_is_dict_insertion_ordered.find("") != sm_is_dict_insertion_ordered.end());
    }

    // Set the namespace to preserve the insertion order of the dictionary keys during flattening.
    static inline Py_ALWAYS_INLINE void SetDictInsertionOrdered(
        const bool &mode,
        const std::string &registry_namespace) {
        const scoped_write_lock_guard lock{sm_is_dict_insertion_ordered_mutex};

        if (mode) [[likely]] {
            sm_is_dict_insertion_ordered.insert(registry_namespace);
        } else [[unlikely]] {
            sm_is_dict_insertion_ordered.erase(registry_namespace);
        }
    }

    friend void BuildModule(py::module_ &mod);  // NOLINT[runtime/references]

private:
    using RegistrationPtr = PyTreeTypeRegistry::RegistrationPtr;
    using ThreadedIdentity = std::pair<const optree::PyTreeSpec *, std::thread::id>;

    struct Node {
        PyTreeKind kind = PyTreeKind::Leaf;

        // Arity for non-Leaf types.
        ssize_t arity = 0;

        // Kind-specific metadata.
        // For a NamedTuple/PyStructSequence, contains the tuple type object.
        // For a Dict, contains a sorted list of keys.
        // For a OrderedDict, contains a list of keys.
        // For a DefaultDict, contains a tuple of (default_factory, sorted list of keys).
        // For a Deque, contains the `maxlen` attribute.
        // For a Custom type, contains the metadata returned by the `flatten_func` function.
        py::object node_data{};

        // The tuple of path entries.
        // For a Custom type, contains the path entries returned by the `flatten_func` function.
        // This is optional, if not specified, `range(arity)` is used.
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
    std::vector<Node> m_traversal{};

    // Whether to treat `None` as a leaf. If false, `None` is a non-leaf node with arity 0.
    bool m_none_is_leaf = false;

    // The registry namespace used to resolve the custom pytree node types.
    std::string m_namespace{};

    // Helper that returns the string representation of a node kind.
    static std::string NodeKindToString(const Node &node);

    // Helper that manufactures an instance of a node given its children.
    static py::object MakeNode(const Node &node,
                               const py::object children[],  // NOLINT[hicpp-avoid-c-arrays]
                               const size_t &num_children);

    // Helper that identifies the path entry class for a node.
    static py::object GetPathEntryType(const Node &node);

    // Recursive helper used to implement Flatten().
    bool FlattenInto(const py::handle &handle,
                     std::vector<py::object> &leaves,  // NOLINT[runtime/references]
                     const std::optional<py::function> &leaf_predicate,
                     const bool &none_is_leaf,
                     const std::string &registry_namespace);

    template <bool NoneIsLeaf, bool DictShouldBeSorted, typename Span>
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

    template <bool NoneIsLeaf,
              bool DictShouldBeSorted,
              typename LeafSpan,
              typename PathSpan,
              typename Stack>
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

    template <bool PassRawNode = true>
    [[nodiscard]] py::object WalkImpl(
        const py::iterable &leaves,
        const std::optional<py::function> &f_node = std::nullopt,
        const std::optional<py::function> &f_leaf = std::nullopt) const;

    template <typename Span, typename Stack>
    [[nodiscard]] ssize_t PathsImpl(Span &paths,   // NOLINT[runtime/references]
                                    Stack &stack,  // NOLINT[runtime/references]
                                    const ssize_t &pos,
                                    const ssize_t &depth) const;

    template <typename Span, typename Stack>
    [[nodiscard]] ssize_t AccessorsImpl(Span &accessors,  // NOLINT[runtime/references]
                                        Stack &stack,     // NOLINT[runtime/references]
                                        const ssize_t &pos,
                                        const ssize_t &depth) const;

    [[nodiscard]] std::string ToStringImpl() const;

    [[nodiscard]] ssize_t HashValueImpl() const;

    template <bool NoneIsLeaf>
    static std::unique_ptr<PyTreeSpec> MakeFromCollectionImpl(const py::handle &handle,
                                                              std::string registry_namespace);

    // Used in tp_traverse for GC support.
    static int PyTpTraverse(PyObject *self_base, visitproc visit, void *arg);

    // A set of namespaces that preserve the insertion order of the dictionary keys during
    // flattening.
    static inline std::unordered_set<std::string> sm_is_dict_insertion_ordered{};
    static inline read_write_mutex sm_is_dict_insertion_ordered_mutex{};
};

class PyTreeIter {
public:
    explicit PyTreeIter(const py::object &tree,
                        const std::optional<py::function> &leaf_predicate,
                        const bool &none_is_leaf,
                        const std::string &registry_namespace)
        : m_root{tree},
          m_agenda{{{tree, 0}}},
          m_leaf_predicate{leaf_predicate},
          m_none_is_leaf{none_is_leaf},
          m_namespace{registry_namespace},
          m_is_dict_insertion_ordered{PyTreeSpec::IsDictInsertionOrdered(registry_namespace)} {}

    PyTreeIter() = delete;
    ~PyTreeIter() = default;

    PyTreeIter(const PyTreeIter &) = delete;
    PyTreeIter &operator=(const PyTreeIter &) = delete;
    PyTreeIter(PyTreeIter &&) = delete;
    PyTreeIter &operator=(PyTreeIter &&) = delete;

    [[nodiscard]] PyTreeIter &Iter() noexcept { return *this; }

    [[nodiscard]] py::object Next();

    friend void BuildModule(py::module_ &mod);  // NOLINT[runtime/references]

private:
    const py::object m_root;
    std::vector<std::pair<py::object, ssize_t>> m_agenda;
    const std::optional<py::function> m_leaf_predicate;
    const bool m_none_is_leaf;
    const std::string m_namespace;
    const bool m_is_dict_insertion_ordered;
#ifdef Py_GIL_DISABLED
    mutable mutex m_mutex{};
#endif

    template <bool NoneIsLeaf>
    [[nodiscard]] py::object NextImpl();

    // Used in tp_traverse for GC support.
    static int PyTpTraverse(PyObject *self_base, visitproc visit, void *arg);
};

}  // namespace optree
