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

#include <algorithm>  // std::copy, std::reverse
#include <iterator>   // std::back_inserter
#include <memory>     // std::unique_ptr, std::make_unique
#include <optional>   // std::optional
#include <sstream>    // std::ostringstream
#include <string>     // std::string
#include <tuple>      // std::tuple
#include <utility>    // std::move
#include <vector>     // std::vector

#include "optree/optree.h"

namespace optree {

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ py::object PyTreeSpec::MakeNode(const Node& node,
                                           // NOLINTNEXTLINE[cppcoreguidelines-avoid-c-arrays]
                                           const py::object children[],
                                           const size_t& num_children) {
    EXPECT_EQ(py::ssize_t_cast(num_children), node.arity, "Node arity did not match.");
    EXPECT_TRUE(children != nullptr || num_children == 0, "Node children is null.");

    switch (node.kind) {
        case PyTreeKind::Leaf:
            INTERNAL_ERROR("PyTreeSpec::MakeNode() not implemented for leaves.");

        case PyTreeKind::None: {
            return py::none();
        }

        case PyTreeKind::Tuple:
        case PyTreeKind::NamedTuple:
        case PyTreeKind::StructSequence: {
            py::tuple tuple{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                TupleSetItem(tuple, i, children[i]);
            }
            if (node.kind == PyTreeKind::NamedTuple) [[unlikely]] {
                const scoped_critical_section cs{node.node_data};
                return node.node_data(*tuple);
            }
            if (node.kind == PyTreeKind::StructSequence) [[unlikely]] {
                const scoped_critical_section cs{node.node_data};
                return node.node_data(std::move(tuple));
            }
            return tuple;
        }

        case PyTreeKind::List:
        case PyTreeKind::Deque: {
            py::list list{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                ListSetItem(list, i, children[i]);
            }
            if (node.kind == PyTreeKind::Deque) [[unlikely]] {
                return PyDequeTypeObject(std::move(list), py::arg("maxlen") = node.node_data);
            }
            return list;
        }

        case PyTreeKind::Dict:
        case PyTreeKind::OrderedDict:
        case PyTreeKind::DefaultDict: {
            py::dict dict{};
            const scoped_critical_section2 cs{node.node_data, node.original_keys};
            if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
                EXPECT_EQ(TupleGetSize(node.node_data), 2, "Number of metadata mismatch.");
            }
            const auto keys = (node.kind != PyTreeKind::DefaultDict
                                   ? py::reinterpret_borrow<py::list>(node.node_data)
                                   : TupleGetItemAs<py::list>(node.node_data, 1));
            if (node.original_keys) [[unlikely]] {
                for (ssize_t i = 0; i < node.arity; ++i) {
                    DictSetItem(dict, ListGetItem(node.original_keys, i), py::none());
                }
            }
            for (ssize_t i = 0; i < node.arity; ++i) {
                // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                DictSetItem(dict, ListGetItem(keys, i), children[i]);
            }
            if (node.kind == PyTreeKind::OrderedDict) [[unlikely]] {
                return PyOrderedDictTypeObject(std::move(dict));
            }
            if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
                const py::object default_factory = TupleGetItem(node.node_data, 0);
                return EVALUATE_WITH_LOCK_HELD(
                    PyDefaultDictTypeObject(default_factory, std::move(dict)),
                    default_factory);
            }
            return dict;
        }

        case PyTreeKind::Custom: {
            const py::tuple tuple{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                TupleSetItem(tuple, i, children[i]);
            }
            return EVALUATE_WITH_LOCK_HELD2(node.custom->unflatten_func(node.node_data, tuple),
                                            node.node_data,
                                            node.custom->unflatten_func);
        }

        default:
            INTERNAL_ERROR();
    }
}

/*static*/ py::object PyTreeSpec::GetPathEntryType(const Node& node) {
    switch (node.kind) {
        case PyTreeKind::Leaf:
        case PyTreeKind::None: {
            return py::none();
        }

        case PyTreeKind::Tuple:
        case PyTreeKind::List:
        case PyTreeKind::Deque: {
            PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
            return storage
                .call_once_and_store_result(
                    []() -> py::object { return py::getattr(GetCxxModule(), "SequenceEntry"); })
                .get_stored();
        }

        case PyTreeKind::Dict:
        case PyTreeKind::OrderedDict:
        case PyTreeKind::DefaultDict: {
            PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
            return storage
                .call_once_and_store_result(
                    []() -> py::object { return py::getattr(GetCxxModule(), "MappingEntry"); })
                .get_stored();
        }

        case PyTreeKind::NamedTuple: {
            PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
            return storage
                .call_once_and_store_result(
                    []() -> py::object { return py::getattr(GetCxxModule(), "NamedTupleEntry"); })
                .get_stored();
        }

        case PyTreeKind::StructSequence: {
            PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
            return storage
                .call_once_and_store_result([]() -> py::object {
                    return py::getattr(GetCxxModule(), "StructSequenceEntry");
                })
                .get_stored();
        }

        case PyTreeKind::Custom: {
            EXPECT_NE(node.custom, nullptr, "The custom registration is null.");
            EXPECT_TRUE(node.custom->path_entry_type, "The path entry type is null.");
            return node.custom->path_entry_type;
        }

        default:
            INTERNAL_ERROR();
    }
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ std::tuple<ssize_t, ssize_t, ssize_t, ssize_t> PyTreeSpec::BroadcastToCommonSuffixImpl(
    std::vector<Node>& nodes,
    const std::vector<Node>& traversal,
    const ssize_t& pos,
    const std::vector<Node>& other_traversal,
    const ssize_t& other_pos) {
    const Node& root = traversal.at(pos);
    const Node& other_root = other_traversal.at(other_pos);
    EXPECT_GE(pos + 1,
              root.num_nodes,
              "PyTreeSpec::BroadcastToCommonSuffix() walked off start of array "
              "for the current PyTreeSpec.");
    EXPECT_GE(other_pos + 1,
              other_root.num_nodes,
              "PyTreeSpec::BroadcastToCommonSuffix() walked off start of array "
              "for the other PyTreeSpec.");

    ssize_t cur = pos - 1;
    ssize_t other_cur = other_pos - 1;

    if (root.kind == PyTreeKind::Leaf) [[likely]] {
        std::copy(other_traversal.crend() - (other_pos + 1),
                  other_traversal.crend() - (other_pos - other_root.num_nodes + 1),
                  std::back_inserter(nodes));
        other_cur -= other_root.num_nodes - 1;
        return {pos - cur, other_pos - other_cur, other_root.num_nodes, other_root.num_leaves};
    }
    if (other_root.kind == PyTreeKind::Leaf) [[likely]] {
        std::copy(traversal.crend() - (pos + 1),
                  traversal.crend() - (pos - root.num_nodes + 1),
                  std::back_inserter(nodes));
        cur -= root.num_nodes - 1;
        return {pos - cur, other_pos - other_cur, root.num_nodes, root.num_leaves};
    }
    if (root.kind == PyTreeKind::None) [[unlikely]] {
        if (other_root.kind != PyTreeKind::None) [[unlikely]] {
            std::ostringstream oss{};
            oss << "PyTreeSpecs have incompatible node types; expected type: "
                << NodeKindToString(root) << ", got: " << NodeKindToString(other_root) << ".";
            throw py::value_error(oss.str());
        }

        nodes.emplace_back(root);
        return {pos - cur, other_pos - other_cur, root.num_nodes, root.num_leaves};
    }

    Node node{
        .kind = root.kind,
        .arity = root.arity,
        .node_data = root.node_data,
        .custom = root.custom,
        .num_leaves = 0,
        .num_nodes = 1,
        .original_keys = root.original_keys,
    };
    switch (root.kind) {
        case PyTreeKind::Tuple:
        case PyTreeKind::List:
        case PyTreeKind::Deque: {
            if (root.kind != other_root.kind) [[unlikely]] {
                std::ostringstream oss{};
                oss << "PyTreeSpecs have incompatible node types; expected type: "
                    << NodeKindToString(root) << ", got: " << NodeKindToString(other_root) << ".";
                throw py::value_error(oss.str());
            }
            if (root.arity != other_root.arity) [[unlikely]] {
                std::ostringstream oss{};
                oss << NodeKindToString(root) << " arity mismatch; expected: " << root.arity
                    << ", got: " << other_root.arity << ".";
                throw py::value_error(oss.str());
            }
            break;
        }

        case PyTreeKind::Dict:
        case PyTreeKind::OrderedDict:
        case PyTreeKind::DefaultDict: {
            if (other_root.kind != PyTreeKind::Dict && other_root.kind != PyTreeKind::OrderedDict &&
                other_root.kind != PyTreeKind::DefaultDict) [[unlikely]] {
                std::ostringstream oss{};
                oss << "PyTreeSpecs have incompatible node types; expected type: "
                    << NodeKindToString(root) << ", got: " << NodeKindToString(other_root) << ".";
                throw py::value_error(oss.str());
            }

            const scoped_critical_section2 cs{root.node_data, other_root.node_data};
            const auto expected_keys = (root.kind != PyTreeKind::DefaultDict
                                            ? py::reinterpret_borrow<py::list>(root.node_data)
                                            : TupleGetItemAs<py::list>(root.node_data, 1));
            auto other_keys = (other_root.kind != PyTreeKind::DefaultDict
                                   ? py::reinterpret_borrow<py::list>(other_root.node_data)
                                   : TupleGetItemAs<py::list>(other_root.node_data, 1));
            const py::dict dict{};
            for (ssize_t i = 0; i < other_root.arity; ++i) {
                DictSetItem(dict, ListGetItem(other_keys, i), py::int_(i));
            }
            if (!DictKeysEqual(expected_keys, dict)) [[unlikely]] {
                TotalOrderSort(other_keys);
                const auto [missing_keys, extra_keys] = DictKeysDifference(expected_keys, dict);
                std::ostringstream key_difference_sstream{};
                if (ListGetSize(missing_keys) != 0) [[likely]] {
                    key_difference_sstream << ", missing key(s): " << PyRepr(missing_keys);
                }
                if (ListGetSize(extra_keys) != 0) [[likely]] {
                    key_difference_sstream << ", extra key(s): " << PyRepr(extra_keys);
                }
                std::ostringstream oss{};
                oss << "dictionary key mismatch; expected key(s): " << PyRepr(expected_keys)
                    << ", got key(s): " + PyRepr(other_keys) << key_difference_sstream.str() << ".";
                throw py::value_error(oss.str());
            }

            const size_t start_num_nodes = nodes.size();
            nodes.emplace_back(std::move(node));
            auto other_curs = reserved_vector<ssize_t>(other_root.arity);
            for (ssize_t i = 0; i < other_root.arity; ++i) {
                other_curs.emplace_back(other_cur);
                other_cur -= other_traversal.at(other_cur).num_nodes;
            }
            std::reverse(other_curs.begin(), other_curs.end());
            const ssize_t last_other_cur = other_cur;
            for (ssize_t i = root.arity - 1; i >= 0; --i) {
                const py::object key = ListGetItem(expected_keys, i);
                other_cur = other_curs[py::cast<ssize_t>(DictGetItem(dict, key))];
                const auto [num_nodes, other_num_nodes, new_num_nodes, new_num_leaves] =
                    // NOLINTNEXTLINE[misc-no-recursion]
                    BroadcastToCommonSuffixImpl(nodes, traversal, cur, other_traversal, other_cur);
                cur -= num_nodes;
                nodes[start_num_nodes].num_nodes += new_num_nodes;
                nodes[start_num_nodes].num_leaves += new_num_leaves;
            }
            return {pos - cur,
                    other_pos - last_other_cur,
                    nodes[start_num_nodes].num_nodes,
                    nodes[start_num_nodes].num_leaves};
        }

        case PyTreeKind::NamedTuple:
        case PyTreeKind::StructSequence: {
            if (root.kind != other_root.kind) [[unlikely]] {
                std::ostringstream oss{};
                oss << "PyTreeSpecs have incompatible node types; expected type: "
                    << NodeKindToString(root) << ", got: " << NodeKindToString(other_root) << ".";
                throw py::value_error(oss.str());
            }
            if (root.arity != other_root.arity) [[unlikely]] {
                std::ostringstream oss{};
                oss << (root.kind == PyTreeKind::NamedTuple ? "namedtuple" : "PyStructSequence")
                    << " arity mismatch; expected: " << root.arity << ", got: " << other_root.arity
                    << ".";
                throw py::value_error(oss.str());
            }
            if (root.node_data.not_equal(other_root.node_data)) [[unlikely]] {
                std::ostringstream oss{};
                oss << (root.kind == PyTreeKind::NamedTuple ? "namedtuple" : "PyStructSequence")
                    << " type mismatch; expected type: " << NodeKindToString(root)
                    << ", got type: " << NodeKindToString(other_root) << ".";
                throw py::value_error(oss.str());
            }
            break;
        }

        case PyTreeKind::Custom: {
            if (root.kind != other_root.kind) [[unlikely]] {
                std::ostringstream oss{};
                oss << "PyTreeSpecs have incompatible node types; expected type: "
                    << NodeKindToString(root) << ", got: " << NodeKindToString(other_root) << ".";
                throw py::value_error(oss.str());
            }
            if (!root.custom->type.is(other_root.custom->type)) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Custom node type mismatch; expected type: " << NodeKindToString(root)
                    << ", got type: " << NodeKindToString(other_root) << ".";
                throw py::value_error(oss.str());
            }
            if (root.arity != other_root.arity) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Custom type arity mismatch; expected: " << root.arity
                    << ", got: " << other_root.arity << ".";
                throw py::value_error(oss.str());
            }
            {
                const scoped_critical_section2 cs{root.node_data, other_root.node_data};
                if (root.node_data.not_equal(other_root.node_data)) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "Mismatch custom node data; expected: " << PyRepr(root.node_data)
                        << ", got: " << PyRepr(other_root.node_data) << ".";
                    throw py::value_error(oss.str());
                }
            }
            break;
        }

        case PyTreeKind::Leaf:
        case PyTreeKind::None:
        default:
            INTERNAL_ERROR();
    }

    const size_t start_num_nodes = nodes.size();
    nodes.emplace_back(std::move(node));
    for (ssize_t i = root.arity - 1; i >= 0; --i) {
        const auto [num_nodes, other_num_nodes, new_num_nodes, new_num_leaves] =
            // NOLINTNEXTLINE[misc-no-recursion]
            BroadcastToCommonSuffixImpl(nodes, traversal, cur, other_traversal, other_cur);
        cur -= num_nodes;
        other_cur -= other_num_nodes;
        nodes[start_num_nodes].num_nodes += new_num_nodes;
        nodes[start_num_nodes].num_leaves += new_num_leaves;
    }
    return {pos - cur,
            other_pos - other_cur,
            nodes[start_num_nodes].num_nodes,
            nodes[start_num_nodes].num_leaves};
}

std::unique_ptr<PyTreeSpec> PyTreeSpec::BroadcastToCommonSuffix(const PyTreeSpec& other) const {
    PYTREESPEC_SANITY_CHECK(*this);
    PYTREESPEC_SANITY_CHECK(other);

    if (m_none_is_leaf != other.m_none_is_leaf) [[unlikely]] {
        throw py::value_error("PyTreeSpecs must have the same none_is_leaf value.");
    }
    if (!m_namespace.empty() && !other.m_namespace.empty() && m_namespace != other.m_namespace)
        [[unlikely]] {
        std::ostringstream oss{};
        oss << "PyTreeSpecs must have the same namespace, got " << PyRepr(m_namespace) << " vs. "
            << PyRepr(other.m_namespace) << ".";
        throw py::value_error(oss.str());
    }

    auto treespec = std::make_unique<PyTreeSpec>();
    treespec->m_none_is_leaf = m_none_is_leaf;
    if (other.m_namespace.empty()) [[likely]] {
        treespec->m_namespace = m_namespace;
    } else [[unlikely]] {
        treespec->m_namespace = other.m_namespace;
    }

    const ssize_t num_nodes = GetNumNodes();
    const ssize_t other_num_nodes = other.GetNumNodes();

    const auto [num_nodes_walked, other_num_nodes_walked, new_num_nodes, new_num_leaves] =
        BroadcastToCommonSuffixImpl(treespec->m_traversal,
                                    m_traversal,
                                    num_nodes - 1,
                                    other.m_traversal,
                                    other_num_nodes - 1);
    std::reverse(treespec->m_traversal.begin(), treespec->m_traversal.end());
    EXPECT_EQ(num_nodes_walked,
              num_nodes,
              "`pos != 0` at end of PyTreeSpec::BroadcastToCommonSuffix() "
              "for the current PyTreeSpec.");
    EXPECT_EQ(other_num_nodes_walked,
              other_num_nodes,
              "`pos != 0` at end of PyTreeSpec::BroadcastToCommonSuffix() "
              "for the other PyTreeSpec.");
    EXPECT_EQ(new_num_nodes,
              treespec->GetNumNodes(),
              "PyTreeSpec::BroadcastToCommonSuffix() mismatched number of nodes.");
    EXPECT_EQ(new_num_leaves,
              treespec->GetNumLeaves(),
              "PyTreeSpec::BroadcastToCommonSuffix() mismatched number of leaves.");
    treespec->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*treespec);
    return treespec;
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
std::unique_ptr<PyTreeSpec> PyTreeSpec::Transform(const std::optional<py::function>& f_node,
                                                  const std::optional<py::function>& f_leaf) const {
    PYTREESPEC_SANITY_CHECK(*this);

    if (!f_node && !f_leaf) [[unlikely]] {
        return std::make_unique<PyTreeSpec>(*this);
    }

    const auto transform =
        [this, &f_node, &f_leaf](const Node& node) -> std::unique_ptr<PyTreeSpec> {
        auto nodespec = GetOneLevel(node);

        const auto& func = (node.kind == PyTreeKind::Leaf ? f_leaf : f_node);
        if (!func) [[likely]] {
            return nodespec;
        }

        const py::object out = EVALUATE_WITH_LOCK_HELD((*func)(std::move(nodespec)), *func);
        if (!py::isinstance<PyTreeSpec>(out)) [[unlikely]] {
            std::ostringstream oss{};
            oss << "Expected the PyTreeSpec transform function returns a PyTreeSpec, got "
                << PyRepr(out) << " (input: " << GetOneLevel(node)->ToString() << ").";
            throw py::type_error(oss.str());
        }
        return std::make_unique<PyTreeSpec>(thread_safe_cast<PyTreeSpec&>(out));
    };

    auto treespec = std::make_unique<PyTreeSpec>();
    std::string common_registry_namespace = m_namespace;
    ssize_t num_extra_leaves = 0;
    ssize_t num_extra_nodes = 0;
    auto pending_num_leaves_nodes = reserved_vector<std::pair<ssize_t, ssize_t>>(4);
    for (const Node& node : m_traversal) {
        auto transformed = transform(node);
        if (transformed->m_none_is_leaf != m_none_is_leaf) [[unlikely]] {
            std::ostringstream oss{};
            oss << "Expected the PyTreeSpec transform function returns "
                   "a PyTreeSpec with the same value of "
                << (m_none_is_leaf ? "`none_is_leaf=True`" : "`none_is_leaf=False`")
                << " as the input, got " << transformed->ToString()
                << " (input: " << GetOneLevel(node)->ToString() << ").";
            throw py::value_error(oss.str());
        }
        if (!transformed->m_namespace.empty()) [[unlikely]] {
            if (common_registry_namespace.empty()) [[likely]] {
                common_registry_namespace = transformed->m_namespace;
            } else if (transformed->m_namespace != common_registry_namespace) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Expected the PyTreeSpec transform function returns "
                       "a PyTreeSpec with namespace "
                    << PyRepr(common_registry_namespace) << ", got "
                    << PyRepr(transformed->m_namespace) << ".";
                throw py::value_error(oss.str());
            }
        }
        if (node.kind != PyTreeKind::Leaf) [[likely]] {
            if (transformed->GetNumLeaves() != node.arity) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Expected the PyTreeSpec transform function returns "
                       "a PyTreeSpec with the same number of arity as the input ("
                    << node.arity << "), got " << transformed->ToString()
                    << " (input: " << GetOneLevel(node)->ToString() << ").";
                throw py::value_error(oss.str());
            }
            if (transformed->GetNumNodes() != node.arity + 1) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Expected the PyTreeSpec transform function returns an one-level PyTreeSpec "
                       "as the input, got "
                    << transformed->ToString() << " (input: " << GetOneLevel(node)->ToString()
                    << ").";
                throw py::value_error(oss.str());
            }
            auto& subroot = treespec->m_traversal.emplace_back(transformed->m_traversal.back());
            EXPECT_GE(py::ssize_t_cast(pending_num_leaves_nodes.size()),
                      node.arity,
                      "PyTreeSpec::Transform() walked off start of array.");
            subroot.num_leaves = 0;
            subroot.num_nodes = 1;
            for (ssize_t i = 0; i < node.arity; ++i) {
                const auto& [num_leaves, num_nodes] = pending_num_leaves_nodes.back();
                pending_num_leaves_nodes.pop_back();
                subroot.num_leaves += num_leaves;
                subroot.num_nodes += num_nodes;
            }
            pending_num_leaves_nodes.emplace_back(subroot.num_leaves, subroot.num_nodes);
        } else [[unlikely]] {
            std::copy(transformed->m_traversal.cbegin(),
                      transformed->m_traversal.cend(),
                      std::back_inserter(treespec->m_traversal));
            const ssize_t num_leaves = transformed->GetNumLeaves();
            const ssize_t num_nodes = transformed->GetNumNodes();
            num_extra_leaves += num_leaves - 1;
            num_extra_nodes += num_nodes - 1;
            pending_num_leaves_nodes.emplace_back(num_leaves, num_nodes);
        }
    }
    EXPECT_EQ(pending_num_leaves_nodes.size(),
              1,
              "PyTreeSpec::Transform() did not yield a singleton.");

    const auto& root = treespec->m_traversal.back();
    EXPECT_EQ(root.num_leaves,
              GetNumLeaves() + num_extra_leaves,
              "Number of transformed tree leaves mismatch.");
    EXPECT_EQ(root.num_nodes,
              GetNumNodes() + num_extra_nodes,
              "Number of transformed tree nodes mismatch.");
    EXPECT_EQ(root.num_leaves,
              treespec->GetNumLeaves(),
              "Number of transformed tree leaves mismatch.");
    EXPECT_EQ(root.num_nodes,
              treespec->GetNumNodes(),
              "Number of transformed tree nodes mismatch.");
    treespec->m_none_is_leaf = m_none_is_leaf;
    treespec->m_namespace = common_registry_namespace;
    treespec->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*treespec);
    return treespec;
}

std::unique_ptr<PyTreeSpec> PyTreeSpec::Compose(const PyTreeSpec& inner_treespec) const {
    PYTREESPEC_SANITY_CHECK(*this);
    PYTREESPEC_SANITY_CHECK(inner_treespec);

    if (m_none_is_leaf != inner_treespec.m_none_is_leaf) [[unlikely]] {
        throw py::value_error("PyTreeSpecs must have the same none_is_leaf value.");
    }
    if (!m_namespace.empty() && !inner_treespec.m_namespace.empty() &&
        m_namespace != inner_treespec.m_namespace) [[unlikely]] {
        std::ostringstream oss{};
        oss << "PyTreeSpecs must have the same namespace, got " << PyRepr(m_namespace) << " vs. "
            << PyRepr(inner_treespec.m_namespace) << ".";
        throw py::value_error(oss.str());
    }

    auto treespec = std::make_unique<PyTreeSpec>();
    treespec->m_none_is_leaf = m_none_is_leaf;
    if (inner_treespec.m_namespace.empty()) [[likely]] {
        treespec->m_namespace = m_namespace;
    } else [[unlikely]] {
        treespec->m_namespace = inner_treespec.m_namespace;
    }

    const ssize_t num_outer_leaves = GetNumLeaves();
    const ssize_t num_outer_nodes = GetNumNodes();
    const ssize_t num_inner_leaves = inner_treespec.GetNumLeaves();
    const ssize_t num_inner_nodes = inner_treespec.GetNumNodes();
    for (const Node& node : m_traversal) {
        if (node.kind == PyTreeKind::Leaf) [[likely]] {
            std::copy(inner_treespec.m_traversal.cbegin(),
                      inner_treespec.m_traversal.cend(),
                      std::back_inserter(treespec->m_traversal));
        } else [[unlikely]] {
            Node new_node{node};
            new_node.num_leaves = node.num_leaves * num_inner_leaves;
            new_node.num_nodes =
                (node.num_nodes - node.num_leaves) + (node.num_leaves * num_inner_nodes);
            treespec->m_traversal.emplace_back(std::move(new_node));
        }
    }

    const auto& root = treespec->m_traversal.back();
    EXPECT_EQ(root.num_leaves,
              num_outer_leaves * num_inner_leaves,
              "Number of composed tree leaves mismatch.");
    EXPECT_EQ(root.num_nodes,
              (num_outer_nodes - num_outer_leaves) + (num_outer_leaves * num_inner_nodes),
              "Number of composed tree nodes mismatch.");
    treespec->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*treespec);
    return treespec;
}

template <typename Span, typename Stack>
ssize_t PyTreeSpec::PathsImpl(Span& paths,  // NOLINT[misc-no-recursion]
                              Stack& stack,
                              const ssize_t& pos,
                              const ssize_t& depth) const {
    const Node& root = m_traversal.at(pos);
    EXPECT_GE(pos + 1, root.num_nodes, "PyTreeSpec::Paths() walked off start of array.");

    ssize_t cur = pos - 1;
    // NOLINTNEXTLINE[misc-no-recursion]
    const auto recurse = [this, &paths, &stack, &depth](const ssize_t& cur,
                                                        const py::handle& entry) -> ssize_t {
        stack.emplace_back(entry);
        const ssize_t num_nodes = PathsImpl(paths, stack, cur, depth + 1);
        stack.pop_back();
        return num_nodes;
    };

    if (root.node_entries) [[unlikely]] {
        for (ssize_t i = root.arity - 1; i >= 0; --i) {
            cur -= recurse(cur, TupleGetItem(root.node_entries, i));
        }
    } else [[likely]] {
        switch (root.kind) {
            case PyTreeKind::Leaf: {
                py::tuple path{depth};
                for (ssize_t d = 0; d < depth; ++d) {
                    TupleSetItem(path, d, stack[d]);
                }
                paths.emplace_back(std::move(path));
                break;
            }

            case PyTreeKind::None:
                break;

            case PyTreeKind::Tuple:
            case PyTreeKind::List:
            case PyTreeKind::NamedTuple:
            case PyTreeKind::Deque:
            case PyTreeKind::StructSequence:
            case PyTreeKind::Custom: {
                for (ssize_t i = root.arity - 1; i >= 0; --i) {
                    cur -= recurse(cur, py::int_(i));
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                const scoped_critical_section cs{root.node_data};
                const auto keys = (root.kind != PyTreeKind::DefaultDict
                                       ? py::reinterpret_borrow<py::list>(root.node_data)
                                       : TupleGetItemAs<py::list>(root.node_data, 1));
                for (ssize_t i = root.arity - 1; i >= 0; --i) {
                    cur -= recurse(cur, ListGetItem(keys, i));
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }

    return pos - cur;
}

std::vector<py::tuple> PyTreeSpec::Paths() const {
    PYTREESPEC_SANITY_CHECK(*this);

    const ssize_t num_leaves = GetNumLeaves();
    auto paths = reserved_vector<py::tuple>(num_leaves);
    if (num_leaves == 0) [[unlikely]] {
        return paths;
    }
    const ssize_t num_nodes = GetNumNodes();
    if (num_nodes == 1 && num_leaves == 1) [[likely]] {
        paths.emplace_back();
        return paths;
    }
    auto stack = reserved_vector<py::handle>(4);
    const ssize_t num_nodes_walked = PathsImpl(paths, stack, num_nodes - 1, 0);
    std::reverse(paths.begin(), paths.end());
    EXPECT_EQ(num_nodes_walked, num_nodes, "`pos != 0` at end of PyTreeSpec::Paths().");
    EXPECT_EQ(py::ssize_t_cast(paths.size()), num_leaves, "PyTreeSpec::Paths() mismatched leaves.");
    return paths;
}

template <typename Span, typename Stack>
ssize_t PyTreeSpec::AccessorsImpl(Span& accessors,  // NOLINT[misc-no-recursion]
                                  Stack& stack,
                                  const ssize_t& pos,
                                  const ssize_t& depth) const {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
    const py::object& PyTreeAccessor = storage
                                           .call_once_and_store_result([]() -> py::object {
                                               return py::getattr(GetCxxModule(), "PyTreeAccessor");
                                           })
                                           .get_stored();

    const Node& root = m_traversal.at(pos);
    EXPECT_GE(pos + 1, root.num_nodes, "PyTreeSpec::TypedPaths() walked off start of array.");

    ssize_t cur = pos - 1;
    const py::object node_type = GetType(root);
    const PyTreeKind& node_kind = root.kind;
    // NOLINTNEXTLINE[misc-no-recursion]
    const auto recurse = [this, &node_type, &node_kind, &accessors, &stack, &depth](
                             const ssize_t& cur,
                             const py::handle& entry,
                             const py::handle& path_entry_type) -> ssize_t {
        stack.emplace_back(EVALUATE_WITH_LOCK_HELD2(path_entry_type(entry, node_type, node_kind),
                                                    path_entry_type,
                                                    node_type));
        const ssize_t num_nodes = AccessorsImpl(accessors, stack, cur, depth + 1);
        stack.pop_back();
        return num_nodes;
    };

    const py::object path_entry_type = GetPathEntryType(root);
    if (root.node_entries) [[unlikely]] {
        EXPECT_EQ(root.kind,
                  PyTreeKind::Custom,
                  "Node entries are only supported for custom nodes.");
        EXPECT_NE(root.custom, nullptr, "The custom registration is null.");
        for (ssize_t i = root.arity - 1; i >= 0; --i) {
            cur -= recurse(cur, TupleGetItem(root.node_entries, i), path_entry_type);
        }
    } else [[likely]] {
        switch (root.kind) {
            case PyTreeKind::Leaf: {
                const py::tuple typed_path{depth};
                for (ssize_t d = 0; d < depth; ++d) {
                    TupleSetItem(typed_path, d, stack[d]);
                }
                accessors.emplace_back(
                    EVALUATE_WITH_LOCK_HELD(PyTreeAccessor(typed_path), PyTreeAccessor));
                break;
            }

            case PyTreeKind::None:
                break;

            case PyTreeKind::Tuple:
            case PyTreeKind::List:
            case PyTreeKind::NamedTuple:
            case PyTreeKind::Deque:
            case PyTreeKind::StructSequence:
            case PyTreeKind::Custom: {
                for (ssize_t i = root.arity - 1; i >= 0; --i) {
                    cur -= recurse(cur, py::int_(i), path_entry_type);
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                const scoped_critical_section cs{root.node_data};
                const auto keys = (root.kind != PyTreeKind::DefaultDict
                                       ? py::reinterpret_borrow<py::list>(root.node_data)
                                       : TupleGetItemAs<py::list>(root.node_data, 1));
                for (ssize_t i = root.arity - 1; i >= 0; --i) {
                    cur -= recurse(cur, ListGetItem(keys, i), path_entry_type);
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }

    return pos - cur;
}

std::vector<py::object> PyTreeSpec::Accessors() const {
    PYTREESPEC_SANITY_CHECK(*this);

    const ssize_t num_leaves = GetNumLeaves();
    auto accessors = reserved_vector<py::object>(num_leaves);
    if (num_leaves == 0) [[unlikely]] {
        return accessors;
    }

    const ssize_t num_nodes = GetNumNodes();
    auto stack = reserved_vector<py::object>(4);
    const ssize_t num_nodes_walked = AccessorsImpl(accessors, stack, num_nodes - 1, 0);
    std::reverse(accessors.begin(), accessors.end());
    EXPECT_EQ(num_nodes_walked, num_nodes, "`pos != 0` at end of PyTreeSpec::Accessors().");
    EXPECT_EQ(py::ssize_t_cast(accessors.size()),
              num_leaves,
              "PyTreeSpec::Accessors() mismatched leaves.");
    return accessors;
}

py::list PyTreeSpec::Entries() const {
    PYTREESPEC_SANITY_CHECK(*this);

    const Node& root = m_traversal.back();
    if (root.node_entries) [[unlikely]] {
        return py::list{root.node_entries};
    }
    switch (root.kind) {
        case PyTreeKind::Leaf:
        case PyTreeKind::None: {
            return py::list{};
        }

        case PyTreeKind::Tuple:
        case PyTreeKind::List:
        case PyTreeKind::NamedTuple:
        case PyTreeKind::Deque:
        case PyTreeKind::StructSequence:
        case PyTreeKind::Custom: {
            py::list entries{root.arity};
            for (ssize_t i = 0; i < root.arity; ++i) {
                ListSetItem(entries, i, py::int_(i));
            }
            return entries;
        }

        case PyTreeKind::Dict:
        case PyTreeKind::OrderedDict: {
            const scoped_critical_section cs{root.node_data};
            return py::getattr(root.node_data, Py_Get_ID(copy))();
        }
        case PyTreeKind::DefaultDict: {
            const scoped_critical_section cs{root.node_data};
            return py::getattr(TupleGetItem(root.node_data, 1), Py_Get_ID(copy))();
        }

        default:
            INTERNAL_ERROR();
    }
}

py::object PyTreeSpec::Entry(ssize_t index) const {
    PYTREESPEC_SANITY_CHECK(*this);

    const Node& root = m_traversal.back();
    if (index < -root.arity || index >= root.arity) [[unlikely]] {
        throw py::index_error("PyTreeSpec::Entry() index out of range.");
    }
    if (index < 0) [[unlikely]] {
        index += root.arity;
    }

    if (root.node_entries) [[unlikely]] {
        return TupleGetItem(root.node_entries, index);
    }
    switch (root.kind) {
        case PyTreeKind::Tuple:
        case PyTreeKind::List:
        case PyTreeKind::NamedTuple:
        case PyTreeKind::Deque:
        case PyTreeKind::StructSequence:
        case PyTreeKind::Custom: {
            return py::int_(index);
        }

        case PyTreeKind::Dict:
        case PyTreeKind::OrderedDict: {
            const scoped_critical_section cs{root.node_data};
            return ListGetItem(root.node_data, index);
        }
        case PyTreeKind::DefaultDict: {
            const scoped_critical_section cs{root.node_data};
            return ListGetItem(TupleGetItemAs<py::list>(root.node_data, 1), index);
        }

        case PyTreeKind::None:
        case PyTreeKind::Leaf:
        default:
            INTERNAL_ERROR();
    }
}

std::vector<std::unique_ptr<PyTreeSpec>> PyTreeSpec::Children() const {
    PYTREESPEC_SANITY_CHECK(*this);

    const Node& root = m_traversal.back();
    auto children = reserved_vector<std::unique_ptr<PyTreeSpec>>(root.arity);
    children.resize(root.arity);
    ssize_t pos = py::ssize_t_cast(m_traversal.size()) - 1;
    for (ssize_t i = root.arity - 1; i >= 0; --i) {
        children[i] = std::make_unique<PyTreeSpec>();
        children[i]->m_none_is_leaf = m_none_is_leaf;
        children[i]->m_namespace = m_namespace;
        const Node& node = m_traversal.at(pos - 1);
        EXPECT_GE(pos, node.num_nodes, "PyTreeSpec::Children() walked off start of array.");
        std::copy(m_traversal.cbegin() + pos - node.num_nodes,
                  m_traversal.cbegin() + pos,
                  std::back_inserter(children[i]->m_traversal));
        children[i]->m_traversal.shrink_to_fit();
        PYTREESPEC_SANITY_CHECK(*children[i]);
        pos -= node.num_nodes;
    }
    EXPECT_EQ(pos, 0, "`pos != 0` at end of PyTreeSpec::Children().");
    return children;
}

std::unique_ptr<PyTreeSpec> PyTreeSpec::Child(ssize_t index) const {
    PYTREESPEC_SANITY_CHECK(*this);

    const Node& root = m_traversal.back();
    if (index < -root.arity || index >= root.arity) [[unlikely]] {
        throw py::index_error("PyTreeSpec::Child() index out of range.");
    }
    if (index < 0) [[unlikely]] {
        index += root.arity;
    }

    ssize_t pos = py::ssize_t_cast(m_traversal.size()) - 1;
    for (ssize_t i = root.arity - 1; i > index; --i) {
        const Node& node = m_traversal.at(pos - 1);
        EXPECT_GE(pos, node.num_nodes, "PyTreeSpec::Child() walked off start of array.");
        pos -= node.num_nodes;
    }

    auto child = std::make_unique<PyTreeSpec>();
    child->m_none_is_leaf = m_none_is_leaf;
    child->m_namespace = m_namespace;
    const Node& node = m_traversal.at(pos - 1);
    EXPECT_GE(pos, node.num_nodes, "PyTreeSpec::Child() walked off start of array.");
    std::copy(m_traversal.cbegin() + pos - node.num_nodes,
              m_traversal.cbegin() + pos,
              std::back_inserter(child->m_traversal));
    child->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*child);
    return child;
}

py::object PyTreeSpec::GetType(const std::optional<Node>& node) const {
    if (!node) [[likely]] {
        PYTREESPEC_SANITY_CHECK(*this);
    }

    const Node& n = node.value_or(m_traversal.back());
    switch (n.kind) {
        case PyTreeKind::Custom:
            EXPECT_NE(n.custom, nullptr, "The custom registration is null.");
            return py::reinterpret_borrow<py::object>(n.custom->type);
        case PyTreeKind::Leaf:
            return py::none();
        case PyTreeKind::None:
            return PyNoneTypeObject;
        case PyTreeKind::Tuple:
            return PyTupleTypeObject;
        case PyTreeKind::List:
            return PyListTypeObject;
        case PyTreeKind::Dict:
            return PyDictTypeObject;
        case PyTreeKind::NamedTuple:
        case PyTreeKind::StructSequence:
            return py::reinterpret_borrow<py::object>(n.node_data);
        case PyTreeKind::OrderedDict:
            return PyOrderedDictTypeObject;
        case PyTreeKind::DefaultDict:
            return PyDefaultDictTypeObject;
        case PyTreeKind::Deque:
            return PyDequeTypeObject;
        default:
            INTERNAL_ERROR();
    }
}

std::unique_ptr<PyTreeSpec> PyTreeSpec::GetOneLevel(const std::optional<Node>& node) const {
    if (!node) [[likely]] {
        PYTREESPEC_SANITY_CHECK(*this);
    }

    const Node& n = node.value_or(m_traversal.back());
    auto out = std::make_unique<PyTreeSpec>();
    for (ssize_t i = 0; i < n.arity; ++i) {
        out->m_traversal.emplace_back(Node{
            .kind = PyTreeKind::Leaf,
            .arity = 0,
            .num_leaves = 1,
            .num_nodes = 1,
        });
    }
    auto& root = out->m_traversal.emplace_back(n);
    root.num_leaves = (n.kind == PyTreeKind::Leaf ? 1 : n.arity);
    root.num_nodes = n.arity + 1;
    out->m_none_is_leaf = m_none_is_leaf;
    out->m_namespace = m_namespace;
    out->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*out);
    return out;
}

}  // namespace optree
