/*
Copyright 2022-2023 MetaOPT Team. All Rights Reserved.

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

#include "include/treespec.h"

namespace optree {

ssize_t PyTreeSpec::GetNumLeaves() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    return m_traversal.back().num_leaves;
}

ssize_t PyTreeSpec::GetNumNodes() const { return py::ssize_t_cast(m_traversal.size()); }

ssize_t PyTreeSpec::GetNumChildren() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    return m_traversal.back().arity;
}

bool PyTreeSpec::GetNoneIsLeaf() const { return m_none_is_leaf; }

std::string PyTreeSpec::GetNamespace() const { return m_namespace; }

py::object PyTreeSpec::GetType() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    Node node = m_traversal.back();
    switch (node.kind) {
        case PyTreeKind::Custom:
            EXPECT_NE(node.custom, nullptr, "The custom registration is null.");
            return node.custom->type;
        case PyTreeKind::Leaf:
            return py::none();
        case PyTreeKind::None:
            return py::type::of(py::none());
        case PyTreeKind::Tuple:
            return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyTuple_Type));
        case PyTreeKind::List:
            return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyList_Type));
        case PyTreeKind::Dict:
            return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyDict_Type));
        case PyTreeKind::NamedTuple:
        case PyTreeKind::StructSequence:
            return node.node_data;
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

PyTreeKind PyTreeSpec::GetPyTreeKind() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    return m_traversal.back().kind;
}

bool PyTreeSpec::IsLeaf(const bool& strict) const {
    if (strict) [[likely]] {
        return GetNumNodes() == 1 && GetNumLeaves() == 1;
    }
    return GetNumNodes() == 1;
}

bool PyTreeSpec::operator==(const PyTreeSpec& other) const {
    if (m_traversal.size() != other.m_traversal.size() || m_none_is_leaf != other.m_none_is_leaf)
        [[likely]] {
        return false;
    }
    if (!m_namespace.empty() && !other.m_namespace.empty() && m_namespace != other.m_namespace)
        [[likely]] {
        return false;
    }

    // NOLINTNEXTLINE[readability-qualified-auto]
    auto b = other.m_traversal.begin();
    // NOLINTNEXTLINE[readability-qualified-auto]
    for (auto a = m_traversal.begin(); a != m_traversal.end(); ++a, ++b) {
        if (a->kind != b->kind || a->arity != b->arity ||
            static_cast<bool>(a->node_data) != static_cast<bool>(b->node_data) ||
            a->custom != b->custom) [[likely]] {
            return false;
        }
        if (a->node_data && a->node_data.not_equal(b->node_data)) [[likely]] {
            return false;
        }
        EXPECT_EQ(a->num_leaves, b->num_leaves);
        EXPECT_EQ(a->num_nodes, b->num_nodes);
    }
    return true;
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
bool PyTreeSpec::IsPrefix(const PyTreeSpec& other, const bool& strict) const {
    if (m_none_is_leaf != other.m_none_is_leaf) [[unlikely]] {
        return false;
    }
    if (!m_namespace.empty() && !other.m_namespace.empty() && m_namespace != other.m_namespace)
        [[likely]] {
        return false;
    }
    if (GetNumNodes() > other.GetNumNodes()) [[likely]] {
        return false;
    }

    bool all_leaves_match = true;
    std::vector<Node> other_traversal{other.m_traversal.begin(), other.m_traversal.end()};
    // NOLINTNEXTLINE[readability-qualified-auto]
    auto b = other_traversal.rbegin();
    // NOLINTNEXTLINE[readability-qualified-auto]
    for (auto a = m_traversal.rbegin(); a != m_traversal.rend(); ++a, ++b) {
        if (b == other_traversal.rend()) [[unlikely]] {
            return false;
        }
        if (a->kind == PyTreeKind::Leaf) [[unlikely]] {
            all_leaves_match &= b->kind == PyTreeKind::Leaf;
            b += b->num_nodes - 1;
            EXPECT_LT(b, other_traversal.rend(), "PyTreeSpec traversal out of range.");
            continue;
        }
        if (a->arity != b->arity ||
            static_cast<bool>(a->node_data) != static_cast<bool>(b->node_data) ||
            a->custom != b->custom) [[likely]] {
            return false;
        }

        switch (a->kind) {
            case PyTreeKind::None:
            case PyTreeKind::Tuple:
            case PyTreeKind::List:
            case PyTreeKind::Deque: {
                if (a->kind != b->kind) [[likely]] {
                    return false;
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                if (b->kind != PyTreeKind::Dict && b->kind != PyTreeKind::OrderedDict &&
                    b->kind != PyTreeKind::DefaultDict) [[likely]] {
                    return false;
                }
                auto expected_keys = py::reinterpret_borrow<py::list>(
                    a->kind != PyTreeKind::DefaultDict
                        ? a->node_data
                        : GET_ITEM_BORROW<py::tuple>(a->node_data, 1));
                auto other_keys = py::reinterpret_borrow<py::list>(
                    b->kind != PyTreeKind::DefaultDict
                        ? b->node_data
                        : GET_ITEM_BORROW<py::tuple>(b->node_data, 1));
                py::dict dict{};
                for (ssize_t i = 0; i < b->arity; ++i) {
                    dict[GET_ITEM_HANDLE<py::list>(other_keys, i)] = py::int_(i);
                }
                if (!DictKeysEqual(expected_keys, dict)) [[likely]] {
                    return false;
                }
                if (expected_keys.not_equal(other_keys)) [[unlikely]] {
                    auto other_offsets = reserved_vector<ssize_t>(b->arity + 1);
                    auto other_num_nodes = reserved_vector<ssize_t>(b->arity);
                    auto other_cur = b + 1;
                    other_offsets.emplace_back(1);
                    for (ssize_t j = b->arity - 1; j >= 0; --j) {
                        ssize_t num_nodes = other_cur->num_nodes;
                        other_num_nodes.emplace_back(num_nodes);
                        other_offsets.emplace_back(other_offsets.back() + num_nodes);
                        other_cur += num_nodes;
                    }
                    std::reverse(other_num_nodes.begin(), other_num_nodes.end());
                    std::reverse(other_offsets.begin(), other_offsets.end());
                    EXPECT_EQ(
                        other_offsets.front(), b->num_nodes, "PyTreeSpec traversal out of range.");
                    auto reordered_index_to_index = std::unordered_map<ssize_t, ssize_t>{};
                    for (ssize_t i = a->arity - 1; i >= 0; --i) {
                        py::object key = GET_ITEM_BORROW<py::list>(expected_keys, i);
                        reordered_index_to_index.emplace(i, dict[key].cast<ssize_t>());
                    }
                    auto reordered_other_num_nodes = reserved_vector<ssize_t>(b->arity);
                    reordered_other_num_nodes.resize(b->arity);
                    for (const auto& [i, j] : reordered_index_to_index) {
                        reordered_other_num_nodes[i] = other_num_nodes[j];
                    }
                    auto reordered_other_offsets = reserved_vector<ssize_t>(b->arity + 1);
                    reordered_other_offsets.emplace_back(1);
                    for (ssize_t i = a->arity - 1; i >= 0; --i) {
                        reordered_other_offsets.emplace_back(reordered_other_offsets.back() +
                                                             reordered_other_num_nodes[i]);
                    }
                    std::reverse(reordered_other_offsets.begin(), reordered_other_offsets.end());
                    EXPECT_EQ(reordered_other_offsets.front(),
                              b->num_nodes,
                              "PyTreeSpec traversal out of range.");
                    auto original_b = other.m_traversal.rbegin() + (b - other_traversal.rbegin());
                    for (const auto& [i, j] : reordered_index_to_index) {
                        std::copy(original_b + other_offsets[j + 1],
                                  original_b + other_offsets[j],
                                  b + reordered_other_offsets[i + 1]);
                    }
                }
                break;
            }

            case PyTreeKind::NamedTuple:
            case PyTreeKind::StructSequence:
            case PyTreeKind::Custom: {
                if (a->kind != b->kind || (a->node_data && a->node_data.not_equal(b->node_data)))
                    [[likely]] {
                    return false;
                }
                break;
            }

            case PyTreeKind::Leaf:
            default:
                INTERNAL_ERROR();
        }

        if (a->num_nodes > b->num_nodes) [[likely]] {
            return false;
        }
    }
    EXPECT_EQ(b, other_traversal.rend(), "PyTreeSpec traversal did not yield a singleton.");
    return (!strict || !all_leaves_match);
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
        std::copy(other_traversal.rend() - (other_pos + 1),
                  other_traversal.rend() - (other_pos - other_root.num_nodes + 1),
                  std::back_inserter(nodes));
        other_cur -= other_root.num_nodes - 1;
        return {pos - cur, other_pos - other_cur, other_root.num_nodes, other_root.num_leaves};
    }
    if (other_root.kind == PyTreeKind::Leaf) [[likely]] {
        std::copy(traversal.rend() - (pos + 1),
                  traversal.rend() - (pos - root.num_nodes + 1),
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

    Node node;
    node.kind = root.kind;
    node.arity = root.arity;
    node.num_leaves = 0;
    node.num_nodes = 1;
    node.node_data = root.node_data;
    node.custom = root.custom;
    node.ordered_keys = root.ordered_keys;
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

            auto expected_keys = py::reinterpret_borrow<py::list>(
                root.kind != PyTreeKind::DefaultDict
                    ? root.node_data
                    : GET_ITEM_BORROW<py::tuple>(root.node_data, 1));
            auto other_keys = py::reinterpret_borrow<py::list>(
                other_root.kind != PyTreeKind::DefaultDict
                    ? other_root.node_data
                    : GET_ITEM_BORROW<py::tuple>(other_root.node_data, 1));
            py::dict dict{};
            for (ssize_t i = 0; i < other_root.arity; ++i) {
                dict[GET_ITEM_HANDLE<py::list>(other_keys, i)] = py::int_(i);
            }
            if (!DictKeysEqual(expected_keys, dict)) [[unlikely]] {
                TotalOrderSort(other_keys);
                auto [missing_keys, extra_keys] = DictKeysDifference(expected_keys, dict);
                std::ostringstream key_difference_sstream{};
                if (GET_SIZE<py::list>(missing_keys) != 0) [[likely]] {
                    key_difference_sstream << ", missing key(s): "
                                           << static_cast<std::string>(py::repr(missing_keys));
                }
                if (GET_SIZE<py::list>(extra_keys) != 0) [[likely]] {
                    key_difference_sstream << ", extra key(s): "
                                           << static_cast<std::string>(py::repr(extra_keys));
                }
                std::ostringstream oss{};
                oss << "dictionary key mismatch; expected key(s): "
                    << static_cast<std::string>(py::repr(expected_keys))
                    << ", got key(s): " + static_cast<std::string>(py::repr(other_keys))
                    << key_difference_sstream.str() << ".";
                throw py::value_error(oss.str());
            }

            size_t start_num_nodes = nodes.size();
            nodes.emplace_back(std::move(node));
            auto other_curs = reserved_vector<ssize_t>(other_root.arity);
            for (ssize_t i = 0; i < other_root.arity; ++i) {
                other_curs.emplace_back(other_cur);
                other_cur -= other_traversal.at(other_cur).num_nodes;
            }
            std::reverse(other_curs.begin(), other_curs.end());
            ssize_t last_other_cur = other_cur;
            for (ssize_t i = root.arity - 1; i >= 0; --i) {
                py::object key = GET_ITEM_BORROW<py::list>(expected_keys, i);
                other_cur = other_curs[dict[key].cast<ssize_t>()];
                auto [num_nodes, other_num_nodes, new_num_nodes, new_num_leaves] =
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
            if (root.node_data.not_equal(other_root.node_data)) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Mismatch custom node data; expected: "
                    << static_cast<std::string>(py::repr(root.node_data))
                    << ", got: " << static_cast<std::string>(py::repr(other_root.node_data)) << ".";
                throw py::value_error(oss.str());
            }
            break;
        }

        case PyTreeKind::Leaf:
        case PyTreeKind::None:
        default:
            INTERNAL_ERROR();
    }

    size_t start_num_nodes = nodes.size();
    nodes.emplace_back(std::move(node));
    for (ssize_t i = root.arity - 1; i >= 0; --i) {
        auto [num_nodes, other_num_nodes, new_num_nodes, new_num_leaves] =
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
    if (m_none_is_leaf != other.m_none_is_leaf) [[unlikely]] {
        throw py::value_error("PyTreeSpecs must have the same none_is_leaf value.");
    }
    if (!m_namespace.empty() && !other.m_namespace.empty() && m_namespace != other.m_namespace)
        [[unlikely]] {
        std::ostringstream oss{};
        oss << "PyTreeSpecs must have the same namespace. Got "
            << static_cast<std::string>(py::repr(py::str(m_namespace))) << " vs. "
            << static_cast<std::string>(py::repr(py::str(other.m_namespace))) << ".";
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

    auto [num_nodes_walked, other_num_nodes_walked, new_num_nodes, new_num_leaves] =
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
    return treespec;
}

std::unique_ptr<PyTreeSpec> PyTreeSpec::Compose(const PyTreeSpec& inner_treespec) const {
    if (m_none_is_leaf != inner_treespec.m_none_is_leaf) [[unlikely]] {
        throw py::value_error("PyTreeSpecs must have the same none_is_leaf value.");
    }
    if (!m_namespace.empty() && !inner_treespec.m_namespace.empty() &&
        m_namespace != inner_treespec.m_namespace) [[unlikely]] {
        std::ostringstream oss{};
        oss << "PyTreeSpecs must have the same namespace. Got "
            << static_cast<std::string>(py::repr(py::str(m_namespace))) << " vs. "
            << static_cast<std::string>(py::repr(py::str(inner_treespec.m_namespace))) << ".";
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
            std::copy(inner_treespec.m_traversal.begin(),
                      inner_treespec.m_traversal.end(),
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
    return treespec;
}

template <typename Span, typename Stack>
// NOLINTNEXTLINE[misc-no-recursion]
ssize_t PyTreeSpec::PathsImpl(Span& paths,
                              Stack& stack,
                              const ssize_t& pos,
                              const ssize_t& depth) const {
    const Node& root = m_traversal.at(pos);
    EXPECT_GE(pos + 1, root.num_nodes, "PyTreeSpec::Paths() walked off start of array.");

    ssize_t cur = pos - 1;
    // NOLINTNEXTLINE[misc-no-recursion]
    auto recurse = [this, &paths, &stack, &depth](const ssize_t& cur,
                                                  const py::handle& entry) -> ssize_t {
        stack.emplace_back(entry);
        const ssize_t num_nodes = PathsImpl(paths, stack, cur, depth + 1);
        stack.pop_back();
        return num_nodes;
    };

    if (root.node_entries) [[unlikely]] {
        for (ssize_t i = root.arity - 1; i >= 0; --i) {
            cur -= recurse(cur, GET_ITEM_HANDLE<py::tuple>(root.node_entries, i));
        }
    } else [[likely]] {
        switch (root.kind) {
            case PyTreeKind::Leaf: {
                py::tuple path{depth};
                for (ssize_t d = 0; d < depth; ++d) {
                    SET_ITEM<py::tuple>(path, d, stack[d]);
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
                auto keys = py::reinterpret_borrow<py::list>(
                    root.kind != PyTreeKind::DefaultDict
                        ? root.node_data
                        : GET_ITEM_BORROW<py::tuple>(root.node_data, 1));
                for (ssize_t i = root.arity - 1; i >= 0; --i) {
                    cur -= recurse(cur, GET_ITEM_HANDLE<py::list>(keys, i));
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
    auto paths = std::vector<py::tuple>{};
    const ssize_t num_leaves = GetNumLeaves();
    if (num_leaves == 0) [[unlikely]] {
        return paths;
    }
    const ssize_t num_nodes = GetNumNodes();
    if (num_nodes == 1 && num_leaves == 1) [[likely]] {
        paths.emplace_back();
        return paths;
    }
    paths.reserve(num_leaves);
    auto stack = reserved_vector<py::handle>(4);
    const ssize_t num_nodes_walked = PathsImpl(paths, stack, num_nodes - 1, 0);
    std::reverse(paths.begin(), paths.end());
    EXPECT_EQ(num_nodes_walked, num_nodes, "`pos != 0` at end of PyTreeSpec::Paths().");
    EXPECT_EQ(py::ssize_t_cast(paths.size()), num_leaves, "PyTreeSpec::Paths() mismatched leaves.");
    return paths;
}

py::list PyTreeSpec::Entries() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
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
            auto entries = py::list{root.arity};
            for (ssize_t i = 0; i < root.arity; ++i) {
                SET_ITEM<py::list>(entries, i, py::int_(i));
            }
            return entries;
        }

        case PyTreeKind::Dict:
        case PyTreeKind::OrderedDict: {
            return py::list{root.node_data};
        }
        case PyTreeKind::DefaultDict: {
            return py::list{GET_ITEM_BORROW<py::tuple>(root.node_data, 1)};
        }

        default:
            INTERNAL_ERROR();
    }
}

py::object PyTreeSpec::Entry(ssize_t index) const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    const Node& root = m_traversal.back();
    if (index < -root.arity || index >= root.arity) [[unlikely]] {
        throw py::index_error("PyTreeSpec::Entry() index out of range.");
    }
    if (index < 0) [[unlikely]] {
        index += root.arity;
    }

    if (root.node_entries) [[unlikely]] {
        return GET_ITEM_BORROW<py::tuple>(root.node_entries, index);
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
            return GET_ITEM_BORROW<py::list>(root.node_data, index);
        }
        case PyTreeKind::DefaultDict: {
            return GET_ITEM_BORROW<py::list>(GET_ITEM_BORROW<py::tuple>(root.node_data, 1), index);
        }

        case PyTreeKind::None:
        case PyTreeKind::Leaf:
        default:
            INTERNAL_ERROR();
    }
}

std::vector<std::unique_ptr<PyTreeSpec>> PyTreeSpec::Children() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
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
        std::copy(m_traversal.begin() + pos - node.num_nodes,
                  m_traversal.begin() + pos,
                  std::back_inserter(children[i]->m_traversal));
        pos -= node.num_nodes;
    }
    EXPECT_EQ(pos, 0, "`pos != 0` at end of PyTreeSpec::Children().");
    return children;
}

std::unique_ptr<PyTreeSpec> PyTreeSpec::Child(ssize_t index) const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
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
    std::copy(m_traversal.begin() + pos - node.num_nodes,
              m_traversal.begin() + pos,
              std::back_inserter(child->m_traversal));
    return child;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::Tuple(const std::vector<PyTreeSpec>& treespecs,
                                                         const bool& none_is_leaf) {
    std::string registry_namespace{};
    for (const PyTreeSpec& treespec : treespecs) {
        if (treespec.m_none_is_leaf != none_is_leaf) [[unlikely]] {
            throw py::value_error(none_is_leaf ? "Expected treespecs with `node_is_leaf=True`."
                                               : "Expected treespecs with `node_is_leaf=False`.");
        }
        if (!treespec.m_namespace.empty()) [[unlikely]] {
            if (registry_namespace.empty()) [[likely]] {
                registry_namespace = treespec.m_namespace;
            } else if (registry_namespace != treespec.m_namespace) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Expected treespecs with the same namespace. Got "
                    << static_cast<std::string>(py::repr(py::str(registry_namespace))) << " vs. "
                    << static_cast<std::string>(py::repr(py::str(treespec.m_namespace))) << ".";
                throw py::value_error(oss.str());
            }
        }
    }

    auto out = std::make_unique<PyTreeSpec>();
    ssize_t num_leaves = 0;
    for (const PyTreeSpec& treespec : treespecs) {
        std::copy(treespec.m_traversal.begin(),
                  treespec.m_traversal.end(),
                  std::back_inserter(out->m_traversal));
        num_leaves += treespec.GetNumLeaves();
    }
    Node node;
    node.kind = PyTreeKind::Tuple;
    node.arity = py::ssize_t_cast(treespecs.size());
    node.num_leaves = num_leaves;
    node.num_nodes = py::ssize_t_cast(out->m_traversal.size()) + 1;
    out->m_traversal.emplace_back(std::move(node));
    out->m_none_is_leaf = none_is_leaf;
    out->m_namespace = registry_namespace;
    return out;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::Leaf(const bool& none_is_leaf) {
    auto out = std::make_unique<PyTreeSpec>();
    Node node;
    node.kind = PyTreeKind::Leaf;
    node.arity = 0;
    node.num_leaves = 1;
    node.num_nodes = 1;
    out->m_traversal.emplace_back(std::move(node));
    out->m_none_is_leaf = none_is_leaf;
    return out;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::None(const bool& none_is_leaf) {
    if (none_is_leaf) [[unlikely]] {
        return Leaf(none_is_leaf);
    }
    auto out = std::make_unique<PyTreeSpec>();
    Node node;
    node.kind = PyTreeKind::None;
    node.arity = 0;
    node.num_leaves = 0;
    node.num_nodes = 1;
    out->m_traversal.emplace_back(std::move(node));
    out->m_none_is_leaf = none_is_leaf;
    return out;
}

/*static*/ std::string PyTreeSpec::NodeKindToString(const Node& node) {
    switch (node.kind) {
        case PyTreeKind::Leaf:
            return "leaf type";
        case PyTreeKind::None:
            return "NoneType";
        case PyTreeKind::Tuple:
            return "tuple";
        case PyTreeKind::List:
            return "list";
        case PyTreeKind::Dict:
            return "dict";
        case PyTreeKind::OrderedDict:
            return "OrderedDict";
        case PyTreeKind::DefaultDict:
            return "defaultdict";
        case PyTreeKind::NamedTuple:
        case PyTreeKind::StructSequence:
            return static_cast<std::string>(py::repr(node.node_data));
        case PyTreeKind::Deque:
            return "deque";
        case PyTreeKind::Custom:
            EXPECT_NE(node.custom, nullptr, "The custom registration is null.");
            return static_cast<std::string>(py::repr(node.custom->type));
        default:
            INTERNAL_ERROR();
    }
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ py::object PyTreeSpec::MakeNode(const PyTreeSpec::Node& node,
                                           const py::object* children,
                                           const size_t& num_children) {
    EXPECT_EQ(py::ssize_t_cast(num_children), node.arity, "Node arity did not match.");
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
                SET_ITEM<py::tuple>(tuple, i, children[i]);
            }
            if (node.kind == PyTreeKind::NamedTuple) [[unlikely]] {
                return node.node_data(*tuple);
            }
            if (node.kind == PyTreeKind::StructSequence) [[unlikely]] {
                return node.node_data(std::move(tuple));
            }
            return tuple;
        }

        case PyTreeKind::List:
        case PyTreeKind::Deque: {
            py::list list{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                SET_ITEM<py::list>(list, i, children[i]);
            }
            if (node.kind == PyTreeKind::Deque) [[unlikely]] {
                return PyDequeTypeObject(list, py::arg("maxlen") = node.node_data);
            }
            return list;
        }

        case PyTreeKind::Dict: {
            py::dict dict{};
            auto keys = py::reinterpret_borrow<py::list>(node.node_data);
            if (node.ordered_keys) [[unlikely]] {
                for (ssize_t i = 0; i < node.arity; ++i) {
                    dict[GET_ITEM_HANDLE<py::list>(node.ordered_keys, i)] = py::none();
                }
            }
            for (ssize_t i = 0; i < node.arity; ++i) {
                // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                dict[GET_ITEM_HANDLE<py::list>(keys, i)] = children[i];
            }
            return dict;
        }

        case PyTreeKind::OrderedDict: {
            py::list items{node.arity};
            auto keys = py::reinterpret_borrow<py::list>(node.node_data);
            for (ssize_t i = 0; i < node.arity; ++i) {
                SET_ITEM<py::list>(
                    items,
                    i,
                    // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                    py::make_tuple(GET_ITEM_HANDLE<py::list>(keys, i), children[i]));
            }
            return PyOrderedDictTypeObject(items);
        }

        case PyTreeKind::DefaultDict: {
            py::dict dict{};
            py::object default_factory = GET_ITEM_BORROW<py::tuple>(node.node_data, 0);
            py::list keys = GET_ITEM_BORROW<py::tuple>(node.node_data, 1);
            if (node.ordered_keys) [[unlikely]] {
                for (ssize_t i = 0; i < node.arity; ++i) {
                    dict[GET_ITEM_HANDLE<py::list>(node.ordered_keys, i)] = py::none();
                }
            }
            for (ssize_t i = 0; i < node.arity; ++i) {
                // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                dict[GET_ITEM_HANDLE<py::list>(keys, i)] = children[i];
            }
            return PyDefaultDictTypeObject(default_factory, dict);
        }

        case PyTreeKind::Custom: {
            py::tuple tuple{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                SET_ITEM<py::tuple>(tuple, i, children[i]);
            }
            return node.custom->unflatten_func(node.node_data, tuple);
        }

        default:
            INTERNAL_ERROR();
    }
}

template <bool NoneIsLeaf>
/*static*/ PyTreeKind PyTreeSpec::GetKind(const py::handle& handle,
                                          PyTreeTypeRegistry::Registration const** custom,
                                          const std::string& registry_namespace) {
    const PyTreeTypeRegistry::Registration* registration =
        PyTreeTypeRegistry::Lookup<NoneIsLeaf>(py::type::handle_of(handle), registry_namespace);
    if (registration) [[likely]] {
        if (registration->kind == PyTreeKind::Custom) [[unlikely]] {
            *custom = registration;
        } else [[likely]] {
            *custom = nullptr;
        }
        return registration->kind;
    }
    *custom = nullptr;
    if (IsNamedTupleInstance(handle)) [[unlikely]] {
        return PyTreeKind::NamedTuple;
    }
    if (IsStructSequenceInstance(handle)) [[unlikely]] {
        return PyTreeKind::StructSequence;
    }
    return PyTreeKind::Leaf;
}

template PyTreeKind PyTreeSpec::GetKind<NONE_IS_NODE>(const py::handle&,
                                                      PyTreeTypeRegistry::Registration const**,
                                                      const std::string&);
template PyTreeKind PyTreeSpec::GetKind<NONE_IS_LEAF>(const py::handle&,
                                                      PyTreeTypeRegistry::Registration const**,
                                                      const std::string&);

/*static*/ void PyTreeSpec::HashCombineNode(ssize_t& seed, const PyTreeSpec::Node& node) {
    ssize_t data_hash = 0;
    switch (node.kind) {
        case PyTreeKind::Custom:
            // We don't hash node_data custom node types since they may not hashable.
            break;

        case PyTreeKind::Leaf:
        case PyTreeKind::None:
        case PyTreeKind::Tuple:
        case PyTreeKind::List:
        case PyTreeKind::NamedTuple:
        case PyTreeKind::Deque:
        case PyTreeKind::StructSequence: {
            data_hash = py::hash(node.node_data ? node.node_data : py::none());
            break;
        }

        case PyTreeKind::Dict:
        case PyTreeKind::OrderedDict:
        case PyTreeKind::DefaultDict: {
            if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
                EXPECT_EQ(
                    GET_SIZE<py::tuple>(node.node_data), 2, "Number of auxiliary data mismatch.");
                py::object default_factory = GET_ITEM_BORROW<py::tuple>(node.node_data, 0);
                data_hash = py::hash(default_factory);
            }
            auto keys = py::reinterpret_borrow<py::list>(
                node.kind != PyTreeKind::DefaultDict
                    ? node.node_data
                    : GET_ITEM_BORROW<py::tuple>(node.node_data, 1));
            EXPECT_EQ(
                GET_SIZE<py::list>(keys), node.arity, "Number of keys and entries does not match.");
            for (const py::handle& key : keys) {
                HashCombine(data_hash, py::hash(key));
            }
            break;
        }

        default:
            INTERNAL_ERROR();
    }

    HashCombine(seed, node.kind);
    HashCombine(seed, node.arity);
    HashCombine(seed, node.custom);
    HashCombine(seed, node.num_leaves);
    HashCombine(seed, node.num_nodes);
    HashCombine(seed, data_hash);
}

ssize_t PyTreeSpec::HashValueImpl() const {
    ssize_t seed = 0;
    for (const Node& node : m_traversal) {
        HashCombineNode(seed, node);
    }
    HashCombine(seed, m_none_is_leaf);
    HashCombine(seed, m_namespace);
    return seed;
}

ssize_t PyTreeSpec::HashValue() const {
    std::pair<const PyTreeSpec*, std::thread::id> indent{this, std::this_thread::get_id()};
    if (sm_hash_running.find(indent) != sm_hash_running.end()) [[unlikely]] {
        return 0;
    }

    sm_hash_running.insert(indent);
    try {
        ssize_t result = HashValueImpl();
        sm_hash_running.erase(indent);
        return result;
    } catch (...) {
        sm_hash_running.erase(indent);
        std::rethrow_exception(std::current_exception());
    }
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
std::string PyTreeSpec::ToStringImpl() const {
    auto agenda = reserved_vector<std::string>(4);
    for (const Node& node : m_traversal) {
        EXPECT_GE(py::ssize_t_cast(agenda.size()), node.arity, "Too few elements for container.");

        std::ostringstream children_sstream{};
        {
            bool first = true;
            for (auto it = agenda.end() - node.arity; it != agenda.end(); ++it) {
                if (!first) [[likely]] {
                    children_sstream << ", ";
                }
                children_sstream << *it;
                first = false;
            }
        }
        std::string children = children_sstream.str();

        std::ostringstream sstream{};
        switch (node.kind) {
            case PyTreeKind::Leaf: {
                agenda.emplace_back("*");
                continue;
            }

            case PyTreeKind::None: {
                sstream << "None";
                break;
            }

            case PyTreeKind::Tuple: {
                sstream << "(" << children;
                // Tuples with only one element must have a trailing comma.
                if (node.arity == 1) [[unlikely]] {
                    sstream << ",";
                }
                sstream << ")";
                break;
            }

            case PyTreeKind::List: {
                sstream << "[" << children << "]";
                break;
            }

            case PyTreeKind::Dict: {
                EXPECT_EQ(GET_SIZE<py::list>(node.node_data),
                          node.arity,
                          "Number of keys and entries does not match.");
                sstream << "{";
                bool first = true;
                auto child_iter = agenda.end() - node.arity;
                for (const py::handle& key : node.node_data) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << static_cast<std::string>(py::repr(key)) << ": " << *child_iter;
                    ++child_iter;
                    first = false;
                }
                sstream << "}";
                break;
            }

            case PyTreeKind::NamedTuple: {
                py::object type = node.node_data;
                auto fields = py::reinterpret_borrow<py::tuple>(py::getattr(type, "_fields"));
                EXPECT_EQ(GET_SIZE<py::tuple>(fields),
                          node.arity,
                          "Number of fields and entries does not match.");
                std::string kind = static_cast<std::string>(py::str(py::getattr(type, "__name__")));
                sstream << kind << "(";
                bool first = true;
                auto child_iter = agenda.end() - node.arity;
                for (const py::handle& field : fields) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << static_cast<std::string>(py::str(field)) << "=" << *child_iter;
                    ++child_iter;
                    first = false;
                }
                sstream << ")";
                break;
            }

            case PyTreeKind::OrderedDict: {
                EXPECT_EQ(GET_SIZE<py::list>(node.node_data),
                          node.arity,
                          "Number of keys and entries does not match.");
                sstream << "OrderedDict([";
                bool first = true;
                auto child_iter = agenda.end() - node.arity;
                for (const py::handle& key : node.node_data) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << "(" << static_cast<std::string>(py::repr(key)) << ", " << *child_iter
                            << ")";
                    ++child_iter;
                    first = false;
                }
                sstream << "])";
                break;
            }

            case PyTreeKind::DefaultDict: {
                EXPECT_EQ(
                    GET_SIZE<py::tuple>(node.node_data), 2, "Number of auxiliary data mismatch.");
                py::object default_factory = GET_ITEM_BORROW<py::tuple>(node.node_data, 0);
                auto keys =
                    py::reinterpret_borrow<py::list>(GET_ITEM_BORROW<py::tuple>(node.node_data, 1));
                EXPECT_EQ(GET_SIZE<py::list>(keys),
                          node.arity,
                          "Number of keys and entries does not match.");
                sstream << "defaultdict(" << static_cast<std::string>(py::repr(default_factory))
                        << ", {";
                bool first = true;
                auto child_iter = agenda.end() - node.arity;
                for (const py::handle& key : keys) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << static_cast<std::string>(py::repr(key)) << ": " << *child_iter;
                    ++child_iter;
                    first = false;
                }
                sstream << "})";
                break;
            }

            case PyTreeKind::Deque: {
                sstream << "deque([" << children << "]";
                if (!node.node_data.is_none()) [[unlikely]] {
                    sstream << ", maxlen=" << static_cast<std::string>(py::str(node.node_data));
                }
                sstream << ")";
                break;
            }

            case PyTreeKind::StructSequence: {
                py::object type = node.node_data;
                auto* members = reinterpret_cast<PyTypeObject*>(type.ptr())->tp_members;
                std::string kind = reinterpret_cast<PyTypeObject*>(type.ptr())->tp_name;
                sstream << kind << "(";
                bool first = true;
                auto child_iter = agenda.end() - node.arity;
                for (ssize_t i = 0; i < node.arity; ++i) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    // NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                    sstream << members[i].name << "=" << *child_iter;
                    ++child_iter;
                    first = false;
                }
                sstream << ")";
                break;
            }

            case PyTreeKind::Custom: {
                std::string kind =
                    static_cast<std::string>(py::str(py::getattr(node.custom->type, "__name__")));
                sstream << "CustomTreeNode(" << kind << "[";
                if (node.node_data) [[likely]] {
                    sstream << static_cast<std::string>(py::repr(node.node_data));
                }
                sstream << "], [" << children << "])";
                break;
            }

            default:
                INTERNAL_ERROR();
        }

        agenda.erase(agenda.end() - node.arity, agenda.end());
        agenda.push_back(sstream.str());
    }

    EXPECT_EQ(agenda.size(), 1, "PyTreeSpec traversal did not yield a singleton.");
    std::ostringstream oss{};
    oss << "PyTreeSpec(" << agenda.back();
    if (m_none_is_leaf) [[unlikely]] {
        oss << ", NoneIsLeaf";
    }
    if (!m_namespace.empty()) [[unlikely]] {
        oss << ", namespace=" << static_cast<std::string>(py::repr(py::str(m_namespace)));
    }
    oss << ")";
    return oss.str();
}

std::string PyTreeSpec::ToString() const {
    std::pair<const PyTreeSpec*, std::thread::id> indent{this, std::this_thread::get_id()};
    if (sm_repr_running.find(indent) != sm_repr_running.end()) [[unlikely]] {
        return "...";
    }

    sm_repr_running.insert(indent);
    try {
        std::string representation = ToStringImpl();
        sm_repr_running.erase(indent);
        return representation;
    } catch (...) {
        sm_repr_running.erase(indent);
        std::rethrow_exception(std::current_exception());
    }
}

py::object PyTreeSpec::ToPicklable() const {
    py::tuple node_states{GetNumNodes()};
    ssize_t i = 0;
    for (const auto& node : m_traversal) {
        SET_ITEM<py::tuple>(node_states,
                            i++,
                            py::make_tuple(static_cast<ssize_t>(node.kind),
                                           node.arity,
                                           node.node_data ? node.node_data : py::none(),
                                           node.node_entries ? node.node_entries : py::none(),
                                           node.custom != nullptr ? node.custom->type : py::none(),
                                           node.num_leaves,
                                           node.num_nodes,
                                           node.ordered_keys ? node.ordered_keys : py::none()));
    }
    return py::make_tuple(std::move(node_states), py::bool_(m_none_is_leaf), py::str(m_namespace));
}

// NOLINTBEGIN[cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers]
// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::FromPicklableImpl(const py::object& picklable) {
    auto state = py::reinterpret_steal<py::tuple>(picklable);
    if (state.size() != 3) [[unlikely]] {
        throw std::runtime_error("Malformed pickled PyTreeSpec.");
    }
    bool none_is_leaf = false;
    std::string registry_namespace{};
    auto out = std::make_unique<PyTreeSpec>();
    out->m_none_is_leaf = none_is_leaf = state[1].cast<bool>();
    out->m_namespace = registry_namespace = state[2].cast<std::string>();
    auto node_states = py::reinterpret_borrow<py::tuple>(state[0]);
    for (const auto& item : node_states) {
        auto t = item.cast<py::tuple>();
        Node& node = out->m_traversal.emplace_back();
        node.kind = static_cast<PyTreeKind>(t[0].cast<ssize_t>());
        if (t.size() != 7) [[unlikely]] {
            if (t.size() == 8) [[likely]] {
                if (t[7].is_none()) [[likely]] {
                    if (node.kind == PyTreeKind::Dict || node.kind == PyTreeKind::DefaultDict)
                        [[unlikely]] {
                        throw std::runtime_error("Malformed pickled PyTreeSpec.");
                    }
                } else [[unlikely]] {
                    if (node.kind == PyTreeKind::Dict || node.kind == PyTreeKind::DefaultDict)
                        [[likely]] {
                        node.ordered_keys = t[7].cast<py::list>();
                    } else [[unlikely]] {
                        throw std::runtime_error("Malformed pickled PyTreeSpec.");
                    }
                }
            } else [[unlikely]] {
                throw std::runtime_error("Malformed pickled PyTreeSpec.");
            }
        }
        node.arity = t[1].cast<ssize_t>();
        switch (node.kind) {
            case PyTreeKind::Leaf:
            case PyTreeKind::None:
            case PyTreeKind::Tuple:
            case PyTreeKind::List: {
                if (!t[2].is_none()) [[unlikely]] {
                    throw std::runtime_error("Malformed pickled PyTreeSpec.");
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict: {
                node.node_data = t[2].cast<py::list>();
                break;
            }

            case PyTreeKind::NamedTuple:
            case PyTreeKind::StructSequence: {
                node.node_data = t[2].cast<py::type>();
                break;
            }

            case PyTreeKind::DefaultDict:
            case PyTreeKind::Deque:
            case PyTreeKind::Custom: {
                node.node_data = t[2];
                break;
            }

            default:
                INTERNAL_ERROR();
        }
        if (node.kind == PyTreeKind::Custom) [[unlikely]] {  // NOLINT
            if (!t[3].is_none()) [[unlikely]] {
                node.node_entries = t[3].cast<py::tuple>();
            }
            if (t[4].is_none()) [[unlikely]] {
                node.custom = nullptr;
            } else [[likely]] {
                if (none_is_leaf) [[unlikely]] {
                    node.custom =
                        PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(t[4], registry_namespace);
                } else [[likely]] {
                    node.custom =
                        PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(t[4], registry_namespace);
                }
            }
            if (node.custom == nullptr) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Unknown custom type in pickled PyTreeSpec: "
                    << static_cast<std::string>(py::repr(t[4]));
                if (!registry_namespace.empty()) [[likely]] {
                    oss << " in namespace "
                        << static_cast<std::string>(py::repr(py::str(registry_namespace)));
                } else [[unlikely]] {
                    oss << " in the global namespace";
                }
                oss << ".";
                throw std::runtime_error(oss.str());
            }
        } else if (!t[3].is_none() || !t[4].is_none()) [[unlikely]] {
            throw std::runtime_error("Malformed pickled PyTreeSpec.");
        }
        node.num_leaves = t[5].cast<ssize_t>();
        node.num_nodes = t[6].cast<ssize_t>();
    }
    return out;
}
// NOLINTEND[cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers]

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::FromPicklable(const py::object& picklable) {
    return FromPicklableImpl(picklable);
}

size_t PyTreeSpec::ThreadIndentTypeHash::operator()(
    const std::pair<const PyTreeSpec*, std::thread::id>& p) const {
    size_t seed = 0;
    HashCombine(seed, p.first);
    HashCombine(seed, p.second);
    return seed;
}

}  // namespace optree
