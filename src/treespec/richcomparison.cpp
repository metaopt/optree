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

#include <algorithm>      // std::copy, std::reverse
#include <unordered_map>  // std::unordered_map
#include <vector>         // std::vector

#include "optree/optree.h"

namespace optree {

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
bool PyTreeSpec::IsPrefix(const PyTreeSpec &other, const bool &strict) const {
    PYTREESPEC_SANITY_CHECK(*this);
    PYTREESPEC_SANITY_CHECK(other);

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
    std::vector<Node> other_traversal{other.m_traversal};
    // NOLINTNEXTLINE[readability-qualified-auto]
    auto b = other_traversal.rbegin();
    // NOLINTNEXTLINE[readability-qualified-auto]
    for (auto a = m_traversal.crbegin(); a != m_traversal.crend(); ++a, ++b) {
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
                const scoped_critical_section2 cs(a->node_data, b->node_data);
                const auto expected_keys = (a->kind != PyTreeKind::DefaultDict
                                                ? py::reinterpret_borrow<py::list>(a->node_data)
                                                : TupleGetItemAs<py::list>(a->node_data, 1));
                const auto other_keys = (b->kind != PyTreeKind::DefaultDict
                                             ? py::reinterpret_borrow<py::list>(b->node_data)
                                             : TupleGetItemAs<py::list>(b->node_data, 1));
                const py::dict dict{};
                for (ssize_t i = 0; i < b->arity; ++i) {
                    DictSetItem(dict, ListGetItem(other_keys, i), py::int_(i));
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
                        const ssize_t num_nodes = other_cur->num_nodes;
                        other_num_nodes.emplace_back(num_nodes);
                        other_offsets.emplace_back(other_offsets.back() + num_nodes);
                        other_cur += num_nodes;
                    }
                    std::reverse(other_num_nodes.begin(), other_num_nodes.end());
                    std::reverse(other_offsets.begin(), other_offsets.end());
                    EXPECT_EQ(other_offsets.front(),
                              b->num_nodes,
                              "PyTreeSpec traversal out of range.");
                    auto reordered_index_to_index = std::unordered_map<ssize_t, ssize_t>{};
                    for (ssize_t i = a->arity - 1; i >= 0; --i) {
                        const py::object key = ListGetItem(expected_keys, i);
                        reordered_index_to_index.emplace(i,
                                                         py::cast<ssize_t>(DictGetItem(dict, key)));
                    }
                    auto reordered_other_num_nodes = reserved_vector<ssize_t>(b->arity);
                    reordered_other_num_nodes.resize(b->arity);
                    for (const auto &[i, j] : reordered_index_to_index) {
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
                    auto original_b = other.m_traversal.crbegin() + (b - other_traversal.crbegin());
                    for (const auto &[i, j] : reordered_index_to_index) {
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
                const scoped_critical_section2 cs(a->node_data, b->node_data);
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
    EXPECT_EQ(b, other_traversal.crend(), "PyTreeSpec traversal did not yield a singleton.");
    return !strict || !all_leaves_match;
}

bool PyTreeSpec::EqualTo(const PyTreeSpec &other) const {
    PYTREESPEC_SANITY_CHECK(*this);
    PYTREESPEC_SANITY_CHECK(other);

    if (m_traversal.size() != other.m_traversal.size() || m_none_is_leaf != other.m_none_is_leaf)
        [[likely]] {
        return false;
    }
    if (!m_namespace.empty() && !other.m_namespace.empty() && m_namespace != other.m_namespace)
        [[likely]] {
        return false;
    }
    if (GetNumNodes() != other.GetNumNodes() || GetNumLeaves() != other.GetNumLeaves()) [[likely]] {
        return false;
    }

    // NOLINTNEXTLINE[readability-qualified-auto]
    auto b = other.m_traversal.cbegin();
    // NOLINTNEXTLINE[readability-qualified-auto]
    for (auto a = m_traversal.cbegin(); a != m_traversal.cend(); ++a, ++b) {
        if (a->kind != b->kind || a->arity != b->arity ||
            static_cast<bool>(a->node_data) != static_cast<bool>(b->node_data) ||
            a->custom != b->custom) [[likely]] {
            return false;
        }
        const scoped_critical_section2 cs(a->node_data, b->node_data);
        if (a->node_data && a->node_data.not_equal(b->node_data)) [[likely]] {
            return false;
        }
        EXPECT_EQ(a->num_leaves, b->num_leaves);
        EXPECT_EQ(a->num_nodes, b->num_nodes);
    }
    return true;
}

}  // namespace optree
