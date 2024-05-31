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

#include <memory>     // std::unique_ptr, std::make_unique
#include <optional>   // std::optional
#include <sstream>    // std::ostringstream
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <tuple>      // std::tuple, std::make_tuple
#include <utility>    // std::move, std::pair, std::make_pair
#include <vector>     // std::vector

#include "include/exceptions.h"
#include "include/registry.h"
#include "include/treespec.h"
#include "include/utils.h"

namespace optree {

template <bool NoneIsLeaf, bool DictShouldBeSorted, typename Span>
// NOLINTNEXTLINE[readability-function-cognitive-complexity]
bool PyTreeSpec::FlattenIntoImpl(const py::handle& handle,
                                 Span& leaves,
                                 const ssize_t& depth,
                                 const std::optional<py::function>& leaf_predicate,
                                 const std::string& registry_namespace) {
    if (depth > MAX_RECURSION_DEPTH) [[unlikely]] {
        PyErr_SetString(PyExc_RecursionError,
                        "Maximum recursion depth exceeded during flattening the tree.");
        throw py::error_already_set();
    }

    bool found_custom = false;
    Node node;
    ssize_t start_num_nodes = py::ssize_t_cast(m_traversal.size());
    ssize_t start_num_leaves = py::ssize_t_cast(leaves.size());
    if (leaf_predicate && py::cast<bool>((*leaf_predicate)(handle))) [[unlikely]] {
        leaves.emplace_back(py::reinterpret_borrow<py::object>(handle));
    } else [[likely]] {
        node.kind =
            PyTreeTypeRegistry::GetKind<NoneIsLeaf>(handle, node.custom, registry_namespace);
        // NOLINTNEXTLINE[misc-no-recursion]
        auto recurse = [this, &found_custom, &leaf_predicate, &registry_namespace, &leaves, &depth](
                           const py::handle& child) -> void {
            found_custom |= FlattenIntoImpl<NoneIsLeaf, DictShouldBeSorted>(
                child, leaves, depth + 1, leaf_predicate, registry_namespace);
        };
        switch (node.kind) {
            case PyTreeKind::Leaf: {
                leaves.emplace_back(py::reinterpret_borrow<py::object>(handle));
                break;
            }

            case PyTreeKind::None: {
                if constexpr (!NoneIsLeaf) {
                    break;
                }
                INTERNAL_ERROR(
                    "NoneIsLeaf is true, but PyTreeTypeRegistry::GetKind() returned "
                    "`PyTreeKind::None`.");
            }

            case PyTreeKind::Tuple: {
                node.arity = GET_SIZE<py::tuple>(handle);
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::tuple>(handle, i));
                }
                break;
            }

            case PyTreeKind::List: {
                node.arity = GET_SIZE<py::list>(handle);
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::list>(handle, i));
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                auto dict = py::reinterpret_borrow<py::dict>(handle);
                node.arity = GET_SIZE<py::dict>(dict);
                py::list keys = DictKeys(dict);
                if (node.kind != PyTreeKind::OrderedDict) [[likely]] {
                    node.original_keys = py::getattr(keys, Py_Get_ID(copy))();
                    if constexpr (DictShouldBeSorted) {
                        TotalOrderSort(keys);
                    }
                }
                for (const py::handle& key : keys) {
                    recurse(dict[key]);
                }
                if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
                    node.node_data = py::make_tuple(py::getattr(handle, Py_Get_ID(default_factory)),
                                                    std::move(keys));
                } else [[likely]] {
                    node.node_data = std::move(keys);
                }
                break;
            }

            case PyTreeKind::NamedTuple:
            case PyTreeKind::StructSequence: {
                auto tuple = py::reinterpret_borrow<py::tuple>(handle);
                node.arity = GET_SIZE<py::tuple>(tuple);
                node.node_data = py::type::of(tuple);
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::tuple>(tuple, i));
                }
                break;
            }

            case PyTreeKind::Deque: {
                auto list = py::cast<py::list>(handle);
                node.arity = GET_SIZE<py::list>(list);
                node.node_data = py::getattr(handle, Py_Get_ID(maxlen));
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::list>(list, i));
                }
                break;
            }

            case PyTreeKind::Custom: {
                found_custom = true;
                py::tuple out = py::cast<py::tuple>(node.custom->flatten_func(handle));
                const ssize_t num_out = GET_SIZE<py::tuple>(out);
                if (num_out != 2 && num_out != 3) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "PyTree custom flatten function for type " << PyRepr(node.custom->type)
                        << " should return a 2- or 3-tuple, got " << num_out << ".";
                    throw std::runtime_error(oss.str());
                }
                node.arity = 0;
                node.node_data = GET_ITEM_BORROW<py::tuple>(out, ssize_t(1));
                for (const py::handle& child :
                     py::cast<py::iterable>(GET_ITEM_BORROW<py::tuple>(out, ssize_t(0)))) {
                    ++node.arity;
                    recurse(child);
                }
                if (num_out == 3) [[likely]] {
                    py::object node_entries = GET_ITEM_BORROW<py::tuple>(out, ssize_t(2));
                    if (!node_entries.is_none()) [[likely]] {
                        node.node_entries = py::cast<py::tuple>(std::move(node_entries));
                        const ssize_t num_entries = GET_SIZE<py::tuple>(node.node_entries);
                        if (num_entries != node.arity) [[unlikely]] {
                            std::ostringstream oss{};
                            oss << "PyTree custom flatten function for type "
                                << PyRepr(node.custom->type)
                                << " returned inconsistent number of children (" << node.arity
                                << ") and number of entries (" << num_entries << ").";
                            throw std::runtime_error(oss.str());
                        }
                    }
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }
    node.num_nodes = py::ssize_t_cast(m_traversal.size()) - start_num_nodes + 1;
    node.num_leaves = leaves.size() - start_num_leaves;
    m_traversal.emplace_back(std::move(node));
    return found_custom;
}

bool PyTreeSpec::FlattenInto(const py::handle& handle,
                             std::vector<py::object>& leaves,
                             const std::optional<py::function>& leaf_predicate,
                             const bool& none_is_leaf,
                             const std::string& registry_namespace) {
    if (none_is_leaf) [[unlikely]] {
        if (!IsDictInsertionOrdered(registry_namespace)) [[likely]] {
            return FlattenIntoImpl<NONE_IS_LEAF, /*DictShouldBeSorted=*/true>(
                handle, leaves, 0, leaf_predicate, registry_namespace);
        } else [[unlikely]] {
            return FlattenIntoImpl<NONE_IS_LEAF, /*DictShouldBeSorted=*/false>(
                handle, leaves, 0, leaf_predicate, registry_namespace);
        }
    } else [[likely]] {
        if (!IsDictInsertionOrdered(registry_namespace)) [[likely]] {
            return FlattenIntoImpl<NONE_IS_NODE, /*DictShouldBeSorted=*/true>(
                handle, leaves, 0, leaf_predicate, registry_namespace);
        } else [[unlikely]] {
            return FlattenIntoImpl<NONE_IS_NODE, /*DictShouldBeSorted=*/false>(
                handle, leaves, 0, leaf_predicate, registry_namespace);
        }
    }
}

/*static*/ std::pair<std::vector<py::object>, std::unique_ptr<PyTreeSpec>> PyTreeSpec::Flatten(
    const py::object& tree,
    const std::optional<py::function>& leaf_predicate,
    const bool& none_is_leaf,
    const std::string& registry_namespace) {
    auto leaves = reserved_vector<py::object>(4);
    auto treespec = std::make_unique<PyTreeSpec>();
    treespec->m_none_is_leaf = none_is_leaf;
    if (treespec->FlattenInto(tree, leaves, leaf_predicate, none_is_leaf, registry_namespace) ||
        IsDictInsertionOrdered(registry_namespace, /*inherit_global_namespace=*/false))
        [[unlikely]] {
        treespec->m_namespace = registry_namespace;
    }
    treespec->m_traversal.shrink_to_fit();
    return std::make_pair(std::move(leaves), std::move(treespec));
}

template <bool NoneIsLeaf,
          bool DictShouldBeSorted,
          typename LeafSpan,
          typename PathSpan,
          typename Stack>
// NOLINTNEXTLINE[readability-function-cognitive-complexity]
bool PyTreeSpec::FlattenIntoWithPathImpl(const py::handle& handle,
                                         LeafSpan& leaves,
                                         PathSpan& paths,
                                         Stack& stack,
                                         const ssize_t& depth,
                                         const std::optional<py::function>& leaf_predicate,
                                         const std::string& registry_namespace) {
    if (depth > MAX_RECURSION_DEPTH) [[unlikely]] {
        PyErr_SetString(PyExc_RecursionError,
                        "Maximum recursion depth exceeded during flattening the tree.");
        throw py::error_already_set();
    }

    bool found_custom = false;
    Node node;
    ssize_t start_num_nodes = py::ssize_t_cast(m_traversal.size());
    ssize_t start_num_leaves = py::ssize_t_cast(leaves.size());
    if (leaf_predicate && py::cast<bool>((*leaf_predicate)(handle))) [[unlikely]] {
        py::tuple path{depth};
        for (ssize_t d = 0; d < depth; ++d) {
            SET_ITEM<py::tuple>(path, d, stack[d]);
        }
        leaves.emplace_back(py::reinterpret_borrow<py::object>(handle));
        paths.emplace_back(std::move(path));
    } else [[likely]] {
        node.kind =
            PyTreeTypeRegistry::GetKind<NoneIsLeaf>(handle, node.custom, registry_namespace);
        // NOLINTNEXTLINE[misc-no-recursion]
        auto recurse = [this,
                        &found_custom,
                        &leaf_predicate,
                        &registry_namespace,
                        &leaves,
                        &paths,
                        &stack,
                        &depth](const py::handle& child, const py::handle& entry) -> void {
            stack.emplace_back(entry);
            found_custom |= FlattenIntoWithPathImpl<NoneIsLeaf, DictShouldBeSorted>(
                child, leaves, paths, stack, depth + 1, leaf_predicate, registry_namespace);
            stack.pop_back();
        };
        switch (node.kind) {
            case PyTreeKind::Leaf: {
                py::tuple path{depth};
                for (ssize_t d = 0; d < depth; ++d) {
                    SET_ITEM<py::tuple>(path, d, stack[d]);
                }
                leaves.emplace_back(py::reinterpret_borrow<py::object>(handle));
                paths.emplace_back(std::move(path));
                break;
            }

            case PyTreeKind::None: {
                if constexpr (!NoneIsLeaf) {
                    break;
                }
                INTERNAL_ERROR(
                    "NoneIsLeaf is true, but PyTreeTypeRegistry::GetKind() returned "
                    "PyTreeKind::None`.");
            }

            case PyTreeKind::Tuple: {
                node.arity = GET_SIZE<py::tuple>(handle);
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::tuple>(handle, i), py::int_(i));
                }
                break;
            }

            case PyTreeKind::List: {
                node.arity = GET_SIZE<py::list>(handle);
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::list>(handle, i), py::int_(i));
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                auto dict = py::reinterpret_borrow<py::dict>(handle);
                py::list keys = DictKeys(dict);
                if (node.kind != PyTreeKind::OrderedDict) [[likely]] {
                    node.original_keys = py::getattr(keys, Py_Get_ID(copy))();
                    if constexpr (DictShouldBeSorted) {
                        TotalOrderSort(keys);
                    }
                }
                for (const py::handle& key : keys) {
                    recurse(dict[key], key);
                }
                node.arity = GET_SIZE<py::dict>(dict);
                if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
                    node.node_data = py::make_tuple(py::getattr(handle, Py_Get_ID(default_factory)),
                                                    std::move(keys));
                } else [[likely]] {
                    node.node_data = std::move(keys);
                }
                break;
            }

            case PyTreeKind::NamedTuple:
            case PyTreeKind::StructSequence: {
                auto tuple = py::reinterpret_borrow<py::tuple>(handle);
                node.arity = GET_SIZE<py::tuple>(tuple);
                node.node_data = py::type::of(tuple);
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::tuple>(tuple, i), py::int_(i));
                }
                break;
            }

            case PyTreeKind::Deque: {
                auto list = py::cast<py::list>(handle);
                node.arity = GET_SIZE<py::list>(list);
                node.node_data = py::getattr(handle, Py_Get_ID(maxlen));
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::list>(list, i), py::int_(i));
                }
                break;
            }

            case PyTreeKind::Custom: {
                found_custom = true;
                py::tuple out = py::cast<py::tuple>(node.custom->flatten_func(handle));
                const ssize_t num_out = GET_SIZE<py::tuple>(out);
                if (num_out != 2 && num_out != 3) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "PyTree custom flatten function for type " << PyRepr(node.custom->type)
                        << " should return a 2- or 3-tuple, got " << num_out << ".";
                    throw std::runtime_error(oss.str());
                }
                node.arity = 0;
                node.node_data = GET_ITEM_BORROW<py::tuple>(out, ssize_t(1));
                py::object node_entries;
                if (num_out == 3) [[likely]] {
                    node_entries = GET_ITEM_BORROW<py::tuple>(out, ssize_t(2));
                } else [[unlikely]] {
                    node_entries = py::none();
                }
                if (node_entries.is_none()) [[unlikely]] {
                    for (const py::handle& child :
                         py::cast<py::iterable>(GET_ITEM_BORROW<py::tuple>(out, ssize_t(0)))) {
                        recurse(child, py::int_(node.arity++));
                    }
                } else [[likely]] {
                    node.node_entries = py::cast<py::tuple>(std::move(node_entries));
                    node.arity = GET_SIZE<py::tuple>(node.node_entries);
                    ssize_t num_children = 0;
                    for (const py::handle& child :
                         py::cast<py::iterable>(GET_ITEM_BORROW<py::tuple>(out, ssize_t(0)))) {
                        if (num_children >= node.arity) [[unlikely]] {
                            throw std::runtime_error(
                                "PyTree custom flatten function for type " +
                                PyRepr(node.custom->type) +
                                " returned inconsistent number of children and number of entries.");
                        }
                        recurse(child,
                                GET_ITEM_BORROW<py::tuple>(node.node_entries, num_children++));
                    }
                    if (num_children != node.arity) [[unlikely]] {
                        std::ostringstream oss{};
                        oss << "PyTree custom flatten function for type "
                            << PyRepr(node.custom->type)
                            << " returned inconsistent number of children (" << num_children
                            << ") and number of entries (" << node.arity << ").";
                        throw std::runtime_error(oss.str());
                    }
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }
    node.num_nodes = py::ssize_t_cast(m_traversal.size()) - start_num_nodes + 1;
    node.num_leaves = leaves.size() - start_num_leaves;
    m_traversal.emplace_back(std::move(node));
    return found_custom;
}

bool PyTreeSpec::FlattenIntoWithPath(const py::handle& handle,
                                     std::vector<py::object>& leaves,
                                     std::vector<py::tuple>& paths,
                                     const std::optional<py::function>& leaf_predicate,
                                     const bool& none_is_leaf,
                                     const std::string& registry_namespace) {
    auto stack = reserved_vector<py::handle>(4);
    if (none_is_leaf) [[unlikely]] {
        if (!IsDictInsertionOrdered(registry_namespace)) [[likely]] {
            return FlattenIntoWithPathImpl<NONE_IS_LEAF, /*DictShouldBeSorted=*/true>(
                handle, leaves, paths, stack, 0, leaf_predicate, registry_namespace);
        } else [[unlikely]] {
            return FlattenIntoWithPathImpl<NONE_IS_LEAF, /*DictShouldBeSorted=*/false>(
                handle, leaves, paths, stack, 0, leaf_predicate, registry_namespace);
        }
    } else [[likely]] {
        if (!IsDictInsertionOrdered(registry_namespace)) [[likely]] {
            return FlattenIntoWithPathImpl<NONE_IS_NODE, /*DictShouldBeSorted=*/true>(
                handle, leaves, paths, stack, 0, leaf_predicate, registry_namespace);
        } else [[unlikely]] {
            return FlattenIntoWithPathImpl<NONE_IS_NODE, /*DictShouldBeSorted=*/false>(
                handle, leaves, paths, stack, 0, leaf_predicate, registry_namespace);
        }
    }
}

/*static*/ std::tuple<std::vector<py::tuple>, std::vector<py::object>, std::unique_ptr<PyTreeSpec>>
PyTreeSpec::FlattenWithPath(const py::object& tree,
                            const std::optional<py::function>& leaf_predicate,
                            const bool& none_is_leaf,
                            const std::string& registry_namespace) {
    auto leaves = reserved_vector<py::object>(4);
    auto paths = reserved_vector<py::tuple>(4);
    auto treespec = std::make_unique<PyTreeSpec>();
    treespec->m_none_is_leaf = none_is_leaf;
    if (treespec->FlattenIntoWithPath(
            tree, leaves, paths, leaf_predicate, none_is_leaf, registry_namespace) ||
        IsDictInsertionOrdered(registry_namespace, /*inherit_global_namespace=*/false))
        [[unlikely]] {
        treespec->m_namespace = registry_namespace;
    }
    treespec->m_traversal.shrink_to_fit();
    return std::make_tuple(std::move(paths), std::move(leaves), std::move(treespec));
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
py::list PyTreeSpec::FlattenUpTo(const py::object& full_tree) const {
    const ssize_t num_leaves = GetNumLeaves();

    auto agenda = reserved_vector<py::object>(4);
    agenda.emplace_back(py::reinterpret_borrow<py::object>(full_tree));

    auto it = m_traversal.rbegin();
    py::list leaves{num_leaves};
    ssize_t leaf = num_leaves - 1;
    while (!agenda.empty()) {
        if (it == m_traversal.rend()) [[unlikely]] {
            std::ostringstream oss{};
            oss << "Tree structures did not match; expected: " << ToString()
                << ", got: " << PyRepr(full_tree) << ".";
            throw py::value_error(oss.str());
        }
        const Node& node = *it;
        py::object object = std::move(agenda.back());
        agenda.pop_back();
        ++it;

        switch (node.kind) {
            case PyTreeKind::Leaf: {
                EXPECT_GE(leaf, 0, "Leaf count mismatch.");
                SET_ITEM<py::list>(leaves, leaf, object);
                --leaf;
                break;
            }

            case PyTreeKind::None: {
                if (m_none_is_leaf) [[unlikely]] {
                    INTERNAL_ERROR(
                        "NoneIsLeaf is true, but PyTreeTypeRegistry::GetKind() returned "
                        "`PyTreeKind::None`.");
                }
                if (!object.is_none()) [[likely]] {
                    std::ostringstream oss{};
                    oss << "Expected None, got " << PyRepr(object) << ".";
                    throw py::value_error(oss.str());
                }
                break;
            }

            case PyTreeKind::Tuple: {
                AssertExact<py::tuple>(object);
                auto tuple = py::reinterpret_borrow<py::tuple>(object);
                if (GET_SIZE<py::tuple>(tuple) != node.arity) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "tuple arity mismatch; expected: " << node.arity
                        << ", got: " << GET_SIZE<py::tuple>(tuple) << "; tuple: " << PyRepr(object)
                        << ".";
                    throw py::value_error(oss.str());
                }
                for (ssize_t i = 0; i < node.arity; ++i) {
                    agenda.emplace_back(GET_ITEM_BORROW<py::tuple>(tuple, i));
                }
                break;
            }

            case PyTreeKind::List: {
                AssertExact<py::list>(object);
                auto list = py::reinterpret_borrow<py::list>(object);
                if (GET_SIZE<py::list>(list) != node.arity) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "list arity mismatch; expected: " << node.arity
                        << ", got: " << GET_SIZE<py::list>(list) << "; list: " << PyRepr(object)
                        << ".";
                    throw py::value_error(oss.str());
                }
                for (ssize_t i = 0; i < node.arity; ++i) {
                    agenda.emplace_back(GET_ITEM_BORROW<py::list>(list, i));
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                AssertExactStandardDict(object);
                auto dict = py::reinterpret_borrow<py::dict>(object);
                py::list expected_keys =
                    (node.kind != PyTreeKind::DefaultDict
                         ? node.node_data
                         : GET_ITEM_BORROW<py::tuple>(node.node_data, ssize_t(1)));
                if (!DictKeysEqual(expected_keys, dict)) [[unlikely]] {
                    py::list keys = SortedDictKeys(dict);
                    auto [missing_keys, extra_keys] = DictKeysDifference(expected_keys, dict);
                    std::ostringstream key_difference_sstream{};
                    if (GET_SIZE<py::list>(missing_keys) != 0) [[likely]] {
                        key_difference_sstream << ", missing key(s): " << PyRepr(missing_keys);
                    }
                    if (GET_SIZE<py::list>(extra_keys) != 0) [[likely]] {
                        key_difference_sstream << ", extra key(s): " << PyRepr(extra_keys);
                    }
                    std::ostringstream oss{};
                    oss << "dictionary key mismatch; expected key(s): " << PyRepr(expected_keys)
                        << ", got key(s): " + PyRepr(keys) << key_difference_sstream.str() << "; ";
                    if (node.kind == PyTreeKind::Dict) [[likely]] {
                        oss << "dict";
                    } else if (node.kind == PyTreeKind::OrderedDict) [[likely]] {
                        oss << "OrderedDict";
                    } else [[unlikely]] {
                        oss << "defaultdict";
                    }
                    oss << ": " << PyRepr(object) << ".";
                    throw py::value_error(oss.str());
                }
                for (const py::handle& key : expected_keys) {
                    agenda.emplace_back(dict[key]);
                }
                break;
            }

            case PyTreeKind::NamedTuple: {
                AssertExactNamedTuple(object);
                auto tuple = py::reinterpret_borrow<py::tuple>(object);
                if (GET_SIZE<py::tuple>(tuple) != node.arity) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "namedtuple arity mismatch; expected: " << node.arity
                        << ", got: " << GET_SIZE<py::tuple>(tuple) << "; tuple: " << PyRepr(object)
                        << ".";
                    throw py::value_error(oss.str());
                }
                if (py::type::handle_of(object).not_equal(node.node_data)) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "namedtuple type mismatch; expected type: " << PyRepr(node.node_data)
                        << ", got type: " << PyRepr(py::type::handle_of(object))
                        << "; tuple: " << PyRepr(object) << ".";
                    throw py::value_error(oss.str());
                }
                for (ssize_t i = 0; i < node.arity; ++i) {
                    agenda.emplace_back(GET_ITEM_BORROW<py::tuple>(tuple, i));
                }
                break;
            }

            case PyTreeKind::Deque: {
                AssertExactDeque(object);
                auto list = py::cast<py::list>(object);
                if (GET_SIZE<py::list>(list) != node.arity) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "deque arity mismatch; expected: " << node.arity
                        << ", got: " << GET_SIZE<py::list>(list) << "; deque: " << PyRepr(object)
                        << ".";
                    throw py::value_error(oss.str());
                }
                for (ssize_t i = 0; i < node.arity; ++i) {
                    agenda.emplace_back(GET_ITEM_BORROW<py::list>(list, i));
                }
                break;
            }

            case PyTreeKind::StructSequence: {
                AssertExactStructSequence(object);
                auto tuple = py::reinterpret_borrow<py::tuple>(object);
                if (GET_SIZE<py::tuple>(tuple) != node.arity) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "PyStructSequence arity mismatch; expected: " << node.arity
                        << ", got: " << GET_SIZE<py::tuple>(tuple) << "; tuple: " << PyRepr(object)
                        << ".";
                    throw py::value_error(oss.str());
                }
                if (py::type::handle_of(object).not_equal(node.node_data)) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "PyStructSequence type mismatch; expected type: "
                        << PyRepr(node.node_data)
                        << ", got type: " << PyRepr(py::type::handle_of(object))
                        << "; tuple: " << PyRepr(object) << ".";
                    throw py::value_error(oss.str());
                }
                for (ssize_t i = 0; i < node.arity; ++i) {
                    agenda.emplace_back(GET_ITEM_BORROW<py::tuple>(tuple, i));
                }
                break;
            }

            case PyTreeKind::Custom: {
                RegistrationPtr registration{nullptr};
                if (m_none_is_leaf) [[unlikely]] {
                    registration =
                        PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(py::type::of(object), m_namespace);
                } else [[likely]] {
                    registration =
                        PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(py::type::of(object), m_namespace);
                }
                if (registration != node.custom) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "Custom node type mismatch; expected type: " << PyRepr(node.custom->type)
                        << ", got type: " << PyRepr(py::type::handle_of(object))
                        << "; value: " << PyRepr(object) << ".";
                    throw py::value_error(oss.str());
                }
                py::tuple out = py::cast<py::tuple>(node.custom->flatten_func(object));
                const ssize_t num_out = GET_SIZE<py::tuple>(out);
                if (num_out != 2 && num_out != 3) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "PyTree custom flatten function for type " << PyRepr(node.custom->type)
                        << " should return a 2- or 3-tuple, got " << num_out << ".";
                    throw std::runtime_error(oss.str());
                }
                py::object node_data = GET_ITEM_BORROW<py::tuple>(out, ssize_t(1));
                if (node.node_data.not_equal(node_data)) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "Mismatch custom node data; expected: " << PyRepr(node.node_data)
                        << ", got: " << PyRepr(node_data) << "; value: " << PyRepr(object) << ".";
                    throw py::value_error(oss.str());
                }
                ssize_t arity = 0;
                for (const py::handle& child :
                     py::cast<py::iterable>(GET_ITEM_BORROW<py::tuple>(out, ssize_t(0)))) {
                    ++arity;
                    agenda.emplace_back(py::reinterpret_borrow<py::object>(child));
                }
                if (arity != node.arity) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "Custom type arity mismatch; expected: " << node.arity
                        << ", got: " << arity << "; value: " << PyRepr(object) << ".";
                    throw py::value_error(oss.str());
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }
    if (it != m_traversal.rend() || leaf != -1) [[unlikely]] {
        std::ostringstream oss{};
        oss << "Tree structures did not match; expected: " << ToString()
            << ", got: " << PyRepr(full_tree) << ".";
        throw py::value_error(oss.str());
    }
    return leaves;
}

template <bool NoneIsLeaf>
bool IsLeafImpl(const py::handle& handle,
                const std::optional<py::function>& leaf_predicate,
                const std::string& registry_namespace) {
    PyTreeTypeRegistry::RegistrationPtr custom{nullptr};
    return ((leaf_predicate && py::cast<bool>((*leaf_predicate)(handle))) ||
            (PyTreeTypeRegistry::GetKind<NoneIsLeaf>(handle, custom, registry_namespace) ==
             PyTreeKind::Leaf));
}

bool IsLeaf(const py::object& object,
            const std::optional<py::function>& leaf_predicate,
            const bool& none_is_leaf,
            const std::string& registry_namespace) {
    if (none_is_leaf) [[unlikely]] {
        return IsLeafImpl<NONE_IS_LEAF>(object, leaf_predicate, registry_namespace);
    } else [[likely]] {
        return IsLeafImpl<NONE_IS_NODE>(object, leaf_predicate, registry_namespace);
    }
}

template <bool NoneIsLeaf>
bool AllLeavesImpl(const py::iterable& iterable,
                   const std::optional<py::function>& leaf_predicate,
                   const std::string& registry_namespace) {
    PyTreeTypeRegistry::RegistrationPtr custom{nullptr};
    for (const py::handle& handle : iterable) {
        if (leaf_predicate && py::cast<bool>((*leaf_predicate)(handle))) [[unlikely]] {
            continue;
        }
        if (PyTreeTypeRegistry::GetKind<NoneIsLeaf>(handle, custom, registry_namespace) !=
            PyTreeKind::Leaf) [[unlikely]] {
            return false;
        }
    }
    return true;
}

bool AllLeaves(const py::iterable& iterable,
               const std::optional<py::function>& leaf_predicate,
               const bool& none_is_leaf,
               const std::string& registry_namespace) {
    if (none_is_leaf) [[unlikely]] {
        return AllLeavesImpl<NONE_IS_LEAF>(iterable, leaf_predicate, registry_namespace);
    } else [[likely]] {
        return AllLeavesImpl<NONE_IS_NODE>(iterable, leaf_predicate, registry_namespace);
    }
}

}  // namespace optree
