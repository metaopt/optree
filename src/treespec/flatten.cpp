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

#include "include/treespec.h"

namespace optree {

template <bool NoneIsLeaf, typename Span>
bool PyTreeSpec::FlattenIntoImpl(const py::handle& handle,
                                 Span& leaves,
                                 const ssize_t& depth,
                                 const std::optional<py::function>& leaf_predicate,
                                 const std::string& registry_namespace) {
    if (depth > MAX_RECURSION_DEPTH) [[unlikely]] {  // NOLINT
        PyErr_SetString(PyExc_RecursionError,
                        "maximum recursion depth exceeded during flattening the tree");
        throw py::error_already_set();
    }

    bool found_custom{false};
    Node node;
    ssize_t start_num_nodes = (ssize_t)m_traversal.size();
    ssize_t start_num_leaves = leaves.size();
    if (leaf_predicate && (*leaf_predicate)(handle).cast<bool>()) [[unlikely]] {
        leaves.emplace_back(py::reinterpret_borrow<py::object>(handle));
    } else [[likely]] {  // NOLINT
        node.kind = GetKind<NoneIsLeaf>(handle, &node.custom, registry_namespace);
        auto recurse = [this, &found_custom, &leaf_predicate, &registry_namespace, &leaves, &depth](
                           py::handle child) {
            found_custom |= FlattenIntoImpl<NoneIsLeaf>(
                child, leaves, depth + 1, leaf_predicate, registry_namespace);
        };
        switch (node.kind) {
            case PyTreeKind::None:
                if (!NoneIsLeaf) break;
                [[fallthrough]];
            case PyTreeKind::Leaf:
                leaves.emplace_back(py::reinterpret_borrow<py::object>(handle));
                break;

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
                py::dict dict = py::reinterpret_borrow<py::dict>(handle);
                py::list keys;
                if (node.kind == PyTreeKind::OrderedDict) [[unlikely]] {
                    keys = DictKeys(dict);
                } else [[likely]] {  // NOLINT
                    keys = SortedDictKeys(dict);
                }
                for (const py::handle& key : keys) {
                    recurse(dict[key]);
                }
                node.arity = GET_SIZE<py::dict>(dict);
                if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
                    node.node_data =
                        py::make_tuple(py::getattr(handle, "default_factory"), std::move(keys));
                } else [[likely]] {  // NOLINT
                    node.node_data = std::move(keys);
                }
                break;
            }

            case PyTreeKind::NamedTuple: {
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
                node.arity = GET_SIZE<py::tuple>(tuple);
                node.node_data = py::reinterpret_borrow<py::object>(tuple.get_type());
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::tuple>(tuple, i));
                }
                break;
            }

            case PyTreeKind::Deque: {
                py::list list = handle.cast<py::list>();
                node.arity = GET_SIZE<py::list>(list);
                node.node_data = py::getattr(handle, "maxlen");
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::list>(list, i));
                }
                break;
            }

            case PyTreeKind::Custom: {
                found_custom = true;
                py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(handle));
                const size_t num_out = GET_SIZE<py::tuple>(out);
                if (num_out != 2 && num_out != 3) [[unlikely]] {  // NOLINT
                    throw std::runtime_error(
                        "PyTree custom to_iterable function should return a pair or a triple.");
                }
                node.arity = 0;
                node.node_data = GET_ITEM_BORROW<py::tuple>(out, 1);
                for (const py::handle& child :
                     py::cast<py::iterable>(GET_ITEM_BORROW<py::tuple>(out, 0))) {
                    ++node.arity;
                    recurse(child);
                }
                if (num_out == 3) [[likely]] {  // NOLINT
                    py::object node_entries = GET_ITEM_BORROW<py::tuple>(out, 2);
                    if (!node_entries.is_none()) [[likely]] {  // NOLINT
                        node.node_entries = py::cast<py::tuple>(std::move(node_entries));
                        const ssize_t num_entries = GET_SIZE<py::tuple>(node.node_entries);
                        if (num_entries != node.arity) [[unlikely]] {
                            throw std::runtime_error(absl::StrFormat(
                                "PyTree custom to_iterable function returned inconsistent number "
                                "of children (%ld) and number of entries (%ld).",
                                node.arity,
                                num_entries));
                        }
                    }
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }
    node.num_nodes = (ssize_t)m_traversal.size() - start_num_nodes + 1;
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
        return FlattenIntoImpl<NONE_IS_LEAF>(handle, leaves, 0, leaf_predicate, registry_namespace);
    } else [[likely]] {  // NOLINT
        return FlattenIntoImpl<NONE_IS_NODE>(handle, leaves, 0, leaf_predicate, registry_namespace);
    }
}

/*static*/ std::pair<std::vector<py::object>, std::unique_ptr<PyTreeSpec>> PyTreeSpec::Flatten(
    const py::handle& tree,
    const std::optional<py::function>& leaf_predicate,
    const bool& none_is_leaf,
    const std::string& registry_namespace) {
    std::vector<py::object> leaves;
    auto treespec = std::make_unique<PyTreeSpec>();
    treespec->m_none_is_leaf = none_is_leaf;
    if (treespec->FlattenInto(tree, leaves, leaf_predicate, none_is_leaf, registry_namespace))
        [[unlikely]] {
        treespec->m_namespace = registry_namespace;
    }
    return std::make_pair(std::move(leaves), std::move(treespec));
}

template <bool NoneIsLeaf, typename Span, typename Stack>
bool PyTreeSpec::FlattenIntoWithPathImpl(const py::handle& handle,
                                         Span& leaves,
                                         Span& paths,
                                         Stack& stack,
                                         const ssize_t& depth,
                                         const std::optional<py::function>& leaf_predicate,
                                         const std::string& registry_namespace) {
    if (depth > MAX_RECURSION_DEPTH) [[unlikely]] {  // NOLINT
        PyErr_SetString(PyExc_RecursionError,
                        "maximum recursion depth exceeded during flattening the tree");
        throw py::error_already_set();
    }

    bool found_custom{false};
    Node node;
    ssize_t start_num_nodes = (ssize_t)m_traversal.size();
    ssize_t start_num_leaves = leaves.size();
    if (leaf_predicate && (*leaf_predicate)(handle).cast<bool>()) [[unlikely]] {
        leaves.emplace_back(py::reinterpret_borrow<py::object>(handle));
    } else [[likely]] {  // NOLINT
        node.kind = GetKind<NoneIsLeaf>(handle, &node.custom, registry_namespace);
        auto recurse = [this,
                        &found_custom,
                        &leaf_predicate,
                        &registry_namespace,
                        &leaves,
                        &paths,
                        &stack,
                        &depth](py::handle child, py::handle entry) {
            stack.emplace_back(entry);
            found_custom |= FlattenIntoWithPathImpl<NoneIsLeaf>(
                child, leaves, paths, stack, depth + 1, leaf_predicate, registry_namespace);
            stack.pop_back();
        };
        switch (node.kind) {
            case PyTreeKind::None:
                if (!NoneIsLeaf) break;
                [[fallthrough]];
            case PyTreeKind::Leaf: {
                py::tuple path = py::tuple{depth};
                for (ssize_t d = 0; d < depth; ++d) {
                    SET_ITEM<py::tuple>(path, d, stack[d]);
                }
                leaves.emplace_back(py::reinterpret_borrow<py::object>(handle));
                paths.emplace_back(std::move(path));
                break;
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
                py::dict dict = py::reinterpret_borrow<py::dict>(handle);
                py::list keys;
                if (node.kind == PyTreeKind::OrderedDict) [[unlikely]] {
                    keys = DictKeys(dict);
                } else [[likely]] {  // NOLINT
                    keys = SortedDictKeys(dict);
                }
                for (const py::handle& key : keys) {
                    recurse(dict[key], key);
                }
                node.arity = GET_SIZE<py::dict>(dict);
                if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
                    node.node_data =
                        py::make_tuple(py::getattr(handle, "default_factory"), std::move(keys));
                } else [[likely]] {  // NOLINT
                    node.node_data = std::move(keys);
                }
                break;
            }

            case PyTreeKind::NamedTuple: {
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
                node.arity = GET_SIZE<py::tuple>(tuple);
                node.node_data = py::reinterpret_borrow<py::object>(tuple.get_type());
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::tuple>(tuple, i), py::int_(i));
                }
                break;
            }

            case PyTreeKind::Deque: {
                py::list list = handle.cast<py::list>();
                node.arity = GET_SIZE<py::list>(list);
                node.node_data = py::getattr(handle, "maxlen");
                for (ssize_t i = 0; i < node.arity; ++i) {
                    recurse(GET_ITEM_HANDLE<py::list>(list, i), py::int_(i));
                }
                break;
            }

            case PyTreeKind::Custom: {
                found_custom = true;
                py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(handle));
                const size_t num_out = GET_SIZE<py::tuple>(out);
                if (num_out != 2 && num_out != 3) [[unlikely]] {
                    throw std::runtime_error(
                        "PyTree custom to_iterable function should return a pair or a triple.");
                }
                node.arity = 0;
                node.node_data = GET_ITEM_BORROW<py::tuple>(out, 1);
                py::object node_entries;
                if (num_out == 3) [[likely]] {  // NOLINT
                    node_entries = GET_ITEM_BORROW<py::tuple>(out, 2);
                } else [[unlikely]] {  // NOLINT
                    node_entries = py::none();
                }
                if (node_entries.is_none()) [[unlikely]] {  // NOLINT
                    for (const py::handle& child :
                         py::cast<py::iterable>(GET_ITEM_BORROW<py::tuple>(out, 0))) {
                        recurse(child, py::int_(node.arity++));
                    }
                } else [[likely]] {  // NOLINT
                    node.node_entries = py::cast<py::tuple>(std::move(node_entries));
                    node.arity = GET_SIZE<py::tuple>(node.node_entries);
                    ssize_t num_children = 0;
                    for (const py::handle& child :
                         py::cast<py::iterable>(GET_ITEM_BORROW<py::tuple>(out, 0))) {
                        if (num_children >= node.arity) [[unlikely]] {
                            throw std::runtime_error(
                                "PyTree custom to_iterable function returned too many children "
                                "than number of entries.");
                        }
                        recurse(child,
                                GET_ITEM_BORROW<py::tuple>(node.node_entries, num_children++));
                    }
                    if (num_children != node.arity) [[unlikely]] {
                        throw std::runtime_error(absl::StrFormat(
                            "PyTree custom to_iterable function returned inconsistent number of "
                            "children (%ld) and number of entries (%ld).",
                            num_children,
                            node.arity));
                    }
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }
    node.num_nodes = (ssize_t)m_traversal.size() - start_num_nodes + 1;
    node.num_leaves = leaves.size() - start_num_leaves;
    m_traversal.emplace_back(std::move(node));
    return found_custom;
}

bool PyTreeSpec::FlattenIntoWithPath(const py::handle& handle,
                                     std::vector<py::object>& leaves,
                                     std::vector<py::object>& paths,
                                     const std::optional<py::function>& leaf_predicate,
                                     const bool& none_is_leaf,
                                     const std::string& registry_namespace) {
    std::vector<py::handle> stack;
    if (none_is_leaf) [[unlikely]] {
        return FlattenIntoWithPathImpl<NONE_IS_LEAF>(
            handle, leaves, paths, stack, 0, leaf_predicate, registry_namespace);
    } else [[likely]] {  // NOLINT
        return FlattenIntoWithPathImpl<NONE_IS_NODE>(
            handle, leaves, paths, stack, 0, leaf_predicate, registry_namespace);
    }
}

/*static*/ std::tuple<std::vector<py::object>, std::vector<py::object>, std::unique_ptr<PyTreeSpec>>
PyTreeSpec::FlattenWithPath(const py::handle& tree,
                            const std::optional<py::function>& leaf_predicate,
                            const bool& none_is_leaf,
                            const std::string& registry_namespace) {
    std::vector<py::object> leaves;
    std::vector<py::object> paths;
    auto treespec = std::make_unique<PyTreeSpec>();
    treespec->m_none_is_leaf = none_is_leaf;
    if (treespec->FlattenIntoWithPath(
            tree, leaves, paths, leaf_predicate, none_is_leaf, registry_namespace)) [[unlikely]] {
        treespec->m_namespace = registry_namespace;
    };
    return std::make_tuple(std::move(paths), std::move(leaves), std::move(treespec));
}

py::list PyTreeSpec::FlattenUpToImpl(const py::handle& full_tree) const {
    const ssize_t num_leaves = PyTreeSpec::num_leaves();

    std::vector<py::object> agenda;
    agenda.emplace_back(py::reinterpret_borrow<py::object>(full_tree));

    auto it = m_traversal.rbegin();
    py::list leaves{num_leaves};
    ssize_t leaf = num_leaves - 1;
    while (!agenda.empty()) {
        if (it == m_traversal.rend()) [[unlikely]] {
            throw std::invalid_argument(absl::StrFormat(
                "Tree structures did not match: %s vs %s.", py::repr(full_tree), ToString()));
        }
        const Node& node = *it;
        py::object object = std::move(agenda.back());
        agenda.pop_back();
        ++it;

        switch (node.kind) {
            case PyTreeKind::None:
                break;

            case PyTreeKind::Leaf:
                EXPECT_GE(leaf, 0, "Leaf count mismatch.");
                leaves[leaf] = py::reinterpret_borrow<py::object>(object);
                --leaf;
                break;

            case PyTreeKind::Tuple: {
                AssertExact<py::tuple>(object);
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
                if (GET_SIZE<py::tuple>(tuple) != node.arity) [[unlikely]] {
                    throw std::invalid_argument(
                        absl::StrFormat("Tuple arity mismatch: %ld != %ld; tuple: %s.",
                                        GET_SIZE<py::tuple>(tuple),
                                        node.arity,
                                        py::repr(object)));
                }
                for (ssize_t i = 0; i < node.arity; ++i) {
                    agenda.emplace_back(GET_ITEM_BORROW<py::tuple>(tuple, i));
                }
                break;
            }

            case PyTreeKind::List: {
                AssertExact<py::list>(object);
                py::list list = py::reinterpret_borrow<py::list>(object);
                if (GET_SIZE<py::list>(list) != node.arity) [[unlikely]] {
                    throw std::invalid_argument(
                        absl::StrFormat("List arity mismatch: %ld != %ld; list: %s.",
                                        GET_SIZE<py::list>(list),
                                        node.arity,
                                        py::repr(object)));
                }
                for (ssize_t i = 0; i < node.arity; ++i) {
                    agenda.emplace_back(GET_ITEM_BORROW<py::list>(list, i));
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict: {
                if (node.kind == PyTreeKind::OrderedDict) [[unlikely]] {
                    AssertExactOrderedDict(object);
                } else [[likely]] {  // NOLINT
                    AssertExact<py::dict>(object);
                }
                py::dict dict = py::reinterpret_borrow<py::dict>(object);
                py::list keys;
                if (node.kind == PyTreeKind::OrderedDict) [[unlikely]] {
                    keys = DictKeys(dict);
                } else [[likely]] {  // NOLINT
                    keys = SortedDictKeys(dict);
                }
                if (keys.not_equal(node.node_data)) [[unlikely]] {
                    throw std::invalid_argument(
                        absl::StrFormat("Dict key mismatch; expected keys: %s; dict: %s.",
                                        py::repr(node.node_data),
                                        py::repr(object)));
                }
                for (const py::handle& key : keys) {
                    agenda.emplace_back(dict[key]);
                }
                break;
            }

            case PyTreeKind::NamedTuple: {
                AssertExactNamedTuple(object);
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
                if (GET_SIZE<py::tuple>(tuple) != node.arity) [[unlikely]] {
                    throw std::invalid_argument(
                        absl::StrFormat("Named tuple arity mismatch: %ld != %ld; tuple: %s.",
                                        GET_SIZE<py::tuple>(tuple),
                                        node.arity,
                                        py::repr(object)));
                }
                if (tuple.get_type().not_equal(node.node_data)) [[unlikely]] {
                    throw std::invalid_argument(
                        absl::StrFormat("Named tuple type mismatch: expected type: %s, tuple: %s.",
                                        py::repr(node.node_data),
                                        py::repr(object)));
                }
                for (ssize_t i = 0; i < node.arity; ++i) {
                    agenda.emplace_back(GET_ITEM_BORROW<py::tuple>(tuple, i));
                }
                break;
            }

            case PyTreeKind::DefaultDict: {
                AssertExactDefaultDict(object);
                py::dict dict = py::reinterpret_borrow<py::dict>(object);
                py::list keys = SortedDictKeys(dict);
                py::object default_factory = py::getattr(object, "default_factory");
                py::object expected_default_factory = GET_ITEM_BORROW<py::tuple>(node.node_data, 0);
                py::list expected_keys = GET_ITEM_BORROW<py::tuple>(node.node_data, 1);
                if (default_factory.not_equal(expected_default_factory)) [[unlikely]] {
                    throw std::invalid_argument(absl::StrFormat(
                        "Defaultdict factory mismatch; expected factory: %s; defaultdict: %s.",
                        py::repr(expected_default_factory),
                        py::repr(object)));
                }
                if (keys.not_equal(expected_keys)) [[unlikely]] {
                    throw std::invalid_argument(absl::StrFormat(
                        "Defaultdict key mismatch; expected keys: %s; defaultdict: %s.",
                        py::repr(expected_keys),
                        py::repr(object)));
                }
                for (const py::handle& key : keys) {
                    agenda.emplace_back(dict[key]);
                }
                break;
            }

            case PyTreeKind::Deque: {
                AssertExactDeque(object);
                py::list list = py::cast<py::list>(object);
                if (GET_SIZE<py::list>(list) != node.arity) [[unlikely]] {
                    throw std::invalid_argument(
                        absl::StrFormat("Deque arity mismatch: %ld != %ld; deque: %s.",
                                        GET_SIZE<py::list>(list),
                                        node.arity,
                                        py::repr(object)));
                }
                for (ssize_t i = 0; i < node.arity; ++i) {
                    agenda.emplace_back(GET_ITEM_BORROW<py::list>(list, i));
                }
                break;
            }

            case PyTreeKind::Custom: {
                const PyTreeTypeRegistry::Registration* registration;
                if (m_none_is_leaf) [[unlikely]] {
                    registration =
                        PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(object.get_type(), m_namespace);
                } else [[likely]] {  // NOLINT
                    registration =
                        PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(object.get_type(), m_namespace);
                }
                if (registration != node.custom) [[unlikely]] {
                    throw std::invalid_argument(
                        absl::StrFormat("Custom node type mismatch: expected type: %s, value: %s.",
                                        py::repr(node.custom->type),
                                        py::repr(object)));
                }
                py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(object));
                const size_t num_out = GET_SIZE<py::tuple>(out);
                if (num_out != 2 && num_out != 3) [[unlikely]] {
                    throw std::runtime_error(
                        "PyTree custom to_iterable function should return a pair or a triple.");
                }
                if (node.node_data.not_equal(GET_ITEM_BORROW<py::tuple>(out, 1))) [[unlikely]] {
                    throw std::invalid_argument(
                        absl::StrFormat("Mismatch custom node data: %s != %s; value: %s.",
                                        py::repr(node.node_data),
                                        py::repr(GET_ITEM_BORROW<py::tuple>(out, 1)),
                                        py::repr(object)));
                }
                ssize_t arity = 0;
                for (const py::handle& child :
                     py::cast<py::iterable>(GET_ITEM_BORROW<py::tuple>(out, 0))) {
                    ++arity;
                    agenda.emplace_back(py::reinterpret_borrow<py::object>(child));
                }
                if (arity != node.arity) [[unlikely]] {
                    throw std::invalid_argument(
                        absl::StrFormat("Custom type arity mismatch: %ld != %ld; value: %s.",
                                        arity,
                                        node.arity,
                                        py::repr(object)));
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }
    if (it != m_traversal.rend() || leaf != -1) [[unlikely]] {
        throw std::invalid_argument(absl::StrFormat(
            "Tree structures did not match: %s vs %s.", py::repr(full_tree), ToString()));
    }
    return leaves;
}

py::list PyTreeSpec::FlattenUpTo(const py::handle& full_tree) const {
    return FlattenUpToImpl(full_tree);
}

template <bool NoneIsLeaf>
/*static*/ bool PyTreeSpec::AllLeavesImpl(const py::iterable& iterable,
                                          const std::string& registry_namespace) {
    const PyTreeTypeRegistry::Registration* custom;
    for (const py::handle& h : iterable) {
        if (GetKind<NoneIsLeaf>(h, &custom, registry_namespace) != PyTreeKind::Leaf) [[unlikely]] {
            return false;
        }
    }
    return true;
}

/*static*/ bool PyTreeSpec::AllLeaves(const py::iterable& iterable,
                                      const bool& none_is_leaf,
                                      const std::string& registry_namespace) {
    if (none_is_leaf) [[unlikely]] {
        return AllLeavesImpl<NONE_IS_LEAF>(iterable, registry_namespace);
    } else [[likely]] {  // NOLINT
        return AllLeavesImpl<NONE_IS_NODE>(iterable, registry_namespace);
    }
}

}  // namespace optree
