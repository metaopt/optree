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

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#include "include/treespec.h"

namespace optree {

template <typename Span>
void PyTreeSpec::FlattenIntoImpl(const py::handle& handle,
                                 Span& leaves,
                                 const std::optional<py::function>& leaf_predicate) {
    Node node;
    ssize_t start_num_nodes = traversal.size();
    ssize_t start_num_leaves = leaves.size();
    if (leaf_predicate && (*leaf_predicate)(handle).cast<bool>()) {
        leaves.emplace_back(py::reinterpret_borrow<py::object>(handle));
    } else {
        node.kind = GetKind(handle, &node.custom);
        auto recurse = [this, &leaf_predicate, &leaves](py::handle child) {
            FlattenIntoImpl(child, leaves, leaf_predicate);
        };
        switch (node.kind) {
            case PyTreeKind::Leaf:
            case PyTreeKind::None:
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

            case PyTreeKind::Dict: {
                py::dict dict = py::reinterpret_borrow<py::dict>(handle);
                py::list keys = SortedDictKeys(dict);
                for (const py::handle& key : keys) {
                    recurse(dict[key]);
                }
                node.arity = GET_SIZE<py::dict>(handle);
                node.node_data = std::move(keys);
                break;
            }

            case PyTreeKind::NamedTuple: {
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
                node.arity = GET_SIZE<py::tuple>(tuple);
                node.node_data = py::reinterpret_borrow<py::object>(tuple.get_type());
                for (const py::handle& entry : tuple) {
                    recurse(entry);
                }
                break;
            }

            case PyTreeKind::Custom: {
                py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(handle));
                if (out.size() != 2) {
                    throw std::runtime_error(
                        "PyTree custom to_iterable function should return a pair.");
                }
                node.arity = 0;
                node.node_data = out[1];
                for (const py::handle& child : py::cast<py::iterable>(out[0])) {
                    ++node.arity;
                    recurse(child);
                }
                break;
            }

            default:
                throw std::logic_error("Unreachable code.");
        }
    }
    node.num_nodes = traversal.size() - start_num_nodes + 1;
    node.num_leaves = leaves.size() - start_num_leaves;
    traversal.emplace_back(std::move(node));
}

void PyTreeSpec::FlattenInto(const py::handle& handle,
                             absl::InlinedVector<py::object, 2>& leaves,
                             const std::optional<py::function>& leaf_predicate) {
    FlattenIntoImpl(handle, leaves, leaf_predicate);
}

void PyTreeSpec::FlattenInto(const py::handle& handle,
                             std::vector<py::object>& leaves,
                             const std::optional<py::function>& leaf_predicate) {
    FlattenIntoImpl(handle, leaves, leaf_predicate);
}

/*static*/ std::pair<std::vector<py::object>, std::unique_ptr<PyTreeSpec>> PyTreeSpec::Flatten(
    const py::handle& tree, const std::optional<py::function>& leaf_predicate) {
    std::vector<py::object> leaves;
    auto treespec = std::make_unique<PyTreeSpec>();
    treespec->FlattenInto(tree, leaves, leaf_predicate);
    return std::make_pair(std::move(leaves), std::move(treespec));
}

py::list PyTreeSpec::FlattenUpToImpl(const py::handle& full_tree) const {
    const ssize_t num_leaves = PyTreeSpec::num_leaves();

    std::vector<py::object> agenda;
    agenda.emplace_back(py::reinterpret_borrow<py::object>(full_tree));

    auto it = traversal.rbegin();
    py::list leaves{num_leaves};
    ssize_t leaf = num_leaves - 1;
    while (!agenda.empty()) {
        if (it == traversal.rend()) {
            throw std::invalid_argument(absl::StrFormat(
                "Tree structures did not match: %s vs %s.", py::repr(full_tree), ToString()));
        }
        const Node& node = *it;
        py::object object = std::move(agenda.back());
        agenda.pop_back();
        ++it;

        switch (node.kind) {
            case PyTreeKind::Leaf:
                if (leaf < 0) {
                    throw std::logic_error("Leaf count mismatch.");
                }
                leaves[leaf] = py::reinterpret_borrow<py::object>(object);
                --leaf;
                break;

            case PyTreeKind::None:
                break;

            case PyTreeKind::Tuple: {
                AssertExact<py::tuple>(object);
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
                if ((ssize_t)tuple.size() != node.arity) {
                    throw std::invalid_argument(
                        absl::StrFormat("Tuple arity mismatch: %ld != %ld; tuple: %s.",
                                        tuple.size(),
                                        node.arity,
                                        py::repr(object)));
                }
                for (py::handle entry : tuple) {
                    agenda.emplace_back(py::reinterpret_borrow<py::object>(entry));
                }
                break;
            }

            case PyTreeKind::List: {
                AssertExact<py::list>(object);
                py::list list = py::reinterpret_borrow<py::list>(object);
                if ((ssize_t)list.size() != node.arity) {
                    throw std::invalid_argument(
                        absl::StrFormat("List arity mismatch: %ld != %ld; list: %s.",
                                        list.size(),
                                        node.arity,
                                        py::repr(object)));
                }
                for (py::handle entry : list) {
                    agenda.emplace_back(py::reinterpret_borrow<py::object>(entry));
                }
                break;
            }

            case PyTreeKind::Dict: {
                AssertExact<py::dict>(object);
                py::dict dict = py::reinterpret_borrow<py::dict>(object);
                py::list keys = SortedDictKeys(dict);
                if (keys.not_equal(node.node_data)) {
                    throw std::invalid_argument(
                        absl::StrFormat("Dict key mismatch; expected keys: %s; dict: %s.",
                                        py::repr(node.node_data),
                                        py::repr(object)));
                }
                for (py::handle key : keys) {
                    agenda.emplace_back(dict[key]);
                }
                break;
            }

            case PyTreeKind::NamedTuple: {
                AssertExactNamedTuple(object);
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
                if ((ssize_t)tuple.size() != node.arity) {
                    throw std::invalid_argument(
                        absl::StrFormat("Named tuple arity mismatch: %ld != %ld; tuple: %s.",
                                        tuple.size(),
                                        node.arity,
                                        py::repr(object)));
                }
                if (tuple.get_type().not_equal(node.node_data)) {
                    throw std::invalid_argument(
                        absl::StrFormat("Named tuple type mismatch: expected type: %s, tuple: %s.",
                                        py::repr(node.node_data),
                                        py::repr(object)));
                }
                for (py::handle entry : tuple) {
                    agenda.emplace_back(py::reinterpret_borrow<py::object>(entry));
                }
                break;
            }

            case PyTreeKind::Custom: {
                auto* registration = PyTreeTypeRegistry::Lookup(object.get_type());
                if (registration != node.custom) {
                    throw std::invalid_argument(
                        absl::StrFormat("Custom node type mismatch: expected type: %s, value: %s.",
                                        py::repr(node.custom->type),
                                        py::repr(object)));
                }
                py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(object));
                if (out.size() != 2) {
                    throw std::runtime_error(
                        "PyTree custom to_iterable function should return a pair.");
                }
                if (node.node_data.not_equal(out[1])) {
                    throw std::invalid_argument(
                        absl::StrFormat("Mismatch custom node data: %s != %s; value: %s.",
                                        py::repr(node.node_data),
                                        py::repr(out[1]),
                                        py::repr(object)));
                }
                ssize_t arity = 0;
                for (py::handle entry : py::cast<py::iterable>(out[0])) {
                    ++arity;
                    agenda.emplace_back(py::reinterpret_borrow<py::object>(entry));
                }
                if (arity != node.arity) {
                    throw std::invalid_argument(
                        absl::StrFormat("Custom type arity mismatch: %ld != %ld; value: %s.",
                                        arity,
                                        node.arity,
                                        py::repr(object)));
                }
                break;
            }

            default:
                throw std::logic_error("Unreachable code.");
        }
    }
    if (it != traversal.rend() || leaf != -1) {
        throw std::invalid_argument(absl::StrFormat(
            "Tree structures did not match: %s vs %s.", py::repr(full_tree), ToString()));
    }
    return leaves;
}

py::list PyTreeSpec::FlattenUpTo(const py::handle& full_tree) const {
    return FlattenUpToImpl(full_tree);
}

/*static*/ bool PyTreeSpec::AllLeavesImpl(const py::iterable& iterable) {
    const PyTreeTypeRegistry::Registration* custom;
    for (const py::handle& h : iterable) {
        if (GetKind(h, &custom) != PyTreeKind::Leaf) return false;
    }
    return true;
}

/*static*/ bool PyTreeSpec::AllLeaves(const py::iterable& iterable) {
    return AllLeavesImpl(iterable);
}

}  // namespace optree
