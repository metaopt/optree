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
#include "include/utils.h"

namespace optree {

py::object PyTreeSpec::Walk(const py::function& f_node,
                            const py::handle& f_leaf,
                            const py::iterable& leaves) const {
    std::vector<py::object> agenda;
    auto it = leaves.begin();
    const bool f_leaf_identity = f_leaf.is_none();
    for (const Node& node : traversal) {
        switch (node.kind) {
            case PyTreeKind::Leaf: {
                if (it == leaves.end()) [[unlikely]] {
                    throw std::invalid_argument("Too few leaves for PyTreeSpec");
                }

                py::object leaf = py::reinterpret_borrow<py::object>(*it);
                agenda.emplace_back(f_leaf_identity ? std::move(leaf) : f_leaf(std::move(leaf)));
                ++it;
                break;
            }

            case PyTreeKind::None:
            case PyTreeKind::Tuple:
            case PyTreeKind::NamedTuple:
            case PyTreeKind::List:
            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict:
            case PyTreeKind::Deque:
            case PyTreeKind::Custom: {
                if ((ssize_t)agenda.size() < node.arity) [[unlikely]] {
                    throw std::logic_error("Too few elements for custom type.");
                }
                py::tuple tuple{node.arity};
                for (ssize_t i = node.arity - 1; i >= 0; --i) {
                    SET_ITEM<py::tuple>(tuple, i, std::move(agenda.back()));
                    agenda.pop_back();
                }
                agenda.emplace_back(f_node(tuple, node.node_data ? node.node_data : py::none()));
            }

            default:
                throw std::logic_error("Unreachable code.");
        }
    }
    if (it != leaves.end()) [[unlikely]] {
        throw std::invalid_argument("Too many leaves for PyTreeSpec.");
    }
    if (agenda.size() != 1) [[unlikely]] {
        throw std::logic_error("PyTreeSpec traversal did not yield a singleton.");
    }
    return std::move(agenda.back());
}

std::unique_ptr<PyTreeSpec> PyTreeSpec::Compose(const PyTreeSpec& inner_treespec) const {
    auto outer_treespec = std::make_unique<PyTreeSpec>();
    outer_treespec->none_is_leaf = none_is_leaf;
    for (const Node& node : traversal) {
        if (node.kind == PyTreeKind::Leaf) [[likely]] {
            absl::c_copy(inner_treespec.traversal, std::back_inserter(outer_treespec->traversal));
        } else [[unlikely]] {  // NOLINT
            outer_treespec->traversal.emplace_back(std::move(node));
        }
    }
    const auto& root = traversal.back();
    const auto& inner_root = inner_treespec.traversal.back();
    auto& outer_root = outer_treespec->traversal.back();
    outer_root.num_nodes =
        (root.num_nodes - root.num_leaves) + (inner_root.num_nodes * root.num_leaves);
    outer_root.num_leaves *= inner_root.num_leaves;
    return outer_treespec;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::Tuple(const std::vector<PyTreeSpec>& treespecs,
                                                         const bool& none_is_leaf) {
    for (const PyTreeSpec& treespec : treespecs) {
        if (treespec.none_is_leaf != none_is_leaf) [[unlikely]] {
            throw std::invalid_argument(absl::StrFormat(
                "Expected TreeSpecs with `node_is_leaf=%s`.", (none_is_leaf ? "True" : "False")));
        }
    }

    auto out = std::make_unique<PyTreeSpec>();
    ssize_t num_leaves = 0;
    for (const PyTreeSpec& treespec : treespecs) {
        absl::c_copy(treespec.traversal, std::back_inserter(out->traversal));
        num_leaves += treespec.num_leaves();
    }
    Node node;
    node.kind = PyTreeKind::Tuple;
    node.arity = treespecs.size();
    node.num_leaves = num_leaves;
    node.num_nodes = out->traversal.size() + 1;
    out->traversal.emplace_back(std::move(node));
    out->none_is_leaf = none_is_leaf;
    return out;
}

std::vector<std::unique_ptr<PyTreeSpec>> PyTreeSpec::Children() const {
    std::vector<std::unique_ptr<PyTreeSpec>> children;
    if (traversal.empty()) [[likely]] {
        return children;
    }
    const Node& root = traversal.back();
    children.resize(root.arity);
    ssize_t pos = traversal.size() - 1;
    for (ssize_t i = root.arity - 1; i >= 0; --i) {
        children[i] = std::make_unique<PyTreeSpec>();
        const Node& node = traversal.at(pos - 1);
        if (pos < node.num_nodes) [[unlikely]] {
            throw std::logic_error("PyTreeSpec::Children() walked off start of array.");
        }
        std::copy(traversal.begin() + pos - node.num_nodes,
                  traversal.begin() + pos,
                  std::back_inserter(children[i]->traversal));
        pos -= node.num_nodes;
    }
    if (pos != 0) [[unlikely]] {
        throw std::logic_error("pos != 0 at end of PyTreeSpec::Children().");
    }
    return children;
}

}  // namespace optree