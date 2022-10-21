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

}  // namespace optree
