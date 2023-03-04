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

py::object PyTreeSpec::Walk(const py::function& f_node,
                            const py::handle& f_leaf,
                            const py::iterable& leaves) const {
    auto agenda = std::vector<py::object>{};
    auto it = leaves.begin();
    const bool f_leaf_identity = f_leaf.is_none();
    for (const Node& node : m_traversal) {
        switch (node.kind) {
            case PyTreeKind::Leaf: {
                if (it == leaves.end()) [[unlikely]] {
                    throw py::value_error("Too few leaves for PyTreeSpec.");
                }

                auto leaf = py::reinterpret_borrow<py::object>(*it);
                agenda.emplace_back(f_leaf_identity ? std::move(leaf) : f_leaf(std::move(leaf)));
                ++it;
                break;
            }

            case PyTreeKind::None:
            case PyTreeKind::Tuple:
            case PyTreeKind::List:
            case PyTreeKind::Dict:
            case PyTreeKind::NamedTuple:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict:
            case PyTreeKind::Deque:
            case PyTreeKind::StructSequence:
            case PyTreeKind::Custom: {
                EXPECT_GE(py::ssize_t_cast(agenda.size()),
                          node.arity,
                          "Too few elements for custom type.");
                py::tuple tuple{node.arity};
                for (ssize_t i = node.arity - 1; i >= 0; --i) {
                    SET_ITEM<py::tuple>(tuple, i, agenda.back());
                    agenda.pop_back();
                }
                agenda.emplace_back(
                    f_node(std::move(tuple), node.node_data ? node.node_data : py::none()));
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }
    if (it != leaves.end()) [[unlikely]] {
        throw py::value_error("Too many leaves for PyTreeSpec.");
    }

    EXPECT_EQ(agenda.size(), 1, "PyTreeSpec traversal did not yield a singleton.");
    return std::move(agenda.back());
}

}  // namespace optree
