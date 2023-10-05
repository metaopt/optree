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

template <typename Span>
py::object PyTreeSpec::UnflattenImpl(const Span& leaves) const {
    auto agenda = reserved_vector<py::object>(4);
    auto it = leaves.begin();
    ssize_t leaf_count = 0;
    for (const Node& node : m_traversal) {
        EXPECT_GE(
            py::ssize_t_cast(agenda.size()), node.arity, "Too few elements for PyTreeSpec node.");

        switch (node.kind) {
            case PyTreeKind::Leaf: {
                if (it == leaves.end()) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "Too few leaves for PyTreeSpec; expected: " << GetNumLeaves()
                        << ", got: " << leaf_count << ".";
                    throw py::value_error(oss.str());
                }
                agenda.emplace_back(py::reinterpret_borrow<py::object>(*it));
                ++it;
                ++leaf_count;
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
                const ssize_t size = py::ssize_t_cast(agenda.size());
                py::object out = MakeNode(node, &agenda[size - node.arity], node.arity);
                agenda.resize(size - node.arity);
                agenda.emplace_back(std::move(out));
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }
    if (it != leaves.end()) [[unlikely]] {
        std::ostringstream oss{};
        oss << "Too many leaves for PyTreeSpec; expected: " << GetNumLeaves() << ".";
        throw py::value_error(oss.str());
    }
    EXPECT_EQ(agenda.size(), 1, "PyTreeSpec traversal did not yield a singleton.");
    return std::move(agenda.back());
}

py::object PyTreeSpec::Unflatten(const py::iterable& leaves) const { return UnflattenImpl(leaves); }

}  // namespace optree
