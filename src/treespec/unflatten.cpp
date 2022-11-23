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

template <typename Span>
py::object PyTreeSpec::UnflattenImpl(const Span& leaves) const {
    absl::InlinedVector<py::object, 4> agenda;
    auto it = leaves.begin();
    ssize_t leaf_count = 0;
    for (const Node& node : m_traversal) {
        if ((ssize_t)agenda.size() < node.arity) [[unlikely]] {
            throw std::logic_error("Too few elements for PyTreeSpec node.");
        }
        switch (node.kind) {
            case PyTreeKind::NONE:
            case PyTreeKind::LEAF: {
                if (node.kind == PyTreeKind::LEAF || m_none_is_leaf) [[likely]] {  // NOLINT
                    if (it == leaves.end()) [[unlikely]] {
                        throw std::invalid_argument(
                            absl::StrFormat("Too few leaves for PyTreeSpec; expected %ld, got %ld.",
                                            num_leaves(),
                                            leaf_count));
                    }
                    agenda.emplace_back(py::reinterpret_borrow<py::object>(*it));
                    ++it;
                    ++leaf_count;
                    break;
                }
            }

            case PyTreeKind::TUPLE:
            case PyTreeKind::NAMED_TUPLE:
            case PyTreeKind::LIST:
            case PyTreeKind::DICT:
            case PyTreeKind::ORDERED_DICT:
            case PyTreeKind::DEFAULT_DICT:
            case PyTreeKind::DEQUE:
            case PyTreeKind::CUSTOM: {
                const ssize_t size = (ssize_t)agenda.size();
                absl::Span<py::object> span;
                if (node.arity > 0) [[likely]] {
                    span = absl::Span<py::object>(&agenda[size - node.arity], node.arity);
                }
                py::object out = MakeNode(node, span);
                agenda.resize(size - node.arity);
                agenda.emplace_back(std::move(out));
                break;
            }

            default:
                throw std::logic_error("Unreachable code.");
        }
    }
    if (it != leaves.end()) [[unlikely]] {
        throw std::invalid_argument(
            absl::StrFormat("Too many leaves for PyTreeSpec; expected %ld.", num_leaves()));
    }
    if (agenda.size() != 1) [[unlikely]] {
        throw std::logic_error("PyTreeSpec traversal did not yield a singleton.");
    }
    return std::move(agenda.back());
}

py::object PyTreeSpec::Unflatten(const py::iterable& leaves) const { return UnflattenImpl(leaves); }

}  // namespace optree
