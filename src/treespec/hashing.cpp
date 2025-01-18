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

#include <exception>      // std::rethrow_exception, std::current_exception
#include <thread>         // std::this_thread::get_id
#include <unordered_set>  // std::unordered_set

#include "optree/optree.h"

namespace optree {

ssize_t PyTreeSpec::HashValueImpl() const {
    ssize_t seed = 0;

    HashCombine(seed, GetNumLeaves());
    HashCombine(seed, GetNumNodes());
    HashCombine(seed, m_none_is_leaf);
    HashCombine(seed, m_namespace);

    for (const Node& node : m_traversal) {
        HashCombine(seed, node.kind);
        HashCombine(seed, node.arity);
        HashCombine(seed, node.num_leaves);
        HashCombine(seed, node.num_nodes);

        switch (node.kind) {
            case PyTreeKind::Custom: {
                // We don't hash node_data of custom node types since they may not hashable.
                const auto& type = GetType(node);
                HashCombine(seed, EVALUATE_WITH_LOCK_HELD(py::hash(type), type));
                break;
            }

            case PyTreeKind::Leaf:
            case PyTreeKind::None:
            case PyTreeKind::Tuple:
            case PyTreeKind::List:
            case PyTreeKind::NamedTuple:
            case PyTreeKind::Deque:
            case PyTreeKind::StructSequence: {
                HashCombine(
                    seed,
                    EVALUATE_WITH_LOCK_HELD(py::hash(node.node_data ? node.node_data : py::none()),
                                            node.node_data));
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                const scoped_critical_section cs{node.node_data};
                if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
                    EXPECT_EQ(TupleGetSize(node.node_data), 2, "Number of metadata mismatch.");
                    const py::object default_factory = TupleGetItem(node.node_data, 0);
                    HashCombine(
                        seed,
                        EVALUATE_WITH_LOCK_HELD(py::hash(default_factory), default_factory));
                }
                const auto keys = (node.kind != PyTreeKind::DefaultDict
                                       ? py::reinterpret_borrow<py::list>(node.node_data)
                                       : TupleGetItemAs<py::list>(node.node_data, 1));
                EXPECT_EQ(ListGetSize(keys),
                          node.arity,
                          "Number of keys and entries does not match.");
                for (const py::handle& key : keys) {
                    HashCombine(seed, EVALUATE_WITH_LOCK_HELD(py::hash(key), key));
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }
    return seed;
}

ssize_t PyTreeSpec::HashValue() const {
    PYTREESPEC_SANITY_CHECK(*this);

    static std::unordered_set<ThreadedIdentity> running{};
    static read_write_mutex mutex{};

    const ThreadedIdentity ident{this, std::this_thread::get_id()};
    {
        const scoped_read_lock_guard lock{mutex};
        if (running.find(ident) != running.end()) [[unlikely]] {
            return 0;
        }
    }

    {
        const scoped_write_lock_guard lock{mutex};
        running.insert(ident);
    }
    try {
        const ssize_t result = HashValueImpl();
        {
            const scoped_write_lock_guard lock{mutex};
            running.erase(ident);
        }
        return result;
    } catch (...) {
        {
            const scoped_write_lock_guard lock{mutex};
            running.erase(ident);
        }
        std::rethrow_exception(std::current_exception());
    }
}

}  // namespace optree
