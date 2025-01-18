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
#include <memory>         // std::unique_ptr, std::make_unique
#include <sstream>        // std::ostringstream
#include <stdexcept>      // std::runtime_error
#include <string>         // std::string
#include <thread>         // std::this_thread::get_id
#include <unordered_set>  // std::unordered_set

#include "optree/optree.h"

namespace optree {

/*static*/ std::string PyTreeSpec::NodeKindToString(const Node& node) {
    switch (node.kind) {
        case PyTreeKind::Leaf:
            return "leaf type";
        case PyTreeKind::None:
            return "NoneType";
        case PyTreeKind::Tuple:
            return "tuple";
        case PyTreeKind::List:
            return "list";
        case PyTreeKind::Dict:
            return "dict";
        case PyTreeKind::OrderedDict:
            return "OrderedDict";
        case PyTreeKind::DefaultDict:
            return "defaultdict";
        case PyTreeKind::NamedTuple:
        case PyTreeKind::StructSequence:
            return PyRepr(node.node_data);
        case PyTreeKind::Deque:
            return "deque";
        case PyTreeKind::Custom:
            EXPECT_NE(node.custom, nullptr, "The custom registration is null.");
            return PyRepr(node.custom->type);
        default:
            INTERNAL_ERROR();
    }
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
std::string PyTreeSpec::ToStringImpl() const {
    auto agenda = reserved_vector<std::string>(4);
    for (const Node& node : m_traversal) {
        EXPECT_GE(py::ssize_t_cast(agenda.size()), node.arity, "Too few elements for container.");

        std::ostringstream children_sstream{};
        {
            bool first = true;
            for (auto it = agenda.cend() - node.arity; it != agenda.cend(); ++it) {
                if (!first) [[likely]] {
                    children_sstream << ", ";
                }
                children_sstream << *it;
                first = false;
            }
        }
        const std::string children = children_sstream.str();

        std::ostringstream sstream{};
        switch (node.kind) {
            case PyTreeKind::Leaf: {
                sstream << "*";
                break;
            }

            case PyTreeKind::None: {
                sstream << "None";
                break;
            }

            case PyTreeKind::Tuple: {
                sstream << "(" << children;
                // Tuples with only one element must have a trailing comma.
                if (node.arity == 1) [[unlikely]] {
                    sstream << ",";
                }
                sstream << ")";
                break;
            }

            case PyTreeKind::List: {
                sstream << "[" << children << "]";
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict: {
                const scoped_critical_section cs{node.node_data};
                EXPECT_EQ(ListGetSize(node.node_data),
                          node.arity,
                          "Number of keys and entries does not match.");
                if (node.kind == PyTreeKind::OrderedDict) [[unlikely]] {
                    sstream << "OrderedDict(";
                }
                if (node.kind == PyTreeKind::Dict || node.arity > 0) [[likely]] {
                    sstream << "{";
                }
                bool first = true;
                auto child_iter = agenda.cend() - node.arity;
                for (const py::handle& key : node.node_data) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << PyRepr(key) << ": " << *child_iter;
                    ++child_iter;
                    first = false;
                }
                if (node.kind == PyTreeKind::Dict || node.arity > 0) [[likely]] {
                    sstream << "}";
                }
                if (node.kind == PyTreeKind::OrderedDict) [[unlikely]] {
                    sstream << ")";
                }
                break;
            }

            case PyTreeKind::NamedTuple: {
                const py::object type = node.node_data;
                const auto fields = NamedTupleGetFields(type);
                EXPECT_EQ(TupleGetSize(fields),
                          node.arity,
                          "Number of fields and entries does not match.");
                const std::string kind =
                    PyStr(EVALUATE_WITH_LOCK_HELD(py::getattr(type, Py_Get_ID(__name__)), type));
                sstream << kind << "(";
                bool first = true;
                auto child_iter = agenda.cend() - node.arity;
                for (const py::handle& field : fields) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << PyStr(field) << "=" << *child_iter;
                    ++child_iter;
                    first = false;
                }
                sstream << ")";
                break;
            }

            case PyTreeKind::DefaultDict: {
                const scoped_critical_section cs(node.node_data);
                EXPECT_EQ(TupleGetSize(node.node_data), 2, "Number of metadata mismatch.");
                const py::object default_factory = TupleGetItem(node.node_data, 0);
                const auto keys = TupleGetItemAs<py::list>(node.node_data, 1);
                EXPECT_EQ(ListGetSize(keys),
                          node.arity,
                          "Number of keys and entries does not match.");
                sstream << "defaultdict(" << PyRepr(default_factory) << ", {";
                bool first = true;
                auto child_it = agenda.cend() - node.arity;
                for (const py::handle& key : keys) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << PyRepr(key) << ": " << *child_it;
                    ++child_it;
                    first = false;
                }
                sstream << "})";
                break;
            }

            case PyTreeKind::Deque: {
                sstream << "deque([" << children << "]";
                if (!node.node_data.is_none()) [[unlikely]] {
                    sstream << ", maxlen=" << PyRepr(node.node_data);
                }
                sstream << ")";
                break;
            }

            case PyTreeKind::StructSequence: {
                const py::object type = node.node_data;
                const auto fields = StructSequenceGetFields(type);
                EXPECT_EQ(TupleGetSize(fields),
                          node.arity,
                          "Number of fields and entries does not match.");
                const py::object module_name = EVALUATE_WITH_LOCK_HELD(
                    py::getattr(type, Py_Get_ID(__module__), Py_Get_ID(__main__)),
                    type);
                if (!module_name.is_none()) [[likely]] {
                    const std::string name = PyStr(module_name);
                    if (!(name.empty() || name == "__main__" || name == "builtins" ||
                          name == "__builtins__")) [[likely]] {
                        sstream << name << ".";
                    }
                }
                const py::object qualname =
                    EVALUATE_WITH_LOCK_HELD(py::getattr(type, Py_Get_ID(__qualname__)), type);
                sstream << PyStr(qualname) << "(";
                bool first = true;
                auto child_iter = agenda.cend() - node.arity;
                for (const py::handle& field : fields) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << PyStr(field) << "=" << *child_iter;
                    ++child_iter;
                    first = false;
                }
                sstream << ")";
                break;
            }

            case PyTreeKind::Custom: {
                const std::string kind = PyStr(
                    EVALUATE_WITH_LOCK_HELD(py::getattr(node.custom->type, Py_Get_ID(__name__)),
                                            node.custom->type));
                sstream << "CustomTreeNode(" << kind << "[";
                if (node.node_data) [[likely]] {
                    sstream << PyRepr(node.node_data);
                }
                sstream << "], [" << children << "])";
                break;
            }

            default:
                INTERNAL_ERROR();
        }

        agenda.resize(agenda.size() - node.arity);
        agenda.emplace_back(sstream.str());
    }

    EXPECT_EQ(agenda.size(), 1, "PyTreeSpec traversal did not yield a singleton.");
    std::ostringstream oss{};
    oss << "PyTreeSpec(" << agenda.back();
    if (m_none_is_leaf) [[unlikely]] {
        oss << ", NoneIsLeaf";
    }
    if (!m_namespace.empty()) [[unlikely]] {
        oss << ", namespace=" << PyRepr(m_namespace);
    }
    oss << ")";
    return oss.str();
}

std::string PyTreeSpec::ToString() const {
    PYTREESPEC_SANITY_CHECK(*this);

    static std::unordered_set<ThreadedIdentity> running{};
    static read_write_mutex mutex{};

    const ThreadedIdentity ident{this, std::this_thread::get_id()};
    {
        const scoped_read_lock_guard lock{mutex};
        if (running.find(ident) != running.end()) [[unlikely]] {
            return "...";
        }
    }

    {
        const scoped_write_lock_guard lock{mutex};
        running.insert(ident);
    }
    try {
        std::string representation = ToStringImpl();
        {
            const scoped_write_lock_guard lock{mutex};
            running.erase(ident);
        }
        return representation;
    } catch (...) {
        {
            const scoped_write_lock_guard lock{mutex};
            running.erase(ident);
        }
        std::rethrow_exception(std::current_exception());
    }
}

py::object PyTreeSpec::ToPickleable() const {
    PYTREESPEC_SANITY_CHECK(*this);

    const py::tuple node_states{GetNumNodes()};
    ssize_t i = 0;
    for (const auto& node : m_traversal) {
        const scoped_critical_section2 cs{
            node.custom != nullptr ? py::handle{node.custom->type.ptr()} : py::handle{},
            node.node_data};
        TupleSetItem(node_states,
                     i++,
                     py::make_tuple(py::int_(static_cast<ssize_t>(node.kind)),
                                    py::int_(node.arity),
                                    node.node_data ? node.node_data : py::none(),
                                    node.node_entries ? node.node_entries : py::none(),
                                    node.custom != nullptr ? node.custom->type : py::none(),
                                    py::int_(node.num_leaves),
                                    py::int_(node.num_nodes),
                                    node.original_keys ? node.original_keys : py::none()));
    }
    return py::make_tuple(node_states, py::bool_(m_none_is_leaf), py::str(m_namespace));
}

// NOLINTBEGIN[cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers]
// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::FromPickleable(const py::object& pickleable) {
    const auto state = thread_safe_cast<py::tuple>(pickleable);
    if (state.size() != 3) [[unlikely]] {
        throw std::runtime_error("Malformed pickled PyTreeSpec.");
    }
    bool none_is_leaf = false;
    std::string registry_namespace{};
    auto out = std::make_unique<PyTreeSpec>();
    out->m_none_is_leaf = none_is_leaf = thread_safe_cast<bool>(state[1]);
    out->m_namespace = registry_namespace = thread_safe_cast<std::string>(state[2]);
    const auto node_states = thread_safe_cast<py::tuple>(state[0]);
    for (const auto& item : node_states) {
        const auto t = thread_safe_cast<py::tuple>(item);
        Node& node = out->m_traversal.emplace_back();
        node.kind = static_cast<PyTreeKind>(thread_safe_cast<ssize_t>(t[0]));
        if (t.size() != 7) [[unlikely]] {
            if (t.size() == 8) [[likely]] {
                if (t[7].is_none()) [[likely]] {
                    if (node.kind == PyTreeKind::Dict || node.kind == PyTreeKind::DefaultDict)
                        [[unlikely]] {
                        throw std::runtime_error("Malformed pickled PyTreeSpec.");
                    }
                } else [[unlikely]] {
                    if (node.kind == PyTreeKind::Dict || node.kind == PyTreeKind::DefaultDict)
                        [[likely]] {
                        node.original_keys = thread_safe_cast<py::list>(t[7]);
                    } else [[unlikely]] {
                        throw std::runtime_error("Malformed pickled PyTreeSpec.");
                    }
                }
            } else [[unlikely]] {
                throw std::runtime_error("Malformed pickled PyTreeSpec.");
            }
        }
        node.arity = thread_safe_cast<ssize_t>(t[1]);
        switch (node.kind) {
            case PyTreeKind::Leaf:
            case PyTreeKind::None:
            case PyTreeKind::Tuple:
            case PyTreeKind::List: {
                if (!t[2].is_none()) [[unlikely]] {
                    throw std::runtime_error("Malformed pickled PyTreeSpec.");
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict: {
                node.node_data = thread_safe_cast<py::list>(t[2]);
                break;
            }

            case PyTreeKind::NamedTuple:
            case PyTreeKind::StructSequence: {
                node.node_data = thread_safe_cast<py::type>(t[2]);
                break;
            }

            case PyTreeKind::DefaultDict:
            case PyTreeKind::Deque:
            case PyTreeKind::Custom: {
                node.node_data = t[2];
                break;
            }

            default:
                INTERNAL_ERROR();
        }
        if (node.kind == PyTreeKind::Custom) [[unlikely]] {  // NOLINT
            if (!t[3].is_none()) [[unlikely]] {
                node.node_entries = thread_safe_cast<py::tuple>(t[3]);
            }
            if (t[4].is_none()) [[unlikely]] {
                node.custom = nullptr;
            } else [[likely]] {
                if (none_is_leaf) [[unlikely]] {
                    node.custom =
                        PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(t[4], registry_namespace);
                } else [[likely]] {
                    node.custom =
                        PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(t[4], registry_namespace);
                }
            }
            if (node.custom == nullptr) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Unknown custom type in pickled PyTreeSpec: " << PyRepr(t[4]);
                if (!registry_namespace.empty()) [[likely]] {
                    oss << " in namespace " << PyRepr(registry_namespace);
                } else [[unlikely]] {
                    oss << " in the global namespace";
                }
                oss << ".";
                throw std::runtime_error(oss.str());
            }
        } else if (!t[3].is_none() || !t[4].is_none()) [[unlikely]] {
            throw std::runtime_error("Malformed pickled PyTreeSpec.");
        }
        node.num_leaves = thread_safe_cast<ssize_t>(t[5]);
        node.num_nodes = thread_safe_cast<ssize_t>(t[6]);
    }
    out->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*out);
    return out;
}
// NOLINTEND[cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers]

}  // namespace optree
