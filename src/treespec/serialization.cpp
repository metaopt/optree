/*
Copyright 2022-2026 MetaOPT Team. All Rights Reserved.

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
#include <utility>        // std::pair, std::move

#include "optree/optree.h"

namespace optree {

/*static*/ std::string PyTreeSpec::NodeKindToString(const Node &node) {
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
        case PyTreeKind::FrozenDict:
            return "frozendict";
        case PyTreeKind::Custom:
            EXPECT_NE(node.custom, nullptr, "The custom registration is null.");
            return PyRepr(node.custom->type);
        case PyTreeKind::NumKinds:
        default:
            INTERNAL_ERROR();
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::string PyTreeSpec::ToStringImpl() const {
    auto agenda = reserved_vector<std::string>(4);
    for (const Node &node : m_traversal) {
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
            case PyTreeKind::OrderedDict:
            case PyTreeKind::FrozenDict: {
                const scoped_critical_section cs{node.node_data};
                EXPECT_EQ(ListGetSize(node.node_data),
                          node.arity,
                          "Number of keys and entries does not match.");
                if (node.kind == PyTreeKind::OrderedDict) [[unlikely]] {
                    sstream << "OrderedDict(";
                } else if (node.kind == PyTreeKind::FrozenDict) [[unlikely]] {
                    sstream << "frozendict(";
                }
                if (node.kind == PyTreeKind::Dict || node.arity > 0) [[likely]] {
                    sstream << "{";
                }
                bool first = true;
                auto child_it = agenda.cend() - node.arity;
                for (const py::handle &key : node.node_data) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << PyRepr(key) << ": " << *child_it;
                    ++child_it;
                    first = false;
                }
                if (node.kind == PyTreeKind::Dict || node.arity > 0) [[likely]] {
                    sstream << "}";
                }
                if (node.kind == PyTreeKind::OrderedDict || node.kind == PyTreeKind::FrozenDict)
                    [[unlikely]] {
                    sstream << ")";
                }
                break;
            }

            case PyTreeKind::NamedTuple: {
                const py::object type = node.node_data;
                const auto fields = NamedTupleGetFields(type);
                // The field names are read from the (mutable) `_fields` attribute at repr time, so
                // a caller may have changed them after the treespec was built. Report the mismatch
                // as a `ValueError`, not an internal error, since the cause is external.
                if (TupleGetSize(fields) != node.arity) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "Number of fields (" << TupleGetSize(fields) << ") of namedtuple type "
                        << PyRepr(type) << " does not match the arity (" << node.arity
                        << ") of the treespec node. The `_fields` attribute may have been modified "
                           "after the treespec was created.";
                    throw py::value_error(oss.str());
                }
                const std::string kind =
                    PyStr(EVALUATE_WITH_LOCK_HELD(py::getattr(type, "__name__"), type));
                sstream << kind << "(";
                bool first = true;
                auto child_it = agenda.cend() - node.arity;
                for (const py::handle &field : fields) {
                    if (!first) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << PyStr(field) << "=" << *child_it;
                    ++child_it;
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
                for (const py::handle &key : keys) {
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
                sstream << "deque(";
                if (node.arity > 0) [[likely]] {
                    sstream << "[" << children << "]";
                }
                if (!node.node_data.is_none()) [[unlikely]] {
                    if (node.arity > 0) [[likely]] {
                        sstream << ", ";
                    }
                    sstream << "maxlen=" << PyRepr(node.node_data);
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
                const py::object module_name =
                    EVALUATE_WITH_LOCK_HELD(py::getattr(type, "__module__", py::none()), type);
                if (!module_name.is_none()) [[likely]] {
                    const std::string name = PyStr(module_name);
                    if (!(name.empty() || name == "__main__" || name == "builtins" ||
                          name == "__builtins__")) [[likely]] {
                        sstream << name << ".";
                    }
                }
                const py::object qualname =
                    EVALUATE_WITH_LOCK_HELD(py::getattr(type, "__qualname__"), type);
                sstream << PyStr(qualname) << "(";
                ssize_t index = 0;
                auto child_it = agenda.cend() - node.arity;
                for (const py::handle &field : fields) {
                    if (index > 0) [[likely]] {
                        sstream << ", ";
                    }
                    const std::string name = PyStr(field);
                    // An unnamed slot has no valid identifier, so render it angle-bracketed like
                    // CPython's `<lambda>` rather than as the bare marker, which would read as an
                    // invalid keyword argument. The index disambiguates multiple unnamed slots.
                    if (name == PyStructSequenceUnnamedField()) [[unlikely]] {
                        sstream << "<unnamed@" << index << ">";
                    } else [[likely]] {
                        sstream << name;
                    }
                    sstream << "=" << *child_it;
                    ++child_it;
                    ++index;
                }
                sstream << ")";
                break;
            }

            case PyTreeKind::Custom: {
                const std::string kind =
                    PyStr(EVALUATE_WITH_LOCK_HELD(py::getattr(node.custom->type, "__name__"),
                                                  node.custom->type));
                sstream << "CustomTreeNode(" << kind << "[";
                if (node.node_data) [[likely]] {
                    sstream << PyRepr(node.node_data);
                }
                sstream << "], [" << children << "])";
                break;
            }

            case PyTreeKind::NumKinds:
            default:
                INTERNAL_ERROR();
        }

        agenda.resize(agenda.size() - node.arity);
        agenda.emplace_back(sstream.str());
    }

    EXPECT_EQ(agenda.size(), 1U, "PyTreeSpec traversal did not yield a singleton.");
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
        const scoped_read_lock lock{mutex};
        if (running.find(ident) != running.end()) [[unlikely]] {
            return "...";
        }
    }

    {
        const scoped_write_lock lock{mutex};
        running.insert(ident);
    }
    try {
        std::string representation = ToStringImpl();
        {
            const scoped_write_lock lock{mutex};
            running.erase(ident);
        }
        return representation;
    } catch (...) {
        {
            const scoped_write_lock lock{mutex};
            running.erase(ident);
        }
        std::rethrow_exception(std::current_exception());
    }
}

py::object PyTreeSpec::ToPicklable() const {
    PYTREESPEC_SANITY_CHECK(*this);

    const py::tuple node_states{GetNumNodes()};
    ssize_t i = 0;
    for (const auto &node : m_traversal) {
        const scoped_critical_section2 cs{
            node.custom != nullptr ? py::handle{node.custom->type} : py::handle{},
            node.node_data};

        // Copy the node's mutable containers so the pickled state cannot alias (and, if the caller
        // mutates it, corrupt) the immutable spec. Only dict-like keys are mutable and internal; a
        // namedtuple/PyStructSequence type or custom metadata is left as-is.
        py::object node_data =
            node.node_data ? py::reinterpret_borrow<py::object>(node.node_data) : py::none();
        if (node.kind == PyTreeKind::Dict || node.kind == PyTreeKind::OrderedDict ||
            node.kind == PyTreeKind::FrozenDict) [[unlikely]] {
            node_data = ListCopy(node.node_data);
        } else if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
            const auto metadata = py::reinterpret_borrow<py::tuple>(node.node_data);
            node_data =
                py::make_tuple(TupleGetItem(metadata, 0), ListCopy(TupleGetItem(metadata, 1)));
        }

        TupleSetItem(node_states,
                     i++,
                     py::make_tuple(py::int_(static_cast<ssize_t>(node.kind)),
                                    py::int_(node.arity),
                                    std::move(node_data),
                                    node.node_entries ? node.node_entries : py::none(),
                                    node.custom != nullptr ? node.custom->type : py::none(),
                                    py::int_(node.num_leaves),
                                    py::int_(node.num_nodes),
                                    node.original_keys
                                        ? static_cast<py::object>(DictCopy(node.original_keys))
                                        : static_cast<py::object>(py::none())));
    }
    return py::make_tuple(node_states, py::bool_(m_none_is_leaf), py::str(m_namespace));
}

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::FromPicklable(const py::object &picklable) {
    const auto malformed = [](const std::string &reason) -> std::runtime_error {
        return std::runtime_error("Malformed pickled PyTreeSpec: " + reason + ".");
    };
    // `DistinctCount` hashes the keys, so an unhashable one raises `TypeError`. Report it as a
    // malformed pickle like every other structural defect instead of letting it escape.
    const auto distinct_count = [&malformed](const py::handle &keys) -> ssize_t {
        try {
            return DistinctCount(keys);
        } catch (py::error_already_set &ex) {
            if (!ex.matches(PyExc_TypeError)) [[unlikely]] {
                throw;
            }
            throw malformed("the keys are not hashable");
        }
    };

    const auto state = thread_safe_cast<py::tuple>(picklable);
    if (state.size() != 3) [[unlikely]] {
        throw malformed("the state is not a 3-tuple");
    }
    bool none_is_leaf = false;
    std::string registry_namespace{};
    auto out = std::make_unique<PyTreeSpec>();
    out->m_none_is_leaf = none_is_leaf = thread_safe_cast<bool>(state[1]);
    out->m_namespace = registry_namespace = thread_safe_cast<std::string>(state[2]);
    const auto node_states = thread_safe_cast<py::tuple>(state[0]);
    for (const auto &item : node_states) {
        const auto node_state = thread_safe_cast<py::tuple>(item);
        const auto node_state_size = node_state.size();
        if (node_state_size != 7 && node_state_size != 8) [[unlikely]] {
            throw malformed("a node state is not a 7- or 8-tuple");
        }

        Node &node = out->m_traversal.emplace_back();
        const auto kind_value = thread_safe_cast<ssize_t>(node_state[0]);
        if (kind_value < 0 || kind_value >= static_cast<ssize_t>(PyTreeKind::NumKinds))
            [[unlikely]] {
            throw malformed("the node kind is out of range");
        }
        node.kind = static_cast<PyTreeKind>(kind_value);
#if !defined(OPTREE_HAS_FROZENDICT)
        // The enum carries `FrozenDict` on every build, so the range check above lets it through.
        // Reject a cross-version state here rather than demoting it to a mutable `dict` in
        // `MakeNode`. The state is well-formed, hence an unsupported build, not a malformed pickle.
        if (node.kind == PyTreeKind::FrozenDict) [[unlikely]] {
            throw py::value_error(
                "Cannot restore a PyTreeSpec containing a `frozendict` node: this build of "
                "optree was compiled without `frozendict` support (requires Python 3.15+).");
        }
#endif
        node.arity = thread_safe_cast<ssize_t>(node_state[1]);
        if (node.arity < 0) [[unlikely]] {
            throw malformed("the node arity is negative");
        }
        if (node_state_size == 8) [[likely]] {
            const auto &original_keys = node_state[7];
            if (original_keys.is_none()) [[likely]] {
                if (node.kind == PyTreeKind::Dict || node.kind == PyTreeKind::DefaultDict ||
                    node.kind == PyTreeKind::FrozenDict) [[unlikely]] {
                    throw malformed("a dict node is missing its original keys");
                }
            } else [[unlikely]] {
                if (node.kind == PyTreeKind::Dict || node.kind == PyTreeKind::DefaultDict ||
                    node.kind == PyTreeKind::FrozenDict) [[likely]] {
                    // `DictFromKeys` builds a new dict, so the caller cannot mutate it afterwards.
                    node.original_keys = DictFromKeys(original_keys);
                } else [[unlikely]] {
                    throw malformed("a non-dict node must not have original keys");
                }
            }
        }

        node.num_leaves = thread_safe_cast<ssize_t>(node_state[5]);
        node.num_nodes = thread_safe_cast<ssize_t>(node_state[6]);
        if (node.num_leaves < 0 || node.num_nodes < 1) [[unlikely]] {
            throw malformed("a node has a negative or invalid size");
        }

        const auto &node_data = node_state[2];
        const auto &node_entries = node_state[3];
        const auto &custom_type = node_state[4];
        switch (node.kind) {
            case PyTreeKind::Leaf:
            case PyTreeKind::None: {
                if (!node_data.is_none()) [[unlikely]] {
                    throw malformed("a leaf or none node must not have node data");
                }
                // A leaf or none node is childless; a nonzero arity would let it absorb preceding
                // subtrees (folding consistently) and silently drop leaves on unflatten.
                if (node.arity != 0) [[unlikely]] {
                    throw malformed("a leaf or none node must have arity 0");
                }
                // With `none_is_leaf`, None is flattened as a leaf, so a flattened tree never
                // contains a None-kind node; a reconstructed one would later trip an InternalError
                // in `FlattenUpTo` (`GetKind` never returns `None` under `NoneIsLeaf`).
                if (node.kind == PyTreeKind::None && none_is_leaf) [[unlikely]] {
                    throw malformed("a none node cannot appear when none_is_leaf is set");
                }
                break;
            }

            case PyTreeKind::Tuple:
            case PyTreeKind::List: {
                if (!node_data.is_none()) [[unlikely]] {
                    throw malformed("a tuple or list node must not have node data");
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::FrozenDict: {
                // Copy the keys instead of borrowing them.
                node.node_data = ListCopy(thread_safe_cast<py::list>(node_data));
                if (ListGetSize(node.node_data) != node.arity) [[unlikely]] {
                    throw malformed("the number of keys does not match the arity");
                }
                // The keys must be hashable and distinct; a duplicate or unhashable key would
                // collapse or fail when the dict is rebuilt, desyncing the keys from the children.
                if (distinct_count(node.node_data) != node.arity) [[unlikely]] {
                    throw malformed("the keys are not distinct");
                }
                break;
            }

            case PyTreeKind::NamedTuple: {
                node.node_data = thread_safe_cast<py::type>(node_data);
                if (!IsNamedTupleClass(node.node_data)) [[unlikely]] {
                    throw malformed("the node data is not a namedtuple type");
                }
                if (TupleGetSize(NamedTupleGetFields(node.node_data)) != node.arity) [[unlikely]] {
                    throw malformed("the number of fields does not match the arity");
                }
                break;
            }

            case PyTreeKind::StructSequence: {
                node.node_data = thread_safe_cast<py::type>(node_data);
                if (!IsStructSequenceClass(node.node_data)) [[unlikely]] {
                    throw malformed("the node data is not a PyStructSequence type");
                }
                if (TupleGetSize(StructSequenceGetFields(node.node_data)) != node.arity)
                    [[unlikely]] {
                    throw malformed("the number of fields does not match the arity");
                }
                break;
            }

            case PyTreeKind::DefaultDict: {
                // A default dict stores its metadata as a 2-tuple `(default_factory, sorted_keys)`.
                // `MakeNode` reads it with raw tuple/list accessors, so validate the shape here to
                // avoid type-confusion on malformed input.
                const auto metadata = thread_safe_cast<py::tuple>(node_data);
                if (metadata.size() != 2) [[unlikely]] {
                    throw malformed("the defaultdict metadata is not a 2-tuple");
                }
                // `default_factory` is passed to `defaultdict(...)`, which requires None or
                // callable.
                const auto default_factory = TupleGetItem(metadata, 0);
                if (!(default_factory.is_none() ||
                      static_cast<bool>(PyCallable_Check(default_factory.ptr())))) [[unlikely]] {
                    throw malformed("the `default_factory` is not callable");
                }
                // Copy the keys and rebuild the metadata tuple around the copy (see the dict case).
                const auto keys = ListCopy(thread_safe_cast<py::list>(metadata[1]));
                if (ListGetSize(keys) != node.arity) [[unlikely]] {
                    throw malformed("the number of keys does not match the arity");
                }
                if (distinct_count(keys) != node.arity) [[unlikely]] {
                    throw malformed("the keys are not distinct");
                }
                node.node_data = py::make_tuple(default_factory, keys);
                break;
            }

            case PyTreeKind::Deque: {
                // A deque's `maxlen` is None (unbounded) or a non-negative int bounding its length,
                // so it must be at least the node's arity.
                if (!node_data.is_none()) [[likely]] {
                    if (PyLong_Check(node_data.ptr()) == 0 ||
                        thread_safe_cast<ssize_t>(node_data) < node.arity) [[unlikely]] {
                        throw malformed("the deque maxlen is invalid");
                    }
                }
                node.node_data = node_data;
                break;
            }

            case PyTreeKind::Custom: {
                node.node_data = node_data;
                break;
            }

            case PyTreeKind::NumKinds:
            default:
                INTERNAL_ERROR();
        }
        if (node.kind == PyTreeKind::Custom) [[unlikely]] {
            if (!node_entries.is_none()) [[unlikely]] {
                node.node_entries = thread_safe_cast<py::tuple>(node_entries);
            }
            if (custom_type.is_none()) [[unlikely]] {
                node.custom = nullptr;
            } else [[likely]] {
                if (none_is_leaf) [[unlikely]] {
                    node.custom =
                        PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(custom_type, registry_namespace);
                } else [[likely]] {
                    node.custom =
                        PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(custom_type, registry_namespace);
                }
            }
            if (node.custom == nullptr) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Unknown custom type in pickled PyTreeSpec: " << PyRepr(custom_type);
                if (!registry_namespace.empty()) [[likely]] {
                    oss << " in namespace " << PyRepr(registry_namespace);
                } else [[unlikely]] {
                    oss << " in the global namespace";
                }
                oss << ".";
                throw std::runtime_error(oss.str());
            }
            // A built-in registration (NoneType/tuple/list/dict/...) lives in the same map but has
            // empty flatten/unflatten callables, which `MakeNode`'s custom branch would call.
            if (node.custom->kind != PyTreeKind::Custom) [[unlikely]] {
                throw malformed("the custom type is a built-in type");
            }
        } else if (!node_entries.is_none() || !custom_type.is_none()) [[unlikely]] {
            throw malformed("a non-custom node must not have node entries or a custom type");
        }
        if (node.original_keys) [[unlikely]] {
            if (DictGetSize(node.original_keys) != node.arity) [[unlikely]] {
                throw malformed("the number of original keys does not match the arity");
            }
            // `original_keys` records the insertion order of the same keys stored (sorted) in
            // node_data; its key set must match, or unflatten would map children onto keys the dict
            // never had.
            const auto keys = (node.kind == PyTreeKind::DefaultDict
                                   ? TupleGetItemAs<py::list>(node.node_data, 1)
                                   : py::reinterpret_borrow<py::list>(node.node_data));
            if (!DictKeysEqual(keys, py::reinterpret_borrow<py::dict>(node.original_keys)))
                [[unlikely]] {
                throw malformed("the keys do not match the original keys");
            }
        }
        if (node.node_entries && !node.node_entries.is_none() &&
            TupleGetSize(node.node_entries) != node.arity) [[unlikely]] {
            throw malformed("the number of node entries does not match the arity");
        }
    }

    // Validate that the reconstructed traversal is structurally consistent.
    // `PYTREESPEC_SANITY_CHECK` only checks the final node, so a malformed pickle could otherwise
    // smuggle in inconsistent arity / num_nodes / num_leaves that cause out-of-bounds access when
    // the spec is later used. Walk the post-order traversal, folding each node's children off a
    // stack of subtree sizes.
    {
        auto subtree_sizes =
            reserved_vector</*(num_nodes, num_leaves)*/ std::pair<ssize_t, ssize_t>>(
                out->m_traversal.size());
        for (const Node &node : out->m_traversal) {
            if (static_cast<ssize_t>(subtree_sizes.size()) < node.arity) [[unlikely]] {
                throw malformed("a node has more children than available subtrees");
            }
            ssize_t children_num_nodes = 0;
            ssize_t children_num_leaves = 0;
            for (ssize_t i = 0; i < node.arity; ++i) {
                children_num_nodes += subtree_sizes.back().first;
                children_num_leaves += subtree_sizes.back().second;
                subtree_sizes.pop_back();
            }
            const ssize_t expected_num_nodes = children_num_nodes + 1;
            const ssize_t expected_num_leaves =
                (node.kind == PyTreeKind::Leaf ? ssize_t{1} : children_num_leaves);
            if (node.num_nodes != expected_num_nodes || node.num_leaves != expected_num_leaves)
                [[unlikely]] {
                throw malformed("a node's size is inconsistent with its children");
            }
            subtree_sizes.emplace_back(node.num_nodes, node.num_leaves);
        }
        if (subtree_sizes.size() != 1) [[unlikely]] {
            throw malformed("the traversal does not yield a single tree");
        }
    }

    out->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*out);
    return out;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

}  // namespace optree
