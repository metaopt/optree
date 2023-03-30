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

#include <Python.h>
#include <structmember.h>  // PyMemberDef

namespace optree {

ssize_t PyTreeSpec::GetNumLeaves() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    return m_traversal.back().num_leaves;
}

ssize_t PyTreeSpec::GetNumNodes() const { return py::ssize_t_cast(m_traversal.size()); }

ssize_t PyTreeSpec::GetNumChildren() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    return m_traversal.back().arity;
}

bool PyTreeSpec::GetNoneIsLeaf() const { return m_none_is_leaf; }

std::string PyTreeSpec::GetNamespace() const { return m_namespace; }

py::object PyTreeSpec::GetType() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    Node node = m_traversal.back();
    switch (node.kind) {
        case PyTreeKind::Custom:
            EXPECT_NE(node.custom, nullptr, "The custom registration is null.");
            return node.custom->type;
        case PyTreeKind::Leaf:
            return py::none();
        case PyTreeKind::None:
            return py::type::of(py::none());
        case PyTreeKind::Tuple:
            return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyTuple_Type));
        case PyTreeKind::List:
            return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyList_Type));
        case PyTreeKind::Dict:
            return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(&PyDict_Type));
        case PyTreeKind::NamedTuple:
        case PyTreeKind::StructSequence:
            return node.node_data;
        case PyTreeKind::OrderedDict:
            return PyOrderedDictTypeObject;
        case PyTreeKind::DefaultDict:
            return PyDefaultDictTypeObject;
        case PyTreeKind::Deque:
            return PyDequeTypeObject;
        default:
            INTERNAL_ERROR();
    }
}

bool PyTreeSpec::IsLeaf(const bool& strict) const {
    if (strict) [[likely]] {
        return GetNumNodes() == 1 && GetNumLeaves() == 1;
    }
    return GetNumNodes() == 1;
}

bool PyTreeSpec::operator==(const PyTreeSpec& other) const {
    if (m_traversal.size() != other.m_traversal.size() || m_none_is_leaf != other.m_none_is_leaf)
        [[likely]] {
        return false;
    }
    if (!m_namespace.empty() && !other.m_namespace.empty() && m_namespace != other.m_namespace)
        [[likely]] {
        return false;
    }

    // NOLINTNEXTLINE[readability-qualified-auto]
    auto b = other.m_traversal.begin();
    // NOLINTNEXTLINE[readability-qualified-auto]
    for (auto a = m_traversal.begin(); a != m_traversal.end(); ++a, ++b) {
        if (a->kind != b->kind || a->arity != b->arity ||
            (a->node_data.ptr() == nullptr) != (b->node_data.ptr() == nullptr) ||
            a->custom != b->custom) [[likely]] {
            return false;
        }
        if (a->node_data && a->node_data.not_equal(b->node_data)) [[likely]] {
            return false;
        }
        EXPECT_EQ(a->num_leaves, b->num_leaves);
        EXPECT_EQ(a->num_nodes, b->num_nodes);
    }
    return true;
}

bool PyTreeSpec::IsPrefix(const PyTreeSpec& other, const bool& strict) const {
    if (m_none_is_leaf != other.m_none_is_leaf) [[unlikely]] {
        return false;
    }
    if (!m_namespace.empty() && !other.m_namespace.empty() && m_namespace != other.m_namespace)
        [[likely]] {
        return false;
    }
    if (GetNumNodes() > other.GetNumNodes()) [[likely]] {
        return false;
    }

    bool all_leaves_match = true;
    // NOLINTNEXTLINE[readability-qualified-auto]
    auto b = other.m_traversal.rbegin();
    // NOLINTNEXTLINE[readability-qualified-auto]
    for (auto a = m_traversal.rbegin(); a != m_traversal.rend(); ++a, ++b) {
        if (b == other.m_traversal.rend()) [[unlikely]] {
            return false;
        }
        if (a->kind == PyTreeKind::Leaf) [[unlikely]] {
            all_leaves_match &= b->kind == PyTreeKind::Leaf;
            b += b->num_nodes - 1;
            EXPECT_LT(b, other.m_traversal.rend(), "PyTreeSpec traversal out of range.");
            continue;
        }
        if (a->kind != b->kind || a->arity != b->arity ||
            (a->node_data.ptr() == nullptr) != (b->node_data.ptr() == nullptr) ||
            a->custom != b->custom) [[likely]] {
            return false;
        }
        if (a->node_data && a->node_data.not_equal(b->node_data)) [[likely]] {
            return false;
        }
        if (a->num_nodes > b->num_nodes) [[likely]] {
            return false;
        }
    }
    EXPECT_EQ(b, other.m_traversal.rend(), "PyTreeSpec traversal did not yield a singleton.");
    return !strict || !all_leaves_match;
}

std::unique_ptr<PyTreeSpec> PyTreeSpec::Compose(const PyTreeSpec& inner_treespec) const {
    if (m_none_is_leaf != inner_treespec.m_none_is_leaf) [[unlikely]] {
        throw py::value_error("PyTreeSpecs must have the same none_is_leaf value.");
    }
    if (!m_namespace.empty() && !inner_treespec.m_namespace.empty() &&
        m_namespace != inner_treespec.m_namespace) [[unlikely]] {
        throw py::value_error(
            absl::StrFormat("PyTreeSpecs must have the same namespace. Got %s vs. %s.",
                            py::repr(py::str(m_namespace)),
                            py::repr(py::str(inner_treespec.m_namespace))));
    }

    auto treespec = std::make_unique<PyTreeSpec>();
    treespec->m_none_is_leaf = m_none_is_leaf;
    if (inner_treespec.m_namespace.empty()) [[likely]] {
        treespec->m_namespace = m_namespace;
    } else [[unlikely]] {
        treespec->m_namespace = inner_treespec.m_namespace;
    }

    const ssize_t num_outer_leaves = GetNumLeaves();
    const ssize_t num_outer_nodes = GetNumNodes();
    const ssize_t num_inner_leaves = inner_treespec.GetNumLeaves();
    const ssize_t num_inner_nodes = inner_treespec.GetNumNodes();
    for (const Node& node : m_traversal) {
        if (node.kind == PyTreeKind::Leaf) [[likely]] {
            absl::c_copy(inner_treespec.m_traversal, std::back_inserter(treespec->m_traversal));
        } else [[unlikely]] {
            Node new_node{node};
            new_node.num_leaves = node.num_leaves * num_inner_leaves;
            new_node.num_nodes =
                (node.num_nodes - node.num_leaves) + (node.num_leaves * num_inner_nodes);
            treespec->m_traversal.emplace_back(std::move(new_node));
        }
    }

    const auto& root = treespec->m_traversal.back();
    EXPECT_EQ(root.num_leaves,
              num_outer_leaves * num_inner_leaves,
              "Number of composed tree leaves mismatch.");
    EXPECT_EQ(root.num_nodes,
              (num_outer_nodes - num_outer_leaves) + (num_outer_leaves * num_inner_nodes),
              "Number of composed tree nodes mismatch.");
    return treespec;
}

template <typename Span, typename Stack>
// NOLINTNEXTLINE[misc-no-recursion]
ssize_t PyTreeSpec::PathsImpl(Span& paths,
                              Stack& stack,
                              const ssize_t& pos,
                              const ssize_t& depth) const {
    const Node& root = m_traversal.at(pos);
    EXPECT_GE(pos + 1, root.num_nodes, "PyTreeSpec::Paths() walked off start of array.");

    ssize_t cur = pos - 1;
    // NOLINTNEXTLINE[misc-no-recursion]
    auto recurse = [this, &paths, &stack, &depth](const ssize_t& cur, const py::handle& entry) {
        stack.emplace_back(entry);
        const ssize_t num_nodes = PathsImpl(paths, stack, cur, depth + 1);
        stack.pop_back();
        return num_nodes;
    };

    if (root.node_entries) [[unlikely]] {
        for (ssize_t i = root.arity - 1; i >= 0; --i) {
            cur -= recurse(cur, GET_ITEM_HANDLE<py::tuple>(root.node_entries, i));
        }
    } else [[likely]] {
        switch (root.kind) {
            case PyTreeKind::None:
                break;
            case PyTreeKind::Leaf: {
                py::tuple path{depth};
                for (ssize_t d = 0; d < depth; ++d) {
                    SET_ITEM<py::tuple>(path, d, stack[d]);
                }
                paths.emplace_back(std::move(path));
                break;
            }

            case PyTreeKind::Tuple:
            case PyTreeKind::List:
            case PyTreeKind::NamedTuple:
            case PyTreeKind::Deque:
            case PyTreeKind::StructSequence:
            case PyTreeKind::Custom: {
                for (ssize_t i = root.arity - 1; i >= 0; --i) {
                    cur -= recurse(cur, py::int_(i));
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                py::list keys;
                if (root.kind != PyTreeKind::DefaultDict) [[likely]] {
                    keys = py::reinterpret_borrow<py::list>(root.node_data);
                } else [[unlikely]] {
                    keys = GET_ITEM_BORROW<py::tuple>(root.node_data, 1);
                }
                for (ssize_t i = root.arity - 1; i >= 0; --i) {
                    cur -= recurse(cur, GET_ITEM_HANDLE<py::list>(keys, i));
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }

    return root.num_nodes;
}

std::vector<py::tuple> PyTreeSpec::Paths() const {
    auto paths = std::vector<py::tuple>{};
    const ssize_t num_leaves = GetNumLeaves();
    if (num_leaves == 0) [[unlikely]] {
        return paths;
    }
    const ssize_t num_nodes = GetNumNodes();
    if (num_nodes == 1 && num_leaves == 1) [[likely]] {
        paths.emplace_back();
        return paths;
    }
    auto stack = std::vector<py::handle>{};
    const ssize_t num_nodes_walked = PathsImpl(paths, stack, num_nodes - 1, 0);
    std::reverse(paths.begin(), paths.end());
    EXPECT_EQ(num_nodes_walked, num_nodes, "`pos != 0` at end of PyTreeSpec::Paths().");
    EXPECT_EQ(py::ssize_t_cast(paths.size()), num_leaves, "PyTreeSpec::Paths() mismatched leaves.");
    return paths;
}

py::list PyTreeSpec::Entries() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    const Node& root = m_traversal.back();
    if (root.node_entries) [[unlikely]] {
        return py::list{root.node_entries};
    }
    switch (root.kind) {
        case PyTreeKind::None:
        case PyTreeKind::Leaf: {
            return py::list{};
        }

        case PyTreeKind::Tuple:
        case PyTreeKind::List:
        case PyTreeKind::NamedTuple:
        case PyTreeKind::Deque:
        case PyTreeKind::StructSequence:
        case PyTreeKind::Custom: {
            auto entries = py::list{root.arity};
            for (ssize_t i = 0; i < root.arity; ++i) {
                SET_ITEM<py::list>(entries, i, py::int_(i));
            }
            return entries;
        }

        case PyTreeKind::Dict:
        case PyTreeKind::OrderedDict: {
            return py::list{root.node_data};
        }
        case PyTreeKind::DefaultDict: {
            return py::list{GET_ITEM_BORROW<py::tuple>(root.node_data, 1)};
        }

        default:
            INTERNAL_ERROR();
    }
}

std::vector<std::unique_ptr<PyTreeSpec>> PyTreeSpec::Children() const {
    EXPECT_FALSE(m_traversal.empty(), "The tree node traversal is empty.");
    const Node& root = m_traversal.back();
    auto children = std::vector<std::unique_ptr<PyTreeSpec>>{};
    children.resize(root.arity);
    ssize_t pos = py::ssize_t_cast(m_traversal.size()) - 1;
    for (ssize_t i = root.arity - 1; i >= 0; --i) {
        children[i] = std::make_unique<PyTreeSpec>();
        children[i]->m_none_is_leaf = m_none_is_leaf;
        children[i]->m_namespace = m_namespace;
        const Node& node = m_traversal.at(pos - 1);
        EXPECT_GE(pos, node.num_nodes, "PyTreeSpec::Children() walked off start of array.");
        std::copy(m_traversal.begin() + pos - node.num_nodes,
                  m_traversal.begin() + pos,
                  std::back_inserter(children[i]->m_traversal));
        pos -= node.num_nodes;
    }
    EXPECT_EQ(pos, 0, "`pos != 0` at end of PyTreeSpec::Children().");
    return children;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::Tuple(const std::vector<PyTreeSpec>& treespecs,
                                                         const bool& none_is_leaf) {
    std::string registry_namespace;
    for (const PyTreeSpec& treespec : treespecs) {
        if (treespec.m_none_is_leaf != none_is_leaf) [[unlikely]] {
            throw py::value_error(absl::StrFormat("Expected treespecs with `node_is_leaf=%s`.",
                                                  (none_is_leaf ? "True" : "False")));
        }
        if (!treespec.m_namespace.empty()) [[unlikely]] {
            if (registry_namespace.empty()) [[likely]] {
                registry_namespace = treespec.m_namespace;
            } else if (registry_namespace != treespec.m_namespace) [[unlikely]] {
                throw py::value_error(
                    absl::StrFormat("Expected treespecs with the same namespace. Got %s vs. %s.",
                                    py::repr(py::str(registry_namespace)),
                                    py::repr(py::str(treespec.m_namespace))));
            }
        }
    }

    auto out = std::make_unique<PyTreeSpec>();
    ssize_t num_leaves = 0;
    for (const PyTreeSpec& treespec : treespecs) {
        absl::c_copy(treespec.m_traversal, std::back_inserter(out->m_traversal));
        num_leaves += treespec.GetNumLeaves();
    }
    Node node;
    node.kind = PyTreeKind::Tuple;
    node.arity = py::ssize_t_cast(treespecs.size());
    node.num_leaves = num_leaves;
    node.num_nodes = py::ssize_t_cast(out->m_traversal.size()) + 1;
    out->m_traversal.emplace_back(std::move(node));
    out->m_none_is_leaf = none_is_leaf;
    out->m_namespace = registry_namespace;
    return out;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::Leaf(const bool& none_is_leaf) {
    auto out = std::make_unique<PyTreeSpec>();
    Node node;
    node.kind = PyTreeKind::Leaf;
    node.arity = 0;
    node.num_leaves = 1;
    node.num_nodes = 1;
    out->m_traversal.emplace_back(std::move(node));
    out->m_none_is_leaf = none_is_leaf;
    return out;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::None(const bool& none_is_leaf) {
    if (none_is_leaf) [[unlikely]] {
        return Leaf(none_is_leaf);
    }
    auto out = std::make_unique<PyTreeSpec>();
    Node node;
    node.kind = PyTreeKind::None;
    node.arity = 0;
    node.num_leaves = 0;
    node.num_nodes = 1;
    out->m_traversal.emplace_back(std::move(node));
    out->m_none_is_leaf = none_is_leaf;
    return out;
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ py::object PyTreeSpec::MakeNode(const PyTreeSpec::Node& node,
                                           const absl::Span<py::object>& children) {
    EXPECT_EQ(py::ssize_t_cast(children.size()), node.arity, "Node arity did not match.");
    switch (node.kind) {
        case PyTreeKind::Leaf:
            INTERNAL_ERROR("MakeNode not implemented for leaves.");

        case PyTreeKind::None:
            return py::none();

        case PyTreeKind::Tuple:
        case PyTreeKind::NamedTuple:
        case PyTreeKind::StructSequence: {
            py::tuple tuple{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                SET_ITEM<py::tuple>(tuple, i, children[i]);
            }
            if (node.kind == PyTreeKind::NamedTuple) [[unlikely]] {
                return node.node_data(*tuple);
            }
            if (node.kind == PyTreeKind::StructSequence) [[unlikely]] {
                return node.node_data(std::move(tuple));
            }
            return tuple;
        }

        case PyTreeKind::List:
        case PyTreeKind::Deque: {
            py::list list{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                SET_ITEM<py::list>(list, i, children[i]);
            }
            if (node.kind == PyTreeKind::Deque) [[unlikely]] {
                return PyDequeTypeObject(list, py::arg("maxlen") = node.node_data);
            }
            return list;
        }

        case PyTreeKind::Dict: {
            py::dict dict;
            auto keys = py::reinterpret_borrow<py::list>(node.node_data);
            if (node.ordered_keys) [[unlikely]] {
                for (ssize_t i = 0; i < node.arity; ++i) {
                    dict[GET_ITEM_HANDLE<py::list>(node.ordered_keys, i)] = py::none();
                }
            }
            for (ssize_t i = 0; i < node.arity; ++i) {
                dict[GET_ITEM_HANDLE<py::list>(keys, i)] = std::move(children[i]);
            }
            return dict;
        }

        case PyTreeKind::OrderedDict: {
            py::list items{node.arity};
            auto keys = py::reinterpret_borrow<py::list>(node.node_data);
            for (ssize_t i = 0; i < node.arity; ++i) {
                SET_ITEM<py::list>(
                    items,
                    i,
                    py::make_tuple(GET_ITEM_HANDLE<py::list>(keys, i), std::move(children[i])));
            }
            return PyOrderedDictTypeObject(items);
        }

        case PyTreeKind::DefaultDict: {
            py::dict dict;
            py::object default_factory = GET_ITEM_BORROW<py::tuple>(node.node_data, 0);
            py::list keys = GET_ITEM_BORROW<py::tuple>(node.node_data, 1);
            if (node.ordered_keys) [[unlikely]] {
                for (ssize_t i = 0; i < node.arity; ++i) {
                    dict[GET_ITEM_HANDLE<py::list>(node.ordered_keys, i)] = py::none();
                }
            }
            for (ssize_t i = 0; i < node.arity; ++i) {
                dict[GET_ITEM_HANDLE<py::list>(keys, i)] = std::move(children[i]);
            }
            return PyDefaultDictTypeObject(default_factory, dict);
        }

        case PyTreeKind::Custom: {
            py::tuple tuple{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                SET_ITEM<py::tuple>(tuple, i, children[i]);
            }
            return node.custom->from_iterable(node.node_data, tuple);
        }

        default:
            INTERNAL_ERROR();
    }
}

template <bool NoneIsLeaf>
/*static*/ PyTreeKind PyTreeSpec::GetKind(const py::handle& handle,
                                          PyTreeTypeRegistry::Registration const** custom,
                                          const std::string& registry_namespace) {
    const PyTreeTypeRegistry::Registration* registration =
        PyTreeTypeRegistry::Lookup<NoneIsLeaf>(py::type::handle_of(handle), registry_namespace);
    if (registration) [[likely]] {
        if (registration->kind == PyTreeKind::Custom) [[unlikely]] {
            *custom = registration;
        } else [[likely]] {
            *custom = nullptr;
        }
        return registration->kind;
    }
    *custom = nullptr;
    if (IsNamedTupleInstance(handle)) [[unlikely]] {
        return PyTreeKind::NamedTuple;
    }
    if (IsStructSequenceInstance(handle)) [[unlikely]] {
        return PyTreeKind::StructSequence;
    }
    return PyTreeKind::Leaf;
}

template PyTreeKind PyTreeSpec::GetKind<NONE_IS_NODE>(const py::handle&,
                                                      PyTreeTypeRegistry::Registration const**,
                                                      const std::string&);
template PyTreeKind PyTreeSpec::GetKind<NONE_IS_LEAF>(const py::handle&,
                                                      PyTreeTypeRegistry::Registration const**,
                                                      const std::string&);

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
std::string PyTreeSpec::ToString() const {
    auto agenda = std::vector<std::string>{};
    for (const Node& node : m_traversal) {
        EXPECT_GE(py::ssize_t_cast(agenda.size()), node.arity, "Too few elements for container.");

        std::string children = absl::StrJoin(agenda.end() - node.arity, agenda.end(), ", ");
        std::string representation;
        switch (node.kind) {
            case PyTreeKind::Leaf:
                agenda.emplace_back("*");
                continue;

            case PyTreeKind::None:
                representation = "None";
                break;

            case PyTreeKind::Tuple:
                // Tuples with only one element must have a trailing comma.
                if (node.arity == 1) [[unlikely]] {
                    children += ",";
                }
                representation = absl::StrCat("(", children, ")");
                break;

            case PyTreeKind::List:
            case PyTreeKind::Deque:
                representation = absl::StrCat("[", children, "]");
                if (node.kind == PyTreeKind::Deque) [[unlikely]] {
                    if (node.node_data.is_none()) [[likely]] {
                        representation = absl::StrCat("deque(", representation, ")");
                    } else [[unlikely]] {
                        representation = absl::StrFormat(
                            "deque(%s, maxlen=%s)", representation, py::str(node.node_data));
                    }
                }
                break;

            case PyTreeKind::Dict: {
                EXPECT_EQ(GET_SIZE<py::list>(node.node_data),
                          node.arity,
                          "Number of keys and entries does not match.");
                representation = "{";
                std::string separator;
                auto child_iter = agenda.end() - node.arity;
                for (const py::handle& key : node.node_data) {
                    absl::StrAppendFormat(
                        &representation, "%s%s: %s", separator, py::repr(key), *child_iter);
                    ++child_iter;
                    separator = ", ";
                }
                representation += "}";
                break;
            }

            case PyTreeKind::NamedTuple: {
                py::object type = node.node_data;
                auto fields = py::reinterpret_borrow<py::tuple>(py::getattr(type, "_fields"));
                EXPECT_EQ(GET_SIZE<py::tuple>(fields),
                          node.arity,
                          "Number of fields and entries does not match.");
                std::string kind = static_cast<std::string>(py::str(py::getattr(type, "__name__")));
                representation = absl::StrFormat("%s(", kind);
                std::string separator;
                auto child_iter = agenda.end() - node.arity;
                for (const py::handle& field : fields) {
                    absl::StrAppendFormat(
                        &representation,
                        "%s%s=%s",
                        separator,
                        static_cast<std::string>(py::reinterpret_borrow<py::str>(field)),
                        *child_iter);
                    ++child_iter;
                    separator = ", ";
                }
                representation += ")";
                break;
            }

            case PyTreeKind::OrderedDict: {
                EXPECT_EQ(GET_SIZE<py::list>(node.node_data),
                          node.arity,
                          "Number of keys and entries does not match.");
                representation = absl::StrFormat("OrderedDict([");
                std::string separator;
                auto child_iter = agenda.end() - node.arity;
                for (const py::handle& key : node.node_data) {
                    absl::StrAppendFormat(
                        &representation, "%s(%s, %s)", separator, py::repr(key), *child_iter);
                    ++child_iter;
                    separator = ", ";
                }
                representation += "])";
                break;
            }

            case PyTreeKind::DefaultDict: {
                EXPECT_EQ(
                    GET_SIZE<py::tuple>(node.node_data), 2, "Number of auxiliary data mismatch.");
                py::object default_factory = GET_ITEM_BORROW<py::tuple>(node.node_data, 0);
                auto keys =
                    py::reinterpret_borrow<py::list>(GET_ITEM_BORROW<py::tuple>(node.node_data, 1));
                EXPECT_EQ(GET_SIZE<py::list>(keys),
                          node.arity,
                          "Number of keys and entries does not match.");
                representation = absl::StrFormat("defaultdict(%s, {", py::repr(default_factory));
                std::string separator;
                auto child_iter = agenda.end() - node.arity;
                for (const py::handle& key : keys) {
                    absl::StrAppendFormat(
                        &representation, "%s%s: %s", separator, py::repr(key), *child_iter);
                    ++child_iter;
                    separator = ", ";
                }
                representation += "})";
                break;
            }

            case PyTreeKind::StructSequence: {
                py::object type = node.node_data;
                auto* members = reinterpret_cast<PyTypeObject*>(type.ptr())->tp_members;
                std::string kind = reinterpret_cast<PyTypeObject*>(type.ptr())->tp_name;
                representation = absl::StrFormat("%s(", kind);
                std::string separator;
                auto child_iter = agenda.end() - node.arity;
                for (ssize_t i = 0; i < node.arity; ++i) {
                    absl::StrAppendFormat(
                        &representation,
                        "%s%s=%s",
                        separator,
                        members[i].name,  // NOLINT[cppcoreguidelines-pro-bounds-pointer-arithmetic]
                        *child_iter);
                    ++child_iter;
                    separator = ", ";
                }
                representation += ")";
                break;
            }

            case PyTreeKind::Custom: {
                py::object type = node.custom->type;
                std::string kind = static_cast<std::string>(py::str(py::getattr(type, "__name__")));
                std::string data;
                if (node.node_data) [[likely]] {
                    data = absl::StrFormat("[%s]", py::repr(node.node_data));
                }
                representation =
                    absl::StrFormat("CustomTreeNode(%s%s, [%s])", kind, data, children);
                break;
            }

            default:
                INTERNAL_ERROR();
        }

        agenda.erase(agenda.end() - node.arity, agenda.end());
        agenda.push_back(std::move(representation));
    }

    EXPECT_EQ(agenda.size(), 1, "PyTreeSpec traversal did not yield a singleton.");
    return absl::StrCat(
        "PyTreeSpec(",
        agenda.back(),
        (m_none_is_leaf ? ", NoneIsLeaf" : ""),
        (m_namespace.empty() ? ""
                             : absl::StrFormat(", namespace=%s", py::repr(py::str(m_namespace)))),
        ")");
}

py::object PyTreeSpec::ToPicklable() const {
    py::tuple node_states{GetNumNodes()};
    ssize_t i = 0;
    for (const auto& node : m_traversal) {
        SET_ITEM<py::tuple>(node_states,
                            i++,
                            py::make_tuple(static_cast<ssize_t>(node.kind),
                                           node.arity,
                                           node.node_data ? node.node_data : py::none(),
                                           node.node_entries ? node.node_entries : py::none(),
                                           node.custom != nullptr ? node.custom->type : py::none(),
                                           node.num_leaves,
                                           node.num_nodes,
                                           node.ordered_keys ? node.ordered_keys : py::none()));
    }
    return py::make_tuple(std::move(node_states), py::bool_(m_none_is_leaf), py::str(m_namespace));
}

// NOLINTBEGIN[cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers]
// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::FromPicklableImpl(const py::object& picklable) {
    auto state = py::reinterpret_steal<py::tuple>(picklable);
    if (state.size() != 3) [[unlikely]] {
        throw std::runtime_error("Malformed pickled PyTreeSpec.");
    }
    bool none_is_leaf = false;
    std::string registry_namespace;
    auto out = std::make_unique<PyTreeSpec>();
    out->m_none_is_leaf = none_is_leaf = state[1].cast<bool>();
    out->m_namespace = registry_namespace = state[2].cast<std::string>();
    auto node_states = py::reinterpret_borrow<py::tuple>(state[0]);
    for (const auto& item : node_states) {
        auto t = item.cast<py::tuple>();
        Node& node = out->m_traversal.emplace_back();
        node.kind = static_cast<PyTreeKind>(t[0].cast<ssize_t>());
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
                        node.ordered_keys = t[7].cast<py::list>();
                    } else [[unlikely]] {
                        throw std::runtime_error("Malformed pickled PyTreeSpec.");
                    }
                }
            } else [[unlikely]] {
                throw std::runtime_error("Malformed pickled PyTreeSpec.");
            }
        }
        node.arity = t[1].cast<ssize_t>();
        switch (node.kind) {
            case PyTreeKind::Leaf:
            case PyTreeKind::None:
            case PyTreeKind::Tuple:
            case PyTreeKind::List:
                if (!t[2].is_none()) [[unlikely]] {
                    throw std::runtime_error("Malformed pickled PyTreeSpec.");
                }
                break;

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
                node.node_data = t[2].cast<py::list>();
                break;

            case PyTreeKind::NamedTuple:
            case PyTreeKind::StructSequence:
                node.node_data = t[2].cast<py::type>();
                break;

            case PyTreeKind::DefaultDict:
            case PyTreeKind::Deque:
            case PyTreeKind::Custom:
                node.node_data = t[2];
                break;

            default:
                INTERNAL_ERROR();
        }
        if (node.kind == PyTreeKind::Custom) [[unlikely]] {  // NOLINT
            if (!t[3].is_none()) [[unlikely]] {
                node.node_entries = t[3].cast<py::tuple>();
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
                throw std::runtime_error(absl::StrFormat(
                    "Unknown custom type in pickled PyTreeSpec: %s.", py::repr(t[4])));
            }
        } else if (!t[3].is_none() || !t[4].is_none()) [[unlikely]] {
            throw std::runtime_error("Malformed pickled PyTreeSpec.");
        }
        node.num_leaves = t[5].cast<ssize_t>();
        node.num_nodes = t[6].cast<ssize_t>();
    }
    return out;
}
// NOLINTEND[cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers]

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::FromPicklable(const py::object& picklable) {
    return FromPicklableImpl(picklable);
}

}  // namespace optree
