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

ssize_t PyTreeSpec::num_leaves() const {
    if (m_traversal.empty()) [[unlikely]] {
        return 0;
    }
    return m_traversal.back().num_leaves;
}

ssize_t PyTreeSpec::num_nodes() const { return (ssize_t)m_traversal.size(); }

bool PyTreeSpec::get_none_is_leaf() const { return m_none_is_leaf; }

std::string PyTreeSpec::get_namespace() const { return m_namespace; }

bool PyTreeSpec::operator==(const PyTreeSpec& other) const {
    if (m_traversal.size() != other.m_traversal.size() || m_none_is_leaf != other.m_none_is_leaf)
        [[likely]] {
        return false;
    }
    if (!m_namespace.empty() && !other.m_namespace.empty() && m_namespace != other.m_namespace)
        [[likely]] {
        return false;
    }

    auto b = other.m_traversal.begin();
    for (auto a = m_traversal.begin(); a != m_traversal.end(); ++a, ++b) {
        if (a->kind != b->kind || a->arity != b->arity ||
            (a->node_data.ptr() == nullptr) != (b->node_data.ptr() == nullptr) ||
            a->custom != b->custom) [[likely]] {
            return false;
        }
        if (a->node_data && a->node_data.not_equal(b->node_data)) [[likely]] {
            return false;
        }
        // We don't need to test equality of num_leaves and num_nodes since they are derivable from
        // the other node data. The num_leaves and num_nodes fields may also change for custom node
        // types.
    }
    return true;
}

bool PyTreeSpec::operator!=(const PyTreeSpec& other) const { return !(*this == other); }

std::unique_ptr<PyTreeSpec> PyTreeSpec::Compose(const PyTreeSpec& inner_treespec) const {
    if (inner_treespec.m_none_is_leaf != m_none_is_leaf) [[unlikely]] {  // NOLINT
        throw std::invalid_argument("PyTreeSpecs must have the same none_is_leaf value.");
    }
    if (!m_namespace.empty() && !inner_treespec.m_namespace.empty() &&
        m_namespace != inner_treespec.m_namespace) [[unlikely]] {  // NOLINT
        throw std::invalid_argument(
            absl::StrFormat("PyTreeSpecs must have the same namespace. Got %s vs. %s.",
                            py::repr(py::str(m_namespace)),
                            py::repr(py::str(inner_treespec.m_namespace))));
    }
    auto outer_treespec = std::make_unique<PyTreeSpec>();
    outer_treespec->m_none_is_leaf = m_none_is_leaf;
    if (inner_treespec.m_namespace.empty()) [[likely]] {
        outer_treespec->m_namespace = m_namespace;
    } else [[unlikely]] {  // NOLINT
        outer_treespec->m_namespace = inner_treespec.m_namespace;
    }
    for (const Node& node : m_traversal) {
        if (node.kind == PyTreeKind::LEAF) [[likely]] {
            absl::c_copy(inner_treespec.m_traversal,
                         std::back_inserter(outer_treespec->m_traversal));
        } else [[unlikely]] {  // NOLINT
            outer_treespec->m_traversal.emplace_back(node);
        }
    }
    const auto& root = m_traversal.back();
    const auto& inner_root = inner_treespec.m_traversal.back();
    auto& outer_root = outer_treespec->m_traversal.back();
    outer_root.num_nodes =
        (root.num_nodes - root.num_leaves) + (inner_root.num_nodes * root.num_leaves);
    outer_root.num_leaves = root.num_leaves * inner_root.num_leaves;
    return outer_treespec;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::Tuple(const std::vector<PyTreeSpec>& treespecs,
                                                         const bool& none_is_leaf) {
    std::string registry_namespace;
    for (const PyTreeSpec& treespec : treespecs) {
        if (treespec.m_none_is_leaf != none_is_leaf) [[unlikely]] {
            throw std::invalid_argument(absl::StrFormat(
                "Expected TreeSpecs with `node_is_leaf=%s`.", (none_is_leaf ? "True" : "False")));
        }
        if (!treespec.m_namespace.empty()) [[unlikely]] {  // NOLINT
            if (registry_namespace.empty()) [[likely]] {
                registry_namespace = treespec.m_namespace;
            } else if (registry_namespace != treespec.m_namespace) [[unlikely]] {
                throw std::invalid_argument(
                    absl::StrFormat("Expected TreeSpecs with the same namespace. Got %s vs. %s.",
                                    py::repr(py::str(registry_namespace)),
                                    py::repr(py::str(treespec.m_namespace))));
            }
        }
    }

    auto out = std::make_unique<PyTreeSpec>();
    ssize_t num_leaves = 0;
    for (const PyTreeSpec& treespec : treespecs) {
        absl::c_copy(treespec.m_traversal, std::back_inserter(out->m_traversal));
        num_leaves += treespec.num_leaves();
    }
    Node node;
    node.kind = PyTreeKind::TUPLE;
    node.arity = (ssize_t)treespecs.size();
    node.num_leaves = num_leaves;
    node.num_nodes = (ssize_t)out->m_traversal.size() + 1;
    out->m_traversal.emplace_back(std::move(node));
    out->m_none_is_leaf = none_is_leaf;
    out->m_namespace = registry_namespace;
    return out;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::Leaf(const bool& none_is_leaf) {
    auto out = std::make_unique<PyTreeSpec>();
    Node node;
    node.kind = PyTreeKind::LEAF;
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
    node.kind = PyTreeKind::NONE;
    node.arity = 0;
    node.num_leaves = 0;
    node.num_nodes = 1;
    out->m_traversal.emplace_back(std::move(node));
    out->m_none_is_leaf = none_is_leaf;
    return out;
}

std::vector<std::unique_ptr<PyTreeSpec>> PyTreeSpec::Children() const {
    std::vector<std::unique_ptr<PyTreeSpec>> children;
    if (m_traversal.empty()) [[likely]] {
        return children;
    }
    const Node& root = m_traversal.back();
    children.resize(root.arity);
    ssize_t pos = (ssize_t)m_traversal.size() - 1;
    for (ssize_t i = root.arity - 1; i >= 0; --i) {
        children[i] = std::make_unique<PyTreeSpec>();
        children[i]->m_none_is_leaf = m_none_is_leaf;
        children[i]->m_namespace = m_namespace;
        const Node& node = m_traversal.at(pos - 1);
        if (pos < node.num_nodes) [[unlikely]] {
            throw std::logic_error("PyTreeSpec::Children() walked off start of array.");
        }
        std::copy(m_traversal.begin() + pos - node.num_nodes,
                  m_traversal.begin() + pos,
                  std::back_inserter(children[i]->m_traversal));
        pos -= node.num_nodes;
    }
    if (pos != 0) [[unlikely]] {
        throw std::logic_error("pos != 0 at end of PyTreeSpec::Children().");
    }
    return children;
}

/*static*/ py::object PyTreeSpec::MakeNode(const PyTreeSpec::Node& node,
                                           const absl::Span<py::object>& children) {
    if ((ssize_t)children.size() != node.arity) [[unlikely]] {
        throw std::logic_error("Node arity did not match.");
    }
    switch (node.kind) {
        case PyTreeKind::LEAF:
            throw std::logic_error("MakeNode not implemented for leaves.");

        case PyTreeKind::NONE:
            return py::none();

        case PyTreeKind::TUPLE:
        case PyTreeKind::NAMED_TUPLE: {
            py::tuple tuple{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                SET_ITEM<py::tuple>(tuple, i, children[i]);
            }
            if (node.kind == PyTreeKind::NAMED_TUPLE) [[unlikely]] {
                return node.node_data(*tuple);
            }
            return std::move(tuple);
        }

        case PyTreeKind::LIST:
        case PyTreeKind::DEQUE: {
            py::list list{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                SET_ITEM<py::list>(list, i, children[i]);
            }
            if (node.kind == PyTreeKind::DEQUE) [[unlikely]] {
                return PyDequeTypeObject(list, py::arg("maxlen") = node.node_data);
            }
            return std::move(list);
        }

        case PyTreeKind::DICT: {
            py::dict dict;
            py::list keys = py::reinterpret_borrow<py::list>(node.node_data);
            for (ssize_t i = 0; i < node.arity; ++i) {
                dict[GET_ITEM_HANDLE<py::list>(keys, i)] = std::move(children[i]);
            }
            return std::move(dict);
        }

        case PyTreeKind::ORDERED_DICT: {
            py::list items{node.arity};
            py::list keys = py::reinterpret_borrow<py::list>(node.node_data);
            for (ssize_t i = 0; i < node.arity; ++i) {
                SET_ITEM<py::list>(
                    items,
                    i,
                    py::make_tuple(GET_ITEM_HANDLE<py::list>(keys, i), std::move(children[i])));
            }
            return PyOrderedDictTypeObject(items);
        }

        case PyTreeKind::DEFAULT_DICT: {
            py::dict dict;
            py::object default_factory = GET_ITEM_BORROW<py::tuple>(node.node_data, 0);
            py::list keys = GET_ITEM_BORROW<py::tuple>(node.node_data, 1);
            for (ssize_t i = 0; i < node.arity; ++i) {
                dict[GET_ITEM_HANDLE<py::list>(keys, i)] = std::move(children[i]);
            }
            return PyDefaultDictTypeObject(default_factory, dict);
        }

        case PyTreeKind::CUSTOM: {
            py::tuple tuple{node.arity};
            for (ssize_t i = 0; i < node.arity; ++i) {
                SET_ITEM<py::tuple>(tuple, i, children[i]);
            }
            return node.custom->from_iterable(node.node_data, tuple);
        }

        default:
            throw std::logic_error("Unreachable code.");
    }
}

template <bool NoneIsLeaf>
/*static*/ PyTreeKind PyTreeSpec::GetKind(const py::handle& handle,
                                          PyTreeTypeRegistry::Registration const** custom,
                                          const std::string& registry_namespace) {
    const PyTreeTypeRegistry::Registration* registration =
        PyTreeTypeRegistry::Lookup<NoneIsLeaf>(handle.get_type(), registry_namespace);
    if (registration) [[likely]] {  // NOLINT
        if (registration->kind == PyTreeKind::CUSTOM) [[unlikely]] {
            *custom = registration;
        } else [[likely]] {  // NOLINT
            *custom = nullptr;
        }
        return registration->kind;
    }
    *custom = nullptr;
    if (IsNamedTuple(handle)) [[unlikely]] {
        return PyTreeKind::NAMED_TUPLE;
    }
    return PyTreeKind::LEAF;
}

template PyTreeKind PyTreeSpec::GetKind<NONE_IS_NODE>(const py::handle&,
                                                      PyTreeTypeRegistry::Registration const**,
                                                      const std::string&);
template PyTreeKind PyTreeSpec::GetKind<NONE_IS_LEAF>(const py::handle&,
                                                      PyTreeTypeRegistry::Registration const**,
                                                      const std::string&);

std::string PyTreeSpec::ToString() const {
    std::vector<std::string> agenda;
    for (const Node& node : m_traversal) {
        if ((ssize_t)agenda.size() < node.arity) [[unlikely]] {
            throw std::logic_error("Too few elements for container.");
        }

        std::string children = absl::StrJoin(agenda.end() - node.arity, agenda.end(), ", ");
        std::string representation;
        switch (node.kind) {
            case PyTreeKind::LEAF:
                agenda.emplace_back("*");
                continue;

            case PyTreeKind::NONE:
                representation = "None";
                break;

            case PyTreeKind::TUPLE:
                // Tuples with only one element must have a trailing comma.
                if (node.arity == 1) [[unlikely]] {
                    children += ",";
                }
                representation = absl::StrCat("(", children, ")");
                break;

            case PyTreeKind::LIST:
            case PyTreeKind::DEQUE:
                representation = absl::StrCat("[", children, "]");
                if (node.kind == PyTreeKind::DEQUE) [[unlikely]] {  // NOLINT
                    if (node.node_data.is_none()) [[likely]] {
                        representation = absl::StrCat("deque(", representation, ")");
                    } else [[unlikely]] {  // NOLINT
                        representation = absl::StrFormat(
                            "deque(%s, maxlen=%s)", representation, py::str(node.node_data));
                    }
                }
                break;

            case PyTreeKind::DICT: {
                if ((ssize_t)py::len(node.node_data) != node.arity) [[unlikely]] {
                    throw std::logic_error("Number of keys and entries does not match.");
                }
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

            case PyTreeKind::NAMED_TUPLE: {
                py::object type = node.node_data;
                py::tuple fields = py::reinterpret_borrow<py::tuple>(py::getattr(type, "_fields"));
                if ((ssize_t)py::len(fields) != node.arity) [[unlikely]] {
                    throw std::logic_error("Number of fields and entries does not match.");
                }
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

            case PyTreeKind::ORDERED_DICT: {
                if ((ssize_t)py::len(node.node_data) != node.arity) [[unlikely]] {
                    throw std::logic_error("Number of keys and entries does not match.");
                }
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

            case PyTreeKind::DEFAULT_DICT: {
                if ((ssize_t)GET_SIZE<py::tuple>(node.node_data) != 2) [[unlikely]] {
                    throw std::logic_error("Number of auxiliary data mismatch.");
                }
                py::object factory = GET_ITEM_BORROW<py::tuple>(node.node_data, 0);
                py::list keys =
                    py::reinterpret_borrow<py::list>(GET_ITEM_BORROW<py::tuple>(node.node_data, 1));
                if ((ssize_t)GET_SIZE<py::list>(keys) != node.arity) [[unlikely]] {
                    throw std::logic_error("Number of keys and entries does not match.");
                }
                representation = absl::StrFormat("defaultdict(%s, {", py::repr(factory));
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

            case PyTreeKind::CUSTOM: {
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
                throw std::logic_error("Unreachable code.");
        }

        agenda.erase(agenda.end() - node.arity, agenda.end());
        agenda.push_back(std::move(representation));
    }
    if (agenda.size() != 1) [[unlikely]] {
        throw std::logic_error("PyTreeSpec traversal did not yield a singleton.");
    }
    return absl::StrCat(
        "PyTreeSpec(",
        agenda.back(),
        (m_none_is_leaf ? ", NoneIsLeaf" : ""),
        (m_namespace.empty() ? ""
                             : absl::StrFormat(", namespace=%s", py::repr(py::str(m_namespace)))),
        ")");
}

py::object PyTreeSpec::ToPicklable() const {
    py::tuple node_states{num_nodes()};
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
                                           node.num_nodes));
    }
    return py::make_tuple(std::move(node_states), py::bool_(m_none_is_leaf), py::str(m_namespace));
}

/*static*/ PyTreeSpec PyTreeSpec::FromPicklableImpl(const py::object& picklable) {
    py::tuple state = py::reinterpret_steal<py::tuple>(picklable);
    if (state.size() != 3) [[unlikely]] {
        throw std::runtime_error("Malformed pickled PyTreeSpec.");
    }
    bool none_is_leaf;
    std::string registry_namespace;
    PyTreeSpec treespec;
    treespec.m_none_is_leaf = none_is_leaf = state[1].cast<bool>();
    treespec.m_namespace = registry_namespace = state[2].cast<std::string>();
    py::tuple node_states = py::reinterpret_borrow<py::tuple>(state[0]);
    for (const auto& item : node_states) {
        auto t = item.cast<py::tuple>();
        if (t.size() != 7) [[unlikely]] {
            throw std::runtime_error("Malformed pickled PyTreeSpec.");
        }
        Node& node = treespec.m_traversal.emplace_back();
        node.kind = static_cast<PyTreeKind>(t[0].cast<ssize_t>());
        node.arity = t[1].cast<ssize_t>();
        switch (node.kind) {
            case PyTreeKind::NAMED_TUPLE:
                node.node_data = t[2].cast<py::type>();
                break;
            case PyTreeKind::DICT:
            case PyTreeKind::ORDERED_DICT:
                node.node_data = t[2].cast<py::list>();
                break;
            case PyTreeKind::DEFAULT_DICT:
            case PyTreeKind::DEQUE:
            case PyTreeKind::CUSTOM:
                node.node_data = t[2];
                break;
            default:
                if (!t[2].is_none()) [[unlikely]] {
                    throw std::runtime_error("Malformed pickled PyTreeSpec.");
                }
                break;
        }
        if (node.kind == PyTreeKind::CUSTOM) [[unlikely]] {  // NOLINT
            if (!t[3].is_none()) [[unlikely]] {
                node.node_entries = t[3].cast<py::tuple>();
            }
            if (t[4].is_none()) [[unlikely]] {
                node.custom = nullptr;
            } else [[likely]] {  // NOLINT
                if (none_is_leaf) [[unlikely]] {
                    node.custom =
                        PyTreeTypeRegistry::Lookup<NONE_IS_LEAF>(t[4], registry_namespace);
                } else [[likely]] {  // NOLINT
                    node.custom =
                        PyTreeTypeRegistry::Lookup<NONE_IS_NODE>(t[4], registry_namespace);
                }
            }
            if (node.custom == nullptr) [[unlikely]] {
                throw std::runtime_error(absl::StrCat("Unknown custom type in pickled PyTreeSpec: ",
                                                      static_cast<std::string>(py::repr(t[4])),
                                                      "."));
            }
        } else if (!t[3].is_none() || !t[4].is_none()) [[unlikely]] {
            throw std::runtime_error("Malformed pickled PyTreeSpec.");
        }
        node.num_leaves = t[5].cast<ssize_t>();
        node.num_nodes = t[6].cast<ssize_t>();
    }
    return treespec;
}

/*static*/ PyTreeSpec PyTreeSpec::FromPicklable(const py::object& picklable) {
    return FromPicklableImpl(picklable);
}

}  // namespace optree
