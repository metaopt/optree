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

#include "include/treedef.h"

#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include "include/utils.h"

namespace optree {

namespace py = pybind11;

Py_ssize_t PyTreeDef::num_leaves() const {
    if (traversal.empty()) {
        return 0;
    }
    return traversal.back().num_leaves;
}

Py_ssize_t PyTreeDef::num_nodes() const { return traversal.size(); }

bool PyTreeDef::operator==(const PyTreeDef& other) const {
    if (traversal.size() != other.traversal.size()) {
        return false;
    }
    for (size_t i = 0; i < traversal.size(); ++i) {
        const Node& a = traversal[i];
        const Node& b = other.traversal[i];
        if (a.kind != b.kind || a.arity != b.arity ||
            (a.node_data.ptr() == nullptr) != (b.node_data.ptr() == nullptr) ||
            a.custom != b.custom) {
            return false;
        }
        if (a.node_data && a.node_data.not_equal(b.node_data)) {
            return false;
        }
        // We don't need to test equality of num_leaves and num_nodes since they
        // are derivable from the other node data.
    }
    return true;
}

bool PyTreeDef::operator!=(const PyTreeDef& other) const { return !(*this == other); }

/*static*/ py::object PyTreeDef::MakeNode(const PyTreeDef::Node& node,
                                          absl::Span<py::object> children) {
    if ((Py_ssize_t)children.size() != node.arity) {
        throw std::logic_error("Node arity mismatch.");
    }
    switch (node.kind) {
        case PyTreeKind::Leaf:
            throw std::logic_error("MakeNode not implemented for leaves.");

        case PyTreeKind::None:
            return py::none();

        case PyTreeKind::Tuple:
        case PyTreeKind::NamedTuple: {
            py::tuple tuple(node.arity);
            for (Py_ssize_t i = 0; i < node.arity; ++i) {
                tuple[i] = std::move(children[i]);
            }
            if (node.kind == PyTreeKind::NamedTuple) {
                return node.node_data(*tuple);
            } else {
                return std::move(tuple);
            }
        }

        case PyTreeKind::List: {
            py::list list(node.arity);
            for (Py_ssize_t i = 0; i < node.arity; ++i) {
                list[i] = std::move(children[i]);
            }
            return std::move(list);
        }

        case PyTreeKind::Dict: {
            py::dict dict;
            py::list keys = py::reinterpret_borrow<py::list>(node.node_data);
            for (Py_ssize_t i = 0; i < node.arity; ++i) {
                dict[keys[i]] = std::move(children[i]);
            }
            return std::move(dict);
            break;
        }
        case PyTreeKind::Custom: {
            py::tuple tuple(node.arity);
            for (Py_ssize_t i = 0; i < node.arity; ++i) {
                tuple[i] = std::move(children[i]);
            }
            return node.custom->from_iterable(node.node_data, tuple);
        }
    }
    throw std::logic_error("Unreachable code.");
}

/*static*/ PyTreeKind PyTreeDef::GetKind(const py::handle& obj,
                                         PyTreeTypeRegistry::Registration const** custom) {
    const PyTreeTypeRegistry::Registration* registration =
        PyTreeTypeRegistry::Lookup(obj.get_type());
    if (registration) {
        if (registration->kind == PyTreeKind::Custom) {
            *custom = registration;
        } else {
            *custom = nullptr;
        }
        return registration->kind;
    } else if (py::isinstance<py::tuple>(obj) && py::hasattr(obj, "_fields")) {
        // We can only identify namedtuples heuristically, here by the presence of
        // a _fields attribute.
        return PyTreeKind::NamedTuple;
    } else {
        return PyTreeKind::Leaf;
    }
}

template <typename T>
void PyTreeDef::FlattenIntoImpl(py::handle handle,
                                T& leaves,
                                const std::optional<py::function>& leaf_predicate) {
    Node node;
    Py_ssize_t start_num_nodes = traversal.size();
    Py_ssize_t start_num_leaves = leaves.size();
    if (leaf_predicate && (*leaf_predicate)(handle).cast<bool>()) {
        leaves.push_back(py::reinterpret_borrow<py::object>(handle));
    } else {
        node.kind = GetKind(handle, &node.custom);
        auto recurse = [this, &leaf_predicate, &leaves](py::handle child) {
            FlattenInto(child, leaves, leaf_predicate);
        };
        switch (node.kind) {
            case PyTreeKind::None:
                leaves.push_back(py::reinterpret_borrow<py::object>(handle));
                break;
            case PyTreeKind::Tuple: {
                node.arity = PyTuple_GET_SIZE(handle.ptr());
                for (Py_ssize_t i = 0; i < node.arity; ++i) {
                    recurse(PyTuple_GET_ITEM(handle.ptr(), i));
                }
                break;
            }
            case PyTreeKind::List: {
                node.arity = PyList_GET_SIZE(handle.ptr());
                for (Py_ssize_t i = 0; i < node.arity; ++i) {
                    recurse(PyList_GET_ITEM(handle.ptr(), i));
                }
                break;
            }
            case PyTreeKind::Dict: {
                py::dict dict = py::reinterpret_borrow<py::dict>(handle);
                py::list keys = py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
                try {
                    // Sort directly if possible.
                    if (PyList_Sort(keys.ptr())) {
                        throw py::error_already_set();
                    }
                } catch (py::error_already_set& ex1) {
                    if (ex1.matches(PyExc_TypeError)) {
                        // Found incomparable keys (e.g. `int` vs. `str`, or user-defined types).
                        try {
                            // Sort with `keys.sort(key=lambda o: (o.__class__.__qualname__, o))`.
                            auto sort_key_fn = py::cpp_function([](const py::object& o) {
                                return py::make_tuple(o.get_type().attr("__qualname__"), o);
                            });
                            keys.attr("sort")(py::arg("key") = sort_key_fn);
                        } catch (py::error_already_set& ex2) {
                            if (ex2.matches(PyExc_TypeError)) {
                                // Found incomparable user-defined key types.
                                // The keys remain in the insertion order.
                                PyErr_Clear();
                            } else {
                                throw;
                            }
                        }
                    } else {
                        throw;
                    }
                }
                for (py::handle key : keys) {
                    recurse(dict[key]);
                }
                node.arity = dict.size();
                node.node_data = std::move(keys);
                break;
            }
            case PyTreeKind::Custom: {
                py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(handle));
                if (out.size() != 2) {
                    throw std::runtime_error(
                        "PyTree custom to_iterable function should return a pair.");
                }
                node.node_data = out[1];
                node.arity = 0;
                for (py::handle entry : py::cast<py::iterable>(out[0])) {
                    ++node.arity;
                    recurse(entry);
                }
                break;
            }
            case PyTreeKind::NamedTuple: {
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
                node.arity = tuple.size();
                node.node_data = py::reinterpret_borrow<py::object>(tuple.get_type());
                for (py::handle entry : tuple) {
                    recurse(entry);
                }
                break;
            }
            default:
                DCHECK(node.kind == PyTreeKind::Leaf);
                leaves.push_back(py::reinterpret_borrow<py::object>(handle));
        }
    }
    node.num_nodes = traversal.size() - start_num_nodes + 1;
    node.num_leaves = leaves.size() - start_num_leaves;
    traversal.push_back(std::move(node));
}

void PyTreeDef::FlattenInto(py::handle handle,
                            absl::InlinedVector<py::object, 2>& leaves,
                            std::optional<py::function> leaf_predicate) {
    FlattenIntoImpl(handle, leaves, leaf_predicate);
}

void PyTreeDef::FlattenInto(py::handle handle,
                            std::vector<py::object>& leaves,
                            std::optional<py::function> leaf_predicate) {
    FlattenIntoImpl(handle, leaves, leaf_predicate);
}

/*static*/ std::pair<std::vector<py::object>, std::unique_ptr<PyTreeDef>> PyTreeDef::Flatten(
    py::handle tree, std::optional<py::function> leaf_predicate) {
    std::vector<py::object> leaves;
    auto treedef = std::make_unique<PyTreeDef>();
    treedef->FlattenInto(tree, leaves, leaf_predicate);
    return std::make_pair(std::move(leaves), std::move(treedef));
}

/*static*/ bool PyTreeDef::AllLeaves(const py::iterable& iterable) {
    const PyTreeTypeRegistry::Registration* custom;
    for (const py::handle& h : iterable) {
        if (GetKind(h, &custom) != PyTreeKind::Leaf) return false;
    }
    return true;
}

template <typename T>
py::object PyTreeDef::UnflattenImpl(T leaves) const {
    absl::InlinedVector<py::object, 4> agenda;
    auto it = leaves.begin();
    Py_ssize_t leaf_count = 0;
    for (const Node& node : traversal) {
        if ((Py_ssize_t)agenda.size() < node.arity) {
            throw std::logic_error("Too few elements for TreeDef node.");
        }
        switch (node.kind) {
            case PyTreeKind::None:
            case PyTreeKind::Leaf:
                if (it == leaves.end()) {
                    throw std::invalid_argument(
                        absl::StrFormat("Too few leaves for PyTreeDef; expected %ld, got %ld.",
                                        num_leaves(),
                                        leaf_count));
                }
                agenda.push_back(py::reinterpret_borrow<py::object>(*it));
                ++it;
                ++leaf_count;
                break;

            case PyTreeKind::Tuple:
            case PyTreeKind::NamedTuple:
            case PyTreeKind::List:
            case PyTreeKind::Dict:
            case PyTreeKind::Custom: {
                const Py_ssize_t size = agenda.size();
                absl::Span<py::object> span;
                if (node.arity > 0) {
                    span = absl::Span<py::object>(&agenda[size - node.arity], node.arity);
                }
                py::object o = MakeNode(node, span);
                agenda.resize(size - node.arity);
                agenda.push_back(o);
                break;
            }
        }
    }
    if (it != leaves.end()) {
        throw std::invalid_argument(
            absl::StrFormat("Too many leaves for PyTreeDef; expected %ld.", num_leaves()));
    }
    if (agenda.size() != 1) {
        throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
    }
    return std::move(agenda.back());
}

py::object PyTreeDef::Unflatten(py::iterable leaves) const { return UnflattenImpl(leaves); }

py::object PyTreeDef::Unflatten(absl::Span<const py::object> leaves) const {
    return UnflattenImpl(leaves);
}

py::list PyTreeDef::FlattenUpTo(py::handle full_tree) const {
    py::list leaves(num_leaves());
    std::vector<py::object> agenda;
    agenda.push_back(py::reinterpret_borrow<py::object>(full_tree));
    auto it = traversal.rbegin();
    Py_ssize_t leaf = num_leaves() - 1;
    while (!agenda.empty()) {
        if (it == traversal.rend()) {
            throw std::invalid_argument(absl::StrFormat(
                "Tree structures did not match: %s vs %s.", py::repr(full_tree), ToString()));
        }
        const Node& node = *it;
        py::object object = agenda.back();
        agenda.pop_back();
        ++it;

        switch (node.kind) {
            case PyTreeKind::Leaf:
                if (leaf < 0) {
                    throw std::logic_error("Leaf count mismatch.");
                }
                leaves[leaf] = py::reinterpret_borrow<py::object>(object);
                --leaf;
                break;

            case PyTreeKind::None:
                break;

            case PyTreeKind::Tuple: {
                if (!PyTuple_CheckExact(object.ptr())) {
                    throw std::invalid_argument(
                        absl::StrFormat("Expected tuple, got %s.", py::repr(object)));
                }
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
                if ((Py_ssize_t)tuple.size() != node.arity) {
                    throw std::invalid_argument(
                        absl::StrFormat("Tuple arity mismatch: %ld != %ld; tuple: %s.",
                                        tuple.size(),
                                        node.arity,
                                        py::repr(object)));
                }
                for (py::handle entry : tuple) {
                    agenda.push_back(py::reinterpret_borrow<py::object>(entry));
                }
                break;
            }

            case PyTreeKind::List: {
                if (!PyList_CheckExact(object.ptr())) {
                    throw std::invalid_argument(
                        absl::StrFormat("Expected list, got %s.", py::repr(object)));
                }
                py::list list = py::reinterpret_borrow<py::list>(object);
                if ((Py_ssize_t)list.size() != node.arity) {
                    throw std::invalid_argument(
                        absl::StrFormat("List arity mismatch: %ld != %ld; list: %s.",
                                        list.size(),
                                        node.arity,
                                        py::repr(object)));
                }
                for (py::handle entry : list) {
                    agenda.push_back(py::reinterpret_borrow<py::object>(entry));
                }
                break;
            }

            case PyTreeKind::Dict: {
                if (!PyDict_CheckExact(object.ptr())) {
                    throw std::invalid_argument(
                        absl::StrFormat("Expected dict, got %s.", py::repr(object)));
                }
                py::dict dict = py::reinterpret_borrow<py::dict>(object);
                py::list keys = py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
                if (PyList_Sort(keys.ptr())) {
                    throw std::runtime_error("Dictionary key sort failed.");
                }
                if (keys.not_equal(node.node_data)) {
                    throw std::invalid_argument(
                        absl::StrFormat("Dict key mismatch; expected keys: %s; dict: %s.",
                                        py::repr(node.node_data),
                                        py::repr(object)));
                }
                for (py::handle key : keys) {
                    agenda.push_back(dict[key]);
                }
                break;
            }

            case PyTreeKind::NamedTuple: {
                if (!py::isinstance<py::tuple>(object) || !py::hasattr(object, "_fields")) {
                    throw std::invalid_argument(
                        absl::StrFormat("Expected named tuple, got %s.", py::repr(object)));
                }
                py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
                if ((Py_ssize_t)tuple.size() != node.arity) {
                    throw std::invalid_argument(
                        absl::StrFormat("Named tuple arity mismatch: %ld != %ld; tuple: %s.",
                                        tuple.size(),
                                        node.arity,
                                        py::repr(object)));
                }
                if (tuple.get_type().not_equal(node.node_data)) {
                    throw std::invalid_argument(
                        absl::StrFormat("Named tuple type mismatch: expected type: %s, tuple: %s.",
                                        py::repr(node.node_data),
                                        py::repr(object)));
                }
                for (py::handle entry : tuple) {
                    agenda.push_back(py::reinterpret_borrow<py::object>(entry));
                }
                break;
            }

            case PyTreeKind::Custom: {
                auto* registration = PyTreeTypeRegistry::Lookup(object.get_type());
                if (registration != node.custom) {
                    throw std::invalid_argument(
                        absl::StrFormat("Custom node type mismatch: expected type: %s, value: %s.",
                                        py::repr(node.custom->type),
                                        py::repr(object)));
                }
                py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(object));
                if (out.size() != 2) {
                    throw std::runtime_error(
                        "PyTree custom to_iterable function should return a pair.");
                }
                if (node.node_data.not_equal(out[1])) {
                    throw std::invalid_argument(
                        absl::StrFormat("Mismatch custom node data: %s != %s; value: %s.",
                                        py::repr(node.node_data),
                                        py::repr(out[1]),
                                        py::repr(object)));
                }
                Py_ssize_t arity = 0;
                for (py::handle entry : py::cast<py::iterable>(out[0])) {
                    ++arity;
                    agenda.push_back(py::reinterpret_borrow<py::object>(entry));
                }
                if (arity != node.arity) {
                    throw std::invalid_argument(
                        absl::StrFormat("Custom type arity mismatch: %ld != %ld; value: %s.",
                                        arity,
                                        node.arity,
                                        py::repr(object)));
                }
                break;
            }
        }
    }
    if (it != traversal.rend() || leaf != -1) {
        throw std::invalid_argument(absl::StrFormat(
            "Tree structures did not match: %s vs %s.", py::repr(full_tree), ToString()));
    }
    return leaves;
}

py::object PyTreeDef::Walk(const py::function& f_node,
                           py::handle f_leaf,
                           py::iterable leaves) const {
    std::vector<py::object> agenda;
    auto it = leaves.begin();
    const bool f_leaf_identity = f_leaf.is_none();
    for (const Node& node : traversal) {
        switch (node.kind) {
            case PyTreeKind::Leaf: {
                if (it == leaves.end()) {
                    throw std::invalid_argument("Too few leaves for PyTreeDef");
                }

                py::object leaf = py::reinterpret_borrow<py::object>(*it);
                agenda.push_back(f_leaf_identity ? std::move(leaf) : f_leaf(std::move(leaf)));
                ++it;
                break;
            }

            case PyTreeKind::None:
            case PyTreeKind::Tuple:
            case PyTreeKind::NamedTuple:
            case PyTreeKind::List:
            case PyTreeKind::Dict:
            case PyTreeKind::Custom: {
                if ((Py_ssize_t)agenda.size() < node.arity) {
                    throw std::logic_error("Too few elements for custom type.");
                }
                py::tuple tuple(node.arity);
                for (Py_ssize_t i = node.arity - 1; i >= 0; --i) {
                    tuple[i] = agenda.back();
                    agenda.pop_back();
                }
                agenda.push_back(f_node(tuple, node.node_data ? node.node_data : py::none()));
            }
        }
    }
    if (it != leaves.end()) {
        throw std::invalid_argument("Too many leaves for PyTreeDef.");
    }
    if (agenda.size() != 1) {
        throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
    }
    return std::move(agenda.back());
}

py::object PyTreeDef::FromIterableTreeHelper(
    py::handle subtree, absl::InlinedVector<PyTreeDef::Node, 1>::const_reverse_iterator* it) const {
    if (*it == traversal.rend()) {
        throw std::invalid_argument("Tree structures did not match.");
    }
    const Node& node = **it;
    ++*it;
    if (node.kind == PyTreeKind::Leaf) {
        return py::reinterpret_borrow<py::object>(subtree);
    }
    py::iterable iterable = py::reinterpret_borrow<py::iterable>(subtree);
    std::vector<py::object> ys;
    ys.reserve(node.arity);
    for (py::handle x : iterable) {
        ys.push_back(py::reinterpret_borrow<py::object>(x));
    }
    if ((Py_ssize_t)ys.size() != node.arity) {
        throw std::invalid_argument("Arity mismatch between trees.");
    }
    for (Py_ssize_t j = node.arity - 1; j >= 0; --j) {
        ys[j] = FromIterableTreeHelper(ys[j], it);
    }

    return MakeNode(node, absl::MakeSpan(ys));
}

py::object PyTreeDef::FromIterableTree(py::handle subtrees) const {
    auto it = traversal.rbegin();
    py::object out = FromIterableTreeHelper(subtrees, &it);
    if (it != traversal.rend()) {
        throw std::invalid_argument("Tree structures did not match.");
    }
    return out;
}

std::unique_ptr<PyTreeDef> PyTreeDef::Compose(const PyTreeDef& inner_treedef) const {
    auto out = std::make_unique<PyTreeDef>();
    for (const Node& n : traversal) {
        if (n.kind == PyTreeKind::Leaf) {
            absl::c_copy(inner_treedef.traversal, std::back_inserter(out->traversal));
        } else {
            out->traversal.push_back(n);
        }
    }
    const auto& root = traversal.back();
    const auto& inner_root = inner_treedef.traversal.back();
    // TODO(tomhennigan): This should update all nodes in the traversal.
    auto& out_root = out->traversal.back();
    out_root.num_nodes =
        (root.num_nodes - root.num_leaves) + (inner_root.num_nodes * root.num_leaves);
    out_root.num_leaves *= inner_root.num_leaves;
    return out;
}

/*static*/ std::unique_ptr<PyTreeDef> PyTreeDef::Tuple(const std::vector<PyTreeDef>& treedefs) {
    auto out = std::make_unique<PyTreeDef>();
    Py_ssize_t num_leaves = 0;
    for (const PyTreeDef& treedef : treedefs) {
        absl::c_copy(treedef.traversal, std::back_inserter(out->traversal));
        num_leaves += treedef.num_leaves();
    }
    Node node;
    node.kind = PyTreeKind::Tuple;
    node.arity = treedefs.size();
    node.num_leaves = num_leaves;
    node.num_nodes = out->traversal.size() + 1;
    out->traversal.push_back(node);
    return out;
}

std::vector<std::unique_ptr<PyTreeDef>> PyTreeDef::Children() const {
    std::vector<std::unique_ptr<PyTreeDef>> children;
    if (traversal.empty()) {
        return children;
    }
    Node const& root = traversal.back();
    children.resize(root.arity);
    Py_ssize_t pos = traversal.size() - 1;
    for (Py_ssize_t i = root.arity - 1; i >= 0; --i) {
        children[i] = std::make_unique<PyTreeDef>();
        const Node& node = traversal.at(pos - 1);
        if (pos < node.num_nodes) {
            throw std::logic_error("Children() walked off start of array.");
        }
        std::copy(traversal.begin() + pos - node.num_nodes,
                  traversal.begin() + pos,
                  std::back_inserter(children[i]->traversal));
        pos -= node.num_nodes;
    }
    if (pos != 0) {
        throw std::logic_error("pos != 0 at end of PyTreeDef::Children.");
    }
    return children;
}

std::string PyTreeDef::ToString() const {
    std::vector<std::string> agenda;
    for (const Node& node : traversal) {
        if ((Py_ssize_t)agenda.size() < node.arity) {
            throw std::logic_error("Too few elements for container.");
        }

        std::string children = absl::StrJoin(agenda.end() - node.arity, agenda.end(), ", ");
        std::string representation;
        switch (node.kind) {
            case PyTreeKind::Leaf:
                agenda.push_back("*");
                continue;
            case PyTreeKind::None:
                representation = "None";
                break;
            case PyTreeKind::Tuple:
                // Tuples with only one element must have a trailing comma.
                if (node.arity == 1) children += ",";
                representation = absl::StrCat("(", children, ")");
                break;
            case PyTreeKind::List:
                representation = absl::StrCat("[", children, "]");
                break;
            case PyTreeKind::Dict: {
                if ((Py_ssize_t)py::len(node.node_data) != node.arity) {
                    throw std::logic_error("Number of keys and entries does not match.");
                }
                representation = "{";
                std::string separator;
                auto child_iter = agenda.end() - node.arity;
                for (const py::handle& key : node.node_data) {
                    absl::StrAppendFormat(
                        &representation, "%s%s: %s", separator, py::repr(key), *child_iter);
                    child_iter++;
                    separator = ", ";
                }
                representation += "}";
                break;
            }

            case PyTreeKind::NamedTuple:
            case PyTreeKind::Custom: {
                std::string kind;
                std::string data;
                if (node.kind == PyTreeKind::NamedTuple) {
                    kind = "namedtuple";
                    if (node.node_data) {
                        // Node data for named tuples is the type.
                        data = absl::StrFormat("[%s]",
                                               py::str(py::getattr(node.node_data, "__name__")));
                    }
                } else {
                    kind = static_cast<std::string>(
                        py::str(py::getattr(node.custom->type, "__name__")));
                    if (node.node_data) {
                        data = absl::StrFormat("[%s]", py::str(node.node_data));
                    }
                }

                representation = absl::StrFormat("CustomNode(%s%s, [%s])", kind, data, children);
                break;
            }
        }
        agenda.erase(agenda.end() - node.arity, agenda.end());
        agenda.push_back(std::move(representation));
    }
    if (agenda.size() != 1) {
        throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
    }
    return absl::StrCat("PyTreeDef(", agenda.back(), ")");
}

py::object PyTreeDef::ToPickleable() const {
    py::list result;
    for (const auto& node : traversal) {
        result.append(py::make_tuple(static_cast<int>(node.kind),
                                     node.arity,
                                     node.node_data ? node.node_data : py::none(),
                                     node.custom != nullptr ? node.custom->type : py::none(),
                                     node.num_leaves,
                                     node.num_nodes));
    }
    return std::move(result);
}

PyTreeDef PyTreeDef::FromPickleable(py::object pickleable) {
    PyTreeDef tree;
    for (const auto& item : pickleable.cast<py::list>()) {
        auto t = item.cast<py::tuple>();
        if (t.size() != 6) {
            throw std::runtime_error("Malformed pickled PyTreeDef.");
        }
        Node& node = tree.traversal.emplace_back();
        node.kind = static_cast<PyTreeKind>(t[0].cast<int>());
        node.arity = t[1].cast<int>();
        switch (node.kind) {
            case PyTreeKind::NamedTuple:
                node.node_data = t[2].cast<py::type>();
                break;
            case PyTreeKind::Dict:
                node.node_data = t[2].cast<py::list>();
                break;
            case PyTreeKind::Custom:
                node.node_data = t[2];
                break;
            default:
                if (!t[2].is_none()) {
                    throw std::runtime_error("Malformed pickled PyTreeDef.");
                }
                break;
        }
        if (node.kind == PyTreeKind::Custom) {
            node.custom = t[3].is_none() ? nullptr : PyTreeTypeRegistry::Lookup(t[3]);
            if (node.custom == nullptr) {
                throw std::runtime_error(absl::StrCat("Unknown custom type in pickled PyTreeDef: ",
                                                      static_cast<std::string>(py::repr(t[3])),
                                                      "."));
            }
        } else {
            if (!t[3].is_none()) {
                throw std::runtime_error("Malformed pickled PyTreeDef.");
            }
        }
        node.num_leaves = t[4].cast<int>();
        node.num_nodes = t[5].cast<int>();
    }
    return tree;
}

}  // namespace optree
