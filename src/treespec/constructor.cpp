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

#include <algorithm>  // std::copy
#include <iterator>   // std::back_inserter
#include <memory>     // std::unique_ptr, std::make_unique
#include <sstream>    // std::ostringstream
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <utility>    // std::move
#include <vector>     // std::vector

#include "optree/optree.h"

namespace optree {

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::MakeLeaf(const bool& none_is_leaf,
                                                            const std::string& /*unused*/) {
    auto out = std::make_unique<PyTreeSpec>();
    out->m_traversal.emplace_back(Node{
        .kind = PyTreeKind::Leaf,
        .arity = 0,
        .num_leaves = 1,
        .num_nodes = 1,
    });
    out->m_none_is_leaf = none_is_leaf;
    out->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*out);
    return out;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::MakeNone(const bool& none_is_leaf,
                                                            const std::string& /*unused*/) {
    if (none_is_leaf) [[unlikely]] {
        return MakeLeaf(none_is_leaf);
    }
    auto out = std::make_unique<PyTreeSpec>();
    out->m_traversal.emplace_back(Node{
        .kind = PyTreeKind::None,
        .arity = 0,
        .num_leaves = 0,
        .num_nodes = 1,
    });
    out->m_none_is_leaf = none_is_leaf;
    out->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*out);
    return out;
}

template <bool NoneIsLeaf>
// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::MakeFromCollectionImpl(
    const py::handle& handle,
    std::string registry_namespace) {
    auto children = reserved_vector<py::object>(4);
    auto treespecs = reserved_vector<PyTreeSpec>(4);

    Node node;
    node.kind = PyTreeTypeRegistry::GetKind<NoneIsLeaf>(handle, node.custom, registry_namespace);

    const auto verify_children = [&handle, &node, &registry_namespace](
                                     const std::vector<py::object>& children,
                                     std::vector<PyTreeSpec>& treespecs) -> void {
        for (const py::object& child : children) {
            if (!py::isinstance<PyTreeSpec>(child)) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Expected a(n) " << NodeKindToString(node) << " of PyTreeSpec(s), got "
                    << PyRepr(handle) << ".";
                throw py::value_error(oss.str());
            }
            treespecs.emplace_back(thread_safe_cast<PyTreeSpec>(child));
        }

        std::string common_registry_namespace{};
        for (const PyTreeSpec& treespec : treespecs) {
            PYTREESPEC_SANITY_CHECK(treespec);
            if (treespec.m_none_is_leaf != NoneIsLeaf) [[unlikely]] {
                throw py::value_error(NoneIsLeaf
                                          ? "Expected treespec(s) with `none_is_leaf=True`."
                                          : "Expected treespec(s) with `none_is_leaf=False`.");
            }
            if (!treespec.m_namespace.empty()) [[unlikely]] {
                if (common_registry_namespace.empty()) [[likely]] {
                    common_registry_namespace = treespec.m_namespace;
                } else if (common_registry_namespace != treespec.m_namespace) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "Expected treespecs with the same namespace, got "
                        << PyRepr(common_registry_namespace) << " vs. "
                        << PyRepr(treespec.m_namespace) << ".";
                    throw py::value_error(oss.str());
                }
            }
        }
        if (!common_registry_namespace.empty()) [[likely]] {
            if (registry_namespace.empty()) [[likely]] {
                registry_namespace = common_registry_namespace;
            } else if (registry_namespace != common_registry_namespace) [[unlikely]] {
                std::ostringstream oss{};
                oss << "Expected treespec(s) with namespace " << PyRepr(registry_namespace)
                    << ", got " << PyRepr(common_registry_namespace) << ".";
                throw py::value_error(oss.str());
            }
        } else if (node.kind != PyTreeKind::Custom) [[likely]] {
            registry_namespace = "";
        }
    };

    switch (node.kind) {
        case PyTreeKind::Leaf: {
            node.arity = 0;
            PyErr_WarnEx(PyExc_UserWarning,
                         "PyTreeSpec::MakeFromCollection() is called on a leaf.",
                         /*stack_level=*/2);
            break;
        }

        case PyTreeKind::None: {
            node.arity = 0;
            if constexpr (!NoneIsLeaf) {
                break;
            }
            INTERNAL_ERROR(
                "NoneIsLeaf is true, but PyTreeTypeRegistry::GetKind() returned "
                "`PyTreeKind::None`.");
        }

        case PyTreeKind::Tuple: {
            node.arity = TupleGetSize(handle);
            for (ssize_t i = 0; i < node.arity; ++i) {
                children.emplace_back(TupleGetItem(handle, i));
            }
            verify_children(children, treespecs);
            break;
        }

        case PyTreeKind::List: {
            {
                const scoped_critical_section cs{handle};
                node.arity = ListGetSize(handle);
                for (ssize_t i = 0; i < node.arity; ++i) {
                    children.emplace_back(ListGetItem(handle, i));
                }
            }
            verify_children(children, treespecs);
            break;
        }

        case PyTreeKind::Dict:
        case PyTreeKind::OrderedDict:
        case PyTreeKind::DefaultDict: {
            py::list keys;
            {
                const scoped_critical_section cs{handle};
                const auto dict = py::reinterpret_borrow<py::dict>(handle);
                node.arity = DictGetSize(dict);
                keys = DictKeys(dict);
                if (node.kind != PyTreeKind::OrderedDict) [[likely]] {
                    node.original_keys = py::getattr(keys, Py_Get_ID(copy))();
                    if (!IsDictInsertionOrdered(registry_namespace)) [[likely]] {
                        TotalOrderSort(keys);
                    }
                }
                for (const py::handle& key : keys) {
                    children.emplace_back(DictGetItem(dict, key));
                }
            }
            verify_children(children, treespecs);
            if (node.kind == PyTreeKind::DefaultDict) [[unlikely]] {
                const scoped_critical_section cs{handle};
                node.node_data = py::make_tuple(py::getattr(handle, Py_Get_ID(default_factory)),
                                                std::move(keys));
            } else [[likely]] {
                node.node_data = std::move(keys);
            }
            break;
        }

        case PyTreeKind::NamedTuple:
        case PyTreeKind::StructSequence: {
            const auto tuple = py::reinterpret_borrow<py::tuple>(handle);
            node.arity = TupleGetSize(tuple);
            node.node_data = py::type::of(tuple);
            for (ssize_t i = 0; i < node.arity; ++i) {
                children.emplace_back(TupleGetItem(tuple, i));
            }
            verify_children(children, treespecs);
            break;
        }

        case PyTreeKind::Deque: {
            const auto list = thread_safe_cast<py::list>(handle);
            node.arity = ListGetSize(list);
            node.node_data =
                EVALUATE_WITH_LOCK_HELD(py::getattr(handle, Py_Get_ID(maxlen)), handle);
            for (ssize_t i = 0; i < node.arity; ++i) {
                children.emplace_back(ListGetItem(list, i));
            }
            verify_children(children, treespecs);
            break;
        }

        case PyTreeKind::Custom: {
            const py::tuple out = EVALUATE_WITH_LOCK_HELD2(
                thread_safe_cast<py::tuple>(node.custom->flatten_func(handle)),
                handle,
                node.custom->flatten_func);
            const ssize_t num_out = TupleGetSize(out);
            if (num_out != 2 && num_out != 3) [[unlikely]] {
                std::ostringstream oss{};
                oss << "PyTree custom flatten function for type " << PyRepr(node.custom->type)
                    << " should return a 2- or 3-tuple, got " << num_out << ".";
                throw std::runtime_error(oss.str());
            }
            node.arity = 0;
            node.node_data = TupleGetItem(out, 1);
            {
                auto children_iterable = thread_safe_cast<py::iterable>(TupleGetItem(out, 0));
                const scoped_critical_section cs{children_iterable};
                for (const py::handle& child : children_iterable) {
                    ++node.arity;
                    children.emplace_back(py::reinterpret_borrow<py::object>(child));
                }
            }
            verify_children(children, treespecs);
            if (num_out == 3) [[likely]] {
                const py::object node_entries = TupleGetItem(out, 2);
                if (!node_entries.is_none()) [[likely]] {
                    node.node_entries = thread_safe_cast<py::tuple>(node_entries);
                    const ssize_t num_entries = TupleGetSize(node.node_entries);
                    if (num_entries != node.arity) [[unlikely]] {
                        std::ostringstream oss{};
                        oss << "PyTree custom flatten function for type "
                            << PyRepr(node.custom->type)
                            << " returned inconsistent number of children (" << node.arity
                            << ") and number of entries (" << num_entries << ").";
                        throw std::runtime_error(oss.str());
                    }
                }
            }
            break;
        }

        default:
            INTERNAL_ERROR();
    }

    auto out = std::make_unique<PyTreeSpec>();
    ssize_t num_leaves = ((node.kind == PyTreeKind::Leaf) ? 1 : 0);
    for (const PyTreeSpec& treespec : treespecs) {
        std::copy(treespec.m_traversal.cbegin(),
                  treespec.m_traversal.cend(),
                  std::back_inserter(out->m_traversal));
        num_leaves += treespec.GetNumLeaves();
    }
    node.num_leaves = num_leaves;
    node.num_nodes = py::ssize_t_cast(out->m_traversal.size()) + 1;
    out->m_traversal.emplace_back(std::move(node));
    out->m_none_is_leaf = NoneIsLeaf;
    out->m_namespace = registry_namespace;
    out->m_traversal.shrink_to_fit();
    PYTREESPEC_SANITY_CHECK(*out);
    return out;
}

/*static*/ std::unique_ptr<PyTreeSpec> PyTreeSpec::MakeFromCollection(
    const py::object& object,
    const bool& none_is_leaf,
    const std::string& registry_namespace) {
    if (none_is_leaf) [[unlikely]] {
        return MakeFromCollectionImpl<NONE_IS_LEAF>(object, registry_namespace);
    } else [[likely]] {
        return MakeFromCollectionImpl<NONE_IS_NODE>(object, registry_namespace);
    }
}

}  // namespace optree
