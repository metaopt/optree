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

#include <optional>   // std::optional
#include <sstream>    // std::ostringstream
#include <stdexcept>  // std::runtime_error
#include <utility>    // std::move

#include "optree/optree.h"

namespace optree {

template <bool NoneIsLeaf>
// NOLINTNEXTLINE[readability-function-cognitive-complexity]
py::object PyTreeIter::NextImpl() {
    while (!m_agenda.empty()) [[likely]] {
        auto [object, depth] = m_agenda.back();
        m_agenda.pop_back();

        if (depth > MAX_RECURSION_DEPTH) [[unlikely]] {
            PyErr_SetString(PyExc_RecursionError,
                            "Maximum recursion depth exceeded during flattening the tree.");
            throw py::error_already_set();
        }

        if (m_leaf_predicate &&
            EVALUATE_WITH_LOCK_HELD2(thread_safe_cast<bool>((*m_leaf_predicate)(object)),
                                     object,
                                     *m_leaf_predicate)) [[unlikely]] {
            return object;
        }

        PyTreeTypeRegistry::RegistrationPtr custom{nullptr};
        const PyTreeKind kind =
            PyTreeTypeRegistry::GetKind<NoneIsLeaf>(object, custom, m_namespace);

        ++depth;
        switch (kind) {
            case PyTreeKind::Leaf: {
                return object;
            }

            case PyTreeKind::None: {
                if constexpr (!NoneIsLeaf) {
                    break;
                }
                INTERNAL_ERROR(
                    "NoneIsLeaf is true, but PyTreeTypeRegistry::GetKind() returned "
                    "`PyTreeKind::None`.");
            }

            case PyTreeKind::Tuple: {
                const ssize_t arity = TupleGetSize(object);
                for (ssize_t i = arity - 1; i >= 0; --i) {
                    m_agenda.emplace_back(TupleGetItem(object, i), depth);
                }
                break;
            }

            case PyTreeKind::List: {
                const scoped_critical_section cs{object};
                const ssize_t arity = ListGetSize(object);
                for (ssize_t i = arity - 1; i >= 0; --i) {
                    m_agenda.emplace_back(ListGetItem(object, i), depth);
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                const scoped_critical_section cs{object};
                const auto dict = py::reinterpret_borrow<py::dict>(object);
                py::list keys = DictKeys(dict);
                if (kind != PyTreeKind::OrderedDict && !m_is_dict_insertion_ordered) [[likely]] {
                    TotalOrderSort(keys);
                }
                if (PyList_Reverse(keys.ptr()) < 0) [[unlikely]] {
                    throw py::error_already_set();
                }
                for (const py::handle &key : keys) {
                    m_agenda.emplace_back(DictGetItem(dict, key), depth);
                }
                break;
            }

            case PyTreeKind::NamedTuple:
            case PyTreeKind::StructSequence: {
                const auto tuple = py::reinterpret_borrow<py::tuple>(object);
                const ssize_t arity = TupleGetSize(tuple);
                for (ssize_t i = arity - 1; i >= 0; --i) {
                    m_agenda.emplace_back(TupleGetItem(tuple, i), depth);
                }
                break;
            }

            case PyTreeKind::Deque: {
                const auto list = thread_safe_cast<py::list>(object);
                const ssize_t arity = ListGetSize(list);
                for (ssize_t i = arity - 1; i >= 0; --i) {
                    m_agenda.emplace_back(ListGetItem(list, i), depth);
                }
                break;
            }

            case PyTreeKind::Custom: {
                const py::tuple out = EVALUATE_WITH_LOCK_HELD2(
                    thread_safe_cast<py::tuple>(custom->flatten_func(object)),
                    object,
                    custom->flatten_func);
                const ssize_t num_out = TupleGetSize(out);
                if (num_out != 2 && num_out != 3) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "PyTree custom flatten function for type " << PyRepr(custom->type)
                        << " should return a 2- or 3-tuple, got " << num_out << ".";
                    throw std::runtime_error(oss.str());
                }
                auto children = thread_safe_cast<py::tuple>(TupleGetItem(out, 0));
                const ssize_t arity = TupleGetSize(children);
                if (num_out == 3) [[likely]] {
                    const py::object node_entries = TupleGetItem(out, 2);
                    if (!node_entries.is_none()) [[likely]] {
                        const ssize_t num_entries =
                            TupleGetSize(thread_safe_cast<py::tuple>(node_entries));
                        if (num_entries != arity) [[unlikely]] {
                            std::ostringstream oss{};
                            oss << "PyTree custom flatten function for type "
                                << PyRepr(custom->type)
                                << " returned inconsistent number of children (" << arity
                                << ") and number of entries (" << num_entries << ").";
                            throw std::runtime_error(oss.str());
                        }
                    }
                }
                for (ssize_t i = arity - 1; i >= 0; --i) {
                    m_agenda.emplace_back(TupleGetItem(children, i), depth);
                }
                break;
            }

            default:
                INTERNAL_ERROR();
        }
    }

    throw py::stop_iteration();
}

py::object PyTreeIter::Next() {
#ifdef Py_GIL_DISABLED
    const scoped_lock_guard lock{m_mutex};
#endif

    if (m_none_is_leaf) [[unlikely]] {
        return NextImpl<NONE_IS_LEAF>();
    } else [[likely]] {
        return NextImpl<NONE_IS_NODE>();
    }
}

template <bool PassRawNode>
// NOLINTNEXTLINE[readability-function-cognitive-complexity]
py::object PyTreeSpec::WalkImpl(const py::iterable &leaves,
                                const std::optional<py::function> &f_node,
                                const std::optional<py::function> &f_leaf) const {
    PYTREESPEC_SANITY_CHECK(*this);

    const scoped_critical_section cs{leaves};
    auto agenda = reserved_vector<py::object>(4);
    auto it = leaves.begin();
    for (const Node &node : m_traversal) {
        switch (node.kind) {
            case PyTreeKind::Leaf: {
                if (it == leaves.end()) [[unlikely]] {
                    throw py::value_error("Too few leaves for PyTreeSpec.");
                }

                const auto leaf = py::reinterpret_borrow<py::object>(*it);
                agenda.emplace_back(
                    f_leaf ? EVALUATE_WITH_LOCK_HELD2((*f_leaf)(leaf), leaf, *f_leaf) : leaf);
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
                const ssize_t size = py::ssize_t_cast(agenda.size());
                EXPECT_GE(size, node.arity, "Too few elements for custom type.");

                if (PassRawNode && f_node) [[likely]] {
                    const py::tuple children{node.arity};
                    for (ssize_t i = node.arity - 1; i >= 0; --i) {
                        TupleSetItem(children, i, agenda.back());
                        agenda.pop_back();
                    }

                    const py::object &node_type = GetType(node);
                    {
                        const scoped_critical_section cs2{node_type};
                        agenda.emplace_back(EVALUATE_WITH_LOCK_HELD2(
                            (*f_node)(node_type,
                                      node.node_data ? node.node_data : py::none(),
                                      children),
                            node.node_data,
                            *f_node));
                    }
                } else [[unlikely]] {
                    const py::object out =
                        MakeNode(node,
                                 node.arity > 0 ? &agenda[size - node.arity] : nullptr,
                                 node.arity);
                    agenda.resize(size - node.arity);
                    agenda.emplace_back(
                        f_node ? EVALUATE_WITH_LOCK_HELD2((*f_node)(out), out, *f_node) : out);
                }
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
    return agenda.back();
}

py::object PyTreeSpec::Traverse(const py::iterable &leaves,
                                const std::optional<py::function> &f_node,
                                const std::optional<py::function> &f_leaf) const {
    return WalkImpl</*PassRawNode=*/false>(leaves, f_node, f_leaf);
}

py::object PyTreeSpec::Walk(const py::iterable &leaves,
                            const std::optional<py::function> &f_node,
                            const std::optional<py::function> &f_leaf) const {
    return WalkImpl</*PassRawNode=*/true>(leaves, f_node, f_leaf);
}

}  // namespace optree
