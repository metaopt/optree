/*
Copyright 2022-2024 MetaOPT Team. All Rights Reserved.

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

#include <sstream>    // std::ostringstream
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <utility>    // std::move

#include "include/exceptions.h"
#include "include/registry.h"
#include "include/treespec.h"
#include "include/utils.h"

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

        if (m_leaf_predicate && py::cast<bool>((*m_leaf_predicate)(object))) [[unlikely]] {
            return object;
        }

        PyTreeTypeRegistry::RegistrationPtr custom{nullptr};
        PyTreeKind kind = PyTreeTypeRegistry::GetKind<NoneIsLeaf>(object, custom, m_namespace);

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
                ssize_t arity = GET_SIZE<py::tuple>(object);
                for (ssize_t i = arity - 1; i >= 0; --i) {
                    m_agenda.emplace_back(GET_ITEM_BORROW<py::tuple>(object, i), depth);
                }
                break;
            }

            case PyTreeKind::List: {
                ssize_t arity = GET_SIZE<py::list>(object);
                for (ssize_t i = arity - 1; i >= 0; --i) {
                    m_agenda.emplace_back(GET_ITEM_BORROW<py::list>(object, i), depth);
                }
                break;
            }

            case PyTreeKind::Dict:
            case PyTreeKind::OrderedDict:
            case PyTreeKind::DefaultDict: {
                auto dict = py::reinterpret_borrow<py::dict>(object);
                py::list keys = DictKeys(dict);
                if (kind != PyTreeKind::OrderedDict && !m_is_dict_insertion_ordered) [[likely]] {
                    TotalOrderSort(keys);
                }
                if (PyList_Reverse(keys.ptr()) < 0) [[unlikely]] {
                    throw py::error_already_set();
                }
                for (const py::handle& key : keys) {
                    m_agenda.emplace_back(dict[key], depth);
                }
                break;
            }

            case PyTreeKind::NamedTuple:
            case PyTreeKind::StructSequence: {
                auto tuple = py::reinterpret_borrow<py::tuple>(object);
                ssize_t arity = GET_SIZE<py::tuple>(tuple);
                for (ssize_t i = arity - 1; i >= 0; --i) {
                    m_agenda.emplace_back(GET_ITEM_BORROW<py::tuple>(tuple, i), depth);
                }
                break;
            }

            case PyTreeKind::Deque: {
                auto list = py::cast<py::list>(object);
                ssize_t arity = GET_SIZE<py::list>(list);
                for (ssize_t i = arity - 1; i >= 0; --i) {
                    m_agenda.emplace_back(GET_ITEM_BORROW<py::list>(list, i), depth);
                }
                break;
            }

            case PyTreeKind::Custom: {
                py::tuple out = py::cast<py::tuple>(custom->flatten_func(object));
                const ssize_t num_out = GET_SIZE<py::tuple>(out);
                if (num_out != 2 && num_out != 3) [[unlikely]] {
                    std::ostringstream oss{};
                    oss << "PyTree custom flatten function for type " << PyRepr(custom->type)
                        << " should return a 2- or 3-tuple, got " << num_out << ".";
                    throw std::runtime_error(oss.str());
                }
                auto children = py::cast<py::tuple>(GET_ITEM_BORROW<py::tuple>(out, ssize_t(0)));
                ssize_t arity = GET_SIZE<py::tuple>(children);
                if (num_out == 3) [[likely]] {
                    py::object node_entries = GET_ITEM_BORROW<py::tuple>(out, ssize_t(2));
                    if (!node_entries.is_none()) [[likely]] {
                        const ssize_t num_entries =
                            GET_SIZE<py::tuple>(py::cast<py::tuple>(std::move(node_entries)));
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
                    m_agenda.emplace_back(GET_ITEM_BORROW<py::tuple>(children, i), depth);
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
    if (m_none_is_leaf) [[unlikely]] {
        return NextImpl<NONE_IS_LEAF>();
    }
    return NextImpl<NONE_IS_NODE>();
}

py::object PyTreeSpec::Walk(const py::function& f_node,
                            const py::object& f_leaf,
                            const py::iterable& leaves) const {
    auto agenda = reserved_vector<py::object>(4);
    auto it = leaves.begin();
    const bool f_leaf_identity = f_leaf.is_none();
    for (const Node& node : m_traversal) {
        switch (node.kind) {
            case PyTreeKind::Leaf: {
                if (it == leaves.end()) [[unlikely]] {
                    throw py::value_error("Too few leaves for PyTreeSpec.");
                }

                auto leaf = py::reinterpret_borrow<py::object>(*it);
                agenda.emplace_back(f_leaf_identity ? std::move(leaf) : f_leaf(std::move(leaf)));
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
                EXPECT_GE(py::ssize_t_cast(agenda.size()),
                          node.arity,
                          "Too few elements for custom type.");
                py::tuple tuple{node.arity};
                for (ssize_t i = node.arity - 1; i >= 0; --i) {
                    SET_ITEM<py::tuple>(tuple, i, agenda.back());
                    agenda.pop_back();
                }
                agenda.emplace_back(
                    f_node(std::move(tuple), node.node_data ? node.node_data : py::none()));
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
    return std::move(agenda.back());
}

}  // namespace optree
