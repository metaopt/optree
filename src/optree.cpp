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

#include "pybind11/pybind11.h"

#include "include/registry.h"
#include "include/treedef.h"

namespace optree {

namespace py = pybind11;

void BuildOptreeModule(py::module& mod) {
    mod.def("flatten", &PyTreeDef::Flatten, py::arg("tree"),
            py::arg("leaf_predicate") = std::nullopt);
    mod.def("tuple", &PyTreeDef::Tuple);
    mod.def("all_leaves", &PyTreeDef::AllLeaves);

    py::class_<PyTreeDef>(mod, "PyTreeDef")
        .def("unflatten", static_cast<py::object (PyTreeDef::*)(py::iterable leaves) const>(
                              &PyTreeDef::Unflatten))
        .def("flatten_up_to", &PyTreeDef::FlattenUpTo)
        .def("compose", &PyTreeDef::Compose)
        .def("walk", &PyTreeDef::Walk,
             "Walk PyTree, calling f_node(node, node_data) at nodes, and f_leaf at leaves",
             py::arg("f_node"), py::arg("f_leaf"), py::arg("leaves"))
        .def("from_iterable_tree", &PyTreeDef::FromIterableTree)
        .def("children", &PyTreeDef::Children)
        .def_property_readonly("num_leaves", &PyTreeDef::num_leaves)
        .def_property_readonly("num_nodes", &PyTreeDef::num_nodes)
        .def("__repr__", &PyTreeDef::ToString)
        .def(
            "__eq__", [](const PyTreeDef& a, const PyTreeDef& b) { return a == b; },
            py::is_operator())
        .def(
            "__ne__", [](const PyTreeDef& a, const PyTreeDef& b) { return a != b; },
            py::is_operator())
        .def("__hash__", [](const PyTreeDef& t) { return absl::HashOf(t); })
        .def(py::pickle([](const PyTreeDef& t) { return t.ToPickleable(); },
                        [](py::object o) { return PyTreeDef::FromPickleable(o); }));

    mod.def("register_node",
            [](py::object type, py::function to_iterable, py::function from_iterable) {
                return PyTreeTypeRegistry::Register(type, to_iterable, from_iterable);
            });
}

}  // namespace optree

PYBIND11_MODULE(_optree, mod) { optree::BuildOptreeModule(mod); }
