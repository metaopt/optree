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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/registry.h"
#include "include/treespec.h"

namespace optree {

void BuildModule(py::module& mod) {  // NOLINT
    mod.def(
        "flatten", &PyTreeSpec::Flatten, py::arg("tree"), py::arg("leaf_predicate") = std::nullopt);
    mod.def("all_leaves", &PyTreeSpec::AllLeaves, py::arg("iterable"));
    mod.def("tuple", &PyTreeSpec::Tuple, py::arg("treespecs"));

    py::class_<PyTreeSpec>(mod, "PyTreeSpec")
        .def_property_readonly("num_leaves", &PyTreeSpec::num_leaves)
        .def_property_readonly("num_nodes", &PyTreeSpec::num_nodes)
        .def("unflatten",
             static_cast<py::object (PyTreeSpec::*)(py::iterable leaves) const>(
                 &PyTreeSpec::Unflatten),
             py::arg("leaves"))
        .def("flatten_up_to", &PyTreeSpec::FlattenUpTo, py::arg("full_trees"))
        .def("compose", &PyTreeSpec::Compose, py::arg("inner_treespec"))
        .def("walk",
             &PyTreeSpec::Walk,
             "Walk PyTree, calling f_node(node, node_data) at nodes, and f_leaf(leaf) at leaves",
             py::arg("f_node"),
             py::arg("f_leaf"),
             py::arg("leaves"))
        .def("children", &PyTreeSpec::Children)
        .def("__str__", &PyTreeSpec::ToString)
        .def("__repr__", &PyTreeSpec::ToString)
        .def(
            "__eq__",
            [](const PyTreeSpec& a, const PyTreeSpec& b) { return a == b; },
            py::is_operator(),
            py::arg("other"))
        .def(
            "__ne__",
            [](const PyTreeSpec& a, const PyTreeSpec& b) { return a != b; },
            py::is_operator(),
            py::arg("other"))
        .def("__hash__", [](const PyTreeSpec& t) { return absl::HashOf(t); })
        .def(py::pickle([](const PyTreeSpec& t) { return t.ToPicklable(); },
                        [](py::object o) { return PyTreeSpec::FromPicklable(o); }));

    mod.def(
        "register_node",
        [](py::object type, py::function to_iterable, py::function from_iterable) {
            return PyTreeTypeRegistry::Register(type, to_iterable, from_iterable);
        },
        py::arg("type"),
        py::arg("to_iterable"),
        py::arg("from_iterable"));
}

}  // namespace optree

PYBIND11_MODULE(_C, mod) { optree::BuildModule(mod); }
