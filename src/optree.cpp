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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/exceptions.h"
#include "include/registry.h"
#include "include/treespec.h"

namespace optree {

void BuildModule(py::module& mod) {  // NOLINT[runtime/references]
    mod.doc() = "Optimized PyTree Utilities.";
    py::register_local_exception<InternalError>(mod, "InternalError", PyExc_SystemError);
    mod.attr("MAX_RECURSION_DEPTH") = py::ssize_t_cast(MAX_RECURSION_DEPTH);
    mod.attr("Py_TPFLAGS_BASETYPE") = py::ssize_t_cast(Py_TPFLAGS_BASETYPE);
#ifdef _GLIBCXX_USE_CXX11_ABI
    // NOLINTNEXTLINE[modernize-use-bool-literals]
    mod.attr("GLIBCXX_USE_CXX11_ABI") = static_cast<bool>(_GLIBCXX_USE_CXX11_ABI);
#else
    mod.attr("GLIBCXX_USE_CXX11_ABI") = false;
#endif

    mod.def("register_node",
            &PyTreeTypeRegistry::Register,
            "Register a Python type. Extends the set of types that are considered internal nodes "
            "in pytrees.",
            py::arg("cls"),
            py::arg("flatten_func"),
            py::arg("unflatten_func"),
            py::arg("namespace") = "")
        .def("flatten",
             &PyTreeSpec::Flatten,
             "Flattens a pytree.",
             py::arg("tree"),
             py::arg("leaf_predicate") = std::nullopt,
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("flatten_with_path",
             &PyTreeSpec::FlattenWithPath,
             "Flatten a pytree and additionally record the paths.",
             py::arg("tree"),
             py::arg("leaf_predicate") = std::nullopt,
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("is_leaf",
             &PyTreeSpec::ObjectIsLeaf,
             "Test whether the given object is a leaf node.",
             py::arg("obj"),
             py::arg("leaf_predicate") = std::nullopt,
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("all_leaves",
             &PyTreeSpec::AllLeaves,
             "Test whether all elements in the given iterable are all leaves.",
             py::arg("iterable"),
             py::arg("leaf_predicate") = std::nullopt,
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("leaf",
             &PyTreeSpec::Leaf,
             "Make a treespec representing a leaf node.",
             py::arg("none_is_leaf") = false)
        .def("none",
             &PyTreeSpec::None,
             "Make a treespec representing a ``None`` node.",
             py::arg("none_is_leaf") = false)
        .def("tuple",
             &PyTreeSpec::Tuple,
             "Make a tuple treespec from a list of child treespecs.",
             py::arg("treespecs"),
             py::arg("none_is_leaf") = false)
        .def("is_namedtuple",
             &IsNamedTuple,
             "Return whether the object is an instance of namedtuple or a subclass of namedtuple.",
             py::arg("obj"))
        .def("is_namedtuple_class",
             &IsNamedTupleClass,
             "Return whether the class is a subclass of namedtuple.",
             py::arg("cls"))
        .def("namedtuple_fields",
             &NamedTupleGetFields,
             "Return the field names of a namedtuple.",
             py::arg("obj"))
        .def("is_structseq",
             &IsStructSequence,
             "Return whether the object is an instance of PyStructSequence or a class of "
             "PyStructSequence.",
             py::arg("obj"))
        .def("is_structseq_class",
             &IsStructSequenceClass,
             "Return whether the object is a class of PyStructSequence.",
             py::arg("cls"))
        .def("structseq_fields",
             &StructSequenceGetFields,
             "Return the field names of a PyStructSequence.",
             py::arg("obj"));

    auto PyTreeKindTypeObject =
        py::enum_<PyTreeKind>(mod, "PyTreeKind", "The kind of a pytree node.")
            .value("CUSTOM", PyTreeKind::Custom, "A custom type.")
            .value("LEAF", PyTreeKind::Leaf, "An opaque leaf node.")
            .value("NONE", PyTreeKind::None, "None.")
            .value("TUPLE", PyTreeKind::Tuple, "A tuple.")
            .value("LIST", PyTreeKind::List, "A list.")
            .value("DICT", PyTreeKind::Dict, "A dict.")
            .value("NAMEDTUPLE", PyTreeKind::NamedTuple, "A collections.namedtuple.")
            .value("ORDEREDDICT", PyTreeKind::OrderedDict, "A collections.OrderedDict.")
            .value("DEFAULTDICT", PyTreeKind::DefaultDict, "A collections.defaultdict.")
            .value("DEQUE", PyTreeKind::Deque, "A collections.deque.")
            .value("STRUCTSEQUENCE", PyTreeKind::StructSequence, "A PyStructSequence.");
    reinterpret_cast<PyTypeObject*>(PyTreeKindTypeObject.ptr())->tp_name = "optree.PyTreeKind";
    py::setattr(PyTreeKindTypeObject.ptr(), "__module__", py::str("optree"));

    auto PyTreeSpecTypeObject =
        py::class_<PyTreeSpec>(mod, "PyTreeSpec", "Representing the structure of the pytree.");
    reinterpret_cast<PyTypeObject*>(PyTreeSpecTypeObject.ptr())->tp_name = "optree.PyTreeSpec";
    py::setattr(PyTreeSpecTypeObject.ptr(), "__module__", py::str("optree"));

    PyTreeSpecTypeObject
        .def_property_readonly(
            "num_leaves", &PyTreeSpec::GetNumLeaves, "Number of leaves in the tree.")
        .def_property_readonly("num_nodes",
                               &PyTreeSpec::GetNumNodes,
                               "Number of nodes in the tree. "
                               "Note that a leaf is also a node but has no children.")
        .def_property_readonly("num_children",
                               &PyTreeSpec::GetNumChildren,
                               "Number of children in the current node. "
                               "Note that a leaf is also a node but has no children.")
        .def_property_readonly(
            "none_is_leaf",
            &PyTreeSpec::GetNoneIsLeaf,
            "Whether to treat None as a leaf. "
            "If false, None is a non-leaf node with arity 0. "
            "Thus None is contained in the treespec rather than in the leaves list.")
        .def_property_readonly(
            "namespace",
            &PyTreeSpec::GetNamespace,
            "The registry namespace used to resolve the custom pytree node types.")
        .def_property_readonly(
            "type",
            &PyTreeSpec::GetType,
            "The type of the current node. Return None if the current node is a leaf.")
        .def_property_readonly("kind", &PyTreeSpec::GetPyTreeKind, "The kind of the current node.")
        .def("unflatten",
             &PyTreeSpec::Unflatten,
             "Reconstruct a pytree from the leaves.",
             py::arg("leaves"))
        .def("flatten_up_to",
             &PyTreeSpec::FlattenUpTo,
             "Flatten the subtrees in ``full_tree`` up to the structure of this treespec "
             "and return a list of subtrees.",
             py::arg("full_tree"))
        .def("broadcast_to_common_suffix",
             &PyTreeSpec::BroadcastToCommonSuffix,
             "Broadcast to the common suffix of this treespec and other treespec.",
             py::arg("other"))
        .def("compose",
             &PyTreeSpec::Compose,
             "Compose two treespecs. Constructs the inner treespec as a subtree at each leaf node.",
             py::arg("inner_treespec"))
        .def("walk",
             &PyTreeSpec::Walk,
             "Walk over the pytree structure, calling ``f_node(children, node_data)`` at nodes, "
             "and ``f_leaf(leaf)`` at leaves.",
             py::arg("f_node"),
             py::arg("f_leaf"),
             py::arg("leaves"))
        .def("is_prefix",
             &PyTreeSpec::IsPrefix,
             "Test whether this treespec is a prefix of the given treespec.",
             py::arg("other"),
             py::arg("strict") = false)
        .def("is_suffix",
             &PyTreeSpec::IsSuffix,
             "Test whether this treespec is a suffix of the given treespec.",
             py::arg("other"),
             py::arg("strict") = false)
        .def("paths", &PyTreeSpec::Paths, "Return a list of paths to the leaves of the treespec.")
        .def("entries", &PyTreeSpec::Entries, "Return a list of one-level entries to the children.")
        .def("entry", &PyTreeSpec::Entry, "Return the entry at the given index.", py::arg("index"))
        .def("children", &PyTreeSpec::Children, "Return a list of treespecs for the children.")
        .def("child",
             &PyTreeSpec::Child,
             "Return the treespec for the child at the given index.",
             py::arg("index"))
        .def("is_leaf",
             &PyTreeSpec::IsLeaf,
             "Test whether the current node is a leaf.",
             py::arg("strict") = true)
        .def("__repr__", &PyTreeSpec::ToString, "Return a string representation of the treespec.")
        .def(
            "__eq__",
            [](const PyTreeSpec& a, const PyTreeSpec& b) { return a == b; },
            "Test for equality to another object.",
            py::is_operator(),
            py::arg("other"))
        .def(
            "__ne__",
            [](const PyTreeSpec& a, const PyTreeSpec& b) { return a != b; },
            "Test for inequality to another object.",
            py::is_operator(),
            py::arg("other"))
        .def(
            "__lt__",
            [](const PyTreeSpec& a, const PyTreeSpec& b) { return a < b; },
            "Test for this treespec is a strict prefix of another object.",
            py::is_operator(),
            py::arg("other"))
        .def(
            "__le__",
            [](const PyTreeSpec& a, const PyTreeSpec& b) { return a <= b; },
            "Test for this treespec is a prefix of another object.",
            py::is_operator(),
            py::arg("other"))
        .def(
            "__gt__",
            [](const PyTreeSpec& a, const PyTreeSpec& b) { return a > b; },
            "Test for this treespec is a strict suffix of another object.",
            py::is_operator(),
            py::arg("other"))
        .def(
            "__ge__",
            [](const PyTreeSpec& a, const PyTreeSpec& b) { return a >= b; },
            "Test for this treespec is a suffix of another object.",
            py::is_operator(),
            py::arg("other"))
        .def("__hash__", &PyTreeSpec::HashValue, "Return the hash of the treespec.")
        .def("__len__", &PyTreeSpec::GetNumLeaves, "Number of leaves in the tree.")
        .def(py::pickle([](const PyTreeSpec& t) { return t.ToPicklable(); },
                        [](const py::object& o) { return PyTreeSpec::FromPicklable(o); }),
             "Serialization support for PyTreeSpec.",
             py::arg("state"));

#ifdef Py_TPFLAGS_IMMUTABLETYPE
    reinterpret_cast<PyTypeObject*>(PyTreeSpecTypeObject.ptr())->tp_flags |=
        Py_TPFLAGS_IMMUTABLETYPE;
    reinterpret_cast<PyTypeObject*>(PyTreeKindTypeObject.ptr())->tp_flags |=
        Py_TPFLAGS_IMMUTABLETYPE;
#endif
    if (PyType_Ready(reinterpret_cast<PyTypeObject*>(PyTreeSpecTypeObject.ptr())) < 0)
        [[unlikely]] {
        INTERNAL_ERROR("`PyType_Ready(&PyTreeSpec_Type)` failed.");
    }
    if (PyType_Ready(reinterpret_cast<PyTypeObject*>(PyTreeKindTypeObject.ptr())) < 0)
        [[unlikely]] {
        INTERNAL_ERROR("`PyType_Ready(&PyTreeKind_Type)` failed.");
    }
}

}  // namespace optree

// NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-type-vararg]
PYBIND11_MODULE(_C, mod) { optree::BuildModule(mod); }
