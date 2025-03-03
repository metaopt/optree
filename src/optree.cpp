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

#include "optree/optree.h"

#include <functional>  // std::{not_,}equal_to, std::less{,_equal}, std::greater{,_equal}
#include <memory>      // std::unique_ptr
#include <optional>    // std::optional, std::nullopt
#include <string>      // std::string

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace optree {

py::module_ GetCxxModule(const std::optional<py::module_>& module) {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::module_> storage;
    return storage
        .call_once_and_store_result([&module]() -> py::module_ {
            EXPECT_TRUE(module, "The module must be provided.");
            return *module;
        })
        .get_stored();
}

void BuildModule(py::module_& mod) {  // NOLINT[runtime/references]
    GetCxxModule(mod);

    mod.doc() = "Optimized PyTree Utilities.";
    py::register_local_exception<InternalError>(mod, "InternalError", PyExc_SystemError);
    mod.attr("MAX_RECURSION_DEPTH") = py::int_(MAX_RECURSION_DEPTH);
    mod.attr("Py_TPFLAGS_BASETYPE") = py::int_(Py_TPFLAGS_BASETYPE);
#ifdef _GLIBCXX_USE_CXX11_ABI
    // NOLINTNEXTLINE[modernize-use-bool-literals]
    mod.attr("GLIBCXX_USE_CXX11_ABI") = py::bool_(static_cast<bool>(_GLIBCXX_USE_CXX11_ABI));
#else
    mod.attr("GLIBCXX_USE_CXX11_ABI") = py::bool_(false);
#endif

    mod.def("register_node",
            &PyTreeTypeRegistry::Register,
            "Register a Python type. Extends the set of types that are considered internal nodes "
            "in pytrees.",
            py::arg("cls"),
            py::pos_only(),
            py::arg("flatten_func"),
            py::arg("unflatten_func"),
            py::arg("path_entry_type"),
            py::arg("namespace") = "")
        .def("unregister_node",
             &PyTreeTypeRegistry::Unregister,
             "Unregister a Python type.",
             py::arg("cls"),
             py::pos_only(),
             py::arg("namespace") = "")
        .def("is_dict_insertion_ordered",
             &PyTreeSpec::IsDictInsertionOrdered,
             "Return whether need to preserve the dict insertion order during flattening.",
             py::arg("namespace") = "",
             py::arg("inherit_global_namespace") = true)
        .def("set_dict_insertion_ordered",
             &PyTreeSpec::SetDictInsertionOrdered,
             "Set whether need to preserve the dict insertion order during flattening.",
             py::arg("mode"),
             py::pos_only(),
             py::arg("namespace") = "")
        .def("flatten",
             &PyTreeSpec::Flatten,
             "Flattens a pytree.",
             py::arg("tree"),
             py::pos_only(),
             py::arg("leaf_predicate") = std::nullopt,
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("flatten_with_path",
             &PyTreeSpec::FlattenWithPath,
             "Flatten a pytree and additionally record the paths.",
             py::arg("tree"),
             py::pos_only(),
             py::arg("leaf_predicate") = std::nullopt,
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("is_leaf",
             &IsLeaf,
             "Test whether the given object is a leaf node.",
             py::arg("obj"),
             py::pos_only(),
             py::arg("leaf_predicate") = std::nullopt,
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("all_leaves",
             &AllLeaves,
             "Test whether all elements in the given iterable are all leaves.",
             py::arg("iterable"),
             py::pos_only(),
             py::arg("leaf_predicate") = std::nullopt,
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("make_leaf",
             &PyTreeSpec::MakeLeaf,
             "Make a treespec representing a leaf node.",
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("make_none",
             &PyTreeSpec::MakeNone,
             "Make a treespec representing a ``None`` node.",
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("make_from_collection",
             &PyTreeSpec::MakeFromCollection,
             "Make a treespec from a collection of child treespecs.",
             py::arg("collection"),
             py::pos_only(),
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def("is_namedtuple",
             &IsNamedTuple,
             "Return whether the object is an instance of namedtuple or a subclass of namedtuple.",
             py::arg("obj"),
             py::pos_only())
        .def("is_namedtuple_instance",
             &IsNamedTupleInstance,
             "Return whether the object is an instance of namedtuple.",
             py::arg("obj"),
             py::pos_only())
        .def("is_namedtuple_class",
             &IsNamedTupleClass,
             "Return whether the class is a subclass of namedtuple.",
             py::arg("cls"),
             py::pos_only())
        .def("namedtuple_fields",
             &NamedTupleGetFields,
             "Return the field names of a namedtuple.",
             py::arg("obj"),
             py::pos_only())
        .def("is_structseq",
             &IsStructSequence,
             "Return whether the object is an instance of PyStructSequence or a class of "
             "PyStructSequence.",
             py::arg("obj"),
             py::pos_only())
        .def("is_structseq_instance",
             &IsStructSequenceInstance,
             "Return whether the object is an instance of PyStructSequence.",
             py::arg("obj"),
             py::pos_only())
        .def("is_structseq_class",
             &IsStructSequenceClass,
             "Return whether the object is a class of PyStructSequence.",
             py::arg("cls"),
             py::pos_only())
        .def("structseq_fields",
             &StructSequenceGetFields,
             "Return the field names of a PyStructSequence.",
             py::arg("obj"),
             py::pos_only());

#if PYBIND11_VERSION_HEX >= 0x020E00F0  // pybind11 2.14.0
#define def_method_pos_only(...) def(__VA_ARGS__ __VA_OPT__(, ) py::pos_only())
#else
#define def_method_pos_only(...) def(__VA_ARGS__)
#endif

    auto PyTreeKindTypeObject =
        py::enum_<PyTreeKind>(mod, "PyTreeKind", "The kind of a pytree node.", py::module_local())
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
    auto* const PyTreeKind_Type = reinterpret_cast<PyTypeObject*>(PyTreeKindTypeObject.ptr());
    PyTreeKind_Type->tp_name = "optree.PyTreeKind";
    py::setattr(PyTreeKindTypeObject.ptr(), Py_Get_ID(__module__), Py_Get_ID(optree));

    auto PyTreeSpecTypeObject = py::class_<PyTreeSpec>(
        mod,
        "PyTreeSpec",
        "Representing the structure of the pytree.",
        // NOLINTBEGIN[readability-function-cognitive-complexity,cppcoreguidelines-avoid-do-while]
        py::custom_type_setup([](PyHeapTypeObject* heap_type) -> void {
            auto* const type = &heap_type->ht_type;
            type->tp_flags |= Py_TPFLAGS_HAVE_GC;
            type->tp_traverse = &PyTreeSpec::PyTpTraverse;
        }),
        // NOLINTEND[readability-function-cognitive-complexity,cppcoreguidelines-avoid-do-while]
        py::module_local());
    auto* const PyTreeSpec_Type = reinterpret_cast<PyTypeObject*>(PyTreeSpecTypeObject.ptr());
    PyTreeSpec_Type->tp_name = "optree.PyTreeSpec";
    py::setattr(PyTreeSpecTypeObject.ptr(), Py_Get_ID(__module__), Py_Get_ID(optree));

    PyTreeSpecTypeObject
        .def("unflatten",
             &PyTreeSpec::Unflatten,
             "Reconstruct a pytree from the leaves.",
             py::arg("leaves"),
             py::pos_only())
        .def("flatten_up_to",
             &PyTreeSpec::FlattenUpTo,
             "Flatten the subtrees in ``full_tree`` up to the structure of this treespec "
             "and return a list of subtrees.",
             py::arg("full_tree"),
             py::pos_only())
        .def("broadcast_to_common_suffix",
             &PyTreeSpec::BroadcastToCommonSuffix,
             "Broadcast to the common suffix of this treespec and other treespec.",
             py::arg("other"),
             py::pos_only())
        .def("transform",
             &PyTreeSpec::Transform,
             "Transform the pytree structure by applying ``f_node(nodespec)`` at nodes and "
             "``f_leaf(leafspec)`` at leaves.",
             py::pos_only(),
             py::arg("f_node") = std::nullopt,
             py::arg("f_leaf") = std::nullopt)
        .def("compose",
             &PyTreeSpec::Compose,
             "Compose two treespecs. Constructs the inner treespec as a subtree at each leaf node.",
             py::arg("inner_treespec"),
             py::pos_only())
        .def("traverse",
             &PyTreeSpec::Traverse,
             "Walk over the pytree structure, calling ``f_leaf(leaf)`` at leaves, "
             "and ``f_node(node)`` at reconstructed non-leaf nodes.",
             py::arg("leaves"),
             py::pos_only(),
             py::arg("f_node") = std::nullopt,
             py::arg("f_leaf") = std::nullopt)
        .def("walk",
             &PyTreeSpec::Walk,
             "Walk over the pytree structure, calling ``f_leaf(leaf)`` at leaves, "
             "and ``f_node(node_type, node_data, children)`` at non-leaf nodes.",
             py::arg("leaves"),
             py::pos_only(),
             py::arg("f_node") = std::nullopt,
             py::arg("f_leaf") = std::nullopt)
        .def_method_pos_only("paths",
                             &PyTreeSpec::Paths,
                             "Return a list of paths to the leaves of the treespec.")
        .def_method_pos_only("accessors",
                             &PyTreeSpec::Accessors,
                             "Return a list of accessors to the leaves in the treespec.")
        .def_method_pos_only("entries",
                             &PyTreeSpec::Entries,
                             "Return a list of one-level entries to the children.")
        .def("entry",
             &PyTreeSpec::Entry,
             "Return the entry at the given index.",
             py::arg("index"),
             py::pos_only())
        .def_method_pos_only("children",
                             &PyTreeSpec::Children,
                             "Return a list of treespecs for the children.")
        .def("child",
             &PyTreeSpec::Child,
             "Return the treespec for the child at the given index.",
             py::arg("index"),
             py::pos_only())
        .def_method_pos_only(
            "one_level",
            [](const PyTreeSpec& t) -> std::optional<std::unique_ptr<PyTreeSpec>> {
                if (t.IsLeaf()) [[unlikely]] {
                    return std::nullopt;
                }
                return t.GetOneLevel();
            },
            "Return the one-level structure of the root node. Return None if the root node "
            "represents a leaf.")
        .def_property_readonly("num_leaves",
                               &PyTreeSpec::GetNumLeaves,
                               "Number of leaves in the tree.")
        .def_property_readonly("num_nodes",
                               &PyTreeSpec::GetNumNodes,
                               "Number of nodes in the tree. "
                               "Note that a leaf is also a node but has no children.")
        .def_property_readonly("num_children",
                               &PyTreeSpec::GetNumChildren,
                               "Number of children of the root node. "
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
            [](const PyTreeSpec& t) -> py::object { return t.GetType(); },
            "The type of the root node. Return None if the root node is a leaf.")
        .def_property_readonly("kind", &PyTreeSpec::GetPyTreeKind, "The kind of the root node.")
        .def("is_leaf",
             &PyTreeSpec::IsLeaf,
             "Test whether the treespec represents a leaf.",
             py::pos_only(),
             py::kw_only(),
             py::arg("strict") = true)
        .def_method_pos_only("is_one_level",
                             &PyTreeSpec::IsOneLevel,
                             "Test whether the treespec represents a one-level tree.")
        .def("is_prefix",
             &PyTreeSpec::IsPrefix,
             "Test whether this treespec is a prefix of the given treespec.",
             py::arg("other"),
             py::pos_only(),
             py::kw_only(),
             py::arg("strict") = false)
        .def("is_suffix",
             &PyTreeSpec::IsSuffix,
             "Test whether this treespec is a suffix of the given treespec.",
             py::arg("other"),
             py::pos_only(),
             py::kw_only(),
             py::arg("strict") = false)
        .def("__eq__",
             std::equal_to<PyTreeSpec>(),
             "Test for equality to another object.",
             py::is_operator(),
             py::arg("other"),
             py::pos_only())
        .def("__ne__",
             std::not_equal_to<PyTreeSpec>(),
             "Test for inequality to another object.",
             py::is_operator(),
             py::arg("other"),
             py::pos_only())
        .def("__lt__",
             std::less<PyTreeSpec>(),
             "Test for this treespec is a strict prefix of another object.",
             py::is_operator(),
             py::arg("other"),
             py::pos_only())
        .def("__le__",
             std::less_equal<PyTreeSpec>(),
             "Test for this treespec is a prefix of another object.",
             py::is_operator(),
             py::arg("other"),
             py::pos_only())
        .def("__gt__",
             std::greater<PyTreeSpec>(),
             "Test for this treespec is a strict suffix of another object.",
             py::is_operator(),
             py::arg("other"),
             py::pos_only())
        .def("__ge__",
             std::greater_equal<PyTreeSpec>(),
             "Test for this treespec is a suffix of another object.",
             py::is_operator(),
             py::arg("other"),
             py::pos_only())
        .def_method_pos_only("__repr__",
                             &PyTreeSpec::ToString,
                             "Return a string representation of the treespec.")
        .def_method_pos_only("__hash__", &PyTreeSpec::HashValue, "Return the hash of the treespec.")
        .def_method_pos_only("__len__", &PyTreeSpec::GetNumLeaves, "Number of leaves in the tree.")
        .def(py::pickle([](const PyTreeSpec& t) -> py::object { return t.ToPickleable(); },
                        [](const py::object& o) -> std::unique_ptr<PyTreeSpec> {
                            return PyTreeSpec::FromPickleable(o);
                        }),
             "Serialization support for PyTreeSpec.",
             py::arg("state"),
             py::pos_only());

    auto PyTreeIterTypeObject = py::class_<PyTreeIter>(
        mod,
        "PyTreeIter",
        "Iterator over the leaves of a pytree.",
        // NOLINTBEGIN[readability-function-cognitive-complexity,cppcoreguidelines-avoid-do-while]
        py::custom_type_setup([](PyHeapTypeObject* heap_type) -> void {
            auto* const type = &heap_type->ht_type;
            type->tp_flags |= Py_TPFLAGS_HAVE_GC;
            type->tp_traverse = &PyTreeIter::PyTpTraverse;
        }),
        // NOLINTEND[readability-function-cognitive-complexity,cppcoreguidelines-avoid-do-while]
        py::module_local());
    auto* const PyTreeIter_Type = reinterpret_cast<PyTypeObject*>(PyTreeIterTypeObject.ptr());
    PyTreeIter_Type->tp_name = "optree.PyTreeIter";
    py::setattr(PyTreeIterTypeObject.ptr(), Py_Get_ID(__module__), Py_Get_ID(optree));

    PyTreeIterTypeObject
        .def(py::init<py::object, std::optional<py::function>, bool, std::string>(),
             "Create a new iterator over the leaves of a pytree.",
             py::arg("tree"),
             py::pos_only(),
             py::arg("leaf_predicate") = std::nullopt,
             py::arg("none_is_leaf") = false,
             py::arg("namespace") = "")
        .def_method_pos_only("__iter__", &PyTreeIter::Iter, "Return the iterator object itself.")
        .def_method_pos_only("__next__", &PyTreeIter::Next, "Return the next leaf in the pytree.");

#undef def_method_pos_only

#ifdef Py_TPFLAGS_IMMUTABLETYPE
    PyTreeKind_Type->tp_flags |= Py_TPFLAGS_IMMUTABLETYPE;
    PyTreeSpec_Type->tp_flags |= Py_TPFLAGS_IMMUTABLETYPE;
    PyTreeIter_Type->tp_flags |= Py_TPFLAGS_IMMUTABLETYPE;
    PyTreeKind_Type->tp_flags &= ~Py_TPFLAGS_READY;
    PyTreeSpec_Type->tp_flags &= ~Py_TPFLAGS_READY;
    PyTreeIter_Type->tp_flags &= ~Py_TPFLAGS_READY;
#endif

    if (PyType_Ready(PyTreeKind_Type) < 0) [[unlikely]] {
        INTERNAL_ERROR("`PyType_Ready(&PyTreeKind_Type)` failed.");
    }
    if (PyType_Ready(PyTreeSpec_Type) < 0) [[unlikely]] {
        INTERNAL_ERROR("`PyType_Ready(&PyTreeSpec_Type)` failed.");
    }
    if (PyType_Ready(PyTreeIter_Type) < 0) [[unlikely]] {
        INTERNAL_ERROR("`PyType_Ready(&PyTreeIter_Type)` failed.");
    }

    py::getattr(py::module_::import("atexit"),
                "register")(py::cpp_function(&PyTreeTypeRegistry::Clear));
}

}  // namespace optree

#if PYBIND11_VERSION_HEX >= 0x020D00F0  // pybind11 2.13.0
// NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-type-vararg]
PYBIND11_MODULE(_C, mod, py::mod_gil_not_used()) { optree::BuildModule(mod); }
#else
// NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-type-vararg]
PYBIND11_MODULE(_C, mod) { optree::BuildModule(mod); }
#endif
