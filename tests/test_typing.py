# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=missing-function-docstring

import sys
import time

import optree
from helpers import CustomNamedTupleSubclass, CustomTuple, Vector2D


class FakeNamedTuple(tuple):
    _fields = ('a', 2, 'b')


class FakeStructSequence(tuple):
    n_sequence_fields = 9
    n_fields = 11
    n_unnamed_fields = 0


def test_is_namedtuple():
    assert not optree.is_namedtuple((1, 2))
    assert not optree.is_namedtuple([1, 2])
    assert not optree.is_namedtuple(sys.float_info)
    assert not optree.is_namedtuple(time.gmtime())
    assert optree.is_namedtuple(CustomTuple(1, 2))
    assert optree.is_namedtuple(CustomNamedTupleSubclass(1, 2))
    assert not optree.is_namedtuple(FakeNamedTuple((1, 2, 3)))
    assert not optree.is_namedtuple(Vector2D(1, 2))
    assert not optree.is_namedtuple(FakeStructSequence((1, 2)))
    assert not optree.is_namedtuple_class(CustomTuple(1, 2))
    assert not optree.is_namedtuple_class(CustomNamedTupleSubclass(1, 2))
    assert not optree.is_namedtuple_class(FakeNamedTuple((1, 2, 3)))

    assert not optree.is_namedtuple(type(sys.float_info))
    assert not optree.is_namedtuple(time.struct_time)
    assert optree.is_namedtuple(CustomTuple)
    assert optree.is_namedtuple(CustomNamedTupleSubclass)
    assert not optree.is_namedtuple(FakeNamedTuple)
    assert not optree.is_namedtuple(Vector2D)
    assert not optree.is_namedtuple_class(type(sys.float_info))
    assert not optree.is_namedtuple_class(time.struct_time)
    assert optree.is_namedtuple_class(CustomTuple)
    assert optree.is_namedtuple_class(CustomNamedTupleSubclass)
    assert not optree.is_namedtuple_class(FakeNamedTuple)
    assert not optree.is_namedtuple_class(Vector2D)
    assert not optree.is_namedtuple_class(FakeStructSequence)


def test_is_structseq():
    assert not optree.is_structseq((1, 2))
    assert not optree.is_structseq([1, 2])
    assert optree.is_structseq(sys.float_info)
    assert optree.is_structseq(time.gmtime())
    assert not optree.is_structseq(CustomTuple(1, 2))
    assert not optree.is_structseq(CustomNamedTupleSubclass(1, 2))
    assert not optree.is_structseq(FakeNamedTuple((1, 2, 3)))
    assert not optree.is_structseq(Vector2D(1, 2))
    assert not optree.is_structseq(FakeStructSequence((1, 2)))
    assert not optree.is_structseq_class(sys.float_info)
    assert not optree.is_structseq_class(time.gmtime())

    assert optree.is_structseq(type(sys.float_info))
    assert optree.is_structseq(time.struct_time)
    assert not optree.is_structseq(CustomTuple)
    assert not optree.is_structseq(CustomNamedTupleSubclass)
    assert not optree.is_structseq(FakeNamedTuple)
    assert not optree.is_structseq(Vector2D)
    assert optree.is_structseq_class(type(sys.float_info))
    assert optree.is_structseq_class(time.struct_time)
    assert not optree.is_structseq_class(CustomTuple)
    assert not optree.is_structseq_class(CustomNamedTupleSubclass)
    assert not optree.is_structseq_class(FakeNamedTuple)
    assert not optree.is_structseq_class(Vector2D)
    assert not optree.is_structseq_class(FakeStructSequence)
