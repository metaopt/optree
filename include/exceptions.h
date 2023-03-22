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

#pragma once

#include <absl/strings/str_format.h>

#include <stdexcept>
#include <string>

#ifndef SOURCE_PATH_PREFIX_SIZE
#define SOURCE_PATH_PREFIX_SIZE 0
#endif

#ifndef FILE_RELPATH
#define FILE_RELPATH (&(__FILE__[SOURCE_PATH_PREFIX_SIZE]))
#endif

#define VFUNC2(__0, __1, NAME, ...) NAME
#define VFUNC3(__0, __1, __2, NAME, ...) NAME

namespace optree {

class InternalError : public std::logic_error {
 public:
    explicit InternalError(const std::string& msg) : std::logic_error(msg) {}
    InternalError(const std::string& msg, const std::string& file, const size_t& lineno)
        : InternalError(absl::StrFormat(
              "%s (at file %s:%lu)\n\n%s",
              msg,
              file,
              lineno,
              "Please file a bug report at https://github.com/metaopt/optree/issues.")) {}
};

}  // namespace optree

#define INTERNAL_ERROR1(message) throw optree::InternalError(message, FILE_RELPATH, __LINE__)
#define INTERNAL_ERROR0() INTERNAL_ERROR1("Unreachable code.")
#define INTERNAL_ERROR(...) /* NOLINTNEXTLINE[whitespace/parens] */ \
    VFUNC2(__0 __VA_OPT__(, ) __VA_ARGS__, INTERNAL_ERROR1, INTERNAL_ERROR0)(__VA_ARGS__)

#define EXPECT2(condition, message) \
    if (!(condition)) [[unlikely]]  \
    INTERNAL_ERROR1(message)
#define EXPECT0() INTERNAL_ERROR0()
#define EXPECT1(condition) EXPECT2(condition, "`" #condition "` failed.")
#define EXPECT(...) /* NOLINTNEXTLINE[whitespace/parens] */ \
    VFUNC3(__0 __VA_OPT__(, ) __VA_ARGS__, EXPECT2, EXPECT1, EXPECT0)(__VA_ARGS__)

#define EXPECT_TRUE(condition, ...) \
    EXPECT(condition __VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_FALSE(condition, ...) \
    EXPECT(!(condition)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_EQ(a, b, ...) \
    EXPECT((a) == (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_NE(a, b, ...) \
    EXPECT((a) != (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_LT(a, b, ...) \
    EXPECT((a) < (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_LE(a, b, ...) \
    EXPECT((a) <= (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_GT(a, b, ...) \
    EXPECT((a) > (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
#define EXPECT_GE(a, b, ...) \
    EXPECT((a) >= (b)__VA_OPT__(, ) __VA_ARGS__)  // NOLINT[whitespace/parens]
