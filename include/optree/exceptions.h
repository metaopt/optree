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

#pragma once

#include <cstddef>    // std::size_t
#include <optional>   // std::optional, std::nullopt
#include <sstream>    // std::ostringstream
#include <stdexcept>  // std::logic_error
#include <string>     // std::string, std::char_traits

namespace optree {

constexpr std::size_t CURRENT_FILE_PATH_SIZE = std::char_traits<char>::length(__FILE__);
constexpr std::size_t CURRENT_FILE_RELPATH_FROM_PROJECT_ROOT_SIZE =
    std::char_traits<char>::length("include/optree/exceptions.h");
static_assert(CURRENT_FILE_PATH_SIZE >= CURRENT_FILE_RELPATH_FROM_PROJECT_ROOT_SIZE,
              "SOURCE_PATH_PREFIX_SIZE must be greater than 0.");
constexpr std::size_t SOURCE_PATH_PREFIX_SIZE =
    CURRENT_FILE_PATH_SIZE - CURRENT_FILE_RELPATH_FROM_PROJECT_ROOT_SIZE;
// NOLINTNEXTLINE[bugprone-reserved-identifier]
#define __FILE_RELPATH_FROM_PROJECT_ROOT__ ((const char*)&(__FILE__[SOURCE_PATH_PREFIX_SIZE]))

class InternalError : public std::logic_error {
public:
    explicit InternalError(const std::string& message) noexcept(noexcept(std::logic_error{message}))
        : std::logic_error{message} {}
    explicit InternalError(const std::string& message,
                           const std::string& file,
                           const std::size_t& lineno,
                           const std::optional<std::string> function =
                               std::nullopt) noexcept(noexcept(std::logic_error{message}))
        : InternalError([&message, &file, &lineno, &function]() -> std::string {
              std::ostringstream oss{};
              oss << message << " (";
              if (function) [[likely]] {
                  oss << "function `" << *function << "` ";
              }
              oss << "at file " << file << ":" << lineno << ")\n\n"
                  << "Please file a bug report at https://github.com/metaopt/optree/issues.";
              return oss.str();
          }()) {}
};

#define VA_FUNC2_(__0, __1, NAME, ...) NAME
#define VA_FUNC3_(__0, __1, __2, NAME, ...) NAME

#if !defined(__GNUC__)
#    define __PRETTY_FUNCTION__ std::nullopt  // NOLINT[bugprone-reserved-identifier]
#endif

#define INTERNAL_ERROR0_() INTERNAL_ERROR1_("Unreachable code.")
#define INTERNAL_ERROR1_(message)                                                                  \
    throw optree::InternalError((message),                                                         \
                                __FILE_RELPATH_FROM_PROJECT_ROOT__,                                \
                                __LINE__,                                                          \
                                __PRETTY_FUNCTION__)
#define INTERNAL_ERROR(...)                                                                        \
    VA_FUNC2_(__0 __VA_OPT__(, ) __VA_ARGS__, INTERNAL_ERROR1_, INTERNAL_ERROR0_)(__VA_ARGS__)

#define EXPECT2_(condition, message)                                                               \
    if (!(condition)) [[unlikely]] {                                                               \
        INTERNAL_ERROR1_(message);                                                                 \
    }
#define EXPECT0_() INTERNAL_ERROR0_()
#define EXPECT1_(condition) EXPECT2_((condition), "`" #condition "` failed.")
#define EXPECT_(...)                                                                               \
    VA_FUNC3_(__0 __VA_OPT__(, ) __VA_ARGS__, EXPECT2_, EXPECT1_, EXPECT0_)(__VA_ARGS__)

#define EXPECT_TRUE(condition, ...) EXPECT_((condition)__VA_OPT__(, ) __VA_ARGS__)
#define EXPECT_FALSE(condition, ...) EXPECT_(!(condition)__VA_OPT__(, ) __VA_ARGS__)
#define EXPECT_EQ(a, b, ...) EXPECT_((a) == (b)__VA_OPT__(, ) __VA_ARGS__)
#define EXPECT_NE(a, b, ...) EXPECT_((a) != (b)__VA_OPT__(, ) __VA_ARGS__)
#define EXPECT_LT(a, b, ...) EXPECT_((a) < (b)__VA_OPT__(, ) __VA_ARGS__)
#define EXPECT_LE(a, b, ...) EXPECT_((a) <= (b)__VA_OPT__(, ) __VA_ARGS__)
#define EXPECT_GT(a, b, ...) EXPECT_((a) > (b)__VA_OPT__(, ) __VA_ARGS__)
#define EXPECT_GE(a, b, ...) EXPECT_((a) >= (b)__VA_OPT__(, ) __VA_ARGS__)

}  // namespace optree
