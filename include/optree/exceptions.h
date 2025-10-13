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

#include <cstddef>          // std::size_t
#include <source_location>  // std::source_location
#include <sstream>          // std::ostringstream
#include <stdexcept>        // std::logic_error
#include <string>           // std::string, std::char_traits
#include <string_view>      // std::string_view

namespace optree {

constexpr std::size_t CURRENT_FILE_PATH_SIZE =
    std::char_traits<char>::length(std::source_location::current().file_name());
constexpr std::size_t CURRENT_FILE_RELPATH_FROM_PROJECT_ROOT_SIZE =
    std::char_traits<char>::length("include/optree/exceptions.h");
static_assert(CURRENT_FILE_PATH_SIZE >= CURRENT_FILE_RELPATH_FROM_PROJECT_ROOT_SIZE,
              "SOURCE_PATH_PREFIX_SIZE must be greater than 0.");
constexpr std::size_t SOURCE_PATH_PREFIX_SIZE =
    CURRENT_FILE_PATH_SIZE - CURRENT_FILE_RELPATH_FROM_PROJECT_ROOT_SIZE;

constexpr std::string_view RelpathFromProjectRoot(const std::string_view &abspath) {
    return abspath.substr(SOURCE_PATH_PREFIX_SIZE);
}
constexpr std::string_view RelpathFromProjectRoot(
    const std::source_location &source_location = std::source_location::current()) {
    return RelpathFromProjectRoot(source_location.file_name());
}

class InternalError : public std::logic_error {
public:
    explicit InternalError(
        const std::string_view &message,
        const std::source_location &source_location = std::source_location::current())
        : std::logic_error{[&message, &source_location]() -> std::string {
              std::ostringstream oss{};
              oss << message << " (in function `" << source_location.function_name() << "` at file "
                  << RelpathFromProjectRoot(source_location) << ":" << source_location.line() << ":"
                  << source_location.column() << ")\n\n"
                  << "Please file a bug report at https://github.com/metaopt/optree/issues.";
              return oss.str();
          }()} {}
};

#define VA_FUNC2_(__0, __1, NAME, ...) NAME
#define VA_FUNC3_(__0, __1, __2, NAME, ...) NAME

#define INTERNAL_ERROR0_() INTERNAL_ERROR1_("Unreachable code.")
#define INTERNAL_ERROR1_(message) throw optree::InternalError(message)
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
