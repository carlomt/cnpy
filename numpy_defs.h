#pragma once

#include <cstddef>

namespace cnpy
{
  namespace constants
  {
    constexpr std::size_t numpy_magic_length = 6;
    constexpr auto numpy_magic = "\x93NUMPY";
    constexpr std::size_t zip_header_length = 30;
    // This needs to be long and not size_t to silence a warning on WIN due to a use it in a seekg(-zip_footer_length) call
    constexpr int zip_footer_length = 22;
  }  // namespace constants
}  // namespace cnpy
