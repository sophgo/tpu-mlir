//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace tpu_mlir {
namespace tpu {

// Function to split a string into tokens based on a
// delimiter
void splitString(std::string &input, char delimiter,
                 std::vector<std::string> &arr) {
  arr.clear();
  // Creating an input string stream from the input string
  std::istringstream stream(input);

  // Temporary string to store each token
  std::string token;

  // Read tokens from the string stream separated by the
  // delimiter
  while (std::getline(stream, token, delimiter)) {
    // Add the token to the array
    arr.push_back(token);
  }
}

} // namespace tpu
} // namespace tpu_mlir
