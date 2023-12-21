//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/TensorManipulation.h"
#include "gtest/gtest.h"

using namespace tpu_mlir;

TEST(TensorManipulation, Chain) {

  std::vector<int> data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                           11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  auto a = TensorManipulation(data, {4, 5});
  auto b = a.resize(SmallVector{2, 2}, SmallVector{3, 3})
               .reshape(4, 9)
               .slice(2, SmallVector{2, 5})
               .transpose(1, 0)
               .slice(3, 2)
               .getTensor();
  // b.dump();
  EXPECT_EQ(b.getShape().size(), 2);
  EXPECT_EQ(b.getShape()[0], 3);
  EXPECT_EQ(b.getShape()[1], 2);
  EXPECT_EQ(b.getStrides().size(), 2);
  EXPECT_EQ(b.getStrides()[0], 2);
  EXPECT_EQ(b.getStrides()[1], 1);
  auto out = b.getData();
  EXPECT_EQ(out.size(), 6);
  int expect_data[] = {3, 8, 4, 9, 5, 10};
  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(expect_data[i], out[i]);
  }
}
