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
  auto a = Tensor(data, {4, 5});
  auto &b = a.resize(std::array{2, 2}, std::array{3, 3})
                .reshape(4, 9)
                .slice(2, std::array{2, 5})
                .transpose(1, 0)
                .slice(3, 2);
  // b.dump();
  EXPECT_EQ(b.getShape().size(), 2);
  EXPECT_EQ(b.getShape()[0], 3);
  EXPECT_EQ(b.getShape()[1], 2);
  EXPECT_EQ(b.getStrides().size(), 2);
  EXPECT_EQ(b.getStrides()[0], 2);
  EXPECT_EQ(b.getStrides()[1], 1);

  {
    auto out = b.getData();
    EXPECT_EQ(out.size(), 6);
    int expect_data[] = {3, 8, 4, 9, 5, 10};
    for (int i = 0; i < 6; i++) {
      EXPECT_EQ(expect_data[i], out[i]);
    }
  }

  auto c = b.asDType<int16_t>();
  EXPECT_EQ(c.size(), 12);

  auto int16_view = (int16_t *)b.getData().data();
  for (int i = 0; i < 12; i++)
    EXPECT_EQ(int16_view[i], c[i]);

  c >>= 1;
  {
    int expect_data[] = {1, 0, 4, 0, 2, 0, 4, 0, 2, 0, 5, 0};
    for (int i = 0; i < 12; i++)
      EXPECT_EQ(c[i], expect_data[i]);
  }

  c |= 0x10;
  {
    int expect_data[] = {17, 16, 20, 16, 18, 16, 20, 16, 18, 16, 21, 16};
    for (int i = 0; i < 12; i++)
      EXPECT_EQ(c[i], expect_data[i]);
  }
}
