//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
template<typename T>
static int remove_value(std::vector<T> & v, T value) {
  int idx = 0;
  for(auto iter = v.begin(); iter != v.end(); iter++, idx++) {
    if (*iter == value) {
      v.erase(iter);
      return idx;
    }
  }
  return -1;
}

static void refresh(std::vector<int64_t> &order, int64_t idx) {
  for(auto &v:order) {
    if (v > idx) {
      v--;
    }
  }
}

void parsePermuteParam(std::vector<int64_t> input_shape, std::vector<int64_t> order,
                        std::vector<int64_t> &shape_4, std::vector<int> &order_4) {
  int num_dims = order.size();
  if (num_dims > 4) {
    // remove dims = 1
    while(num_dims > 4) {
      int idx = remove_value<int64_t>(input_shape, 1);
      if (idx < 0) {
        break;
      }
      remove_value<int64_t>(order, idx);
      refresh(order, idx);
      num_dims--;
    }
    // remove continous order
    while(num_dims > 4) {
      bool done = false;
      for (int i = 0; i < num_dims-1; i++) {
        if (order[i] +1 == order[i+1]) {
          int idx = order[i];
          input_shape[idx] *= input_shape[idx+1];
          input_shape.erase(input_shape.begin() + idx + 1);
          order.erase(order.begin() + i + 1);
          refresh(order, idx + 1);
          num_dims--;
          done = true;
          break;
        }
      }
      if (done == false) {
        break;
      }
    }
    if (num_dims > 4) {
      llvm_unreachable("permute shape not support");
    }
  }
  order_4 = {0, 1, 2, 3};
  shape_4 = {1, 1, 1, 1};
  for (int end = num_dims - 1, idx = 3; end >= 0 && idx >= 0; end--, idx--) {
    shape_4[idx] = input_shape[end];
    order_4[idx] = order[end] + idx - end;
  }
}
void tpu::PermuteOp::codegen_global_cv18xx( int64_t layer_id) {

  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  auto order = Module::getI64Array(this->order());
  std::vector<int64_t> input_shape;
  Module::getShapeVec(input(), input_shape);
  std::vector<int64_t> shape_4;
  std::vector<int> order_4;
  parsePermuteParam(input_shape, *order, shape_4, order_4);
  if (Quant::isUniformQuantized(output())) {
    cvi_backend_tg_permute_kernel( layer_id, ga_input, ga_output,
          shape_4[0], shape_4[1], shape_4[2], shape_4[3],
          order_4[0], order_4[1], order_4[2], order_4[3], CVK_FMT_I8);
  } else {
    cvi_backend_tg_permute_kernel( layer_id, ga_input, ga_output,
          shape_4[0], shape_4[1], shape_4[2], shape_4[3],
          order_4[0], order_4[1], order_4[2], order_4[3], CVK_FMT_BF16);
  }
}

