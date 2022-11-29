//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::PermuteOp::getFLOPs() { return 0; }

LogicalResult top::PermuteOp::init(InferenceParameter &p) { return success(); }
void top::PermuteOp::deinit(InferenceParameter &p) {}

template <typename T> static int remove_value(std::vector<T> &v, int value) {
  int idx = 0;
  for (auto iter = v.begin(); iter != v.end(); iter++, idx++) {
    if (*iter == value) {
      v.erase(iter);
      return idx;
    }
  }
  return -1;
}

static void refresh(std::vector<int> &order, int idx) {
  for (auto &v : order) {
    if (v > idx) {
      v--;
    }
  }
}

LogicalResult top::PermuteOp::inference(InferenceParameter &p) {
  int64_t in, ic, ih, iw;
  std::vector<int64_t> in_shape = Module::getShape(input());
  std::shared_ptr<std::vector<int64_t>> perm = Module::getI64Array(order());
  int num_dims = in_shape.size();
  std::vector<int> order;
  for (int i = 0; i < num_dims; i++) {
    order.emplace_back(perm->at(i));
  }
  if (num_dims > 4) {
    // remove dims = 1
    while (num_dims > 4) {
      int idx = remove_value<int64_t>(in_shape, 1);
      if (idx < 0) {
        break;
      }
      remove_value(order, idx);
      refresh(order, idx);
      num_dims--;
    }
    // remove continous order
    while (num_dims > 4) {
      bool done = false;
      for (int i = 0; i < num_dims - 1; i++) {
        if (order[i] + 1 == order[i + 1]) {
          int idx = order[i];
          in_shape[idx] *= in_shape[idx + 1];
          in_shape.erase(in_shape.begin() + idx + 1);
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

  in = in_shape[0], ic = in_shape[1], ih = in_shape[2], iw = in_shape[3];

  for (int n = 0; n < in; n++) {
    for (int c = 0; c < ic; c++) {
      for (int h = 0; h < ih; h++) {
        for (int w = 0; w < iw; w++) {
          int cur[4] = {n, c, h, w};
          int in_idx = w + h * iw + c * ih * iw + n * ic * ih * iw;
          int out_idx =
              cur[order[3]] + cur[order[2]] * in_shape[order[3]] +
              cur[order[1]] * in_shape[order[3]] * in_shape[order[2]] +
              cur[order[0]] * in_shape[order[3]] * in_shape[order[2]] *
                  in_shape[order[1]];
          p.outputs[0][out_idx] = p.inputs[0][in_idx];
        }
      }
    }
  }
  return success();
}
