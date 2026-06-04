//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaShapeSliceOp(tpu::ShapeSliceOp op) {
  auto input = op.getInput();
  auto output = op.getOutput();
  auto in_shape = module::getShape(input);
  auto num_out = module::getNumElements(output);
  auto elem_size = module::getBytes(output) / num_out;

  // 通过属性名获取切片参数
  auto axes_attr = op->getAttrOfType<mlir::ArrayAttr>("axes");
  auto offset_attr = op->getAttrOfType<mlir::ArrayAttr>("offset");
  auto ends_attr = op->getAttrOfType<mlir::ArrayAttr>("ends");
  auto steps_attr = op->getAttrOfType<mlir::ArrayAttr>("steps");

  std::vector<int64_t> axes, offset, ends, steps;
  if (axes_attr) {
    for (auto v : axes_attr.getValue()) {
      axes.push_back(v.cast<mlir::IntegerAttr>().getInt());
    }
  }
  if (offset_attr) {
    for (auto v : offset_attr.getValue()) {
      offset.push_back(v.cast<mlir::IntegerAttr>().getInt());
    }
  }
  if (ends_attr) {
    for (auto v : ends_attr.getValue()) {
      ends.push_back(v.cast<mlir::IntegerAttr>().getInt());
    }
  }
  if (steps_attr) {
    for (auto v : steps_attr.getValue()) {
      steps.push_back(v.cast<mlir::IntegerAttr>().getInt());
    }
  }

  // 读取输入数据到主机
  size_t in_bytes = module::getBytes(input);
  std::vector<char> in_data(in_bytes);
  CHECK_CUDA(cudaMemcpy(in_data.data(), getCudaData(input), in_bytes,
                        cudaMemcpyDeviceToHost));

  // 生成输出数据
  std::vector<char> out_data(num_out * elem_size);
  if (axes.size() == 1 && axes[0] == 0) {
    int start = offset.empty() ? 0 : static_cast<int>(offset[0]);
    int end = ends.empty() ? static_cast<int>(in_shape[0]) : static_cast<int>(ends[0]);
    int step = steps.empty() ? 1 : static_cast<int>(steps[0]);
    int out_idx = 0;
    for (int i = start; i < end && i < static_cast<int>(in_shape[0]); i += step) {
      if (out_idx >= num_out) break;
      memcpy(out_data.data() + out_idx * elem_size,
             in_data.data() + i * elem_size, elem_size);
      out_idx++;
    }
  } else {
    UNREACHABLE_OP("ShapeSliceOp with complex axes not supported", op);
  }

  // 将输出数据拷贝回设备
  CHECK_CUDA(cudaMemcpy(getCudaData(output), out_data.data(),
                        num_out * elem_size, cudaMemcpyHostToDevice));
}

// void py_cuda::cudaShapeSliceOp(top::ShapeSliceOp op) {
//   UNREACHABLE_OP("top::ShapeSliceOp not expected", op);
// }