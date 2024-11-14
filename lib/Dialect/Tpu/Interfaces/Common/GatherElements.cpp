//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::GatherElementsOp::init(InferenceParameter &p) {
  return success();
}
void tpu::GatherElementsOp::deinit(InferenceParameter &p) {}

// support dim <= 8
static inline void gather_dim8(float *dst, const float *src,
                               const float *indices, const int *indices_shape,
                               int *src_shape, int src_dim, int axis,
                               int dst_dim) {

  int indices_shape8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  int src_shape8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  int indices_axis_stride = 1;

  for (int i = 0; i < src_dim; ++i) {
    src_shape8[i] = src_shape[i];
    indices_shape8[i] = indices_shape[i];
    if (i > axis)
      indices_axis_stride *= src_shape[i];
  }

  for (int i0 = 0; i0 < indices_shape8[0]; ++i0) {
    int tmp0 = 0;
    tmp0 += axis == 0 ? 0 : i0;
    tmp0 *= src_shape8[1];
    for (int i1 = 0; i1 < indices_shape8[1]; ++i1) {
      int tmp1 = tmp0;
      tmp1 += axis == 1 ? 0 : i1;
      tmp1 *= src_shape8[2];
      for (int i2 = 0; i2 < indices_shape8[2]; ++i2) {
        int tmp2 = tmp1;
        tmp2 += axis == 2 ? 0 : i2;
        tmp2 *= src_shape8[3];
        for (int i3 = 0; i3 < indices_shape8[3]; ++i3) {
          int tmp3 = tmp2;
          tmp3 += axis == 3 ? 0 : i3;
          tmp3 *= src_shape8[4];
          for (int i4 = 0; i4 < indices_shape8[4]; ++i4) {
            int tmp4 = tmp3;
            tmp4 += axis == 4 ? 0 : i4;
            tmp4 *= src_shape8[5];
            for (int i5 = 0; i5 < indices_shape8[5]; ++i5) {
              int tmp5 = tmp4;
              tmp5 += axis == 5 ? 0 : i5;
              tmp5 *= src_shape8[6];
              for (int i6 = 0; i6 < indices_shape8[6]; ++i6) {
                int tmp6 = tmp5;
                tmp6 += axis == 6 ? 0 : i6;
                tmp6 *= src_shape8[7];
                for (int i7 = 0; i7 < indices_shape8[7]; ++i7) {
                  int tmp7 = tmp6;
                  tmp7 += axis == 7 ? 0 : i7;
                  int indices_add = (int)(*indices) * indices_axis_stride;
                  *dst = src[tmp7 + indices_add];
                  ++dst;
                  ++indices;
                  // llvm::outs() << tmp << " " << tmp7 << "\n";
                }
              }
            }
          }
        }
      }
    }
  }
}

LogicalResult tpu::GatherElementsOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  const float *indices = p.inputs[1];
  float *dst = p.outputs[0];
  int axis = getAxis();
  auto src_dim = module::getShape(getInput()).size();
  auto dst_dim = module::getShape(getIndices()).size();
  int src_shape[src_dim];
  int indices_shape[dst_dim];
  module::getGlobalShape(getInput(), src_shape, src_dim);
  module::getGlobalShape(getIndices(), indices_shape, dst_dim);

  if (axis < 0) {
    axis += src_dim;
  }

  if (src_dim > 0 && src_dim <= 8 && axis < src_dim) {
    gather_dim8(dst, src, indices, indices_shape, src_shape, src_dim, axis,
                dst_dim);
  } else {
    llvm_unreachable("Not implemented yet");
  }

  return success();
}

mlir::Type tpu::GatherElementsOp::type_verify(uint64_t opd_idx,
                                              TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 1) {
    // indices
    auto opd = op->getOperand(1);
    auto in_op = opd.getDefiningOp();
    if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
      return do_nothing(mode);
    }
    auto stype = module::getStorageType(opd);
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    // auto bitwidth = stype.getIntOrFloatBitWidth();
    return Builder(op).getIntegerType(32);
  }
  return type_verify_case_same(op, opd_idx, mode);
}

bool tpu::GatherElementsOp::support_multi_core() { return false; }
