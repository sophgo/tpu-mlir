//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

// clang-format off
// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7] => batch = 12, M = 5, K = 6, N = 7
// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// case 5: [4, 5, 6] * [6] = [4, 5] => batch =1, M = 20, K = 6, N = 1
// case 6: [4096] * [4096, 12884] = [1,12884] => batch =1, M = 1, K = 4096, N = 12884
// clang-format on
matmul_attr_t tpu::MatMulOp::parseParam() {
  matmul_attr_t p = {0};
  auto a_s = SmallVector<int64_t>(module::getShape(getInput()));
  auto b_s = SmallVector<int64_t>(module::getShape(getRight()));
  auto o_s = SmallVector<int64_t>(module::getShape(getOutput()));
  p.input_zp = getInputZp();
  p.with_bias = !module::isNone(getBias());
  p.do_relu = getDoRelu();
  p.relu_limit = this->getReluLimit().convertToDouble();
  p.right_zp = getRightZp();
  p.right_transpose = getRightTranspose();
  p.left_transpose = getLeftTranspose();
  p.output_transpose = getOutputTranspose();
  p.hdim_is_batch = getHdimIsBatch();
  p.left_reuse = getLeftReuse();
  auto a_dims = a_s.size();
  auto b_dims = b_s.size();
  auto o_dims = o_s.size();
  p.batch = 1;
  p.batch_low = 1;
  if (b_dims == 1) {
    assert(p.right_transpose == false);
    b_s.push_back(1);
    o_s.push_back(1);
    b_dims += 1;
    o_dims += 1;
  }
  if (a_dims == 1){
    assert(p.left_transpose == false);
    a_s.insert(a_s.begin(), 1);
    o_s.insert(o_s.begin(), 1);
    a_dims += 1;
    o_dims += 1;
  }
  // for hdim_is_batch = true,
  // BM1684x: (B0, M, B1, K) x (B0, K, B1, N) = (B0, M, B1, N)
  // CV18xx:  (B0, B1, M, K) x (B0, K, B1, N) = (B0, B1, M, N)
  // up to now bm168x right_trans, left_trans, output_trans always be true
  //           cv18xx support either one to be true
  if (p.right_transpose) {
    if (p.hdim_is_batch) {
      p.K = b_s[b_dims - 3];
      p.N = b_s[b_dims - 1];
      // fix bias_merge_izp size for bm1684x
      if (module::isBM1684XFamily()) {
        p.N = b_s[b_dims - 3];
        p.K = b_s[b_dims - 1];
      }
    } else {
      // trans hw
      p.N = b_s[b_dims - 2];
      p.K = b_s[b_dims - 1];
    }
  } else {
    p.N = b_s[b_dims - 1];
    p.K = b_s[b_dims - 2];
  }

  if (p.left_transpose) {
    if (p.hdim_is_batch) {
      p.M = a_s[a_dims - 3];
    } else {
      // trans hw
      p.M = a_s[a_dims - 1];
      for (int i = 0; i < a_dims - 2; i++) {
        p.batch *= a_s[i];
      }
    }
  } else {
    p.M = a_s[a_dims - 2];
  }
  // parse batch info from output
  for (int i = 0; i < o_dims - 2; i++) {
    p.batch *= o_s[i];
  }
  if (p.hdim_is_batch) {
    p.batch = o_s[0];
    if (!p.output_transpose && module::isCV18xx()) {
      p.batch_low = o_s[1];
    } else {
      p.batch_low = o_s[2];
      p.output_transpose = true; // tmp code remove later
    }
  }
  if (!p.hdim_is_batch) {
    // if right batch dim is broadcast, merge left batch to M
    int right_batch = 1;
    for (int i = 0; i < b_dims - 2; i++) {
      right_batch *= b_s[i];
    }
    if (right_batch != p.batch && right_batch == 1) {
      p.batch = 1;
    }
    if (p.batch > 1 || o_dims <= 2) {
      p.M = o_s[o_dims - 2];
    } else {
      p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                            std::multiplies<int64_t>());
    }
  }
  return p;
}

LogicalResult tpu::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  auto a = parseParam();
  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], a.batch,
                a.batch_low, a.M, a.K, a.N, a.do_relu, a.relu_limit, a.right_zp,
                a.input_zp, a.right_transpose, a.left_transpose,
                a.output_transpose, a.hdim_is_batch);
  p.handle = (void *)matmul;
  return success();
}

void tpu::MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::MatMulOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto matmul = (MatMul *)p.handle;

  matmul->run();
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  bool is_cv18xx = module::isCV18xx();
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    auto qmode = getQuantMode();
    if (is_cv18xx) {
      auto a = parseParam();
      auto full_batch = a.batch * a.batch_low;
      bool is_fc = module::isWeight(getRight());
      i64_array_t rshift_v;
      i64_array_t multiplier_v;
      if (is_fc) {
        rshift_v = module::getI64Array(getRshifts(), full_batch, 0);
        multiplier_v = module::getI64Array(getMultipliers(), full_batch, 1);
      } else {
        rshift_v = module::getI64Array(getRshifts(), 1, 0);
        multiplier_v = module::getI64Array(getMultipliers(), 1, 1);
        rshift_v->resize(full_batch, rshift_v->at(0));
        multiplier_v->resize(full_batch, multiplier_v->at(0));
      }
      int64_t isz = a.M * a.N;
      for (int64_t i = 0; i < full_batch; ++i) {
#pragma omp parallel for schedule(static, omp_schedule(isz))
        for (int64_t j = 0; j < isz; ++j) {
          int64_t offset = i * isz + j;
          int64_t v = 0;
          v = applyMultiplierAndRShift(p.outputs[0][offset],
                                       multiplier_v->at(i), rshift_v->at(i),
                                       qmode, ROUNDING_HALF_AWAY_FROM_ZERO);
          p.outputs[0][offset] = saturate(v, out_type);
        }
      }
    } else {
      auto o_qtype = module::getUniformQuantizedType(getOutput());
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      auto multiplier_v = module::getI64Array(getMultipliers(), 1, 1);
      assert(rshift_v->size() == 1);
      assert(multiplier_v->size() == 1);
      auto num_output = module::getNumElements(getOutput());
      if (qmode == tpu::RequantMode::TFLite_LShift ||
          qmode == tpu::RequantMode::TFLite) {
#pragma omp parallel for schedule(static, omp_schedule(num_output))
        for (int64_t i = 0; i < num_output; i++) {
          // auto v = (((int64_t)(p.outputs[0][i] * mlti) + (1 << (rft - 1))) >>
          // rft);
          auto v = MultiplyByQuantizedMultiplier((int32_t)(p.outputs[0][i]),
                                                 (int32_t)multiplier_v->at(0),
                                                 -(int32_t)rshift_v->at(0)) +
                   o_qtype.getZeroPoint();
          p.outputs[0][i] = saturate(v, out_type);
        }
      } else if (qmode == tpu::RequantMode::MultiplierShift) {
#pragma omp parallel for schedule(static, omp_schedule(num_output))
        for (int i = 0; i < num_output; ++i) {
          auto v = applyMultiplierAndRShift(
                       p.outputs[0][i], multiplier_v->at(0), rshift_v->at(0)) +
                   o_qtype.getZeroPoint();
          p.outputs[0][i] = saturate(v, out_type);
        }
      }
    }
  }

  return success();
}

LogicalResult tpu::MatMulOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    return failure();
  }
  if(module::isBM1684XFamily() && (int)getRunMode(getOperation()) == 1) {
    return failure();
  }

  auto ins = getOperands();
  if (backend::BM168x::getDataType(ins[0]) == DTYPE_FP32) {
    return failure();
  }

  auto Lshape = module::getShape(ins[0]);
  auto Rshape = module::getShape(ins[1]);
  int left_num_dims = module::getShape(ins[0]).size();
  int right_num_dims = module::getShape(ins[1]).size();
  if (((left_num_dims == 4 && Lshape[1] < Lshape[2]) ||
       (left_num_dims == 5 && Lshape[1] < Lshape[3])) &&
      right_num_dims == 2) {
    // GROUP_SMALL_C
    return success();
  } else if (left_num_dims == 3 && right_num_dims == 3) {
    // (1, M, K) x (1, K, N)
    return success();
  } else if (left_num_dims == 4 && right_num_dims == 4 && getHdimIsBatch()) {
    // (B1, M, B2, K) x (B1, K, B2, N)
    return success();
  }
  return failure();
}

LogicalResult tpu::MatMulOp::AllowDataSplit(int64_t axis,
                                            group_type_t group_type) {
  if (axis == 0) {
    return success();
  }

  auto lshape = module::getShape(getInput());
  if (group_type == GROUP_MM && axis == 1 && lshape.size() > 2) {
    return success();
  }

  if (lshape.size() == 4 && axis == 2 && getHdimIsBatch()) {
    return success();
  }

  return failure();
}

mlir::Type tpu::MatMulOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  if (opd_idx == 0 || opd_idx == 1) {
    return type_verify_case_i32(getOperation(), opd_idx, mode);
  }
  return type_verify_case_same(getOperation(), opd_idx, mode);
}

void tpu::MatMulOp::assign_fw_param(void *param) {
  auto p = parseParam();
  if (p.batch == 1) {
    // FW_BMNET_FC
    fw_fc_layer_param_t fw_fc_layer_param = {0};
    fw_fc_layer_param.input_neuron_num = p.K;
    fw_fc_layer_param.output_neuron_num = p.N;
    fw_fc_layer_param.transpose = p.right_transpose;
    fw_fc_layer_param.using_bias = p.with_bias;
    fw_fc_layer_param.if_activated = p.do_relu;
    fw_fc_layer_param.active_type =
        getDoRelu() ? 1 : 0; // not support prelu now
    fw_fc_layer_param.relu_upper_limit = p.relu_limit;
    fw_fc_layer_param.channel_shared = 0; // use for prelu
    fw_fc_layer_param.shared_slope = 0.f; // use for prelu
    fw_fc_layer_param.out_dims = module::getShape(getOutput()).size();
    if (module::isUniformQuantized(getOutput())) {
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      auto multiplier_v = module::getI64Array(getMultipliers(), 1, 1);
      fw_fc_layer_param.rshift_num = (uint8_t)rshift_v->at(0);
      fw_fc_layer_param.opd0_sign = module::isSign(getInput());
      fw_fc_layer_param.opd1_sign = module::isSign(getRight());
      fw_fc_layer_param.opd2_sign =
          fw_fc_layer_param.using_bias ? module::isSign(getBias()) : 0;
      fw_fc_layer_param.perlayer_bias = 0; // not support;

      fw_fc_layer_param.if_use_scale = 0; // 0:FcPerLayerShift 1:FcPerLayerScale
      fw_fc_layer_param.if_asymmetric = false; // not support
      fw_fc_layer_param.if_bias_float =
          (p.with_bias && BM168x::getDataType(getBias()) == DTYPE_FP32) ? 1 : 0;
      fw_fc_layer_param.if_perchannel = 0;
      fw_fc_layer_param.scale = multiplier_v->at(0);
      fw_fc_layer_param.weight_offset = (short)(0 - p.right_zp);
      fw_fc_layer_param.output_offset = 0; // not suppot now
      fw_fc_layer_param.weight_is_datatensor = !module::isWeight(getRight());
    }
    fw_fc_layer_param.version = 1;
    fw_fc_layer_param.res_16b =
        BM168x::getDataType(getOutput()) == DTYPE_INT16 ||
        BM168x::getDataType(getOutput()) == DTYPE_UINT16;
    if (BM168x::getDataType(getOutput()) == DTYPE_FP32 &&
        (BM168x::getDataType(getInput()) == DTYPE_INT8 ||
         BM168x::getDataType(getInput()) == DTYPE_UINT8)) {
      fw_fc_layer_param.res_16b = 2;
      fw_fc_layer_param.using_bias = 2;
    }
    fw_fc_layer_param.output_sign = module::isSign(getOutput());
    // assign param
    memcpy(param, &fw_fc_layer_param, sizeof(fw_fc_layer_param_t));
  } else {
    // FW_BMNET_BATCH_MATMUL
    fw_batch_matmul_layer_param_t fw_batch_matmul_layer_param = {0};
    fw_batch_matmul_layer_param.if_relu = p.do_relu;
    fw_batch_matmul_layer_param.relu_upper_limit = p.relu_limit;
    fw_batch_matmul_layer_param.in0_is_coeff = module::isWeight(getInput());
    fw_batch_matmul_layer_param.in1_is_coeff = module::isWeight(getRight());
    module::getGlobalShape(getInput(), fw_batch_matmul_layer_param.in0_shape);
    module::getGlobalShape(getRight(), fw_batch_matmul_layer_param.in1_shape);
    // assign param
    memcpy(param, &fw_batch_matmul_layer_param,
           sizeof(fw_batch_matmul_layer_param_t));
  }
}
