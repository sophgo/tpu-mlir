//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"

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
  if (a_dims == 1) {
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
      if (module::isBM1684XFamily() || module::isBM1690Family()) {
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
      p.M = p.output_transpose ? o_s[o_dims - 1] : o_s[o_dims - 2];
    } else {
      p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                            std::multiplies<int64_t>());
    }
    int b_temp = 1;
    for (int i = 1; i < b_dims - 2; i++) {
      b_temp *= b_s[i];
    }
    if (a_s[0] == b_s[0] && b_temp == 1 && b_dims > 2) {
      p.batch = b_s[0];
      int a_temp = 1;
      for (int i = 1; i < a_dims - 2; i++) {
        a_temp *= a_s[i];
      }
      // consider left_transpose
      p.M = a_s[o_dims - 2 + p.left_transpose] * a_temp;
    }
  }
  return p;
}

uint64_t tpu::MatMulOp::getL2BufferSize() {
  // calculate L2SRAM buffer size
  auto p = parseParam();
  fc_global_spec_t spec = {0};
  memset(&spec, 0, sizeof(spec));
  spec.if_getting_buffer_size = true;
  uint64_t buffer_size = 0;
  spec.buffer_size_ptr = &buffer_size;
  spec.if_relu = p.do_relu;
  spec.relu_limit = p.relu_limit;
  spec.have_bias = p.with_bias;
  spec.requant_mode = -1;
  spec.R_transpose = p.right_transpose;
  if (module::isUniformQuantized(getInput())) {
    spec.rshift = 0;
    spec.is_asymmetric = 1;
    spec.rzp_is_const = 1;
    spec.rzp_const_val = p.right_zp;
    spec.izp_const_val = p.input_zp;
    if (module::isUniformQuantized(getOutput())) {
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      auto multiplier_v = module::getI64Array(getMultipliers(), 1, 1);
      assert(rshift_v->size() == 1);
      assert(multiplier_v->size() == 1);
      spec.requant_mode = static_cast<int>(getQuantMode());
      spec.mul_val = multiplier_v->at(0);
      spec.shift_val = -rshift_v->at(0);
      auto output_type = module::getUniformQuantizedType(getOutput());
      spec.offset_val = output_type.getZeroPoint();
      spec.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
    }
  }
  auto input_spec = BM168x::get_input_spec(getOperation());
  auto output_spec = BM168x::get_output_spec(getOperation());
  // don't check instruction address when getting buffer size
  BM168x::instance()->dl_set_cmd_check_param(nullptr, false);
  BM168x::call_global_func("backend_api_fc_multi_core_global", &spec,
                           sizeof(spec), input_spec->data(),
                           output_spec->data());
  return buffer_size;
}

matmul_attr_t tpu::MatMulOp::dynparseParam() {
  std::vector<int64_t> in0_shape = module::getShape(getInput());
  std::vector<int64_t> in1_shape = module::getShape(getRight());
  bool l_transpose = getLeftTranspose();
  bool r_transpose = getRightTranspose();
  bool hdim_is_batch = getHdimIsBatch();
  auto in0_shape_new = in0_shape;
  auto in1_shape_new = in1_shape;
  int in0_dims = in0_shape.size();
  int in1_dims = in1_shape.size();
  if (l_transpose) {
    if (hdim_is_batch) {
      in0_shape_new[1] = in0_shape[2];
      in0_shape_new[2] = in0_shape[1];
      l_transpose = false;
    }
  }
  if (r_transpose) {
    if (hdim_is_batch) {
      in1_shape_new[1] = in1_shape[2];
      in1_shape_new[2] = in1_shape[1];
      r_transpose = false;
    }
  }

  auto k = in0_shape_new[in0_dims - 1];
  bool keep_dims_ = getKeepDims();
  int k_idx = in1_dims - (r_transpose ? 1 : 2);
  int n_idx = in1_dims - (r_transpose ? 2 : 1);
  auto n = in1_shape_new[n_idx];
  std::vector<int64_t> out_shape;
  if (in0_dims > in1_dims) {
    out_shape = in0_shape;
  } else if (in0_dims == in1_dims) {
    out_shape = in0_shape_new;
    for (int i = out_shape.size() - 3; i >= 0; i--) {
      out_shape[i] = std::max(in0_shape_new[i], in1_shape_new[i]);
    }
  } else {
    out_shape = in1_shape_new;
    for (int i = 1; i <= 2; i++) {
      out_shape[out_shape.size() - i] = in0_shape_new[in0_dims - i];
    }
  }
  if (in1_dims == 1) {
    ASSERT_THIS(in1_shape_new[0] == k);
    out_shape.pop_back();
  } else if (in1_shape_new[k_idx] == k) {
    if (module::getPlatform() == module::Platform::CAFFE) {
      // for caffe case
      // shape case:[1, 1, 1, 4832] * [4832, 126] = [1, 126]
      // shape case:[8, 1, 1, 4832] * [4832, 136] = [8, 136]
      for (int i = 1; i < out_shape.size(); i++) {
        if (out_shape[i] == 1) {
          out_shape.erase(out_shape.begin() + i);
          i--;
        }
      }
      out_shape[out_shape.size() - 1] = n;
    } else {
      out_shape[out_shape.size() - 1] = n;
    }
  } else if (in1_dims == 2) {
    auto sum = in1_shape_new[k_idx];
    while (out_shape.size() > 0 && sum % out_shape.back() == 0 && sum != 1) {
      sum = sum / out_shape.back();
      out_shape.pop_back();
    }
    if (sum != 1) {
      UNREACHABLE_THIS("shape is illegal");
    }
    out_shape.push_back(n);
  } else {
    out_shape[out_shape.size() - 1] = n;
  }
  if (!keep_dims_) {
    int64_t batch_size = std::accumulate(out_shape.begin(), out_shape.end() - 1,
                                         1, std::multiplies<int64_t>());
    out_shape.resize(2);
    out_shape[0] = batch_size;
    out_shape[1] = n;
  }
  bool o_transpose = getOutputTranspose();
  auto out_shape_new = out_shape;
  if (o_transpose) {
    if (hdim_is_batch) {
      out_shape_new[1] = out_shape[2];
      out_shape_new[2] = out_shape[1];
    }
  }
  module::setShape(getOutput(), out_shape_new);

  auto o_s = SmallVector<int64_t>(module::getShape(getOutput()));
  module::setShape(getOutput(), o_s);
  matmul_attr_t p = {0};
  auto a_s = SmallVector<int64_t>(module::getShape(getInput()));
  auto b_s = SmallVector<int64_t>(module::getShape(getRight()));
  // auto o_s = SmallVector<int64_t>(module::getShape(getOutput()));
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
  if (a_dims == 1) {
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
      if (module::isBM1684XFamily() || module::isBM1690Family()) {
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
    int b_temp = 1;
    for (int i = 1; i < b_dims - 2; i++) {
      b_temp *= b_s[i];
    }
    if (a_s[0] == b_s[0] && b_temp == 1 && b_dims > 2) {
      p.batch = b_s[0];
      int a_temp = 1;
      for (int i = 1; i < a_dims - 2; i++) {
        a_temp *= a_s[i];
      }
      p.M = a_s[o_dims - 2] * a_temp;
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

void tpu::MatMulOp::DumpQuantAgnosticAttrs(llvm::raw_string_ostream &os) {
  for (auto attr : getOperation()->getAttrs()) {
    auto attr_name = attr.getName().str();
    if (attr_name == "ginfo" || attr_name == "rshifts" ||
        attr_name == "multipliers") {
      continue;
    }
    os << attr_name << "=";
    attr.getValue().print(os);
    os << "; ";
  }

  auto rshift_v = module::getI64Array(getRshifts());
  auto multiplier_v = module::getI64Array(getMultipliers());
  assert(rshift_v && multiplier_v);
  if (rshift_v->size() == 1 && rshift_v->at(0) == 0) {
    // do-nothing.
  } else {
    os << "rshifts_len=" << rshift_v->size()
       << "; "; // to distinguish per-channel/per-tensor
  }

  if (multiplier_v->size() == 1 && multiplier_v->at(0) == 1) {
    // do-nothing.
  } else {
    os << "multipliers_len=" << multiplier_v->size() << "; ";
  }
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
  auto matmul = new MatMul();
  auto a = dynparseParam();
  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], a.batch,
                a.batch_low, a.M, a.K, a.N, a.do_relu, a.relu_limit, a.right_zp,
                a.input_zp, a.right_transpose, a.left_transpose,
                a.output_transpose, a.hdim_is_batch);
  matmul->run();
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  bool is_cv18xx = module::isCV18xx();
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isFloat8E4M3FN()) {
      if (!getOutF8Scales().has_value())
        llvm_unreachable("should have out scale for MatMul in f8 mode");
      f64_array_t scales = module::getF64Array(getOutF8Scales().value());
      if (scales->size() == 1) {
        [[maybe_unused]] auto scale_f = scales->at(0);
        [[maybe_unused]] auto scale_f_reciprocal = 1 / scales->at(0);
        F8E4M3(p.outputs[0], p.outputs[0], num_elem, scale_f_reciprocal, true);
      } else {
        auto output_shape = module::getShape(getOutput());
        ASSERT_THIS(scales->size() == output_shape[output_shape.size() - 1]);
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
        for (int i = 0; i < num_elem; i++) {
          p.outputs[0][i] = F8E4M3(
              p.outputs[0][i],
              1.0 / scales->at(i % output_shape[output_shape.size() - 1]),
              true);
        }
      }
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
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
      auto output_shape = module::getShape(getOutput());
      int N = output_shape[output_shape.size() - 1];
      int shift_num = getFuseRq() ? N : 1;
      auto rshift_v = module::getI64Array(getRshifts(), shift_num, 0);
      auto multiplier_v = module::getI64Array(getMultipliers(), shift_num, 1);
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
        if (getFuseRq()) {
#pragma omp parallel for schedule(static, omp_schedule(num_output))
          for (int i = 0; i < num_output; ++i) {
            auto v = applyMultiplierAndRShift(
                         p.outputs[0][i], p.inputs[3][i % N], rshift_v->at(0)) +
                     o_qtype.getZeroPoint();
            p.outputs[0][i] = saturate(v, out_type);
          }
        } else {
#pragma omp parallel for schedule(static, omp_schedule(num_output))
          for (int i = 0; i < num_output; ++i) {
            auto v =
                applyMultiplierAndRShift(p.outputs[0][i], multiplier_v->at(0),
                                         rshift_v->at(0)) +
                o_qtype.getZeroPoint();
            p.outputs[0][i] = saturate(v, out_type);
          }
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
  if ((module::isBM1684XFamily() || module::isBM1690Family()) &&
      getRunMode(getOperation()) == tpu::RunMode::TPU_DYNAMIC) {
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
  bool right_trans = getRightTranspose();
  if (left_num_dims == 4 && right_num_dims == 4 && !right_trans &&
      Lshape[0] == Rshape[0] && Lshape[1] != Rshape[1] &&
      Lshape[2] == Rshape[2] && Lshape[3] == Rshape[3])
    return failure();
  if (((left_num_dims == 4 && Lshape[1] < Lshape[2]) ||
       (left_num_dims == 5 && Lshape[1] < Lshape[3])) &&
      right_num_dims == 2) {
    // GROUP_SMALL_C
    return success();
  } else if (!module::isTrain() && left_num_dims == 3 && right_num_dims == 3) {
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
  if ((group_type == GROUP_MM || group_type == GROUP_SMALL_C) && axis == 1 &&
      lshape.size() > 2) {
    return success();
  }

  if (lshape.size() == 4 && axis == 2 && getHdimIsBatch()) {
    return success();
  }

  return failure();
}

mlir::Type tpu::MatMulOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  if (opd_idx == 0 || opd_idx == 1) {
    return type_verify_case_i16_or_i32(getOperation(), opd_idx, mode);
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

ArrayAttr tpu::MatMulOp::getIndexingMaps() {
  MLIRContext *context = getContext();
  if (module::isWeight(getRight()) &&
      module::getStorageType(getInput()).isInteger(4)) {
    return Builder(getContext()).getAffineMapArrayAttr({});
  }

  auto outShape = module::getShape(getOutput());
  auto inputShape = module::getShape(getInput());
  auto rightShape = module::getShape(getRight());
  // compute the parallel dimensions
  bool hasTS = getLeftTranspose() || getOutputTranspose();
  bool hdimBatch = getHdimIsBatch();
  int maxParallelDims = outShape.size() - 1 - hasTS - hdimBatch;
  if (hdimBatch && outShape.size() == 4) {
    maxParallelDims = 1;
  }

  if (maxParallelDims < 1)
    return Builder(getContext()).getAffineMapArrayAttr({});

  AffineMap outMap =
      AffineMap::getMultiDimIdentityMap(maxParallelDims, context);
  int inputParalleDims =
      std::min(std::max((int)inputShape.size() - 1, 0), maxParallelDims);
  int rightParalleDims =
      std::min(std::max((int)rightShape.size() - 2, 0), maxParallelDims);

  // batch broadcast case: (B, M, K) x (1, K, N), 1 can not be sliced
  bool all_one = true;
  for (int i = 0; i < rightParalleDims; i++) {
    if (rightShape[i] > 1) {
      all_one = false;
      break;
    }
  }
  if (all_one) {
    rightParalleDims = 0;
  }

  AffineMap inputMap =
      AffineMap::get(maxParallelDims, 0,
                     outMap.getResults().slice(0, inputParalleDims), context);

  AffineMap rightMap =
      AffineMap::get(maxParallelDims, 0,
                     outMap.getResults().slice(0, rightParalleDims), context);
  AffineMap emptyMap = AffineMap::get(maxParallelDims, 0, context);

  SmallVector<AffineMap> indexingMaps{inputMap, rightMap};

  for (int i = 2, n = getNumOperands(); i < n; ++i) {
    indexingMaps.push_back(emptyMap);
  }

  indexingMaps.push_back(outMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

bool tpu::MatMulOp::support_multi_core() {
  if (module::isSG2380()) {
    auto in_type = module::getStorageType(getOperand(0));
    if (in_type.isInteger(8)) {
      auto p = parseParam();
      if (p.right_transpose || p.M >= 64) {
        return false;
      }
      uint64_t mm_cycle = p.M * align_up(p.N, 32) * 2u * p.K;
      uint64_t mm2_cycle =
          (uint64_t)align_up(p.M, 32) * align_up(p.N, 4) * align_up(p.K, 16);
      if (mm_cycle <= mm2_cycle) {
        return true;
      }
    }
  }
  auto p = parseParam();
  if (module::isBM1690Family()) {
    auto in_type = module::getStorageType(getOperand(0));
    if (in_type.isInteger(8)) {
      if (p.left_transpose) {
        return false;
      }
      if (p.batch > 1) {
        return false;
      }
      return true;
    }
  }

  if (p.hdim_is_batch || p.batch != 1 || p.do_relu ||
      module::getMode() == module::Mode::F8E4M3 ||
      module::getMode() == module::Mode::F8E5M2 ||
      module::getMode() == module::Mode::F8 ||
      module::getMode() == module::Mode::INT8) {
    return false;
  }

  auto out_stype = module::getStorageType(getOutput());
  if (out_stype.isF16() || out_stype.isBF16()) {
    return true;
  }
  return false;
}
