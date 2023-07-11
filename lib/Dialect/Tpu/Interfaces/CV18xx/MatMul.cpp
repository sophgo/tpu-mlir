//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "tpu_mlir/Support/TPUCompressUtil.h"

using namespace tpu_mlir::backend;

// for bf16
static void transposeBiasFp32(std::vector<float> &bias_f32,
                              std::vector<uint32_t> &bias_u32) {
  // Split into high/low part
  std::vector<uint16_t> bias_fp32_high;
  std::vector<uint16_t> bias_fp32_low;
  float *biasFloatPtr = bias_f32.data();
  int size = bias_f32.size();
  for (int i = 0; i < size; ++i) {
    unsigned short *temp_short_ptr =
        reinterpret_cast<unsigned short *>(biasFloatPtr + i);
    bias_fp32_high.push_back(temp_short_ptr[1]);
    bias_fp32_low.push_back(temp_short_ptr[0]);
  }
  std::vector<uint16_t> bias_reshape_fp32;
  bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_high.begin(),
                           bias_fp32_high.end());
  bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_low.begin(),
                           bias_fp32_low.end());
  // then copy into uint32_t
  assert(bias_u32.size() == bias_f32.size());
  memcpy(bias_u32.data(), bias_reshape_fp32.data(), size * sizeof(uint32_t));
}

// for int8
static void transposeBiasInt32(std::vector<int32_t> &bias_int32,
                               std::vector<uint32_t> &bias_u32) {
  size_t element_num = bias_u32.size();
  int8_t *ptr = reinterpret_cast<int8_t *>(bias_int32.data());
  std::vector<int8_t> b(ptr, ptr + element_num * sizeof(int32_t));
  std::vector<int8_t> b_t(b.size());
  for (size_t i = 0; i < element_num; i++) {
    for (size_t j = 0; j < 4; j++) {
      b_t[j * element_num + i] = b[i * 4 + j];
    }
  }
  memcpy(bias_u32.data(), b_t.data(), b_t.size());
}

void tpu::MatMulOp::codegen_global_cv18xx(int64_t layer_id) {

  OpBuilder builder(getContext());
  auto p = parseParam();
  // TODO get batch_high and batch_low, group_fc bias transpose
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_filter = module::getAddress(getRight());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_bias = GA_INVALID;
  bool is_fc = true;
  if (!getRight().getDefiningOp()) {
    is_fc = false;
  } else {
    is_fc = module::isWeight(getRight());
  }
  if (is_fc) {
    int batch_high = 1;      // fixme
    int batch_low = p.batch; // fixme
    auto filterOp = getRight().getDefiningOp<top::WeightOp>();
    bool do_compress = filterOp.getDoCompress().has_value() &&
                       filterOp.getDoCompress().value();
    WeightCompresser weight_opt(this->getOperation(), do_compress);
    if (module::isUniformQuantized(getOutput())) {
      auto multiplier_v = module::getI64Array(getMultipliers(), p.batch, 1);
      auto rshift_v = module::getI64Array(getRshifts(), p.batch, 0);
      std::vector<int32_t> multiplier_int32;
      std::vector<int32_t> rshift_int32;
      multiplier_int32.assign(multiplier_v->begin(), multiplier_v->end());
      rshift_int32.assign(rshift_v->begin(), rshift_v->end());
      if (p.with_bias) {
        ga_bias = module::getAddress(getBias());
        auto biasOp = getBias().getDefiningOp<top::WeightOp>();
        auto bias_data = biasOp.read<int32_t>();
        std::vector<int32_t> bias_i32(p.N);
        std::vector<uint32_t> tmp_bias_u32(p.N);
        std::vector<uint32_t> bias_u32(bias_data->size());
        for (int b = 0; b < p.batch; ++b) {
          std::copy(bias_data->data() + b * p.N,
                    bias_data->data() + (b + 1) * p.N, bias_i32.data());
          transposeBiasInt32(bias_i32, tmp_bias_u32);
          memcpy(bias_u32.data() + b * p.N, tmp_bias_u32.data(),
                 p.N * sizeof(int32_t));
        }
        biasOp.update(bias_u32, bias_data->size());
      }
      cvi_backend_tg_fixed_fc_kernel(
          layer_id, ga_input, ga_filter, ga_bias, ga_output, p.M, p.K, p.N,
          p.with_bias, p.do_relu, rshift_int32, multiplier_int32,
          &weight_opt.old_data, &weight_opt.new_data, batch_high, batch_low,
          getLeftTranspose(), getRightTranspose(), getOutputTranspose());
    } else {
      // TODO batch_high, batch_low, lstride, ostride, do_quant_bf16
      if (p.with_bias) {
        ga_bias = module::getAddress(getBias());
        auto storage_type = module::getStorageType(getBias());
        if (storage_type.isF32()) {
          std::shared_ptr<std::vector<float_t>> bias_data;
          auto biasOp = getBias().getDefiningOp<top::WeightOp>();
          bias_data = biasOp.read<float_t>();
          std::vector<uint32_t> bias_u32(bias_data->size());
          std::vector<float> tmp_bias(p.N);
          std::vector<uint32_t> tmp_u32(p.N);
          for (int b = 0; b < p.batch; ++b) {
            std::copy(bias_data->data() + b * p.N,
                      bias_data->data() + (b + 1) * p.N, tmp_bias.data());
            transposeBiasFp32(tmp_bias, tmp_u32);
            std::copy(tmp_u32.begin(), tmp_u32.end(),
                      bias_u32.data() + b * p.N);
          }
          biasOp.update(bias_u32, bias_data->size());
          auto new_bias_type = RankedTensorType::get(
              module::getShape(getBias()), builder.getIntegerType(32),
              builder.getI64IntegerAttr(module::getAddress(getBias())));
          getBias().setType(new_bias_type);
        } // else: already modify by other matmul， because they share the bias
          // (only in bf16)
      }
      gaddr_t ga_scale = GA_INVALID;
      gaddr_t ga_zeropoint = GA_INVALID;
      bool do_quant_bf16 = false;
      cvi_backend_tg_bf16_fc_kernel(
          layer_id, ga_input, ga_filter, ga_bias, ga_output, p.M, p.K, p.N,
          p.with_bias, p.do_relu, &weight_opt.old_data, &weight_opt.new_data,
          batch_high, batch_low, getLeftTranspose(), getRightTranspose(),
          getOutputTranspose(), do_quant_bf16, ga_scale, ga_zeropoint);
    }
  } else {
    int batch_high = p.batch;    // fixme
    int batch_low = p.batch_low; // fixme
    // if (getRightTranspose() && getHdimIsBatch()) {
    //   batch_low = p.batch_low;
    //   batch_high = p.batch / batch_low;

    // }
    if (module::isUniformQuantized(getOutput())) {
      auto multiplier_v = module::getI64Array(getMultipliers(), 1, 1);
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      std::vector<int32_t> multiplier_int32;
      std::vector<int32_t> rshift_int32;
      multiplier_int32.assign(multiplier_v->begin(), multiplier_v->end());
      rshift_int32.assign(rshift_v->begin(), rshift_v->end());

      cvi_backend_tg_fixed_fc_kernel(
          layer_id, ga_input, ga_filter, ga_bias, ga_output, p.M, p.K, p.N,
          p.with_bias, p.do_relu, rshift_int32, multiplier_int32, nullptr,
          nullptr, batch_high, batch_low, getLeftTranspose(),
          getRightTranspose(), getOutputTranspose());
    } else {
      // TODO batch_high, batch_low, lt, rt, ot
      cvi_backend_tg_bf16_fc_kernel(
          layer_id, ga_input, ga_filter, GA_INVALID, ga_output, p.M, p.K, p.N,
          false, p.do_relu, nullptr, nullptr, batch_high, batch_low,
          getLeftTranspose(), getRightTranspose(), getOutputTranspose());
    }
  }
}

// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::MatMulOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::MatMulOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                         int64_t d_step, int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info,
                                         int64_t layer_id) {
  llvm_unreachable("Not supported now");
}
