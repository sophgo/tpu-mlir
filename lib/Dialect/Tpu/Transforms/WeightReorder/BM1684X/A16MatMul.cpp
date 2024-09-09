//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"

#ifdef __linux__
#include <dlfcn.h>
#else
#include <windows.h>
#endif

using namespace bm1684x;

typedef struct
{
    struct
    {
        int s, i;
    } Y_row;
    struct
    {
        int s, i;
    } Y_col;
    struct
    {
        int s, i;
    } inner_num;
    struct
    {
        bool activ, weight;
    } hold_in_lmem;
} a16mm_slice_info_t;

typedef struct
{
    int activ_size;
    int weight_size;
    int bias_size;
    int scale_size;
    int zp_size;
    int output_size;
    // the buffer for untransposed output
    int buffer_size1;
    // the buffer for f16 weight data
    int buffer_size2;
    // the buffer for transposed input data
    int buffer_size3;
    // the buffer for unsqueeze int4 data
    // if inner_dim is not sliced, buffer4 shares the same addr with buffer1
    int buffer_size4;
    int half_offset;
    bool load_full_scale;
    int align_size;
} a16mm_size_info_t;

typedef enum
{
    DT_INT8 = (0 << 1) | 1,
    DT_UINT8 = (0 << 1) | 0,
    DT_INT16 = (3 << 1) | 1,
    DT_UINT16 = (3 << 1) | 0,
    DT_FP16 = (1 << 1) | 1,
    DT_BFP16 = (5 << 1) | 1,
    DT_INT32 = (4 << 1) | 1,
    DT_UINT32 = (4 << 1) | 0,
    DT_FP32 = (2 << 1) | 1,
    DT_INT4 = (6 << 1) | 1,
    DT_UINT4 = (6 << 1) | 0,
    DT_FP8E5M2 = (0 << 5) | (7 << 1) | 1,
    DT_FP8E4M3 = (1 << 5) | (7 << 1) | 1,
    DT_FP20 = (8 << 1) | 1,
    DT_TF32 = (9 << 1) | 1,
} data_type_t;

static inline data_type_t tpu_type_convert(DATA_TYPE_T data_type) {
    data_type_t dtype = DT_FP32;
    switch (data_type) {
    case DTYPE_FP32:    dtype = DT_FP32;    break;
    case DTYPE_UINT32:  dtype = DT_UINT32;  break;
    case DTYPE_INT32:   dtype = DT_INT32;   break;
    case DTYPE_FP16:    dtype = DT_FP16;    break;
    case DTYPE_BFP16:   dtype = DT_BFP16;   break;
    case DTYPE_INT16:   dtype = DT_INT16;   break;
    case DTYPE_UINT16:  dtype = DT_UINT16;  break;
    case DTYPE_INT8:    dtype = DT_INT8;    break;
    case DTYPE_UINT8:   dtype = DT_UINT8;   break;
    case DTYPE_INT4:    dtype = DT_INT4;    break;
    case DTYPE_UINT4:   dtype = DT_UINT4;   break;
    default:
        assert(0);
        break;
    }
    return dtype;
}

template <>
LogicalResult WeightReorder<tpu::A16MatMulOp, Float16Type>::matchAndRewriteImpl(
    tpu::A16MatMulOp op, PatternRewriter &rewriter) const {
  if (op.getWeightBits() != 4 || op.getQGroupSize() <= 0) {
    return failure();
  }
  auto scale_stype = module::getStorageType(op.getScale());
  auto scaleOp = op.getScale().getDefiningOp<top::WeightOp>();
  auto zpOp = op.getZp().getDefiningOp<top::WeightOp>();
  auto zp_stype = module::getStorageType(op.getZp());
  auto scale_shape = scaleOp.getType().getShape();
  auto ori_scale_data = scaleOp.read<uint16_t>();
  if (module::isSG2380()) {
    auto ori_zp_data = zpOp.read<uint16_t>();

    int64_t npu_num = backend::Arch::NPU_NUM;
    if (scale_shape[0] % npu_num) {
      llvm_unreachable("invalid scale channel");
    }
    auto w = scale_shape[1];
    auto h = scale_shape[0] / npu_num;

    auto scale_zp_data = std::make_shared<std::vector<uint16_t>>(
        scale_shape[0] * scale_shape[1] * 2, 0);

    for (auto i = 0; i < npu_num; i++) {
      for (auto j = 0; j < h; j++) {
        auto offset_new = 2 * (i * h * w + j * w);
        auto offset_ori = i * w + npu_num * j * w;
        auto scale_zp_data_addr = scale_zp_data->data() + offset_new;
        auto ori_scale_data_addr = ori_scale_data->data() + offset_ori;
        auto ori_zp_data_addr = ori_zp_data->data() + offset_ori;
        for (auto k = 0; k < w; k++) {
          scale_zp_data_addr[2 * k] = ori_zp_data_addr[k];
          scale_zp_data_addr[2 * k + 1] = ori_scale_data_addr[k];
        }
      }
    }

    auto scale_zp_type = RankedTensorType::get({npu_num, h, 2 * w}, scale_stype);
    auto scaleZpOp =
        top::WeightOp::create(op, "reordered_s_zp", *scale_zp_data, scale_zp_type);
    op.setOperand(2, scaleZpOp);
    op.setOperand(3, module::getNoneOp(op));

    /* reorder the weight of a16 matmul */
    // Determine whether to reorder weights based on the split shape, with the
    // splitting method consistent with a16matmul in TPU1686.
    typedef bool (*t_a16mm_data_split_trans)(
        int, int, int, int, bool, bool, bool, int, int, data_type_t,
        data_type_t, a16mm_slice_info_t *, a16mm_size_info_t *);
    t_a16mm_data_split_trans dl_a16mm_data_split_trans;
    auto sg2380_cmodel_handle = dlopen("libcmodel_sg2380.so", RTLD_LAZY);
    assert(sg2380_cmodel_handle != nullptr);
    dl_a16mm_data_split_trans = (t_a16mm_data_split_trans)dlsym(
        sg2380_cmodel_handle, "a16mm_data_split_trans");
    assert(dl_a16mm_data_split_trans != nullptr);

    auto num_core = module::getCoreNum();
    auto input_shape =
        op.getInput().getType().cast<RankedTensorType>().getShape();
    auto weightOp = op.getWeight().getDefiningOp<top::WeightOp>();
    auto weight_shape = weightOp.getType().getShape();
    int input_shape_dim = input_shape.size();
    assert(std::accumulate(input_shape.begin(), input_shape.begin() + (input_shape_dim - 2), 1, std::multiplies<int64_t>()) == 1);
    int final_row_num = input_shape[input_shape_dim - 2];
    int inner_num = input_shape[input_shape_dim - 1];
    int final_col_num = weight_shape[0] / 4;
    bool has_bias = !module::isNone(op.getBias());
    bool has_zp = !module::isNone(op.getZp());
    int q_group_size = op.getQGroupSize();
    int weight_bits = op.getWeightBits();
    data_type_t io_dtype =
        tpu_type_convert(backend::BM168x::getDataType(op.getInput()));
    data_type_t weight_dtype =
        tpu_type_convert(backend::BM168x::getDataType(op.getWeight()));

    a16mm_slice_info_t slice_val = {0};
    a16mm_size_info_t size_info = {0};
    bool activ_trans = final_row_num == 1;
    size_info.load_full_scale = true;

    if (inner_num == 11008) {
        size_info.align_size = backend::BM168x::EU_BYTES;
    }

    bool ret = dl_a16mm_data_split_trans(
        final_row_num, inner_num, final_col_num, backend::Arch::LMEM_BYTES,
        has_bias, has_zp, activ_trans, q_group_size, weight_bits, io_dtype,
        weight_dtype, &slice_val, &size_info);
    if (!ret) {
        size_info.load_full_scale = false;
        ret = dl_a16mm_data_split_trans(
            final_row_num, inner_num, final_col_num, backend::Arch::LMEM_BYTES,
            has_bias, has_zp, activ_trans, q_group_size, weight_bits, io_dtype,
            weight_dtype, &slice_val, &size_info);
    }
    dlclose(sg2380_cmodel_handle);

    // Weights are considered reordered if the following conditions are
    // satisfied:
    // 1. Number of cores is 4.
    // 2. weight_shape[0] is divisible by the number of cores.
    // 3. final_row_num equals 1.
    // 4. inner_num is divisible by slice_val.inner_num.s.
    // 5. The loaded shape (1, slice_n, 1, w) can be decomposed into (n, c, 1,
    // w), and the size of cw is 64k, where c is a multiple of npu_num
    // 6. final_col_num must be divisible by c
    int load_align_size = 64 * 1024;
    int load_weight_w = slice_val.inner_num.s * (weight_bits == 8 ? 1 : 0.5);
    int load_weight_c = load_align_size / load_weight_w;
    int load_weight_n = slice_val.Y_row.s / load_weight_c;

    if (num_core == 4 && weight_shape[0] % num_core == 0 &&
        final_row_num == 1 && final_col_num % load_weight_c == 0 &&
        inner_num % slice_val.inner_num.s == 0 &&
        load_align_size % load_weight_w == 0 && load_weight_c % npu_num == 0 &&
        slice_val.Y_row.s % load_weight_c == 0) {
        int load_weight_n_last =
        ((final_col_num / load_weight_c) % load_weight_n)
            ? ((final_col_num / load_weight_c) % load_weight_n)
            : load_weight_n;
        auto ori_weight_data = weightOp.read<uint8_t>();
        auto weight_data = std::make_shared<std::vector<uint8_t>>(
            weight_shape[0] * weight_shape[1], 0);

        std::vector<int> core_slice_idx = {0, final_col_num, final_col_num * 2,
                                           final_col_num * 3};

        for (int i = 0; i < weight_shape[0] / load_weight_c *
                                (inner_num / slice_val.inner_num.s);
             ++i) {
            int core_id = i % num_core;
            int core_index = i / num_core;
            int slice_row_index = core_index / (load_weight_n * inner_num /
                                                slice_val.inner_num.s);
            int load_weight_n_i =
                ((final_col_num / load_weight_c / load_weight_n ==
                  slice_row_index)
                     ? load_weight_n_last
                     : load_weight_n);
            int slice_inner_index = (core_index % (load_weight_n * inner_num /
                                                   slice_val.inner_num.s)) /
                                    load_weight_n_i;
            int block_row_index = (core_index % (load_weight_n * inner_num /
                                                 slice_val.inner_num.s)) %
                                  load_weight_n_i;
            std::copy_n(
                ori_weight_data->begin() +
                    ((core_slice_idx[core_id] / load_weight_c +
                      block_row_index + slice_inner_index * load_weight_n_i +
                      slice_row_index * inner_num / slice_val.inner_num.s *
                          load_weight_n) *
                     load_align_size),
                load_align_size, weight_data->begin() + i * load_align_size);
        }
        weightOp.update(*weight_data, weight_shape[0] * weight_shape[1]);
    }
  } else {
    auto ori_zp_data = zpOp.read<uint8_t>();

    auto new_scale_data = std::make_shared<std::vector<uint16_t>>(
        scale_shape[0] * scale_shape[1], 0);
    auto new_zp_data = std::make_shared<std::vector<uint8_t>>(
        scale_shape[0] * scale_shape[1], 0);
    int64_t npu_num = backend::Arch::NPU_NUM;
    if (scale_shape[0] % npu_num) {
      llvm_unreachable("invalid scale channel");
    }
    auto w = scale_shape[1];
    auto h = scale_shape[0] / npu_num;

    for (auto i = 0; i < npu_num; i++) {
      for (auto j = 0; j < h; j++) {
        auto offset_new = i * h * w + j * w;
        auto offset_ori = i * w + npu_num * j * w;
        memcpy(new_scale_data->data() + offset_new,
               ori_scale_data->data() + offset_ori, w * sizeof(uint16_t));
        memcpy(new_zp_data->data() + offset_new, ori_zp_data->data() + offset_ori,
               w * sizeof(uint8_t));
      }
    }

    auto new_scale_type = RankedTensorType::get({npu_num, h, w}, scale_stype);
    auto new_zp_type = RankedTensorType::get({npu_num, h, w}, zp_stype);
    auto new_scaleOp =
        top::WeightOp::create(op, "reordered", *new_scale_data, new_scale_type);
    auto new_zpOp =
        top::WeightOp::create(op, "reordered", *new_zp_data, new_zp_type);
    op.setOperand(2, new_scaleOp);
    op.setOperand(3, new_zpOp);
  }
  return success();
}
