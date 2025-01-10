//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupPostTransform.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUNnvlcUtil.h"

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {

static int64_t get_shape_size(std::vector<int64_t> shape) {
  if (shape.empty()) {
    return 1;
  } else {
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }
    return size;
  }
}

void conv3d_weight_transform_bm1684(int IC, int OC, int KT, int KH, int KW,
                                    const void *weight_orig,
                                    const void *weight_trans, int method,
                                    int type_bytes) {
  if (type_bytes == 4) {
    if (method == 0) {
      for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
          for (int kt = 0; kt < KT; ++kt) {
            for (int kh = 0; kh < KH; ++kh) {
              for (int kw = 0; kw < KW; ++kw) {
                long long src = oc * (IC * KT * KH * KW) + ic * (KT * KH * KW) +
                                kt * (KH * KW) + kh * KW + kw;
                long long dst = kt * (OC * KW * align_up(IC * KH, 2)) +
                                oc * (KW * align_up(IC * KH, 2)) +
                                kw * align_up(IC * KH, 2) + ic * KH + kh;
                *((float *)weight_trans + dst) = *((float *)weight_orig + src);
              }
            }
          }
        }
      }
    } else if (method == 1) {
      for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
          for (int kt = 0; kt < KT; ++kt) {
            for (int kh = 0; kh < KH; ++kh) {
              for (int kw = 0; kw < KW; ++kw) {
                long long src = oc * (IC * KT * KH * KW) + ic * (KT * KH * KW) +
                                kt * (KH * KW) + kh * KW + kw;
                long long dst = kt * (OC * KH * KW * align_up(IC, 2)) +
                                oc * (KH * KW * align_up(IC, 2)) +
                                kh * (KW * align_up(IC, 2)) +
                                kw * align_up(IC, 2) + ic;
                *((float *)weight_trans + dst) = *((float *)weight_orig + src);
              }
            }
          }
        }
      }
    } else if (method == 2) {
      for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
          for (int kt = 0; kt < KT; ++kt) {
            for (int kh = 0; kh < KH; ++kh) {
              for (int kw = 0; kw < KW; ++kw) {
                long long src = oc * (IC * KT * KH * KW) + ic * (KT * KH * KW) +
                                kt * (KH * KW) + kh * KW + kw;
                long long dst = oc * KH * KW * align_up(IC * KT, 2) +
                                kh * KW * align_up(IC * KT, 2) +
                                kw * align_up(IC * KT, 2) + ic * KT + kt;
                *((float *)weight_trans + dst) = *((float *)weight_orig + src);
              }
            }
          }
        }
      }
    } else {
      llvm_unreachable("wrong conv weight data type");
    }
  } else if (type_bytes == 1) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int ic = 0; ic < IC; ++ic) {
        for (int kt = 0; kt < KT; ++kt) {
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              long long src = oc * (IC * KT * KH * KW) + ic * (KT * KH * KW) +
                              kt * (KH * KW) + kh * KW + kw;
              long long dst = oc * KH * KW * align_up(IC * KT, 4) +
                              kh * KW * align_up(IC * KT, 4) +
                              kw * align_up(IC * KT, 4);
              if (method == 0) {
                dst += (ic * KT + kt);
              } else if (method == 1) {
                dst += (kt * IC + ic);
              } else {
                llvm_unreachable("wrong conv weight data type");
              }
              *((char *)weight_trans + dst) = *((char *)weight_orig + src);
            }
          }
        }
      }
    }
  } else {
    llvm_unreachable("wrong conv weight data type");
  }
}

template <typename T>
void conv3d_stride_gt_15_weightreorder(Operation *op) {
  auto conv3d_op = dyn_cast<tpu::Conv3DOp>(op);
  auto attr = conv3d_op.parseParam();
  auto filter_op = conv3d_op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_type = module::getStorageType(conv3d_op.getFilter());
  // int64_t OC = attr.oc;
  int64_t IC = attr.ic / attr.groups;
  int64_t KT = attr.kd;
  int64_t KH = attr.kh;
  int64_t KW = attr.kw;
  auto data_type = BM168x::getDataType(conv3d_op.getFilter());
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int npu_num = BM168x::NPU_NUM;
  int64_t IC_PARALLEL = BM168x::ic_num(fmt_bytes);
  if (data_type == DTYPE_FP32 || data_type == DTYPE_INT32 ||
      data_type == DTYPE_UINT32) {
    IC_PARALLEL = 1;
  }

  int stride_h = attr.sh;
  int stride_w = attr.sw;
  int groups = attr.groups;
  bool strideh_gt_15 = stride_h > 15;
  bool stridew_gt_15 = stride_w > 15;
  std::vector<int> cell_h;
  std::vector<int> cell_w;
  std::vector<int> cell_h_sum;
  std::vector<int> cell_w_sum;
  int cell_num_h = 1;
  int cell_num_w = 1;
  int max_cell_h = KH;
  int max_cell_w = KW;

  if (strideh_gt_15) {
    cell_num_h = ceiling_func(KH, 15);
    max_cell_h = ceiling_func(KH, cell_num_h);
    int cur_h = 0;
    int sum_h = 0;
    for (int i = 0; i < cell_num_h; i++) {
      cur_h = KH / cell_num_h + ((i < KH % cell_num_h) ? 1 : 0);
      cell_h.push_back(cur_h);
      cell_h_sum.push_back(sum_h);
      sum_h += cur_h;
    }
  } else {
    cell_h.push_back(max_cell_h);
    cell_h_sum.push_back(0);
  }
  if (stridew_gt_15) {
    cell_num_w = ceiling_func(KW, 15);
    max_cell_w = ceiling_func(KW, cell_num_w);
    int cur_w = 0;
    int sum_w = 0;
    for (int i = 0; i < cell_num_w; i++) {
      cur_w = KW / cell_num_w + ((i < KW % cell_num_w) ? 1 : 0);
      cell_w.push_back(cur_w);
      cell_w_sum.push_back(sum_w);
      sum_w += cur_w;
    }
  } else {
    cell_w.push_back(max_cell_w);
    cell_w_sum.push_back(0);
  }
  int oc_per_groups = attr.oc / attr.groups;
  int ocloops = ceiling_func(oc_per_groups, npu_num);
  int weight_size_per_group =
      ((oc_per_groups < npu_num) ? oc_per_groups
                                 : align_up(oc_per_groups, npu_num)) *
      align_up(IC * KT, IC_PARALLEL) * cell_num_h * max_cell_h * cell_num_w *
      max_cell_w;
  int weight_size = attr.groups * weight_size_per_group;
  auto filter_new = std::make_shared<std::vector<T>>(weight_size, 0);
  auto filter_old = filter_op.read<T>();
  std::vector<int64_t> filter_shape(5);
  // Must be initialized to 0. It is to avoid memory increase when bmodel
  // combine.

  // oc
  for (int group_idx = 0; group_idx < attr.groups; group_idx++) {
    for (int oc = 0; oc < oc_per_groups; oc++) {
      // ic
      for (int ic_idx = 0; ic_idx < ceiling_func(IC * KT, IC_PARALLEL);
           ic_idx++) {
        for (int ic_inner = 0; ic_inner < IC_PARALLEL; ic_inner++) {
          // kh
          for (int cell_h_idx = 0; cell_h_idx < cell_num_h; cell_h_idx++) {
            for (int ih = 0; ih < cell_h[cell_h_idx]; ih++) {
              // kw
              for (int cell_w_idx = 0; cell_w_idx < cell_num_w; cell_w_idx++) {
                for (int iw = 0; iw < cell_w[cell_w_idx]; iw++) {
                  if (ic_idx * IC_PARALLEL + ic_inner >= IC * KT)
                    continue;

                  int orig_offset =
                      (group_idx * oc_per_groups + oc) * (IC * KT) * KH * KW +
                      (ic_idx * IC_PARALLEL + ic_inner) * KH * KW +
                      (cell_h_sum[cell_h_idx] + ih) * KW +
                      (cell_w_sum[cell_w_idx] + iw);
                  int trans_offset =
                      (oc % npu_num) * groups * ocloops *
                          align_up(IC * KT, IC_PARALLEL) * cell_num_h *
                          max_cell_h * cell_num_w * max_cell_w + // npu idx
                      group_idx * ocloops * align_up(IC * KT, IC_PARALLEL) *
                          cell_num_h * max_cell_h * cell_num_w *
                          max_cell_w + // group idx
                      (cell_h_idx * cell_num_w + cell_w_idx) * ocloops *
                          max_cell_h * max_cell_w *
                          align_up(IC * KT, IC_PARALLEL) + // cell idx
                      (oc / npu_num) * cell_h[cell_h_idx] * cell_w[cell_w_idx] *
                          align_up(IC * KT, IC_PARALLEL) + // oc offset
                      ic_idx * IC_PARALLEL * cell_h[cell_h_idx] *
                          cell_w[cell_w_idx] + // ic idx
                      (ih * cell_w[cell_w_idx] + iw) * IC_PARALLEL +
                      ic_inner;
                  filter_new->at(trans_offset) = filter_old->at(orig_offset);
                }
              }
            }
          }
        }
      }
    }
  }
  filter_shape[0] = 1;
  filter_shape[1] = (oc_per_groups < npu_num) ? oc_per_groups : npu_num;
  filter_shape[2] = 1;
  filter_shape[3] = 1;
  filter_shape[4] = groups * cell_num_h * cell_num_w * ocloops * max_cell_h *
                    max_cell_w * align_up(IC * KT, IC_PARALLEL);
  if (filter_shape[4] > MAX_TPU_DIM) {
    if (attr.is_dw) {
      filter_shape[3] = ceiling_func(attr.oc, (int64_t)IC_PARALLEL);
      filter_shape[4] /= filter_shape[3];
    } else {
      filter_shape[3] = IC_PARALLEL;
      filter_shape[4] /= IC_PARALLEL;
    }
  }

  filter_old = filter_new;
  auto filter_ranked_type = RankedTensorType::get(filter_shape, filter_type);
  auto new_filter = top::WeightOp::create(op, "postreordered", *filter_old,
                                          filter_ranked_type);
  op->setOperand(1, new_filter);
}

static void conv3d_post_transform(Operation *op, const LgInfo &lg_info) {
  auto conv3d_op = dyn_cast<tpu::Conv3DOp>(op);
  auto attr = conv3d_op.parseParam();
  auto filter_op = conv3d_op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_type = module::getStorageType(conv3d_op.getFilter());
  int64_t OC = attr.oc;
  int64_t IC = attr.ic / attr.groups;
  int64_t KT = attr.kd;
  int64_t KH = attr.kh;
  int64_t KW = attr.kw;
  auto data_type = BM168x::getDataType(conv3d_op.getFilter());
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  // int npu_num = BM168x::NPU_NUM;
  int64_t IC_PARALLEL = BM168x::ic_num(fmt_bytes);
  std::vector<int64_t> ori_filter_shape = {OC, IC, KT, KH, KW};
  auto ori_type = RankedTensorType::get(ori_filter_shape, filter_type);
  auto out_type = module::getStorageType(conv3d_op.getOutput());

  int stride_h = attr.sh;
  int stride_w = attr.sw;
  // int groups = attr.groups;
  bool strideh_gt_15 = stride_h > 15;
  bool stridew_gt_15 = stride_w > 15;

  // conv3d_op.getFilter().setType(ori_type);
  if (attr.has_bias) {
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto bias_type = module::getStorageType(conv3d_op.getBias());
    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    conv3d_op.getBias().setType(new_type);
  }
  if (module::isBM1684Family()) {
    int method = 0;
    if (attr.dd > 1)
      method = 2;
    else if (attr.ic / attr.groups > 10 || attr.dh > 1)
      method = 1;
    if (lg_info.group_ops.size() > 1)
      // local layer only supports method=1 coeff arrange
      method = 1;
    if (filter_type.isF32()) {
      conv3d_op.getFilter().setType(ori_type);
      auto filter_f32 = filter_op.read<float>();
      std::vector<int64_t> filter_shape;
      if (0 == method) {
        filter_shape.assign({KT, OC, KW, align_up((IC * KH), 2), 1});
      } else if (1 == method) {
        if (lg_info.group_ops.size() > 1) {
          // local layer set D = 1 for fit common conv weight load
          filter_shape.assign({KT, OC, 1, KH * KW, align_up(IC, 2)});
        } else {
          filter_shape.assign({KT, OC, KH * KW, align_up(IC, 2), 1});
        }
      } else if (2 == method) {
        filter_shape.assign({OC, KH, KW, align_up(IC * KT, 2), 1});
      }
      auto filter_new =
          std::make_shared<std::vector<float>>(get_shape_size(filter_shape), 0);
      conv3d_weight_transform_bm1684(IC, OC, KT, KH, KW, filter_f32->data(),
                                     filter_new->data(), method, fmt_bytes);
      filter_f32 = filter_new;
      auto filter_ranked_type =
          RankedTensorType::get(filter_shape, filter_type);
      auto new_filter = top::WeightOp::create(op, "postreordered", *filter_f32,
                                              filter_ranked_type);
      op->setOperand(1, new_filter);
    } else if (filter_type.isInteger(8)) {
      // not support now
    } else {
      llvm_unreachable("wrong conv weight data type");
    }

  } else if (module::isBM1684XFamily() || module::isBM1690Family()) {
    conv3d_op.getFilter().setType(ori_type);
    if (filter_type.isF32() && lg_info.group_ops.size() > 1) {
      // (oc, ic, kt, kh, kw) -> (1, oc, kt, ic, kh*kw)
      auto filter_f32 = filter_op.read<float>();
      std::vector<int64_t> filter_shape = {1, OC, KT, IC, KH * KW};
      auto filter_new =
          std::make_shared<std::vector<float>>(get_shape_size(filter_shape), 0);
      for (int64_t oc = 0; oc < OC; oc++) {
        for (int64_t ic = 0; ic < IC; ic++) {
          for (int64_t kt = 0; kt < KT; kt++) {
            for (int64_t khw = 0; khw < KH * KW; khw++) {
              long long src_offset = oc * (IC * KT * KH * KW) +
                                     ic * (KT * KH * KW) + kt * (KH * KW) + khw;
              long long dst_offset = oc * (IC * KT * KH * KW) +
                                     kt * (IC * KH * KW) + ic * (KH * KW) + khw;
              filter_new->at(dst_offset) = filter_f32->at(src_offset);
            }
          }
        }
      }
      filter_f32 = filter_new;
      auto filter_ranked_type =
          RankedTensorType::get(filter_shape, filter_type);
      auto new_filter = top::WeightOp::create(op, "postreordered", *filter_f32,
                                              filter_ranked_type);
      op->setOperand(1, new_filter);
    } else if (filter_type.isF32() && out_type.isF32() &&
               lg_info.group_ops.size() == 1) {
      if (strideh_gt_15 || stridew_gt_15) {
        conv3d_stride_gt_15_weightreorder<float>(op);
      }
    } else if ((filter_type.isF16() || filter_type.isBF16()) &&
               lg_info.group_ops.size() > 1) {
      // (oc, ic, kt, kh, kw) -> (1, oc, kt, ic/IC_PARALLEL, kh*kw *
      // IC_PARALLEL)
      auto filter_u16 = filter_op.read<uint16_t>();
      std::vector<int64_t> filter_shape = {
          1, OC, KT, ceiling_func(IC, IC_PARALLEL), KH * KW * IC_PARALLEL};
      auto filter_new = std::make_shared<std::vector<uint16_t>>(
          get_shape_size(filter_shape), 0);
      for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < ceiling_func(IC, IC_PARALLEL); ++ic) {
          for (int kt = 0; kt < KT; ++kt) {
            for (int khw = 0; khw < KH * KW; ++khw) {
              for (int inner = 0; inner < IC_PARALLEL; ++inner) {
                if (ic * IC_PARALLEL + inner >= IC)
                  break;
                long long src = oc * IC * KT * KH * KW +
                                (ic * IC_PARALLEL + inner) * KT * KH * KW +
                                kt * (KH * KW) + khw;
                long long dst = oc * KT * align_up(IC, IC_PARALLEL) * KH * KW +
                                kt * align_up(IC, IC_PARALLEL) * KH * KW +
                                ic * IC_PARALLEL * KH * KW + khw * IC_PARALLEL +
                                inner;
                filter_new->at(dst) = filter_u16->at(src);
              }
            }
          }
        }
      }
      filter_u16 = filter_new;
      auto filter_ranked_type =
          RankedTensorType::get(filter_shape, filter_type);
      auto new_filter = top::WeightOp::create(op, "postreordered", *filter_u16,
                                              filter_ranked_type);
      op->setOperand(1, new_filter);
    } else if ((filter_type.isF16() || filter_type.isBF16()) &&
               lg_info.group_ops.size() == 1) {
      if (strideh_gt_15 || stridew_gt_15) {
        conv3d_stride_gt_15_weightreorder<uint16_t>(op);
      } else {
        // (oc, ic, kt, kh, kw) -> (oc, (ic*kt)/IC_PARALLEL, kh, kw,
        // IC_PARALLEL)
        auto filter_u16 = filter_op.read<uint16_t>();
        std::vector<int64_t> filter_shape = {
            1, OC, ceiling_func(IC * KT, IC_PARALLEL), KH * KW, IC_PARALLEL};
        auto filter_new = std::make_shared<std::vector<uint16_t>>(
            get_shape_size(filter_shape), 0);
        for (int oc = 0; oc < OC; ++oc) {
          for (int ic = 0; ic < ceiling_func(IC * KT, IC_PARALLEL); ++ic) {
            for (int khw = 0; khw < KH * KW; ++khw) {
              for (int inner = 0; inner < IC_PARALLEL; ++inner) {
                if (ic * IC_PARALLEL + inner >= IC * KT)
                  break;
                long long src = oc * IC * KT * KH * KW +
                                (ic * IC_PARALLEL + inner) * KH * KW + khw;
                long long dst = oc * align_up(IC * KT, IC_PARALLEL) * KH * KW +
                                ic * IC_PARALLEL * KH * KW + khw * IC_PARALLEL +
                                inner;
                filter_new->at(dst) = filter_u16->at(src);
              }
            }
          }
        }
        filter_u16 = filter_new;
        auto filter_ranked_type =
            RankedTensorType::get(filter_shape, filter_type);
        auto new_filter = top::WeightOp::create(
            op, "postreordered", *filter_u16, filter_ranked_type);
        op->setOperand(1, new_filter);
      }
    } else if (filter_type.isInteger(8) && lg_info.group_ops.size() > 1) {
      // (oc, ic, kt, kh, kw) -> (1, oc, kt, ic/IC_PARALLEL, kh*kw *
      // IC_PARALLEL)
      auto filter_i8 = filter_op.read<int8_t>();
      std::vector<int64_t> filter_shape = {
          1, OC, KT, ceiling_func(IC, IC_PARALLEL), KH * KW * IC_PARALLEL};
      auto filter_new = std::make_shared<std::vector<int8_t>>(
          get_shape_size(filter_shape), 0);
      for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < ceiling_func(IC, IC_PARALLEL); ++ic) {
          for (int kt = 0; kt < KT; ++kt) {
            for (int khw = 0; khw < KH * KW; ++khw) {
              for (int inner = 0; inner < IC_PARALLEL; ++inner) {
                if (ic * IC_PARALLEL + inner >= IC)
                  break;
                long long src = oc * IC * KT * KH * KW +
                                (ic * IC_PARALLEL + inner) * KT * KH * KW +
                                kt * (KH * KW) + khw;
                long long dst = oc * KT * align_up(IC, IC_PARALLEL) * KH * KW +
                                kt * align_up(IC, IC_PARALLEL) * KH * KW +
                                ic * IC_PARALLEL * KH * KW + khw * IC_PARALLEL +
                                inner;
                filter_new->at(dst) = filter_i8->at(src);
              }
            }
          }
        }
      }
      filter_i8 = filter_new;
      auto filter_ranked_type =
          RankedTensorType::get(filter_shape, filter_type);
      auto new_filter = top::WeightOp::create(op, "postreordered", *filter_i8,
                                              filter_ranked_type);
      op->setOperand(1, new_filter);
    } else if (filter_type.isInteger(8) && lg_info.group_ops.size() == 1) {
      if (strideh_gt_15 || stridew_gt_15) {
        llvm_unreachable(
            "Currently conv3d uint8/int8 stride>15 is not supported.");
      } else {
        // (oc, ic, kt, kh, kw) -> (oc, (ic*kt)/IC_PARALLEL, kh, kw,
        // IC_PARALLEL)
        auto filter_i8 = filter_op.read<int8_t>();
        std::vector<int64_t> filter_shape = {
            1, OC, ceiling_func(IC * KT, IC_PARALLEL), KH * KW, IC_PARALLEL};
        auto filter_new = std::make_shared<std::vector<int8_t>>(
            get_shape_size(filter_shape), 0);
        for (int oc = 0; oc < OC; ++oc) {
          for (int ic = 0; ic < ceiling_func(IC * KT, IC_PARALLEL); ++ic) {
            for (int khw = 0; khw < KH * KW; ++khw) {
              for (int inner = 0; inner < IC_PARALLEL; ++inner) {
                if (ic * IC_PARALLEL + inner >= IC * KT)
                  break;
                long long src = oc * IC * KT * KH * KW +
                                (ic * IC_PARALLEL + inner) * KH * KW + khw;
                long long dst = oc * align_up(IC * KT, IC_PARALLEL) * KH * KW +
                                ic * IC_PARALLEL * KH * KW + khw * IC_PARALLEL +
                                inner;
                filter_new->at(dst) = filter_i8->at(src);
              }
            }
          }
        }
        filter_i8 = filter_new;
        auto filter_ranked_type =
            RankedTensorType::get(filter_shape, filter_type);
        auto new_filter = top::WeightOp::create(op, "postreordered", *filter_i8,
                                                filter_ranked_type);
        op->setOperand(1, new_filter);
      }
    }
  }
  return;
}

static void _3D_group_post_transform(const LgInfo &lg_info) {
  if (lg_info.group_ops.size() > 1 && lg_info.type == GROUP_NORMAL)
    return;
  for (auto op : lg_info.group_ops) {
    if (isa<tpu::Conv3DOp>(op)) {
      conv3d_post_transform(op, lg_info);
    }
  }
}
/// The pass of group post transform

static void matmul_left_reuse_setting(const LgInfo &lg_info) {
  for (auto op : lg_info.group_ops) {
    if (isa<tpu::MatMulOp>(op)) {
      auto malmul_op = dyn_cast<tpu::MatMulOp>(op);
      auto in = malmul_op.getInput();
      if (in.hasOneUse()) {
        malmul_op.setLeftReuse(0);
      } else {
        malmul_op.setLeftReuse(1);
      }
    }
  }
}

void getCompressParameter(const uint8_t *ibuf, int32_t isz, bool &signedness,
                          bool &zero_guard, mlir::Type dtype, uint8_t &bias0,
                          uint8_t &bias1) {
  assert(!(dtype.isBF16() &&
           signedness)); // WARNING: signedness MUST be 0 as isBfloat16==True

  // cmd_info->is_bfloat16 = isBfloat16;
  if (!dtype.isBF16() && signedness) {
    // two-side circular shift
    int hist[256] = {0};
    for (size_t i = 0; i < isz; i++) {
      hist[ibuf[i]]++;
    }
    int8_t pos_v = 1;
    while (true) {
      if (hist[((uint8_t)pos_v)] == 0) {
        pos_v++;
      } else {
        break;
      }
    }
    bias0 = (pos_v > 1) ? (pos_v - 1) : 0;
    int8_t neg_v = -1;
    while (true) {
      if (hist[(uint8_t)neg_v] == 0) {
        neg_v--;
      } else {
        break;
      }
    }
    bias1 = (neg_v < -1) ? abs(neg_v + 1) : 0;
    signedness = 1;
  }

  if (dtype.isBF16() || dtype.isF16()) {
    // center shift
    int64_t exp_accum = 0;
    auto bf16_in = reinterpret_cast<const uint16_t *>(ibuf);
    size_t inum = (isz >> 1), cnt = 0;
    for (size_t i = 0; i < inum; i++) {
      uint8_t exp = ((bf16_in[i] >> 7) & 0xFF);
      if (exp != 0) {
        exp_accum += exp;
        cnt++;
      }
    }
    if (cnt > 0) {
      bias0 = (uint8_t)((exp_accum / (float)cnt) + 0.5);
    }
    if (dtype.isBF16()) {
      zero_guard = 1;
    } else if (dtype.isF16()) {
      zero_guard = (inum == cnt) ? 0 : 1;
    }
    signedness = 0;
  }
}

static void nnvlc_transform(const LgInfo &lg_info) {
  if (lg_info.group_ops.size() < 2)
    return;
  for (auto op : lg_info.group_ops) {
    if (isa<tpu::MatMulOp>(op) &&
        !module::getStorageType(op->getResult(0)).isF32()) {
      for (int idx = 0, sz = op->getOperands().size(); idx < sz; idx++) {
        if (isa<top::WeightOp>(op->getOperand(idx).getDefiningOp()) &&
            (module::getStorageType(op->getOperand(idx)).isF16() ||
             module::getStorageType(op->getOperand(idx)).isBF16() ||
             module::getStorageType(op->getOperand(idx)).isInteger(8))) {
          auto right_op = op->getOperand(idx).getDefiningOp<top::WeightOp>();
          auto right_type = module::getStorageType(op->getOperand(idx));
          auto right_shape = right_op.getType().getShape();

          uint8_t bias0 = right_type.isBF16() ? 127 : 0;
          uint8_t bias1;
          bool zero_guard = right_type.isInteger(8) ? 0 : 1;
          bool is_signed = right_type.isBF16() ? 0 : 1;

          if (right_type.isF16() || right_type.isBF16()) {
            auto right_u16 = right_op.read<uint16_t>();
            size_t length = right_shape.size();
            int64_t weight_size = 2;
            for (size_t i = 0; i < length; ++i) {
              weight_size *= right_shape[i];
            }
            int32_t osize;
            getCompressParameter(reinterpret_cast<uint8_t *>(right_u16->data()),
                                 weight_size, is_signed, zero_guard, right_type,
                                 bias0, bias1);
            auto nnvlc_results = nnvlc_encode(
                reinterpret_cast<uint8_t *>(right_u16->data()), weight_size,
                right_type, bias0, bias1, is_signed, zero_guard, osize);
            bool do_compress = std::get<0>(nnvlc_results);
            if (do_compress) {
              uint8_t *obuf = std::get<1>(nnvlc_results);
              auto new_type = RankedTensorType::get(right_shape, right_type);
              auto data = std::vector<uint16_t>(osize / 2);
              memcpy(data.data(), obuf, osize);
              delete[] obuf;
              auto new_op =
                  top::WeightOp::create(op, "right_nnvlc", data, new_type);
              op->setOperand(idx, new_op);
              auto ctx = op->getContext();
              auto builder = OpBuilder(ctx);
              auto rightOp_new =
                  op->getOperand(idx).getDefiningOp<top::WeightOp>();
              rightOp_new->setAttr("do_compress",
                                   builder.getBoolAttr(do_compress));
              rightOp_new->setAttr("bias0",
                                   builder.getI64IntegerAttr((uint64_t)bias0));
              rightOp_new->setAttr("bias1",
                                   builder.getI64IntegerAttr((uint64_t)bias1));
              rightOp_new->setAttr("is_signed", builder.getBoolAttr(is_signed));
              rightOp_new->setAttr("zero_guard",
                                   builder.getBoolAttr(zero_guard));
            } else {
              auto ctx = op->getContext();
              auto builder = OpBuilder(ctx);
              auto rightOp_new =
                  op->getOperand(idx).getDefiningOp<top::WeightOp>();
              rightOp_new->setAttr("do_compress",
                                   builder.getBoolAttr(do_compress));
            }
          } else if (right_type.isInteger(8)) {
            auto filter_i8 = right_op.read<int8_t>();
            size_t length = right_shape.size();
            int64_t weight_size = 1;
            for (size_t i = 0; i < length; ++i) {
              weight_size *= right_shape[i];
            }
            int32_t osize;
            getCompressParameter(reinterpret_cast<uint8_t *>(filter_i8->data()),
                                 weight_size, is_signed, zero_guard, right_type,
                                 bias0, bias1);
            auto nnvlc_results = nnvlc_encode(
                reinterpret_cast<uint8_t *>(filter_i8->data()), weight_size,
                right_type, bias0, bias1, is_signed, zero_guard, osize);
            bool do_compress = std::get<0>(nnvlc_results);
            if (do_compress) {
              uint8_t *obuf = std::get<1>(nnvlc_results);
              auto new_type = RankedTensorType::get(right_shape, right_type);
              auto data = std::vector<int8_t>(osize);
              memcpy(data.data(), obuf, osize);
              delete[] obuf;
              auto new_op =
                  top::WeightOp::create(op, "right_nnvlc", data, new_type);
              op->setOperand(idx, new_op);
              auto ctx = op->getContext();
              auto builder = OpBuilder(ctx);
              auto rightOp_new =
                  op->getOperand(idx).getDefiningOp<top::WeightOp>();
              rightOp_new->setAttr("do_compress",
                                   builder.getBoolAttr(do_compress));
              rightOp_new->setAttr("bias0",
                                   builder.getI64IntegerAttr((uint64_t)bias0));
              rightOp_new->setAttr("bias1",
                                   builder.getI64IntegerAttr((uint64_t)bias1));
              rightOp_new->setAttr("is_signed", builder.getBoolAttr(is_signed));
              rightOp_new->setAttr("zero_guard",
                                   builder.getBoolAttr(zero_guard));
            } else {
              auto ctx = op->getContext();
              auto builder = OpBuilder(ctx);
              auto rightOp_new =
                  op->getOperand(idx).getDefiningOp<top::WeightOp>();
              rightOp_new->setAttr("do_compress",
                                   builder.getBoolAttr(do_compress));
            }
          }
        }
      }
    }

    if (isa<tpu::Conv2DOp>(op) &&
        op->getAttr("use_3ic_optimize").cast<IntegerAttr>().getInt() == 0 &&
        !module::getStorageType(op->getResult(0)).isF32()) {
      uint32_t idx;
      if (isa<top::WeightOp>(op->getOperand(1).getDefiningOp()) &&
          !module::getStorageType(op->getOperand(1)).isF32()) {
        idx = 1;
      } else if (isa<top::WeightOp>(op->getOperand(0).getDefiningOp()) &&
                 !module::getStorageType(op->getOperand(0)).isF32()) {
        idx = 0;
      }
      auto filter_op = op->getOperand(idx).getDefiningOp<top::WeightOp>();
      auto filter_type = module::getStorageType(op->getOperand(idx));
      auto filter_shape = filter_op.getType().getShape();

      uint8_t bias0 = filter_type.isBF16() ? 127 : 0;
      uint8_t bias1;
      bool zero_guard = filter_type.isInteger(8) ? 0 : 1;
      bool is_signed = filter_type.isBF16() ? 0 : 1;

      if (filter_type.isF16() || filter_type.isBF16()) {
        auto filter_u16 = filter_op.read<uint16_t>();
        int32_t weight_size = filter_shape[0] * filter_shape[1] *
                              filter_shape[2] * filter_shape[3] * 2;
        int32_t osize;
        getCompressParameter(reinterpret_cast<uint8_t *>(filter_u16->data()),
                             weight_size, is_signed, zero_guard, filter_type,
                             bias0, bias1);
        auto nnvlc_results = nnvlc_encode(
            reinterpret_cast<uint8_t *>(filter_u16->data()), weight_size,
            filter_type, bias0, bias1, is_signed, zero_guard, osize);
        bool do_compress = std::get<0>(nnvlc_results);
        if (do_compress) {
          uint8_t *obuf = std::get<1>(nnvlc_results);
          auto new_type = RankedTensorType::get(filter_shape, filter_type);
          auto data = std::vector<uint16_t>(osize / 2);
          memcpy(data.data(), obuf, osize);
          delete[] obuf;
          auto new_op =
              top::WeightOp::create(op, "filter_nnvlc", data, new_type);
          op->setOperand(idx, new_op);
          auto ctx = op->getContext();
          auto builder = OpBuilder(ctx);
          auto filterOp_new =
              op->getOperand(idx).getDefiningOp<top::WeightOp>();
          filterOp_new->setAttr("do_compress",
                                builder.getBoolAttr(do_compress));
          filterOp_new->setAttr("bias0",
                                builder.getI64IntegerAttr((uint64_t)bias0));
          filterOp_new->setAttr("bias1",
                                builder.getI64IntegerAttr((uint64_t)bias1));
          filterOp_new->setAttr("is_signed", builder.getBoolAttr(is_signed));
          filterOp_new->setAttr("zero_guard", builder.getBoolAttr(zero_guard));
        } else {
          auto ctx = op->getContext();
          auto builder = OpBuilder(ctx);
          auto filterOp_new =
              op->getOperand(idx).getDefiningOp<top::WeightOp>();
          filterOp_new->setAttr("do_compress",
                                builder.getBoolAttr(do_compress));
        }
      } else if (filter_type.isInteger(8)) {
        auto filter_i8 = filter_op.read<int8_t>();
        int32_t weight_size = filter_shape[0] * filter_shape[1] *
                              filter_shape[2] * filter_shape[3];
        int32_t osize;
        getCompressParameter(reinterpret_cast<uint8_t *>(filter_i8->data()),
                             weight_size, is_signed, zero_guard, filter_type,
                             bias0, bias1);
        auto nnvlc_results = nnvlc_encode(
            reinterpret_cast<uint8_t *>(filter_i8->data()), weight_size,
            filter_type, bias0, bias1, true, zero_guard, osize);
        bool do_compress = std::get<0>(nnvlc_results);
        if (do_compress) {
          uint8_t *obuf = std::get<1>(nnvlc_results);
          auto new_type = RankedTensorType::get(filter_shape, filter_type);
          auto data = std::vector<int8_t>(osize);
          memcpy(data.data(), obuf, osize);
          delete[] obuf;
          auto new_op =
              top::WeightOp::create(op, "filter_reorderd", data, new_type);
          op->setOperand(idx, new_op);
          auto ctx = op->getContext();
          auto builder = OpBuilder(ctx);
          auto filterOp_new =
              op->getOperand(idx).getDefiningOp<top::WeightOp>();
          filterOp_new->setAttr("do_compress",
                                builder.getBoolAttr(do_compress));
          filterOp_new->setAttr("bias0",
                                builder.getI64IntegerAttr((uint64_t)bias0));
          filterOp_new->setAttr("bias1",
                                builder.getI64IntegerAttr((uint64_t)bias1));
          filterOp_new->setAttr("is_signed", builder.getBoolAttr(is_signed));
          filterOp_new->setAttr("zero_guard", builder.getBoolAttr(zero_guard));
        } else {
          auto ctx = op->getContext();
          auto builder = OpBuilder(ctx);
          auto filterOp_new =
              op->getOperand(idx).getDefiningOp<top::WeightOp>();
          filterOp_new->setAttr("do_compress",
                                builder.getBoolAttr(do_compress));
        }
      }
    }
  }
}

class GroupPostTransformPass : public LgPass {
public:
  GroupPostTransformPass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    if (module::isBM1684XFamily() || module::isBM1684Family() ||
        module::isBM1690Family()) {
      for (size_t i = 0; i < pass_ir->lg_infos.size(); ++i) {
        _3D_group_post_transform(pass_ir->lg_infos[i]);
        matmul_left_reuse_setting(pass_ir->lg_infos[i]);
        if (module::isBM1688() && (options_.nnvlc_mode == NnvlcMode::WEIGHT ||
                                   options_.nnvlc_mode == NnvlcMode::ALL)) {
          nnvlc_transform(pass_ir->lg_infos[i]);
        }
      }
    }
    return true;
  }
  virtual std::string name() override { return "GroupPostTransformPass"; }
  virtual std::string brief() override {
    return "Some transform after layer groups is determined";
  }
};

std::unique_ptr<LgPass> CreateGroupPostTransformPass(const LgOptions &options) {
  return std::unique_ptr<LgPass>(new GroupPostTransformPass(options));
}

} // namespace tpu
} // namespace tpu_mlir
