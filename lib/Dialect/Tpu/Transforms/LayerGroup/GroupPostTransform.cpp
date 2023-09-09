//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupPostTransform.h"
#include "tpu_mlir/Support/MathUtils.h"

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
  int64_t IC_PARALLEL = BM168x::ic_num(fmt_bytes);
  std::vector<int64_t> ori_filter_shape = {OC, IC, KT, KH, KW};
  auto ori_type = RankedTensorType::get(ori_filter_shape, filter_type);
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

  } else if (module::isBM1684XFamily() || module::isSG2260Family()) {
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
      // (oc, ic, kt, kh, kw) -> (oc, (ic*kt)/IC_PARALLEL, kh, kw, IC_PARALLEL)
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
      auto new_filter = top::WeightOp::create(op, "postreordered", *filter_u16,
                                              filter_ranked_type);
      op->setOperand(1, new_filter);
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
      // (oc, ic, kt, kh, kw) -> (oc, (ic*kt)/IC_PARALLEL, kh, kw, IC_PARALLEL)
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
class GroupPostTransformPass : public LgPass {
public:
  GroupPostTransformPass() {}
  virtual bool run(LgPassIR *pass_ir) override {
    if (module::isBM1684XFamily() || module::isBM1684Family()
        || module::isSG2260Family()) {
      for (size_t i = 0; i < pass_ir->lg_infos.size(); ++i) {
        _3D_group_post_transform(pass_ir->lg_infos[i]);
        matmul_left_reuse_setting(pass_ir->lg_infos[i]);
      }
    }
    return true;
  }
  virtual std::string name() override { return "GroupPostTransformPass"; }
  virtual std::string brief() override {
    return "Some transform after layer groups is determined";
  }
};

std::unique_ptr<LgPass> CreateGroupPostTransformPass() {
  return std::unique_ptr<LgPass>(new GroupPostTransformPass());
}

} // namespace tpu
} // namespace tpu_mlir
