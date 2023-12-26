//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Distribute/DistributeUtils.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Distribute/Distribute.h"
#include "tpu_mlir/Backend/Arch.h"

#define DEBUG_TYPE "distribute_ops"

using namespace llvm;
namespace tpu_mlir {
namespace tpu {

/**
 * pos_id => Gather -> (Reshape) -----------
 *          -> Slice ------------           \
 *         /                     \           \
 * Reshape --> Slice -> MulConst --> Concat --> Mul --> Add
 *         \                                         /
 *          -> Mul ----------------------------------
 */
static bool isRotaryEmbed(Operation *op, std::vector<Operation *> &begin_ops) {
  LLVM_DEBUG(llvm::dbgs() << "== Start match RotaryEmbed. ==\n");

  auto add0 = dyn_cast<tpu::AddOp>(op);
  if (!add0) {
    LLVM_DEBUG(llvm::dbgs() << "1. This Op is not AddOp: " << *op << "\n");
    return false;
  }

  auto left_mul = dyn_cast<tpu::MulOp>(add0.getInputs()[0].getDefiningOp());
  if (!isa<tpu::MulOp>(left_mul) || !left_mul->hasOneUse()) {
    LLVM_DEBUG({
      if (!isa<tpu::MulOp>(left_mul)) {
        llvm::dbgs() << "2. This Op is not MulOp: " << *left_mul << "\n";
      } else {
        std::vector<Operation *> users(left_mul->user_begin(),
                                       left_mul->user_end());
        llvm::dbgs() << "2. This MulOp is: " << *left_mul << "\n";
        llvm::dbgs() << "This MulOp has " << users.size() << " users:\n";
        for (auto user : users) {
          llvm::dbgs() << *user << "\n";
        }
      }
    });
    return false;
  }

  auto reshape = left_mul->getOperand(0).getDefiningOp();
  if (!isa<tpu::ReshapeOp>(reshape)) {
    LLVM_DEBUG(llvm::dbgs()
               << "3. This Op is not ReshapeOp: " << *reshape << "\n");
    return false;
  }

  auto left_gather = left_mul->getOperand(1).getDefiningOp();
  if (!isa<tpu::GatherOp>(left_gather)) {
    auto cur_op = left_mul->getOperand(1).getDefiningOp();
    while (isa<tpu::ReshapeOp, tpu::UnsqueezeOp>(cur_op)) {
      cur_op = cur_op->getOperand(0).getDefiningOp();
    }
    left_gather = cur_op;
  }
  if (!isa<tpu::GatherOp>(left_gather)) {
    LLVM_DEBUG(llvm::dbgs()
               << "4. This Op is not GatherOp: " << *left_gather << "\n");
    return false;
  }

  auto right_mul = add0.getInputs()[1].getDefiningOp();
  if (!isa<tpu::MulOp>(right_mul) || !right_mul->hasOneUse()) {
    LLVM_DEBUG({
      if (!isa<tpu::MulOp>(right_mul)) {
        llvm::dbgs() << "5. This Op is not MulOp: " << *right_mul << "\n";
      } else {
        std::vector<Operation *> users(right_mul->user_begin(),
                                       right_mul->user_end());
        llvm::dbgs() << "5. This MulOp is: " << *right_mul << "\n";
        llvm::dbgs() << "This MulOp has " << users.size() << " users:\n";
        for (auto user : users) {
          llvm::dbgs() << *user << "\n";
        }
      }
    });
    return false;
  }

  auto right_gather = right_mul->getOperand(1).getDefiningOp();
  if (!isa<tpu::GatherOp>(right_gather)) {
    auto cur_op = right_mul->getOperand(1).getDefiningOp();
    while (isa<tpu::ReshapeOp, tpu::UnsqueezeOp>(cur_op)) {
      cur_op = cur_op->getOperand(0).getDefiningOp();
    }
    right_gather = cur_op;
  }
  if (!isa<tpu::GatherOp>(right_gather)) {
    LLVM_DEBUG(llvm::dbgs()
               << "6. This Op is not GatherOp: " << *right_gather << "\n");
    return false;
  }

  auto concat = right_mul->getOperand(0).getDefiningOp();
  if (!isa<tpu::ConcatOp>(concat)) {
    LLVM_DEBUG(llvm::dbgs()
               << "7. This Op is not ConcatOp: " << *concat << "\n");
    return false;
  }
  auto in0 = concat->getOperand(0);
  auto in1 = concat->getOperand(1);

  auto mul_const = in0.getDefiningOp();
  if (!isa<tpu::MulConstOp>(mul_const) || !mul_const->hasOneUse()) {
    LLVM_DEBUG({
      if (!isa<tpu::MulConstOp>(mul_const)) {
        llvm::dbgs() << "8. This Op is not MulConstOp: " << *mul_const << "\n";
      } else {
        std::vector<Operation *> users(mul_const->user_begin(),
                                       mul_const->user_end());
        llvm::dbgs() << "8. This MulConstOp is: " << *mul_const << "\n";
        llvm::dbgs() << "This MulConstOp has " << users.size() << " users:\n";
        for (auto user : users) {
          llvm::dbgs() << *user << "\n";
        }
      }
    });
    return false;
  }

  auto slice0 = mul_const->getOperand(0).getDefiningOp();
  if (!isa<tpu::SliceOp>(slice0) || !slice0->hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs()
               << "9. This Op is not SliceOp: " << *slice0 << "\n");
    return false;
  }

  auto slice1 = in1.getDefiningOp();
  if (!isa<tpu::SliceOp>(slice1) || !slice1->hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs()
               << "10. This seocnd input of concat is not form SliceOp: "
               << *slice1 << "\n");
    return false;
  }

  if (reshape != slice0->getOperand(0).getDefiningOp() ||
      reshape != slice1->getOperand(0).getDefiningOp()) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "11. The input of the two SliceOp is not from the reshape\n";
      reshape->dump();
      slice0->dump();
      slice1->dump();
    });
    return false;
  }

  begin_ops.push_back(reshape);
  begin_ops.push_back(left_gather);
  begin_ops.push_back(right_gather);

  LLVM_DEBUG(llvm::dbgs() << "== End match RotaryEmbed. ==\n");
  return true;
}

/**
 * GroupQueryAttention:
 *  (Cast) -> Norm -> (Cast) -> MatMulW -> Reshape
 * Note: Norm is LayerNorm, RMSNorm
 */
static bool isAttentionInput(Operation *op,
                             std::vector<Operation *> &begin_ops) {
  LLVM_DEBUG(llvm::dbgs() << "== Start match AttentionInput. ==\n");
  int num_users = std::distance(op->user_begin(), op->user_end());
  if (!isa<tpu::ReshapeOp>(op) || num_users != 3) {
    LLVM_DEBUG({
      if (!isa<tpu::ReshapeOp>(op)) {
        llvm::dbgs() << "1. This Op is not ReshapeOp: " << *op << "\n";
      } else {
        std::vector<Operation *> users(op->user_begin(), op->user_end());
        llvm::dbgs() << "1. This ReshapeOp is: " << *op << "\n";
        llvm::dbgs() << "This ReshapeOp has " << users.size() << " users:\n";
        for (auto user : users) {
          llvm::dbgs() << *user << "\n";
        }
      }
    });
    return false;
  }

  auto matmul0 = op->getOperand(0).getDefiningOp();
  if (!isa<tpu::MatMulOp>(matmul0) || !matmul0->hasOneUse()) {
    LLVM_DEBUG({
      if (!isa<tpu::MatMulOp>(matmul0)) {
        llvm::dbgs() << "2. This Op is not MatMulOp: " << *matmul0 << "\n";
      } else {
        std::vector<Operation *> users(matmul0->user_begin(),
                                       matmul0->user_end());
        llvm::dbgs() << "2. This MatMulOp is: " << *matmul0 << "\n";
        llvm::dbgs() << "This MatMulOp has " << users.size() << " users:\n";
        for (auto user : users) {
          llvm::dbgs() << *user << "\n";
        }
      }
    });
    return false;
  }

  // LayerNormOp or CastOp
  auto top_op = matmul0->getOperand(0).getDefiningOp();
  if (isa<tpu::CastOp>(top_op)) {
    top_op = top_op->getOperand(0).getDefiningOp();
  }

  if (!isa<tpu::LayerNormOp, tpu::RMSNormOp>(top_op)) {
    LLVM_DEBUG(llvm::dbgs() << "The top_op is not NormOp: " << *top_op << "\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "== End match AttentionInput. ==\n");
  begin_ops.push_back(top_op);
  return true;
}

/**
 * 1. Common Attention:
 *  Norm -> MatMulW -> RotaryEmbed --> MatMul
 * 2. For GroupQueryAttention:
 *  Reshape -> Slice -> RotaryEmbed -> MatMul
 * Note: Norm is LayerNorm, RMSNorm
 */
static bool isAttentionQuery(Operation *op, std::vector<Operation *> &begin_ops,
                             bool GQA) {
  LLVM_DEBUG(llvm::dbgs() << "== Start match AttentionQuery. ==\n");
  if (!isa<tpu::AddOp>(op) || !op->hasOneUse()) {
    LLVM_DEBUG({
      if (!isa<tpu::AddOp>(op)) {
        llvm::dbgs() << "1. This Op is not AddOp: " << *op << "\n";
      } else {
        std::vector<Operation *> users(op->user_begin(), op->user_end());
        llvm::dbgs() << "1. This AddOp is: " << *op << "\n";
        llvm::dbgs() << "This AddOp has " << users.size() << " users:\n";
        for (auto user : users) {
          llvm::dbgs() << *user << "\n";
        }
      }
    });
    return false;
  }

  std::vector<Operation *> rotary_begins;
  if (!isRotaryEmbed(op, rotary_begins)) {
    LLVM_DEBUG(llvm::dbgs() << "2. Failed to match RotaryEmbed.\n");
    return false;
  }

  Operation *top_op = rotary_begins[0]->getOperand(0).getDefiningOp();
  if (GQA) {
    // Reshape -> Slice
    if (!isa<tpu::SliceOp>(top_op) || !top_op->hasOneUse()) {
      LLVM_DEBUG({
        if (!isa<tpu::SliceOp>(top_op)) {
          llvm::dbgs() << "3. This Op is not SliceOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "3. This SliceOp is: " << *top_op << "\n";
          llvm::dbgs() << "This SliceOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
    top_op = top_op->getOperand(0).getDefiningOp();
    int num_users = std::distance(top_op->user_begin(), top_op->user_end());
    if (!isa<tpu::ReshapeOp>(top_op) || num_users != 3) {
      LLVM_DEBUG({
        if (!isa<tpu::ReshapeOp>(top_op)) {
          llvm::dbgs() << "4. This Op is not ReshapeOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "4. This ReshapeOp is: " << *top_op << "\n";
          llvm::dbgs() << "This ReshapeOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
  } else {
    // Norm -> MatMulW
    if (!isLargeMatMul(top_op) || !top_op->hasOneUse()) {
      LLVM_DEBUG({
        if (!isa<tpu::MatMulOp, tpu::A16MatMulOp>(top_op)) {
          llvm::dbgs() << "5. This Op is not MatMulOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "5. This MatMulOp is: " << *top_op << "\n";
          llvm::dbgs() << "This MatMulOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
    top_op = top_op->getOperand(0).getDefiningOp();
    int num_users = std::distance(top_op->user_begin(), top_op->user_end());
    if (!isa<tpu::LayerNormOp, tpu::RMSNormOp>(top_op) || num_users != 3) {
      LLVM_DEBUG({
        if (!isa<tpu::LayerNormOp, tpu::RMSNormOp>(top_op)) {
          llvm::dbgs() << "6. This Op is not NormOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "6. This NormOp is: " << *top_op << "\n";
          llvm::dbgs() << "This NormOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "== End match AttentionQuery. ==\n");
  begin_ops.push_back(top_op);
  begin_ops.push_back(rotary_begins[1]);
  begin_ops.push_back(rotary_begins[2]);

  return true;
}

/**
 * 1. Common Attention
 *  (1) Prefill:
 *   Norm -> MatMulW -> RotaryEmbed ---> MatMul
 *                                  \
 *                                    -> (Cast) => present_k
 *  (2) Decode:
 *                 past_k => (Cast) -
 *                                    \
 *   Norm -> MatMulW -> RotaryEmbed ---> Concat -> MatMul
 *                                  \
 *                                    -> (Cast) => present_k
 * 2. GroupQueryAttention: will have tile before MatMul
 * Note: Norm is LayerNorm, RMSNorm
 */
static bool isAttentionKey(Operation *op, std::vector<Operation *> &begin_ops,
                           std::vector<Operation *> &end_ops, bool GQA) {
  Operation *past_k_op = nullptr;
  Operation *present_k_op;
  LLVM_DEBUG(llvm::dbgs() << "== Start match AttentionKey. ==\n");

  if (GQA) {
    // Reshape -> Tile -> Reshape
    while (isa<tpu::ReshapeOp, tpu::TileOp>(op)) {
      if (!op->hasOneUse()) {
        break;
      }
      op = op->getOperand(0).getDefiningOp();
    }
  }

  if (isa<tpu::ConcatOp>(op)) {
    auto in0 = op->getOperand(0);
    auto op0 = op;
    while (!isa<top::InputOp>(in0.getDefiningOp())) {
      op0 = in0.getDefiningOp();
      in0 = op0->getOperand(0);
    }
    past_k_op = op0;
    op = op->getOperand(1).getDefiningOp();
  }

  if (!op->hasOneUse()) {
    present_k_op = op;
    for (auto user : op->getUsers()) {
      if (isa<tpu::CastOp>(user)) {
        present_k_op = user;
      }
    }
  }

  std::vector<Operation *> rotary_begins;
  if (!isRotaryEmbed(op, rotary_begins)) {
    LLVM_DEBUG(llvm::dbgs() << "1. Failed to match RotaryEmbed.\n");
    return false;
  }

  Operation *top_op = rotary_begins[0]->getOperand(0).getDefiningOp();
  if (GQA) {
    // Reshape -> Slice
    if (!isa<tpu::SliceOp>(top_op) || !top_op->hasOneUse()) {
      LLVM_DEBUG({
        if (!isa<tpu::SliceOp>(top_op)) {
          llvm::dbgs() << "2. This Op is not SliceOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "2. This SliceOp is: " << *top_op << "\n";
          llvm::dbgs() << "This SliceOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
    top_op = top_op->getOperand(0).getDefiningOp();
    int num_users = std::distance(top_op->user_begin(), top_op->user_end());
    if (!isa<tpu::ReshapeOp>(top_op) || num_users != 3) {
      LLVM_DEBUG({
        if (!isa<tpu::ReshapeOp>(top_op)) {
          llvm::dbgs() << "3. This Op is not ReshapeOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "3. This ReshapeOp is: " << *top_op << "\n";
          llvm::dbgs() << "This ReshapeOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
  } else {
    // Norm -> MatMulW
    if (!isLargeMatMul(top_op) || !top_op->hasOneUse()) {
      LLVM_DEBUG({
        if (!isa<tpu::MatMulOp, tpu::A16MatMulOp>(top_op)) {
          llvm::dbgs() << "4. This Op is not MatMulOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "4. This MatMulOp is: " << *top_op << "\n";
          llvm::dbgs() << "This MatMulOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
    top_op = top_op->getOperand(0).getDefiningOp();
    int num_users = std::distance(top_op->user_begin(), top_op->user_end());
    if (!isa<tpu::LayerNormOp, tpu::RMSNormOp>(top_op) || num_users != 3) {
      LLVM_DEBUG({
        if (!isa<tpu::LayerNormOp, tpu::RMSNormOp>(top_op)) {
          llvm::dbgs() << "5. This Op is not NormOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "5. This NormOp is: " << *top_op << "\n";
          llvm::dbgs() << "This NormOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "== End match AttentionKey. ==\n");
  begin_ops.push_back(top_op);
  begin_ops.push_back(rotary_begins[1]);
  begin_ops.push_back(rotary_begins[2]);
  if (past_k_op != nullptr) {
    begin_ops.push_back(past_k_op);
  }
  end_ops.push_back(present_k_op);

  return true;
}

/**
 * attn_mask => (Cast,Reshape) -
 *                              \
 * MatMul -> MulConst -----------> Add -> (Cast) -> Softmax -> (Cast) -> MatMul
 */
static bool isAttentionMatrix(Operation *op,
                              std::vector<Operation *> &begin_ops) {
  LLVM_DEBUG(llvm::dbgs() << "== Start match AttentionMatrix. ==\n");
  if (!isa<tpu::CastOp>(op) || !op->hasOneUse()) {
    LLVM_DEBUG({
      if (!isa<tpu::CastOp>(op)) {
        llvm::dbgs() << "1. This Op is not CastOp: " << *op << "\n";
        llvm::dbgs() << "Softmax still uses fp32 yet, so it needs CastOp.\n";
      } else {
        std::vector<Operation *> users(op->user_begin(), op->user_end());
        llvm::dbgs() << "1. This Op is: " << *op << "\n";
        llvm::dbgs() << "This CastOp has " << users.size() << " users:\n";
        for (auto user : users) {
          llvm::dbgs() << *user << "\n";
        }
      }
    });
    return false;
  }
  Operation *attn_mask_op;
  while (!isa<tpu::MatMulOp>(op)) {
    if (isa<tpu::AddOp>(op)) {
      auto in1 = op->getOperand(1);
      auto op1 = op;
      while (!isa<top::InputOp>(in1.getDefiningOp())) {
        op1 = in1.getDefiningOp();
        in1 = op1->getOperand(0);
      }
      attn_mask_op = op1;
    }
    auto in0 = op->getOperand(0);
    op = in0.getDefiningOp();
  }

  LLVM_DEBUG(llvm::dbgs() << "== End match AttentionMatrix. ==\n");
  begin_ops.push_back(op);
  begin_ops.push_back(attn_mask_op);
  return true;
}

/**
 * 1. Common Attention
 *  (1) Prefill:
 *   Norm -> MatMulW -> Resahpe --> MatMul
 *                             \
 *                               -> (Cast) => present_v
 *
 *  (2) Decode:
 *             past_v => (Cast)
 *                              \
 *   Norm -> MatMulW -> Reshape --> Concat -> MatMul
 *                             \
 *                               -> (Cast) => present_v
 * 2. GroupQueryAttention:
 * Note: Norm is LayerNorm, RMSNorm
 */
static bool isAttentionValue(Operation *op, std::vector<Operation *> &begin_ops,
                             std::vector<Operation *> &end_ops, bool GQA) {
  Operation *past_v_op = nullptr;
  Operation *present_v_op;
  LLVM_DEBUG(llvm::dbgs() << "== Start match AttentionValue. ==\n");

  if (GQA) {
    // Reshape -> Tile -> Reshape
    while (isa<tpu::ReshapeOp, tpu::TileOp>(op)) {
      if (!op->hasOneUse()) {
        break;
      }
      op = op->getOperand(0).getDefiningOp();
    }
  }

  if (isa<tpu::ConcatOp>(op)) {
    auto in0 = op->getOperand(0);
    auto op0 = op;
    while (!isa<top::InputOp>(in0.getDefiningOp())) {
      op0 = in0.getDefiningOp();
      in0 = op0->getOperand(0);
    }
    past_v_op = op0;
    op = op->getOperand(1).getDefiningOp();
  }

  if (!op->hasOneUse()) {
    present_v_op = op;
    for (auto user : op->getUsers()) {
      if (isa<tpu::CastOp>(user)) {
        present_v_op = user;
      }
    }
  }

  Operation *top_op = op->getOperand(0).getDefiningOp();
  if (GQA) {
    if (!isa<tpu::SliceOp>(top_op) || !top_op->hasOneUse()) {
      LLVM_DEBUG({
        if (!isa<tpu::SliceOp>(top_op)) {
          llvm::dbgs() << "1. This Op is not SliceOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "1. This SliceOp is: " << *top_op << "\n";
          llvm::dbgs() << "This SliceOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
    top_op = top_op->getOperand(0).getDefiningOp();
    int num_users = std::distance(top_op->user_begin(), top_op->user_end());
    if (!isa<tpu::ReshapeOp>(top_op) && num_users != 3) {
      LLVM_DEBUG({
        if (!isa<tpu::ReshapeOp>(top_op)) {
          llvm::dbgs() << "2. This Op is not ReshapeOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "2. This ReshapeOp is: " << *top_op << "\n";
          llvm::dbgs() << "This ReshapeOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
  } else {
    if (!isLargeMatMul(top_op) || !top_op->hasOneUse()) {
      LLVM_DEBUG({
        if (!isa<tpu::MatMulOp, tpu::A16MatMulOp>(top_op)) {
          llvm::dbgs() << "3. This Op is not MatMulOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "3. This MatMulOp is: " << *top_op << "\n";
          llvm::dbgs() << "This MatMulOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
    top_op = top_op->getOperand(0).getDefiningOp();
    int num_users = std::distance(top_op->user_begin(), top_op->user_end());
    if (!isa<tpu::RMSNormOp, tpu::LayerNormOp>(top_op) && num_users != 3) {
      LLVM_DEBUG({
        if (!isa<tpu::LayerNormOp, tpu::RMSNormOp>(top_op)) {
          llvm::dbgs() << "4. This Op is not NormOp: " << *top_op << "\n";
        } else {
          std::vector<Operation *> users(top_op->user_begin(),
                                         top_op->user_end());
          llvm::dbgs() << "4. This NormOp is: " << *top_op << "\n";
          llvm::dbgs() << "This NormOp has " << users.size() << " users:\n";
          for (auto user : users) {
            llvm::dbgs() << *user << "\n";
          }
        }
      });
      return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "== End match AttentionValue. ==\n");
  begin_ops.push_back(top_op);
  begin_ops.push_back(past_v_op);
  end_ops.push_back(present_v_op);
  return true;
}

/**
 * Matmul -> Reshape -> MatMulW
 */
static bool isAttentionOutput(Operation *op,
                              std::vector<Operation *> &begin_ops) {
  LLVM_DEBUG(llvm::dbgs() << "== Start match AttentionOutput. ==\n");
  if (!isLargeMatMul(op)) {
    LLVM_DEBUG(llvm::dbgs() << "1. This Op is not MatMulOp: " << *op << "\n");
    return false;
  }
  auto reshape0 = op->getOperand(0).getDefiningOp();
  if (!isa<tpu::ReshapeOp>(reshape0) || !reshape0->hasOneUse()) {
    LLVM_DEBUG({
      if (!isa<tpu::ReshapeOp>(reshape0)) {
        llvm::dbgs() << "2. This Op is not ReshapeOp: " << *reshape0 << "\n";
      } else {
        std::vector<Operation *> users(reshape0->user_begin(),
                                       reshape0->user_end());
        llvm::dbgs() << "2. This ReshapeOp is: " << *reshape0 << "\n";
        llvm::dbgs() << "This ReshapeOp has " << users.size() << " users:\n";
        for (auto user : users) {
          llvm::dbgs() << *user << "\n";
        }
      }
    });
    return false;
  }

  auto mm0 = reshape0->getOperand(0).getDefiningOp();
  auto left0 = mm0->getOperand(0).getDefiningOp();
  auto right0 = mm0->getOperand(1).getDefiningOp();
  if (!isa<tpu::MatMulOp>(mm0) || !mm0->hasOneUse()) {
    LLVM_DEBUG({
      if (!isa<tpu::MatMulOp>(mm0)) {
        llvm::dbgs() << "3. This Op is not MatMulOp of QK@V: " << *mm0 << "\n";
      } else {
        std::vector<Operation *> users(mm0->user_begin(), mm0->user_end());
        llvm::dbgs() << "3. This MatMulOp is: " << *mm0 << "\n";
        llvm::dbgs() << "This MatMulOp has " << users.size() << " users:\n";
        for (auto user : users) {
          llvm::dbgs() << *user << "\n";
        }
      }
    });
    return false;
  }
  if (isa<top::WeightOp>(left0) || isa<top::WeightOp>(right0)) {
    LLVM_DEBUG(llvm::dbgs() << "4. QK@V: Left or right input is weight.\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "== End match AttentionOutput. ==\n");
  begin_ops.push_back(mm0);
  return true;
}

/**
 *                     -> AttentionQuery -
 *                    /                   \
 *  => AttentionInput --> AttentionKey ----> AttentionMatrix --> AttentionOutput
 *                    \                                       /
 *                     -> AttentionValue --------------------
 */
bool isAttentionPattern(Operation *op, std::vector<Operation *> &begin_ops,
                        std::vector<Operation *> &end_ops, bool GQA) {
  LLVM_DEBUG(llvm::dbgs() << "== Start match AttentionPattern. ==\n");
  // record the inputs users during backward
  std::vector<Operation *> query_pos_ids_op;
  std::vector<Operation *> key_pos_ids_op;
  Operation *past_k_op;
  Operation *past_v_op;
  Operation *attn_mask_op;
  // outputs
  Operation *present_k_op;
  Operation *present_v_op;

  std::vector<Operation *> begins;
  std::vector<Operation *> ends;
  // matmul0 is QK_out@value
  if (!isAttentionOutput(op, begins)) {
    LLVM_DEBUG(llvm::dbgs() << "1. Failed to match AttentionOutput.\n");
    return false;
  }
  auto matmul0 = dyn_cast<tpu::MatMulOp>(begins[0]);
  auto left0 = matmul0.getInput().getDefiningOp();
  auto right0 = matmul0.getRight().getDefiningOp();
  begins.clear();

  if (!isAttentionValue(right0, begins, ends, GQA)) {
    LLVM_DEBUG(llvm::dbgs() << "2. Failed to match AttentionValue.\n");
    return false;
  }
  // begins: value_top, [past_v]
  auto value_top_op = begins[0];
  past_v_op = begins.size() > 1 ? begins[1] : nullptr;
  present_v_op = ends[0];
  begins.clear();
  ends.clear();

  if (!isAttentionMatrix(left0, begins)) {
    LLVM_DEBUG(llvm::dbgs() << "3. Failed to match AttentionMatrix.\n");
    return false;
  }
  // begins: matmul, attn_mask
  auto matmul1 = cast<tpu::MatMulOp>(begins[0]);
  auto left1 = matmul1.getInput().getDefiningOp();
  auto right1 = matmul1.getRight().getDefiningOp();
  attn_mask_op = begins[1];
  begins.clear();

  if (!isAttentionKey(right1, begins, ends, GQA)) {
    LLVM_DEBUG(llvm::dbgs() << "4. Failed to match AttentionKey.\n");
    return false;
  }
  // begins: key_top, key_pos0, key_pos1, [past_k]
  auto key_top_op = begins[0];
  key_pos_ids_op.push_back(begins[1]);
  key_pos_ids_op.push_back(begins[2]);
  past_k_op = begins.size() > 3 ? begins[3] : nullptr;
  present_k_op = ends[0];
  begins.clear();
  ends.clear();

  if (!isAttentionQuery(left1, begins, GQA)) {
    LLVM_DEBUG(llvm::dbgs() << "5. Failed to match AttentionQuery.\n");
    return false;
  }
  // begins: query_top, query_pos0, query_pos1
  auto query_top_op = begins[0];
  query_pos_ids_op.push_back(begins[1]);
  query_pos_ids_op.push_back(begins[2]);
  begins.clear();
  if (query_pos_ids_op[0] != key_pos_ids_op[0] ||
      query_pos_ids_op[0] != key_pos_ids_op[0]) {
    LLVM_DEBUG(llvm::dbgs()
               << "6. The pos_ids of Query and Key should be the same\n");
    return false;
  }

  if (query_top_op != key_top_op || query_top_op != value_top_op) {
    LLVM_DEBUG(llvm::dbgs()
               << "7. The Query/Key/Value shares the same top_op.\n");
    return false;
  }

  if (GQA) {
    if (!isAttentionInput(query_top_op, begins)) {
      LLVM_DEBUG(llvm::dbgs() << "8. Failed to match AttentionInput.\n");
      return false;
    }
  } else {
    if (isa<tpu::LayerNormOp, tpu::RMSNormOp>(query_top_op)) {
      begins.push_back(query_top_op);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "8. The top_op QKV should be NormOp.\n");
      return false;
    }
  }
  auto top_op = begins[0];
  begins.clear();

  end_ops.push_back(op);
  end_ops.push_back(present_k_op);
  end_ops.push_back(present_v_op);

  begin_ops.push_back(top_op);
  begin_ops.push_back(query_pos_ids_op[0]);
  begin_ops.push_back(query_pos_ids_op[1]);
  begin_ops.push_back(attn_mask_op);
  if (past_k_op != nullptr) {
    begin_ops.push_back(past_k_op);
    begin_ops.push_back(past_v_op);
  }

  LLVM_DEBUG(llvm::dbgs() << "== End match AttentionPattern. ==\n");
  return true;
}

/**
 * Only one operand is activaton and the others are weight or none,
 * such as GatherOp, LayerNormOp, RMSNormOp
 */
std::vector<Operation *> cloneOpWithWeight(PatternRewriter &rewriter,
                                           Operation *next_op, Value &cur_out,
                                           std::string suffix) {
  rewriter.setInsertionPointAfter(next_op);
  auto new_op = rewriter.clone(*next_op);
  module::setLocSuffix(new_op, suffix);
  std::vector<Value> operands(new_op->operand_begin(), new_op->operand_end());
  for (auto [idx, opd] : llvm::enumerate(operands)) {
    auto src_op = opd.getDefiningOp();
    if (auto weight_op = dyn_cast_or_null<top::WeightOp>(src_op)) {
      auto new_weight = weight_op.clone(suffix);
      new_op->setOperand(idx, new_weight);
    } else if (isa_and_nonnull<top::NoneOp>(src_op)) {
      // pass
    } else {
      new_op->setOperand(idx, cur_out);
    }
  }
  cur_out = new_op->getResult(0);
  std::vector<Operation *> next_ops(next_op->user_begin(), next_op->user_end());
  return next_ops;
}

/**
 * case 1: new op shape is the same as the origin output, pass axis < 0
 * case 2: new op shape is sliced from the origin output , pass axis >= 0
 * only one operand is activation,
 * such as ActiveOp, CastOp, MulConstOp, SoftMaxOp, ReshapeOp, TileOp, SliceOp
 */
std::vector<Operation *> cloneCommonOp(PatternRewriter &rewriter,
                                       Operation *next_op, Value &cur_out,
                                       int axis, int num_devices,
                                       int cur_device) {
  auto suffix = std::to_string(cur_device);
  rewriter.setInsertionPointAfter(next_op);
  std::vector<int64_t> new_shape = module::getShape(next_op->getResult(0));
  if (axis > 0) {
    new_shape[axis] = cur_out.getType().cast<RankedTensorType>().getShape()[axis];
  }

  auto new_op = cloneOp(rewriter, next_op, new_shape, suffix);
  new_op->setOperand(0, cur_out);
  cur_out = new_op->getResult(0);
  std::vector<Operation *> next_ops(next_op->user_begin(), next_op->user_end());
  return next_ops;
}

/**
 * the input/output shape is the same and only one operand is activation,
 * such as ActiveOp, CastOp, MulConstOp, SoftMaxOp
 */
Operation *cloneCommonOp(PatternRewriter &rewriter, Operation *next_op,
                         Value &cur_out, std::string &suffix) {
  rewriter.setInsertionPointAfter(next_op);
  std::vector<int64_t> new_shape = module::getShape(cur_out);
  while (isa<tpu::CastOp, tpu::ActiveOp, tpu::MulConstOp, tpu::SoftmaxOp>(
      next_op)) {
    if (isa<tpu::ReshapeOp, tpu::UnsqueezeOp, tpu::SliceOp, tpu::TileOp>(
            next_op)) {
      new_shape = module::getShape(next_op->getResult(0));
    } else {
      new_shape = module::getShape(cur_out);
    }
    auto new_op = cloneOp(rewriter, next_op, new_shape, suffix);
    new_op->setOperand(0, cur_out);
    cur_out = new_op->getResult(0);
    next_op = *next_op->user_begin();
  }
  return next_op;
}

std::vector<Operation *> cloneCommonAxisOp(PatternRewriter &rewriter,
                                           Operation *next_op, Value &cur_out,
                                           int axis, int num_devices,
                                           int cur_device) {
  std::vector<int64_t> new_shape = module::getShape(next_op->getResult(0));
  new_shape[axis] = new_shape[axis] / num_devices;

  Operation *new_op;
  rewriter.setInsertionPointAfterValue(cur_out);
  if (isa<tpu::ReshapeOp, tpu::SliceOp, tpu::TileOp>(next_op)) {
    new_op = rewriter.clone(*next_op);
    new_op->setOperand(0, cur_out);
    for (auto r : new_op->getResults()) {
      module::setShape(r, new_shape);
    }
  }
  std::string suffix = std::to_string(cur_device);
  module::setLocSuffix(new_op, suffix);

  cur_out = new_op->getResult(0);
  std::vector<Operation *> next_ops(next_op->user_begin(), next_op->user_end());
  return next_ops;
}

// axis = -1: do not slice the shape
Operation *cloneMultiInsOp(PatternRewriter &rewriter, Operation *next_op,
                           Value &cur_out, std::vector<Value> &operands,
                           int axis, int num_devices, int cur_device) {
  auto org_out = next_op->getResult(0);
  auto suffix = std::to_string(cur_device);
  auto new_loc = module::getLocLike(org_out, suffix);
  std::vector<int64_t> new_shape = module::getShape(org_out);
  if (axis >= 0) {
    new_shape[axis] = cur_out.getType().cast<RankedTensorType>().getShape()[axis];
  }
  auto new_type = module::getTypeLike(org_out, new_shape);
  std::vector<NamedAttribute> attrs(next_op->getAttrs().begin(),
                                    next_op->getAttrs().end());
  rewriter.setInsertionPointAfter(next_op);
  Operation *new_op;
  if (isa<tpu::ConcatOp>(next_op)) {
    new_op = rewriter.create<tpu::ConcatOp>(new_loc, new_type, operands, attrs);
  } else if (isa<tpu::AddOp>(next_op)) {
    new_op = rewriter.create<tpu::AddOp>(new_loc, new_type, operands, attrs);
  } else if (isa<tpu::MulOp>(next_op)) {
    new_op = rewriter.create<tpu::MulOp>(new_loc, new_type, operands, attrs);
  } else if (isa<tpu::MatMulOp>(next_op)) {
    operands.push_back(next_op->getOperand(2));
    new_op = rewriter.create<tpu::MatMulOp>(new_loc, new_type, operands, attrs);
  }

  cur_out = new_op->getResult(0);
  next_op = *next_op->user_begin();
  return next_op;
}

std::vector<Operation *> cloneRotaryEmbedOp(PatternRewriter &rewriter,
                                            Operation *next_op, Value &cur_out,
                                            int axis,
                                            std::vector<Value> pos_ids,
                                            int num_devices, int cur_device) {
  auto suffix = std::to_string(cur_device);
  std::vector<int64_t> new_shape = module::getShape(next_op->getResult(0));
  new_shape[axis] = new_shape[axis] / num_devices;
  auto users = next_op->getUsers();
  cloneCommonAxisOp(rewriter, next_op, cur_out, axis, num_devices, cur_device);
  std::vector<Value> cat_operands;
  std::vector<Value> add_operands;
  auto start_out = cur_out;
  for (auto user : users) {
    next_op = user;
    cur_out = start_out;
    if (isa<tpu::SliceOp>(next_op)) {
      next_op = cloneCommonAxisOp(rewriter, next_op, cur_out, axis, num_devices,
                                  cur_device)[0];
    }
    if (isa<tpu::MulConstOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, suffix);
    }

    if (isa<tpu::ConcatOp>(next_op)) {
      cat_operands.push_back(cur_out);
      if (cat_operands.size() < 2) {
        continue;
      }
      if (isa<tpu::MulConstOp>(cur_out.getDefiningOp())) {
        std::swap(cat_operands[0], cat_operands[1]);
      }
      next_op = cloneMultiInsOp(rewriter, next_op, cur_out, cat_operands, axis,
                                num_devices, cur_device);
    }

    if (isa<tpu::MulOp>(next_op)) {
      std::vector<Value> mul_operands{cur_out};
      if (isa<tpu::ConcatOp>(next_op->getOperand(0).getDefiningOp())) {
        mul_operands.push_back(pos_ids[0]);
      } else {
        mul_operands.push_back(pos_ids[1]);
      }
      next_op = cloneMultiInsOp(rewriter, next_op, cur_out, mul_operands, axis,
                                num_devices, cur_device);
    }

    if (isa<tpu::AddOp>(next_op)) {
      add_operands.push_back(cur_out);
      if (add_operands.size() < 2) {
        continue;
      }
      if (!isa<tpu::ConcatOp>(cur_out.getDefiningOp())) {
        std::swap(cat_operands[0], cat_operands[1]);
      }
      cloneMultiInsOp(rewriter, next_op, cur_out, add_operands, axis,
                      num_devices, cur_device);
    }
  }

  std::vector<Operation *> next_ops(next_op->user_begin(), next_op->user_end());
  return next_ops;
}

Operation *cloneColParallelMatMul(PatternRewriter &rewriter, Operation *next_op,
                                  Value &cur_out, int num_devices,
                                  int cur_device) {
  auto suffix = std::to_string(cur_device);
  auto filterOp = next_op->getOperand(1).getDefiningOp<top::WeightOp>();
  auto filterShape = module::getShape(filterOp.getOutput());
  auto num_dims = filterShape.size();

  int64_t wbits = 16;
  bool w_trans = false;
  int q_group_size = 0;
  if (auto mm0 = dyn_cast<tpu::A16MatMulOp>(next_op)) {
    wbits = mm0.getWeightBits();
    w_trans = mm0.getWTranspose();
    q_group_size = mm0.getQGroupSize();
  }

  auto N = w_trans ? filterShape[num_dims - 2] : filterShape[num_dims - 1];
  auto length = ceiling_func(N, num_devices);
  if (q_group_size) {
    auto scale_c = ceiling_func(N, num_devices * backend::Arch::NPU_NUM);
    length = q_group_size * scale_c;
  }
  auto offset = cur_device * length;
  length = std::min(length, N - offset);

  auto out = next_op->getResult(0);
  std::vector<int64_t> new_shape = module::getShape(out);
  new_shape[new_shape.size() - 1] = length;
  auto new_type = module::getTypeLike(out, new_shape);
  auto new_loc = module::getLocLike(out, suffix);

  // slice and clone the weight
  auto new_filter =
      module::opSliceAxis(filterOp, num_dims - 1 - w_trans, offset, length);
  std::vector<Value> operands{cur_out, new_filter};

  // slice and clone the bias
  Value bias = next_op->getOperand(2);
  if (isa<tpu::A16MatMulOp>(next_op)) {
    bias = next_op->getOperand(4);
  }
  Value new_bias = bias;
  if (auto biasOp = dyn_cast<top::WeightOp>(bias.getDefiningOp())) {
    int bias_num_dims = module::getShape(bias).size();
    new_bias = module::opSliceAxis(bias, bias_num_dims - 1, offset, length);
  }

  Operation *new_op;
  rewriter.setInsertionPointAfter(next_op);
  if (auto mm0 = dyn_cast<tpu::MatMulOp>(next_op)) {
    operands.push_back(new_bias);
    auto new_mm0 = rewriter.create<tpu::MatMulOp>(new_loc, new_type, operands,
                                                  mm0->getAttrs());
    new_op = new_mm0.getOperation();
  } else if (auto mm0 = dyn_cast<tpu::A16MatMulOp>(next_op)) {
    // clone the scale
    if (q_group_size) {
      assert(
          module::getShape(mm0.getScale()).size() == 2 &&
          "scale and zp weight reorder should not happen before distribute");
    }

    if (isa<top::WeightOp>(mm0.getScale().getDefiningOp())) {
      auto new_scale = module::opSliceAxis(mm0.getScale(), 0, offset, length);
      operands.push_back(new_scale);
    } else {
      operands.push_back(module::getNoneOp(next_op));
    }
    // clone the zp
    if (isa<top::WeightOp>(mm0.getZp().getDefiningOp())) {
      auto new_zp = module::opSliceAxis(mm0.getZp(), 0, offset, length);
      operands.push_back(new_zp);
    } else {
      operands.push_back(module::getNoneOp(next_op));
    }

    operands.push_back(new_bias);

    auto new_mm0 = rewriter.create<tpu::A16MatMulOp>(new_loc, new_type,
                                                     operands, mm0->getAttrs());
    new_op = new_mm0.getOperation();
  } else {
    llvm_unreachable("This Op should be MatMulOp/A16MatMulOp.\n");
  }

  cur_out = new_op->getResult(0);
  next_op = *next_op->user_begin();
  return next_op;
}

Operation *cloneRowParallelMatMul(PatternRewriter &rewriter, Operation *next_op,
                                  Value &cur_out, int num_devices,
                                  int cur_device) {
  auto suffix = std::to_string(cur_device);
  auto filterOp = next_op->getOperand(1).getDefiningOp<top::WeightOp>();
  auto filterShape = module::getShape(filterOp.getOutput());
  auto num_dims = filterShape.size();

  int64_t wbits = 16;
  bool w_trans = false;
  int q_group_size = 0;
  if (auto mm0 = dyn_cast<tpu::A16MatMulOp>(next_op)) {
    wbits = mm0.getWeightBits();
    w_trans = mm0.getWTranspose();
    q_group_size = mm0.getQGroupSize();
  }

  auto K = filterShape[num_dims - 2 + w_trans];
  auto length = ceiling_func(K, num_devices);
  if (q_group_size) {
    auto scale_w = ceiling_func(2 * K, num_devices * q_group_size);
    length = scale_w * q_group_size / 2;
  }
  auto offset = cur_device * length;
  length = std::min(length, K - offset);

  auto out = next_op->getResult(0);
  auto new_loc = module::getLocLike(out, suffix);
  auto new_type = out.getType();

  // slice and clone the weight
  auto new_filter =
      module::opSliceAxis(filterOp, num_dims - 2 + w_trans, offset, length);
  std::vector<Value> operands{cur_out, new_filter};

  // clone the bias
  Value bias = next_op->getOperand(2);
  if (isa<tpu::A16MatMulOp>(next_op)) {
    bias = next_op->getOperand(4);
  }
  Value new_bias = bias;
  if (auto biasOp = dyn_cast<top::WeightOp>(bias.getDefiningOp())) {
    if (cur_device == 0) {
      new_bias = biasOp.clone(suffix);
    } else {
      new_bias = module::getNoneOp(next_op)->getResult(0);
    }
  }

  Operation *new_op;
  rewriter.setInsertionPointAfter(next_op);
  if (auto mm0 = dyn_cast<tpu::MatMulOp>(next_op)) {
    operands.push_back(new_bias);
    auto new_mm0 = rewriter.create<tpu::MatMulOp>(new_loc, new_type, operands,
                                                  mm0->getAttrs());
    new_op = new_mm0.getOperation();
  } else if (auto mm0 = dyn_cast<tpu::A16MatMulOp>(next_op)) {
    // clone the scale
    int scale_length = 0;
    int scale_offset = 0;
    if (q_group_size) {
      assert(
          module::getShape(mm0.getScale()).size() == 2 &&
          "scale and zp weight reorder should not happen before distribute");
      assert(2 * length % q_group_size == 0);

      scale_length = ceiling_func(2 * length, q_group_size);
      scale_offset = cur_device * scale_length;
    }

    if (isa<top::WeightOp>(mm0.getScale().getDefiningOp())) {
      auto new_scale = q_group_size
                           ? module::opSliceAxis(mm0.getScale(), 1,
                                                 scale_offset, scale_length)
                           : cast<top::WeightOp>(mm0.getScale().getDefiningOp())
                                 .clone(suffix);
      operands.push_back(new_scale);
    }
    // clone the zp
    if (isa<top::WeightOp>(mm0.getZp().getDefiningOp())) {
      auto new_zp =
          q_group_size
              ? module::opSliceAxis(mm0.getZp(), 1, scale_offset, scale_length)
              : cast<top::WeightOp>(mm0.getZp().getDefiningOp());
      operands.push_back(new_zp);
    } else {
      operands.push_back(module::getNoneOp(next_op));
    }

    operands.push_back(new_bias);

    auto new_mm0 = rewriter.create<tpu::A16MatMulOp>(new_loc, new_type,
                                                     operands, mm0->getAttrs());
    new_op = new_mm0.getOperation();
  } else {
    llvm_unreachable("This Op should be MatMulOp/A16MatMulOp.\n");
  }

  cur_out = new_op->getResult(0);
  next_op = *next_op->user_begin();
  return next_op;
}

void createMulConstOp(PatternRewriter &rewriter, Value &cur_out,
                      int num_devices, int cur_device) {
  auto suffix = "mulconst_" + std::to_string(cur_device);
  auto name = module::getName(cur_out);
  std::string new_name = name.str() + "_" + suffix;
  auto new_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfterValue(cur_out);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "const_val", rewriter.getF64FloatAttr(1. / num_devices)));
  auto new_type = cur_out.getType();
  auto new_op = rewriter.create<tpu::MulConstOp>(new_loc, new_type,
                                                 ValueRange{cur_out}, attrs);

  cur_out = new_op->getResult(0);
}

Operation *createSliceOp(PatternRewriter &rewriter, Operation *next_op,
                         Value &cur_out, int axis, int num_devices,
                         int cur_device) {
  auto suffix = "slice_" + std::to_string(cur_device);
  auto name = module::getName(cur_out);
  std::string new_name = name.str() + "_" + suffix;
  auto new_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfterValue(cur_out);
  std::vector<NamedAttribute> attrs;

  std::vector<int64_t> new_shape = module::getShape(cur_out);
  std::vector<int64_t> offset_v(new_shape.size(), 0);
  std::vector<int64_t> step_v(new_shape.size(), 1);
  std::vector<int64_t> end_v = new_shape;
  new_shape[axis] = new_shape[axis] / num_devices;
  offset_v[axis] = cur_device * new_shape[axis];
  end_v[axis] = (1 + cur_device) * new_shape[axis];

  attrs.push_back(
      rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offset_v)));
  attrs.push_back(
      rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(step_v)));
  attrs.push_back(
      rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(end_v)));
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(axis)));
  auto new_type = cur_out.getType();
  auto none = module::getNoneOp(next_op);
  auto new_op = rewriter.create<tpu::SliceOp>(
      new_loc, new_type, ValueRange{cur_out, none, none, none, none}, attrs);
  module::setShape(new_op->getResult(0), new_shape);

  cur_out = new_op->getResult(0);
  return next_op;
}

std::vector<Operation *> cloneAttentionInput(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             int num_devices, int cur_device) {
  auto suffix = std::to_string(cur_device);
  while (isa<tpu::CastOp, tpu::LayerNormOp>(next_op)) {
    if (isa<tpu::CastOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, suffix);
    } else {
      next_op = cloneOpWithWeight(rewriter, next_op, cur_out, suffix)[0];
    }
  }
  next_op = cloneColParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                   cur_device);
  std::vector<Operation *> next_ops =
      cloneCommonAxisOp(rewriter, next_op, cur_out, 2, num_devices, cur_device);
  return next_ops;
}

std::vector<Operation *> cloneAttentionQuery(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             std::vector<Value> &pos_ids,
                                             int num_devices, int cur_device,
                                             bool GQA) {
  if (GQA) {
    next_op = cloneCommonAxisOp(rewriter, next_op, cur_out, 2, num_devices,
                                cur_device)[0];
  } else {
    // clone Query MatMul
    next_op = cloneColParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                     cur_device);
  }
  next_op = cloneRotaryEmbedOp(rewriter, next_op, cur_out, 2, pos_ids,
                               num_devices, cur_device)[0];
  std::vector<Operation *> next_ops{next_op};
  return next_ops;
}

std::vector<Operation *>
cloneAttentionKey(PatternRewriter &rewriter, Operation *next_op, Value &cur_out,
                  std::vector<Value> &other_opds, std::vector<Value> &outs,
                  int num_devices, int cur_device, bool GQA) {
  auto suffix = std::to_string(cur_device);
  if (GQA) {
    // clone SliceOp
    next_op = cloneCommonAxisOp(rewriter, next_op, cur_out, 2, num_devices,
                                cur_device)[0];
  } else {
    // clone Key MatMul
    next_op = cloneColParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                     cur_device);
  }

  std::vector<Value> pos_ids{other_opds[0], other_opds[1]};
  auto next_ops = cloneRotaryEmbedOp(rewriter, next_op, cur_out, 2, pos_ids,
                                     num_devices, cur_device);
  if (!isa<tpu::ReshapeOp, tpu::ConcatOp, tpu::MatMulOp>(next_ops[0])) {
    std::swap(next_ops[0], next_ops[1]);
  }

  // clone output key
  auto key_out = cur_out;
  if (isa<tpu::CastOp>(next_ops[1])) {
    auto key_out_op = cloneCommonOp(rewriter, next_ops[1], key_out, suffix);
    assert(isa<tpu::DistributionEndOp>(key_out_op));
  }

  std::vector<Value> operands;
  Value past_k_out = other_opds[2];
  if (isa<tpu::ConcatOp>(next_ops[0])) {
    operands.push_back(past_k_out);
    operands.push_back(cur_out);
    next_op = cloneMultiInsOp(rewriter, next_ops[0], cur_out, operands, 2,
                              num_devices, cur_device);
  } else {
    next_op = next_ops[0];
  }
  // clone key branch
  if (GQA) {
    auto key_shape = module::getShape(cur_out);
    while (isa<tpu::ReshapeOp, tpu::TileOp>(next_op)) {
      if (key_shape[2] != 1) {
        next_op = cloneCommonAxisOp(rewriter, next_op, cur_out, 2, num_devices,
                                    cur_device)[0];
      } else {
        next_op = *next_op->user_begin();
      }
    }
  }
  Value key_branch = cur_out;

  outs.push_back(key_out);
  outs.push_back(key_branch);

  next_ops = {next_op};
  return next_ops;
}

std::vector<Operation *> cloneAttentionMatrix(PatternRewriter &rewriter,
                                              Operation *next_op,
                                              Value &cur_out,
                                              std::vector<Value> &other_opds,
                                              int num_devices, int cur_device) {
  auto suffix = std::to_string(cur_device);
  next_op = cloneCommonOp(rewriter, next_op, cur_out, suffix);
  // add the score with the mask_float
  std::vector<Value> operands;
  operands.push_back(cur_out);
  operands.push_back(other_opds[0]);
  next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, 2,
                            num_devices, cur_device);
  while (isa<tpu::CastOp, tpu::SoftmaxOp>(next_op)) {
    next_op = cloneCommonOp(rewriter, next_op, cur_out, suffix);
  }
  std::vector<Operation *> next_ops{next_op};
  return next_ops;
}

std::vector<Operation *> cloneAttentionValue(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             std::vector<Value> &other_opds,
                                             std::vector<Value> &outs,
                                             int num_devices, int cur_device,
                                             bool GQA) {
  auto suffix = std::to_string(cur_device);
  if (GQA) {
    // clone SliceOp
    next_op = cloneCommonAxisOp(rewriter, next_op, cur_out, 2, num_devices,
                                cur_device)[0];
  } else {
    // clone Value MatMul
    next_op = cloneColParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                     cur_device);
  }

  // clone RehsapeOp
  auto next_ops =
      cloneCommonAxisOp(rewriter, next_op, cur_out, 2, num_devices, cur_device);
  if (!isa<tpu::ReshapeOp, tpu::ConcatOp, tpu::MatMulOp>(next_ops[0])) {
    std::swap(next_ops[0], next_ops[1]);
  }

  // clone output Value
  Value value_out = cur_out;
  if (isa<tpu::CastOp>(next_ops[1])) {
    auto value_out_op = cloneCommonOp(rewriter, next_ops[1], value_out, suffix);
    assert(isa<tpu::DistributionEndOp>(value_out_op));
  }

  std::vector<Value> operands;
  Value past_v_out = other_opds[0];
  if (isa<tpu::ConcatOp>(next_ops[0])) {
    operands.push_back(past_v_out);
    operands.push_back(cur_out);
    next_op = cloneMultiInsOp(rewriter, next_ops[0], cur_out, operands, 2,
                              num_devices, cur_device);
  } else {
    next_op = next_ops[0];
  }

  // clone value branch
  if (GQA) {
    auto value_shape = module::getShape(cur_out);
    while (isa<tpu::ReshapeOp, tpu::TileOp>(next_op)) {
      if (value_shape[2] != 1) {
        next_op = cloneCommonAxisOp(rewriter, next_op, cur_out, 2, num_devices,
                                    cur_device)[0];
      } else {
        next_op = *next_op->user_begin();
      }
    }
  }
  Value value_branch = cur_out;

  outs.push_back(value_out);
  outs.push_back(value_branch);

  next_ops = {next_op};
  return next_ops;
}

std::vector<Operation *> cloneAttentionOutput(PatternRewriter &rewriter,
                                              Operation *next_op,
                                              Value &cur_out, int num_devices,
                                              int cur_device) {
  // clone ReshapeOp
  next_op = cloneCommonAxisOp(rewriter, next_op, cur_out, 2, num_devices,
                              cur_device)[0];
  next_op = cloneRowParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                   cur_device);

  std::vector<Operation *> next_ops{next_op};
  return next_ops;
}

} // namespace tpu
} // namespace tpu_mlir
