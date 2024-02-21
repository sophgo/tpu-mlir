//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/ConvertTopToTpu.h"
namespace tpu_mlir {

void ConvertTopToTpu::match_swin_mlp(std::vector<Operation *> &mlp) {
  mainFunc_.walk([&](Operation *op) {
    if (auto addop = dyn_cast<top::AddOp>(op)) {
      top::AddOp aop = NULL;
      top::MatMulOp mmop = NULL;
      top::ReshapeOp rsop = NULL;
      for (auto in : addop.getOperands()) {
        if (isa<top::MatMulOp>(in.getDefiningOp()))
          mmop = dyn_cast_or_null<top::MatMulOp>(in.getDefiningOp());
        else if (isa<top::AddOp>(in.getDefiningOp()))
          aop = dyn_cast_or_null<top::AddOp>(in.getDefiningOp());
        else if (isa<top::ReshapeOp>(in.getDefiningOp()))
          rsop = dyn_cast_or_null<top::ReshapeOp>(in.getDefiningOp());
        else
          return;
      }
      if (mmop == NULL || (aop == NULL && rsop == NULL) ||
          !isSISO(mmop.getOperation()))
        return;
      if (isa<top::GELUOp>(mmop.getInput().getDefiningOp())) {
        auto geluop =
            dyn_cast_or_null<top::GELUOp>(mmop.getInput().getDefiningOp());
        if (!isSISO(geluop.getOperation()))
          return;
        if (isa<top::MatMulOp>(geluop.getInput().getDefiningOp())) {
          auto mmop1 = dyn_cast_or_null<top::MatMulOp>(
              geluop.getInput().getDefiningOp());
          if (isa<top::LayerNormOp>(mmop1.getInput().getDefiningOp())) {
            auto lnop = dyn_cast_or_null<top::LayerNormOp>(
                mmop1.getInput().getDefiningOp());
            if ((lnop.getInput().getDefiningOp() != aop &&
                 lnop.getInput().getDefiningOp() != rsop) ||
                lnop == NULL || !isSISO(lnop.getOperation())) {
              return;
            }
          }
          if (!mmop.getOutput().hasOneUse() ||
              !geluop.getOutput().hasOneUse() || !mmop1.getOutput().hasOneUse())
            return;
          mlp.push_back(addop);
          return;
        }
      }
    }
  });
}

void ConvertTopToTpu::match_swin_wmsa(std::vector<Operation *> &wmsa,
                                      std::vector<Operation *> &sm_ops) {
  mainFunc_.walk([&](Operation *op) {
    if (isa<top::ReshapeOp>(op) || isa<top::MatMulOp>(op) ||
        isa<top::LayerNormOp>(op)) {
      auto rsop = dyn_cast_or_null<top::ReshapeOp>(op);
      auto mmop = dyn_cast_or_null<top::MatMulOp>(op);
      auto lnop_ = dyn_cast_or_null<top::LayerNormOp>(op);
      top::LayerNormOp lnop = NULL;
      top::AddOp aop = NULL;
      if (rsop != NULL) {
        for (auto user : rsop.getOutput().getUsers()) {
          if (isa<top::LayerNormOp>(user)) {
            lnop = dyn_cast<top::LayerNormOp>(user);
          } else if (isa<top::AddOp>(user)) {
            aop = dyn_cast<top::AddOp>(user);
          }
        }
      } else if (mmop != NULL) {
        if (std::distance(mmop.getResult().user_begin(),
                          mmop.getResult().user_end()) == 1 &&
            isa<top::AddOp>(*(mmop.getResult().user_begin()))) {
          auto aop_ = dyn_cast<top::AddOp>(*(mmop.getResult().user_begin()));
          for (auto user : aop_.getOutput().getUsers()) {
            if (isa<top::LayerNormOp>(user)) {
              lnop = dyn_cast<top::LayerNormOp>(user);
            } else if (isa<top::AddOp>(user)) {
              aop = dyn_cast<top::AddOp>(user);
            }
          }
        } else {
          for (auto user : mmop.getOutput().getUsers()) {
            if (isa<top::LayerNormOp>(user)) {
              lnop = dyn_cast<top::LayerNormOp>(user);
            } else if (isa<top::AddOp>(user)) {
              aop = dyn_cast<top::AddOp>(user);
            }
          }
        }
      } else if (lnop_ != NULL) {
        for (auto user : lnop_.getOutput().getUsers()) {
          if (isa<top::LayerNormOp>(user)) {
            lnop = dyn_cast<top::LayerNormOp>(user);
          } else if (isa<top::AddOp>(user)) {
            aop = dyn_cast<top::AddOp>(user);
          }
        }
      } else
        return;
      if (lnop == NULL || aop == NULL)
        return;
#if 0
      if (!isSISO(lnop.getOperation()))
        return;
#else
      for (auto u : lnop.getResult().getUsers()) {
        // some dirty slices left after some passes took effect before lowerring
        if (isa<top::SliceOp>(u) &&
            std::distance(u->getResult(0).getUsers().begin(),
                          u->getResult(0).getUsers().end()) == 0)
          continue;
        else if (isa<top::ReshapeOp>(u) || isa<top::SwapDimInnerOp>(u))
          continue;
        else
          return;
      }
#endif
      if (!convergence(lnop, aop))
        return;
      if (isa<top::ReshapeOp>(*(lnop.getResult().getUsers().begin())) ||
          isa<top::SwapDimInnerOp>(*(lnop.getResult().getUsers().begin()))) {
        auto rsop1 = dyn_cast_or_null<top::ReshapeOp>(
            *(lnop.getResult().getUsers().begin()));
        auto swdiop = dyn_cast_or_null<top::SwapDimInnerOp>(
            *(lnop.getResult().getUsers().begin()));

        if (swdiop != NULL) {
          for (auto u : swdiop.getResult().getUsers()) {
            if (isa<top::SwapDimInnerOp>(u)) {
              swdiop = dyn_cast_or_null<top::SwapDimInnerOp>(u);
              break;
            } else if (isa<top::SliceOp>(u) &&
                       std::distance(u->getResult(0).getUsers().begin(),
                                     u->getResult(0).getUsers().end()) == 0)
              continue;
            else
              return;
          }
        }

        top::ReshapeOp rsop2 = NULL;
        if (rsop1 != NULL)
          rsop2 = rsop1;
        else if (swdiop != NULL)
          rsop2 = dyn_cast_or_null<top::ReshapeOp>(
              *(swdiop.getResult().getUsers().begin()));
        if (rsop2 == NULL)
          return;
        if (isSISO(rsop2.getOperation()) &&
            isa<top::PermuteOp>(*(rsop2.getResult().getUsers().begin()))) {
          auto p =
              dyn_cast<top::PermuteOp>(*(rsop2.getResult().getUsers().begin()));
          if (isa<top::ReshapeOp>(*(p.getResult().getUsers().begin())))
            rsop2 =
                dyn_cast<top::ReshapeOp>(*(p.getResult().getUsers().begin()));
          else
            return;
        }
        if ((std::distance(rsop2.getOutput().user_begin(),
                           rsop2.getOutput().user_end()) != 3) &&
            (std::distance(rsop2.getOutput().user_begin(),
                           rsop2.getOutput().user_end()) !=
             4)) // chip opt may split matmul to 3, but left the original matmul
                 // not removed
          return;
        top::MatMulOp mmop_[3] = {NULL};
        top::ReshapeOp rsop_[3] = {NULL};
        top::PermuteOp pmop_[3] = {NULL};
        top::MulConstOp mcop_ = NULL;
        top::MatMulOp mmop1 = NULL;
        top::MatMulOp mmop2 = NULL;
        int idx = 0;
        // seems the order or qkv is not fixed in the patterns, try to find them
        // out, 0 mulconst after permute, 1 matmul 1, 2 is matmul after softmax
        for (auto u : rsop2.getResult().getUsers()) {
          if (auto mmop_tmp = dyn_cast<top::MatMulOp>(u)) {
            if (!isSISO(mmop_tmp.getOperation()))
              return;
            if (auto rsop_tmp = dyn_cast<top::ReshapeOp>(
                    *(mmop_tmp.getResult().user_begin()))) {
              if (rsop_tmp.getResult()
                      .getUsers()
                      .empty()) // the original matmul
                continue;
              if (!isSISO(rsop_tmp))
                return;
              if (auto pmop = dyn_cast<top::PermuteOp>(
                      *(rsop_tmp.getResult().user_begin()))) {
                if (!isSISO(pmop))
                  return;
                if (isa<top::MulConstOp>(*(pmop.getResult().user_begin()))) {
                  mcop_ = dyn_cast<top::MulConstOp>(
                      *(pmop.getResult().user_begin()));
                  if (!isSISO(mcop_))
                    return;
                  idx = 0;
                } else if (isa<top::PermuteOp>(
                               *(pmop.getResult().user_begin()))) {
                  pmop = dyn_cast<top::PermuteOp>(
                      *(pmop.getResult().user_begin()));
                  idx = 1;
                } else if (isa<top::MatMulOp>(
                               *(pmop.getResult().user_begin()))) {
                  auto mmop_tmp1 =
                      dyn_cast<top::MatMulOp>(*(pmop.getResult().user_begin()));
                  if (isa<top::AddOp>(*(mmop_tmp1.getResult().user_begin()))) {
                    idx = 1;
                  } else if (isa<top::PermuteOp>(
                                 *(mmop_tmp1.getResult().user_begin()))) {
                    idx = 2;
                  } else {
                    //(*(mmop_tmp1.getResult().user_begin()))->dump();
                    return;
                  }
                } else {
                  return;
                }
                mmop_[idx] = mmop_tmp;
                rsop_[idx] = rsop_tmp;
                pmop_[idx] = pmop;
              } else {
                return;
              }
            } else {
              return;
            }
          } else {
            return;
          }
        }
        if (pmop_[0] == NULL || pmop_[1] == NULL || pmop_[2] == NULL)
          return;

        if (mcop_ != NULL) {
          if (!isa<top::MatMulOp>(*(mcop_.getResult().user_begin())) ||
              !isa<top::MatMulOp>(*(pmop_[1].getResult().user_begin())) ||
              (*mcop_.getResult().user_begin() !=
               *pmop_[1].getResult().user_begin()))
            return;
        } else {
          return;
        }

        mmop1 = dyn_cast<top::MatMulOp>(*(pmop_[1].getResult().user_begin()));
        auto addop1 =
            dyn_cast_or_null<top::AddOp>(*(mmop1.getResult().user_begin()));
        if (addop1 == NULL)
          return;
        top::SoftmaxOp smop = dyn_cast_or_null<top::SoftmaxOp>(
            *(addop1.getOutput().getUsers().begin()));
        if (smop == NULL)
          return;
        sm_ops.push_back(smop.getOperation());
        if (*(smop.getOutput().getUsers().begin()) !=
            *(pmop_[2].getResult().user_begin()))
          return;
        mmop2 =
            dyn_cast_or_null<top::MatMulOp>(*(smop.getResult().user_begin()));
        if (mmop2 == NULL)
          return;
        if (auto pmop1 =
                dyn_cast<top::PermuteOp>(*(mmop2.getResult().user_begin()))) {
          if (auto rsop3 =
                  dyn_cast<top::ReshapeOp>(*(pmop1.getResult().user_begin()))) {
            if (auto mmop3 = dyn_cast<top::MatMulOp>(
                    *(rsop3.getResult().user_begin()))) {
              if (auto rsop4 = dyn_cast<top::ReshapeOp>(
                      *(mmop3.getResult().user_begin()))) {
                if (auto pmop2 = dyn_cast<top::PermuteOp>(
                        *(rsop4.getResult().user_begin()))) {
                  if (auto rsop5 = dyn_cast<top::ReshapeOp>(
                          *(pmop2.getResult().user_begin()))) {
                    if (auto swdi2 = dyn_cast<top::SwapDimInnerOp>(
                            *(rsop5.getResult().user_begin()))) {
                      for (auto u : swdi2.getResult().getUsers()) {
                        if (isa<top::SwapDimInnerOp>(u)) {
                          swdi2 = dyn_cast_or_null<top::SwapDimInnerOp>(u);
                          break;
                        } else {
                          ; // u->dump();
                        }
                      }
                      if (*(swdi2.getResult().user_begin()) == aop) {
                        if (rsop != NULL) {
                          wmsa.push_back(rsop);
                        } else if (mmop != NULL) {
                          wmsa.push_back(mmop);
                        } else if (lnop_ != NULL) {
                          wmsa.push_back(lnop_);
                        } else
                          return;
                      } else
                        return;
                    } else if (*(rsop5.getResult().user_begin()) ==
                               aop) { // maybe no swdi2
                      if (rsop != NULL) {
                        wmsa.push_back(rsop);
                      } else if (mmop != NULL) {
                        wmsa.push_back(mmop);
                      } else if (lnop_ != NULL) {
                        wmsa.push_back(lnop_);
                      } else
                        return;
                    } else
                      return;
                  } else
                    return;
                } else if (*(rsop4.getResult().user_begin()) == aop) {
                  if (rsop != NULL) {
                    wmsa.push_back(rsop);
                  } else if (mmop != NULL) {
                    wmsa.push_back(mmop);
                  } else if (lnop_ != NULL) {
                    wmsa.push_back(lnop_);
                  } else
                    return;
                } else
                  return;
              } else
                return;
            } else
              return;
          } else
            return;
        } else
          return;
      }
    }
  });
}

bool tpu_mlir::ConvertTopToTpu::swin_mix_precision() {
  std::vector<Operation *> mlp;
  std::vector<Operation *> wmsa;
  std::vector<Operation *> dep2space;
  std::vector<Operation *> smops;
  bool patch_embed = false;

  auto match_depth2space_and_patch_embed =
      [&](std::vector<Operation *> &depth2space, bool /*  */ &patch_embed) {
        mainFunc_.walk([&](Operation *op) {
          // match patch_embed
          if (isa<top::InputOp>(op)) {
            Operation *dstOP = *(op->getUsers().begin());
            if (isa<top::ConvOp>(dstOP)) {
              dstOP = *(dstOP->getUsers().begin());
              if (isa<top::ReshapeOp>(dstOP)) {
                auto reshapeOp = dyn_cast<top::ReshapeOp>(dstOP);
                auto in_shape = module::getShape(reshapeOp->getOperand(0));
                auto out_shape = module::getShape(reshapeOp->getResult(0));
                dstOP = *(dstOP->getUsers().begin());
                if (isa<top::PermuteOp>(dstOP) && in_shape.size() == 4 &&
                    out_shape.size() == 3 &&
                    in_shape[3] * in_shape[2] == out_shape[2]) {
                  patch_embed = true;
                }
              }
            }
          }
          // match depth2space
          else if (isa<top::Depth2SpaceOp>(op)) {
            depth2space.push_back(op);
          }
        });
      };
  match_swin_mlp(mlp);
  match_swin_wmsa(wmsa, smops);
  match_depth2space_and_patch_embed(dep2space, patch_embed);

  if (mlp.size() == 12 && patch_embed &&
      dep2space.size() == 3) { // old swin tiny in nnmodels
    bool after_depth2space = false;
    mainFunc_.walk([&](Operation *op) {
      if (op == dep2space.back())
        after_depth2space = true;
      if ((isa<top::AddOp, top::Depth2SpaceOp>(op) ||
           (after_depth2space && isa<top::MatMulOp>(op))) &&
          (LoweringConfig::quantize_map.find(module::getName(op).str()) ==
           LoweringConfig::quantize_map.end())) {
        if (isa<top::MatMulOp>(op)) {
          if (isa<top::AddOp>(*(op->getResults()[0].getUsers().begin()))) {
            LoweringConfig::quantize_map.insert(
                {module::getName(op).str(), module::Mode::F16});
          }
        } else {
          LoweringConfig::quantize_map.insert(
              {module::getName(op).str(), module::Mode::F16});
        }
      }
    });
    // 2. fc1 in all mlp, fc2 in mlp after specific layer
    int cnt = 0;
    for (auto it = mlp.begin(); it != mlp.end(); ++it) {
      cnt++;
      if (auto addop = dyn_cast<top::AddOp>(*it)) {
        top::MatMulOp mmop = NULL;
        for (auto in : addop.getOperands()) {
          if (isa<top::MatMulOp>(in.getDefiningOp())) {
            mmop = dyn_cast_or_null<top::MatMulOp>(in.getDefiningOp());
            if (cnt >= 9) {
              if (LoweringConfig::quantize_map.find(
                      module::getName(mmop.getOperation()).str()) ==
                  LoweringConfig::quantize_map.end())
                LoweringConfig::quantize_map.insert(
                    {module::getName(mmop.getOperation()).str(),
                     module::Mode::F16});
            }
            if (isa<top::GELUOp>(mmop.getInput().getDefiningOp())) {
              auto geluop =
                  dyn_cast<top::GELUOp>(mmop.getInput().getDefiningOp());
              if (auto mmop1 = dyn_cast<top::MatMulOp>(
                      geluop.getInput().getDefiningOp())) {
                if (LoweringConfig::quantize_map.find(
                        module::getName(mmop1.getOperation()).str()) ==
                    LoweringConfig::quantize_map.end())
                  LoweringConfig::quantize_map.insert(
                      {module::getName(mmop1.getOperation()).str(),
                       module::Mode::F16});
              }
            }
          }
        }
      }
    }
    return true;
  }

  if (mlp.size() > 0 && wmsa.size() > 0 && (mlp.size() == wmsa.size())) {
    int cnt = 0;
    for (auto op : wmsa) {
      cnt++;
      auto rsop = dyn_cast_or_null<top::ReshapeOp>(op);
      auto mmop = dyn_cast_or_null<top::MatMulOp>(op);
      auto lnop = dyn_cast_or_null<top::LayerNormOp>(op);
      top::LayerNormOp lnop_ = NULL;

      if (rsop == NULL && mmop == NULL && lnop == NULL)
        return false;
      top::AddOp addop = NULL;
      if (rsop != NULL) {
        for (auto u : rsop.getResult().getUsers()) {
          if (isa<top::AddOp>(u))
            addop = dyn_cast_or_null<top::AddOp>(u);
          if (isa<top::LayerNormOp>(u))
            lnop_ = dyn_cast_or_null<top::LayerNormOp>(u);
        }
      } else if (mmop != NULL) {
        for (auto u : mmop.getResult().getUsers()) {
          if (isa<top::AddOp>(u))
            addop = dyn_cast_or_null<top::AddOp>(u);
          if (isa<top::LayerNormOp>(u))
            lnop_ = dyn_cast_or_null<top::LayerNormOp>(u);
        }
        if (LoweringConfig::quantize_map.find(
                module::getName(mmop.getOperation()).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(mmop.getOperation()).str(), module::Mode::F16});
        }
      } else if (lnop != NULL) {
        for (auto u : lnop.getResult().getUsers()) {
          if (isa<top::AddOp>(u))
            addop = dyn_cast_or_null<top::AddOp>(u);
          if (isa<top::LayerNormOp>(u))
            lnop_ = dyn_cast_or_null<top::LayerNormOp>(u);
        }
      }
      if (addop == NULL || lnop_ == NULL)
        return false;
      if (cnt <= 2) {
        ; // set_block_fp16(lnop_.getOperation(),addop.getOperation());
      }
      if (rsop != NULL) {
        if (LoweringConfig::quantize_map.find(
                module::getName(rsop.getOperation()).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(rsop.getOperation()).str(), module::Mode::F16});
        }
      } else if (mmop != NULL) {
        ; // nned to set mm to float?
      } else if (lnop != NULL) {
        ;
      }
      if (LoweringConfig::quantize_map.find(
              module::getName(addop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(addop.getOperation()).str(), module::Mode::F16});
      }
    }
    cnt = 0;
    for (auto op : mlp) {
      cnt++;
      auto addop = dyn_cast_or_null<top::AddOp>(op);
      if (addop == NULL)
        return false;
      if (LoweringConfig::quantize_map.find(
              module::getName(addop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(addop.getOperation()).str(), module::Mode::F16});
      }
      for (auto in : addop.getOperands()) {
        if (auto mmop = dyn_cast<top::MatMulOp>(in.getDefiningOp())) {
          if (mlp.size() < 24) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(mmop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(mmop.getOperation()).str(),
                   module::Mode::F16});
            }
          }
          if (auto geluop = dyn_cast<top::GELUOp>(
                  mmop.getOperands()[0].getDefiningOp())) {
            if ((cnt < 9 && mlp.size() < 24) || (cnt < 18 && mlp.size() >= 24))
              break;
            if (LoweringConfig::quantize_map.find(
                    module::getName(geluop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end())
              LoweringConfig::quantize_map.insert(
                  {module::getName(geluop.getOperation()).str(),
                   module::Mode::F16});

            if (auto mmop = dyn_cast<top::MatMulOp>(
                    geluop.getInput().getDefiningOp())) {
              if (LoweringConfig::quantize_map.find(
                      module::getName(mmop.getOperation()).str()) ==
                  LoweringConfig::quantize_map.end()) {
                LoweringConfig::quantize_map.insert(
                    {module::getName(mmop.getOperation()).str(),
                     module::Mode::F16});
              }
            }
          } else if (auto rsop = dyn_cast<top::ReshapeOp>(in.getDefiningOp())) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(rsop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(rsop.getOperation()).str(),
                   module::Mode::F16});
            }
          }
        }
      }
    }
    for (auto op : smops) {
      auto smop = dyn_cast<top::SoftmaxOp>(op);
      auto addop = dyn_cast<top::AddOp>(smop.getInput().getDefiningOp());
      if (LoweringConfig::quantize_map.find(
              module::getName(addop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(addop.getOperation()).str(), module::Mode::F16});
      }
    }
    return true;
  }
  return false;
}

void ConvertTopToTpu::match_cswin_cswsa(std::vector<Operation *> &cswsa) {
  mainFunc_.walk([&](Operation *op) {
    if (isa<top::AddOp, top::LayerNormOp>(op)) {
      top::AddOp aop = NULL;
      top::LayerNormOp lnop_ = NULL;

      if (aop == NULL)
        aop = dyn_cast_or_null<top::AddOp>(op);
      if (lnop_ == NULL)
        lnop_ = dyn_cast_or_null<top::LayerNormOp>(op);
      if (aop == NULL && lnop_ == NULL)
        return;
      top::LayerNormOp lnop = NULL;
      top::AddOp addop = NULL;
      if (aop != NULL) {
        for (auto u : aop.getOutput().getUsers()) {
          if (!isa<top::LayerNormOp, top::AddOp>(u)) {
            return;
          }
          if (lnop == NULL) {
            lnop = dyn_cast_or_null<top::LayerNormOp>(u);
          }
          if (addop == NULL) {
            addop = dyn_cast_or_null<top::AddOp>(u);
          }
        }
      } else if (lnop_ != NULL) {
        for (auto u : lnop_.getOutput().getUsers()) {
          if (!isa<top::LayerNormOp, top::AddOp>(u)) {
            return;
          }
          if (lnop == NULL) {
            lnop = dyn_cast_or_null<top::LayerNormOp>(u);
          }
          if (addop == NULL) {
            addop = dyn_cast_or_null<top::AddOp>(u);
          }
        }
      }
      if (lnop == NULL || addop == NULL)
        return;
      int sm_cnt = 0;
      int mm_cnt = 0;
      int trip_mms = 0;
      int trip_slices = 0;
      int six_slices = 0;
      if (!convergence_with_sm_matmul_slice(
              lnop, addop, sm_cnt, mm_cnt, trip_mms, trip_slices, six_slices)) {
        return;
      }
      if (sm_cnt == 4 && mm_cnt == 19 && six_slices == 1) // multi pass in tree
        cswsa.push_back(op);
      else if (sm_cnt == 2 && mm_cnt == 10 &&
               trip_slices == 1) // multi pass in tree
        cswsa.push_back(op);
    }
  });
}

bool ConvertTopToTpu::cswin_mix_precision() {
  std::vector<Operation *> mlp;
  std::vector<Operation *> cswsa;

  match_vit_mlp(mlp);
  match_cswin_cswsa(cswsa);

  if (mlp.size() > 0 && cswsa.size() > 0 && (mlp.size() == cswsa.size())) {
    for (auto op : cswsa) {
      auto addop = dyn_cast_or_null<top::AddOp>(op);
      auto lnop = dyn_cast_or_null<top::LayerNormOp>(op);
      if (addop == NULL && lnop == NULL)
        return false;
      if (addop != NULL) {
        if (LoweringConfig::quantize_map.find(
                module::getName(addop.getOperation()).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(addop.getOperation()).str(), module::Mode::F16});
        }
        for (auto u : addop.getResult().getUsers()) {
          if (auto aop = dyn_cast<top::AddOp>(u)) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(aop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(aop.getOperation()).str(),
                   module::Mode::F16});
            }
          }
        }
      } else if (lnop != NULL) {
        for (auto u : lnop.getResult().getUsers()) {
          if (auto aop = dyn_cast<top::AddOp>(u)) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(aop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(aop.getOperation()).str(),
                   module::Mode::F16});
            }
          }
        }
      }
    }
    for (auto op : mlp) {
      auto addop = dyn_cast_or_null<top::AddOp>(op);
      if (addop == NULL)
        return false;
      if (LoweringConfig::quantize_map.find(
              module::getName(addop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(addop.getOperation()).str(), module::Mode::F16});
      }
      for (auto in : addop.getOperands()) {
        if (auto mmop = dyn_cast<top::MatMulOp>(in.getDefiningOp())) {
          auto geop =
              dyn_cast_or_null<top::GELUOp>(mmop.getInput().getDefiningOp());
          if (geop == NULL)
            return false;
          if (LoweringConfig::quantize_map.find(
                  module::getName(mmop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(mmop.getOperation()).str(),
                 module::Mode::F16});
          }
        }
      }
    }
    return true;
  }

  return false;
}

void ConvertTopToTpu::match_detr_ffn(std::vector<Operation *> &ffn) {
  mainFunc_.walk([&](Operation *op) {
    if (auto addop = dyn_cast<top::AddOp>(op)) {
      top::LayerNormOp lnop = NULL;
      top::MatMulOp mmop = NULL;
      for (auto in : addop.getOperands()) {
        if (isa<top::MatMulOp>(in.getDefiningOp()))
          mmop = dyn_cast_or_null<top::MatMulOp>(in.getDefiningOp());
        else if (isa<top::LayerNormOp>(in.getDefiningOp()))
          lnop = dyn_cast_or_null<top::LayerNormOp>(in.getDefiningOp());
        else
          return;
      }
      if (mmop == NULL || lnop == NULL || !isSISO(mmop.getOperation()))
        return;
      if (isa<top::MatMulOp>(mmop.getInput().getDefiningOp())) {
        auto mmop1 =
            dyn_cast_or_null<top::MatMulOp>(mmop.getInput().getDefiningOp());
        if (isa<top::LayerNormOp>(mmop1.getInput().getDefiningOp())) {
          if (mmop1.getInput().getDefiningOp() != lnop)
            return;
        }
        if (!mmop1.getOutput().hasOneUse() || !mmop1.getDoRelu())
          return;
        ffn.push_back(op);
        return;
      }
    }
  });
}

void ConvertTopToTpu::match_detr_encoder_mha(std::vector<Operation *> &mha,
                                             std::vector<Operation *> &smops) {
  mainFunc_.walk([&](Operation *op) {
    if (isa<top::LayerNormOp, top::PermuteOp>(op)) {
      top::AddOp addu = NULL;
      top::AddOp addl = NULL;
      top::MatMulOp mmop = NULL;
      if (auto lnop = dyn_cast<top::LayerNormOp>(op)) {
        for (auto u : lnop.getOutput().getUsers()) {
          if (!isa<top::MatMulOp, top::AddOp>(u)) {
            return;
          }
          if (isa<top::AddOp>(u)) {
            auto aop_ = dyn_cast<top::AddOp>(u);
            if (std::distance(aop_.getResult().getUsers().begin(),
                              aop_.getResult().getUsers().end()) == 2) {
              addu = aop_;
            } else if (std::distance(aop_.getResult().getUsers().begin(),
                                     aop_.getResult().getUsers().end()) == 1) {
              addl = aop_;
            } else {
              return;
            }
          } else if (isa<top::MatMulOp>(u))
            mmop = dyn_cast<top::MatMulOp>(u);
        }
      } else if (auto pop = dyn_cast<top::PermuteOp>(op)) {
        for (auto u : pop.getOutput().getUsers()) {
          if (!isa<top::MatMulOp, top::AddOp>(u)) {
            return;
          }
          if (isa<top::AddOp>(u)) {
            auto aop_ = dyn_cast<top::AddOp>(u);
            if (std::distance(aop_.getResult().getUsers().begin(),
                              aop_.getResult().getUsers().end()) == 2) {
              addu = aop_;
            } else if (std::distance(aop_.getResult().getUsers().begin(),
                                     aop_.getResult().getUsers().end()) == 1) {
              addl = aop_;
            } else {
              return;
            }
          } else if (isa<top::MatMulOp>(u))
            mmop = dyn_cast<top::MatMulOp>(u);
        }
      }
      if (addu == NULL || addl == NULL || mmop == NULL)
        return;
      int sm_cnt = 0;
      int mm_cnt = 0;
      int trip_mm = 0;
      int trip_slices = 0;
      int six_slices = 0;
      if (!convergence_with_sm_matmul_slice(addu, addl, sm_cnt, mm_cnt, trip_mm,
                                            trip_slices, six_slices)) {
        return;
      }
      if (!convergence_with_sm_matmul_slice(mmop, addl, sm_cnt, mm_cnt, trip_mm,
                                            trip_slices, six_slices)) {
        return;
      }
      if (sm_cnt != 2 || mm_cnt != 10) // multi pass in tree
        return;
      mha.push_back(op);
    }
  });
}

template <typename opType>
bool ConvertTopToTpu::find_in_block(Operation *from, Operation *to,
                                    std::vector<Operation *> &ops) {
  bool res = true;
  auto re = from->getResult(0);
  for (auto r : re.getUsers()) {
    if (isa<top::NoneOp>(r))
      return false;
    else if (r == to)
      return true;
    else if (isa<ReturnOp>(r)) {
      return true;
    }
    if (isa<opType>(r)) {
      if (std::find(ops.begin(), ops.end(), r) == ops.end())
        ops.push_back(r);
    }
    res = res && find_in_block<opType>(r, to, ops);
  }
  return res;
}

void ConvertTopToTpu::float_till_output(Operation *start) {
  if (isa<top::NoneOp>(start) || !(start->getLoc().isa<NameLoc>()))
    return;
  if (LoweringConfig::quantize_map.find(module::getName(start).str()) ==
      LoweringConfig::quantize_map.end()) {
    LoweringConfig::quantize_map.insert(
        {module::getName(start).str(), module::Mode::F16});
  }
  for (auto o : start->getResult(0).getUsers()) {
    if (isa<top::NoneOp>(o))
      continue;
    else {
      float_till_output(o);
    }
  }
}

bool ConvertTopToTpu::match_detr_decoder(std::vector<Operation *> &dec,
                                         std::vector<Operation *> &addops,
                                         std::vector<Operation *> &smops) {
  top::LayerNormOp lnop = NULL;
  top::AddOp aop = NULL;
  top::ConcatOp cop = NULL;
  mainFunc_.walk([&](Operation *op) {
    if (auto lnop_ = dyn_cast<top::LayerNormOp>(op)) {
      if (std::distance(op->getResult(0).getUsers().begin(),
                        op->getResult(0).getUsers().end()) != 7) {
        return;
      }
      int mmcnt = 0;
      int addcnt = 0;
      std::vector<Operation *> mmops;
      top::AddOp aop_ = NULL;
      for (auto u : lnop_.getResult().getUsers()) {
        if (dyn_cast<top::MatMulOp>(u))
          mmcnt++;
        if (dyn_cast<top::AddOp>(u)) {
          addcnt++;
          aop_ = dyn_cast_or_null<top::AddOp>(u);
        }
      }
      if (mmcnt == 6 && addcnt == 1) {
        for (auto m : mmops) {
          auto mm = dyn_cast<top::MatMulOp>(m);
          if (mm.getOperand(1).getDefiningOp() != aop_)
            return;
        }
        lnop = lnop_;
        aop = aop_;
      }
    }
  });
  mainFunc_.walk([&](Operation *op) {
    if (auto concat = dyn_cast<top::ConcatOp>(op)) {
      if (op->getOperands().size() == 6) {
        cop = dyn_cast_or_null<top::ConcatOp>(op);
      }
    }
  });
  if (lnop != NULL && cop != NULL && aop != NULL) {
    int sm_cnt = 0;
    int mm_cnt = 0;
    int trip_mm = 0;
    int trip_slices = 0;
    int six_slices = 0;
    if (!convergence_with_sm_matmul_slice(aop, cop, sm_cnt, mm_cnt, trip_mm,
                                          trip_slices, six_slices)) {
      return false;
    }
    dec.push_back(aop.getOperation());
    dec.push_back(cop.getOperation());
    find_in_block<top::AddOp>(aop, cop, addops);
    find_in_block<top::SoftmaxOp>(aop, cop, smops);
    return true;
  } else
    return false;
}

bool ConvertTopToTpu::detr_mix_precision() {
  std::vector<Operation *> ffn;
  std::vector<Operation *> mha;
  std::vector<Operation *> dec;
  std::vector<Operation *> decoder_addops;
  std::vector<Operation *> decoder_smops;
  std::vector<Operation *> encoder_smops;
  match_detr_ffn(ffn);
  match_detr_encoder_mha(mha, encoder_smops);
  if (!match_detr_decoder(dec, decoder_addops, decoder_smops))
    return false;

  if (ffn.size() != 12 || mha.size() != 6)
    return false;
  for (auto op : ffn) {
    if (auto aop = dyn_cast<top::AddOp>(op)) {
      if (LoweringConfig::quantize_map.find(
              module::getName(aop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(aop.getOperation()).str(), module::Mode::F16});
      }
      for (auto i : aop.getInputs()) {
        if (auto mm0 = dyn_cast<top::MatMulOp>(i.getDefiningOp())) {
          if (LoweringConfig::quantize_map.find(
                  module::getName(mm0.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(mm0.getOperation()).str(), module::Mode::F16});
          }
          if (auto mm1 =
                  dyn_cast<top::MatMulOp>(mm0.getInput().getDefiningOp())) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(mm1.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(mm1.getOperation()).str(),
                   module::Mode::F16});
            }
          }
        }
      }
    }
  }
  for (auto op : mha) {
    if (auto lnop = dyn_cast<top::LayerNormOp>(op)) {
      for (auto u : lnop.getResult().getUsers()) {
        if (auto aop = dyn_cast<top::AddOp>(u)) {
          if (LoweringConfig::quantize_map.find(
                  module::getName(aop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(aop.getOperation()).str(), module::Mode::F16});
          }
        }
      }
    } else if (auto pop = dyn_cast<top::PermuteOp>(op)) {
      for (auto u : pop.getResult().getUsers()) {
        if (auto aop = dyn_cast<top::AddOp>(u)) {
          if (LoweringConfig::quantize_map.find(
                  module::getName(aop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(aop.getOperation()).str(), module::Mode::F16});
          }
        }
      }
      if (auto rop = dyn_cast<top::ReshapeOp>(pop.getInput().getDefiningOp())) {
        if (auto cop = dyn_cast<top::ConvOp>(rop.getInput().getDefiningOp())) {
          if (LoweringConfig::quantize_map.find(
                  module::getName(cop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(cop.getOperation()).str(), module::Mode::F16});
          }
        }
      }
    }
    top::AddOp aopu = NULL;
    top::AddOp aopl = NULL;
    top::MatMulOp mmop = NULL;
    for (auto u : op->getResult(0).getUsers()) {
      if (auto aop_ = dyn_cast<top::AddOp>(u)) {
        if (isa<top::MatMulOp>(*aop_.getResult().getUsers().begin()))
          aopu = aop_;
        else
          aopl = aop_;
      } else if (isa<top::MatMulOp>(u)) {
        mmop = dyn_cast<top::MatMulOp>(u);
      }
    }
    if (aopu != NULL && aopl != NULL && mmop != NULL) {
      set_block_fp16(aopu.getOperation(), aopl.getOperation());
      set_block_fp16(mmop.getOperation(), aopl.getOperation());
    }
  }

  for (auto cop : dec) {
    if (isa<top::ConcatOp>(cop)) {
      if (LoweringConfig::quantize_map.find(module::getName(cop).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(cop).str(), module::Mode::F16});
      }
      for (auto o : cop->getResult(0).getUsers()) {
        float_till_output(o);
      }
    }
  }
  for (auto a : decoder_addops) {
    if (LoweringConfig::quantize_map.find(module::getName(a).str()) ==
        LoweringConfig::quantize_map.end()) {
      LoweringConfig::quantize_map.insert(
          {module::getName(a).str(), module::Mode::F16});
    }
  }
  for (auto a : decoder_smops) {
    if (auto mmop = dyn_cast<top::MatMulOp>(a->getOperand(0).getDefiningOp())) {
      if (LoweringConfig::quantize_map.find(
              module::getName(mmop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(mmop.getOperation()).str(), module::Mode::F16});
      }
    }
  }
  for (auto a : encoder_smops) {
    if (auto mmop = dyn_cast<top::MatMulOp>(a->getOperand(0).getDefiningOp())) {
      if (LoweringConfig::quantize_map.find(
              module::getName(mmop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(mmop.getOperation()).str(), module::Mode::F16});
      }
    }
  }

  return true;
}

void ConvertTopToTpu::match_deit_mha(std::vector<Operation *> &mha) {
  mainFunc_.walk([&](Operation *op) {
    if (isa<top::AddOp>(op)) {
      auto aop = dyn_cast_or_null<top::AddOp>(op);
      if (std::distance(aop.getOutput().getUsers().begin(),
                        aop.getOutput().getUsers().end()) != 2)
        return;
      top::LayerNormOp lnop = NULL;
      top::AddOp addop = NULL;
      for (auto u : aop.getOutput().getUsers()) {
        if (isa<top::LayerNormOp>(u)) {
          lnop = dyn_cast_or_null<top::LayerNormOp>(u);
        } else if (isa<top::AddOp>(u)) {
          addop = dyn_cast_or_null<top::AddOp>(u);
        } else
          return;
      }
      if (lnop == NULL || addop == NULL)
        return;
      if (!convergence(lnop.getOperation(), addop.getOperation()))
        return;
      if (isa<top::AttentionOp>(*(lnop.getResult().user_begin()))) {
        auto atnop =
            dyn_cast<top::AttentionOp>(*(lnop.getResult().user_begin()));
        if (*(atnop.getResult().user_begin()) == addop) {
          mha.push_back(op);
          return;
        }
      }
    }
  });
}

bool ConvertTopToTpu::deit_mix_precision() {
  std::vector<Operation *> mlp;
  std::vector<Operation *> mha;

  match_vit_mlp(mlp); // ending add in mlp
  match_deit_mha(
      mha); // beginging add in mha, infact mostly same with those in mlp

  if (mlp.size() > 0 && mha.size() > 0 && (mlp.size() == mha.size())) {
    for (auto op : mha) {
      auto addop = dyn_cast_or_null<top::AddOp>(op);
      if (addop == NULL)
        return false;
      if (LoweringConfig::quantize_map.find(
              module::getName(addop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(addop.getOperation()).str(), module::Mode::F16});
      }
      for (auto u : addop.getResult().getUsers()) {
        if (auto aop = dyn_cast<top::AddOp>(u)) {
          if (LoweringConfig::quantize_map.find(
                  module::getName(aop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(aop.getOperation()).str(), module::Mode::F16});
          }
        }
      }
    }
    for (auto op : mlp) {
      auto addop = dyn_cast_or_null<top::AddOp>(op);
      if (addop == NULL)
        return false;
      if (LoweringConfig::quantize_map.find(
              module::getName(addop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(addop.getOperation()).str(), module::Mode::F16});
      }
      for (auto in : addop.getOperands()) {
        if (auto mmop = dyn_cast<top::MatMulOp>(in.getDefiningOp())) {
          auto geop =
              dyn_cast_or_null<top::GELUOp>(mmop.getInput().getDefiningOp());
          if (geop == NULL)
            return false;
          if (LoweringConfig::quantize_map.find(
                  module::getName(mmop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(mmop.getOperation()).str(),
                 module::Mode::F16});
          }
        }
      }
    }
    return true;
  }
  return false;
}

void ConvertTopToTpu::match_eva2_mlp(std::vector<Operation *> &mlp) {
  mainFunc_.walk([&](Operation *op) {
    if (auto addop = dyn_cast<top::AddOp>(op)) {
      top::AddOp aop = NULL;
      top::MatMulOp mmop = NULL;
      for (auto in : addop.getOperands()) {
        if (isa<top::MatMulOp>(in.getDefiningOp()))
          mmop = dyn_cast_or_null<top::MatMulOp>(in.getDefiningOp());
        else if (isa<top::AddOp>(in.getDefiningOp()))
          aop = dyn_cast_or_null<top::AddOp>(in.getDefiningOp());
        else
          return;
      }
      if (mmop == NULL || aop == NULL || !isSISO(mmop.getOperation()))
        return;
      top::MulOp mop = NULL;
      if (auto lnop_ =
              dyn_cast<top::LayerNormOp>(mmop.getInput().getDefiningOp()))
        mop = dyn_cast_or_null<top::MulOp>(lnop_.getInput().getDefiningOp());
      else
        mop = dyn_cast_or_null<top::MulOp>(mmop.getInput().getDefiningOp());
      if (mop != NULL) {
        top::SiLUOp silop = NULL;
        top::MatMulOp mmop1 = NULL;
        top::LayerNormOp lnop = NULL;
        for (auto i : mop.getInputs()) {
          if (!isa<top::MatMulOp, top::SiLUOp>(i.getDefiningOp())) {
            return;
          }
          if (silop == NULL)
            silop = dyn_cast_or_null<top::SiLUOp>(i.getDefiningOp());
          if (lnop == NULL)
            mmop1 = dyn_cast_or_null<top::MatMulOp>(i.getDefiningOp());
        }
        if (silop == NULL || mmop1 == NULL)
          return;
        if (!isSISO(mmop1) || !isSISO(silop))
          return;
        auto mmop2 =
            dyn_cast_or_null<top::MatMulOp>(silop.getInput().getDefiningOp());
        lnop = dyn_cast_or_null<top::LayerNormOp>(
            mmop1.getInput().getDefiningOp());
        if (mmop2 && lnop && isSISO(mmop2.getOperation()) &&
            mmop2.getInput().getDefiningOp() == lnop &&
            lnop.getInput().getDefiningOp() == aop) {
          mlp.push_back(op);
        }
      }
    }
  });
}

void ConvertTopToTpu::match_eva2_mhsa(std::vector<Operation *> &mhsa) {
  mainFunc_.walk([&](Operation *op) {
    if (auto aop = dyn_cast<top::AddOp>(op)) {
      top::LayerNormOp lnop = NULL;
      top::AddOp addop = NULL;
      for (auto u : aop.getOutput().getUsers()) {
        if (!isa<top::LayerNormOp, top::AddOp>(u)) {
          return;
        }
        if (lnop == NULL) {
          lnop = dyn_cast_or_null<top::LayerNormOp>(u);
        }
        if (addop == NULL) {
          addop = dyn_cast_or_null<top::AddOp>(u);
        }
      }
      if (lnop == NULL || addop == NULL)
        return;
      int sm_cnt = 0;
      int mm_cnt = 0;
      int trip_mm = 0;
      int trip_slice = 0;
      int six_slice = 0;
      if (!convergence_with_sm_matmul_slice(lnop, addop, sm_cnt, mm_cnt,
                                            trip_mm, trip_slice, six_slice)) {
        return;
      }
      if (sm_cnt != 8 ||
          !((mm_cnt == 29 && trip_mm > 0) ||
            (mm_cnt == 30 && trip_mm == 0))) // multi pass in tree
        return;
      mhsa.push_back(op);
    }
  });
}

bool ConvertTopToTpu::set_block_fp16(Operation *from, Operation *to) {
  bool res = true;
  if (isa<top::MatMulOp>(from)) {
    if (LoweringConfig::quantize_map.find(module::getName(from).str()) ==
        LoweringConfig::quantize_map.end()) {
      LoweringConfig::quantize_map.insert(
          {module::getName(from).str(), module::Mode::F16});
    }
  }
  auto re = from->getResult(0);
  for (auto r : re.getUsers()) {
    if (isa<top::NoneOp>(r))
      return false;
    else if (r == to)
      return true;
    else if (isa<ReturnOp>(r)) {
      return false;
    }
    res &= set_block_fp16(r, to);
  }
  return res;
}

bool ConvertTopToTpu::eva2_mix_precision() {
  std::vector<Operation *> mlp;
  std::vector<Operation *> mhsa;

  match_eva2_mlp(mlp);
  match_eva2_mhsa(mhsa);

  if (mlp.size() == mhsa.size() && (mlp.size() == 12 || mlp.size() == 24)) {
    int cnt = 0;
    for (auto op : mlp) {
      cnt++;
      if (!isa<top::AddOp>(op))
        return false;
      if (LoweringConfig::quantize_map.find(module::getName(op).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(op).str(), module::Mode::F16});
      }
      for (auto i : op->getOperands()) {
        if (auto aop = dyn_cast<top::AddOp>(i.getDefiningOp())) {
          if (LoweringConfig::quantize_map.find(
                  module::getName(aop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(aop.getOperation()).str(), module::Mode::F16});
          }
        }
      }
      for (auto i : op->getOperands()) {
        if (auto mmop = dyn_cast<top::MatMulOp>(i.getDefiningOp())) {
          if (cnt <= 2 || (mlp.size() == 24 && cnt >= 18)) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(mmop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(mmop.getOperation()).str(),
                   module::Mode::F16});
            }
          }
          if (auto mulop =
                  dyn_cast<top::MulOp>(mmop.getInput().getDefiningOp())) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(mulop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(mulop.getOperation()).str(),
                   module::Mode::F16});
            }
          } else if (auto lnop = dyn_cast<top::LayerNormOp>(
                         mmop.getInput().getDefiningOp())) {
            if (auto mulop =
                    dyn_cast<top::MulOp>(lnop.getInput().getDefiningOp())) {
              if (LoweringConfig::quantize_map.find(
                      module::getName(mulop.getOperation()).str()) ==
                  LoweringConfig::quantize_map.end()) {
                LoweringConfig::quantize_map.insert(
                    {module::getName(mulop.getOperation()).str(),
                     module::Mode::F16});
              }
            }
          }
        }
      }
    }

    cnt = 0;
    for (auto op : mhsa) {
      cnt++;
      if (LoweringConfig::quantize_map.find(module::getName(op).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(op).str(), module::Mode::F16});
      }
      if (cnt <= 2 || (mlp.size() == 24 && cnt <= 7)) { // 2 for small
        top::AddOp aop = NULL;
        top::LayerNormOp lnop = NULL;
        for (auto o : op->getResult(0).getUsers()) {
          if (aop == NULL) {
            aop = dyn_cast_or_null<top::AddOp>(o);
          }
          if (lnop == NULL) {
            lnop = dyn_cast_or_null<top::LayerNormOp>(o);
          }
        }
        if (aop != NULL && lnop != NULL) {
          set_block_fp16(lnop.getOperation(), aop.getOperation());
        }
      }
      if (cnt >= 100) { // 9) { // don't enable this intresting backward thing
        for (auto o : op->getResult(0).getUsers()) {
          if (auto aop = dyn_cast<top::AddOp>(o)) {
            for (auto i : aop.getOperands()) {
              if (auto mmop = dyn_cast<top::MatMulOp>(i.getDefiningOp())) {
                if (LoweringConfig::quantize_map.find(
                        module::getName(mmop.getOperation()).str()) ==
                    LoweringConfig::quantize_map.end()) {
                  LoweringConfig::quantize_map.insert(
                      {module::getName(mmop.getOperation()).str(),
                       module::Mode::F16});
                }
              }
            }
          }
        }
      }
    }
    return true;
  }
  return false;
}

} // namespace tpu_mlir
