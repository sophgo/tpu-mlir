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
bool tpu_mlir::ConvertTopToTpu::swin_t_mix_precision() {
  std::vector<Operation *> ffn;
  std::vector<Operation *> depth2space;
  bool is_patch_embed = false;
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
            is_patch_embed = true;
          }
        }
      }
    }
    // match depth2space
    else if (isa<top::Depth2SpaceOp>(op)) {
      depth2space.push_back(op);
    }
    // match ffn
    else if (isa<top::LayerNormOp>(op)) {
      Operation *add0Operand = op->getOperand(0).getDefiningOp();
      Operation *fc1 = *(op->getUsers().begin());
      if (isa<top::MatMulOp>(fc1)) {
        auto dstOp = *(fc1->getUsers().begin());
        if (isa<top::GELUOp>(dstOp)) {
          dstOp = *(dstOp->getUsers().begin());
          if (isa<top::MatMulOp>(dstOp)) {
            dstOp = *(dstOp->getUsers().begin()); // add op
            for (auto add1operand : dstOp->getOperands()) {
              if (add1operand.getDefiningOp() == add0Operand) {
                ffn.push_back(fc1);
              }
            }
          }
        }
      }
    }
  });
  if (is_patch_embed && ffn.size() == 12 && depth2space.size() == 3) {
    llvm::outs() << "is swin_t \n";
    // 1. add & depth2space & after last depth2space to f16
    bool after_depth2space = false;
    mainFunc_.walk([&](Operation *op) {
      if (op == depth2space.back())
        after_depth2space = true;
      if ((isa<top::AddOp, top::Depth2SpaceOp>(op) ||
           (after_depth2space && isa<top::MatMulOp>(op))) &&
          (LoweringConfig::quantize_map.find(module::getName(op).str()) ==
           LoweringConfig::quantize_map.end())) {
        LoweringConfig::quantize_map.insert(
            {module::getName(op).str(), module::Mode::F16});
      }
    });
    // 2. fc1 in all ffn , fc2 in ffn after specific layer
    for (auto it = ffn.begin(); it != ffn.end(); ++it) {
      Operation *fc1 = *it;
      if ((LoweringConfig::quantize_map.find(module::getName(fc1).str()) ==
           LoweringConfig::quantize_map.end())) {
        LoweringConfig::quantize_map.insert(
            {module::getName(fc1).str(), module::Mode::F16});
      }
      if (std::distance(ffn.begin(), it) >= 9) {
        auto fc2 = *(fc1->getUsers().begin()->getUsers().begin());
        if (isa<top::MatMulOp>(fc2) &&
            (LoweringConfig::quantize_map.find(module::getName(fc2).str()) ==
             LoweringConfig::quantize_map.end()))
          LoweringConfig::quantize_map.insert(
              {module::getName(fc2).str(), module::Mode::F16});
      }
    }
    return true;
  }
  return false;
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
      if (auto lnop_ = dyn_cast<top::LayerNormOp>(mmop.getInput().getDefiningOp()))
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
        auto mmop2 = dyn_cast_or_null<top::MatMulOp>(silop.getInput().getDefiningOp());
        lnop = dyn_cast_or_null<top::LayerNormOp>(mmop1.getInput().getDefiningOp());
        if (mmop2 && lnop && isSISO(mmop2.getOperation()) && mmop2.getInput().getDefiningOp() == lnop && lnop.getInput().getDefiningOp() == aop) {
          mlp.push_back(op);
        }
      }
    }
  });
}

void ConvertTopToTpu::match_eva2_mhsa(std::vector<Operation *> &mhsa) {
  mainFunc_.walk([&](Operation *op) {
    if (auto aop = dyn_cast<top::AddOp>(op)){
      top::LayerNormOp lnop = NULL;
      top::AddOp addop = NULL;
      for(auto u: aop.getOutput().getUsers()) {
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
      if (!convergence_with_sm_matmul(lnop, addop, sm_cnt, mm_cnt, trip_mm)) {
        return;
      }
      if (sm_cnt != 8 || mm_cnt != 29 || trip_mm == 0) // multi pass in tree
        return;
      mhsa.push_back(op);
    }
  });
}

bool ConvertTopToTpu::set_block_fp16(Operation* from, Operation *to) {
  bool res = true;
  if (isa<top::MatMulOp>(from)) {
    if (LoweringConfig::quantize_map.find(
            module::getName(from).str()) ==
        LoweringConfig::quantize_map.end()) {
      LoweringConfig::quantize_map.insert(
          {module::getName(from).str(), module::Mode::F16});
        }
  }
  auto re = from->getResult(0);
  for (auto r :re.getUsers()) {
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
    for (auto op: mlp){
      cnt ++;
      if (!isa<top::AddOp>(op))
         return false;
      if (LoweringConfig::quantize_map.find(
              module::getName(op).str()) ==
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
          if (cnt <= 2 ||(mlp.size() == 24 && cnt >= 18)) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(mmop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(mmop.getOperation()).str(), module::Mode::F16});
                }
          }
          if (auto mulop = dyn_cast<top::MulOp>(mmop.getInput().getDefiningOp())) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(mulop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(mulop.getOperation()).str(), module::Mode::F16});
                }
          } else if (auto lnop = dyn_cast<top::LayerNormOp>(mmop.getInput().getDefiningOp())) {
            if (auto mulop = dyn_cast<top::MulOp>(lnop.getInput().getDefiningOp())) {
              if (LoweringConfig::quantize_map.find(
                      module::getName(mulop.getOperation()).str()) ==
                  LoweringConfig::quantize_map.end()) {
                LoweringConfig::quantize_map.insert(
                    {module::getName(mulop.getOperation()).str(), module::Mode::F16});
                  }
            }
          }
        }
      }
    }

    cnt =0;
    for (auto op : mhsa) {
      cnt ++;
      if (LoweringConfig::quantize_map.find(
              module::getName(op).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(op).str(), module::Mode::F16});
      }
      if (cnt <= 2 || (mlp.size() == 24 && cnt <= 7)) { // 2 for small
        top::AddOp aop = NULL;
        top::LayerNormOp lnop = NULL;
        for (auto o: op->getResult(0).getUsers()) {
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
      if (cnt >= 100) { //9) { // don't enable this intresting backward thing
        for (auto o: op->getResult(0).getUsers()) {
          if (auto aop = dyn_cast<top::AddOp>(o)) {
            for (auto i : aop.getOperands()) {
              if (auto mmop = dyn_cast<top::MatMulOp>(i.getDefiningOp())) {
                if (LoweringConfig::quantize_map.find(
                        module::getName(mmop.getOperation()).str()) ==
                    LoweringConfig::quantize_map.end()) {
                  LoweringConfig::quantize_map.insert(
                      {module::getName(mmop.getOperation()).str(), module::Mode::F16});
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
