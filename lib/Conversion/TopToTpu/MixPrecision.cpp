//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/ConvertTopToTpu.h"
#include <tuple>
namespace tpu_mlir {

// SISO is single input single output, not counting weight and none and
// input/output
bool ConvertTopToTpu::isSISO(Operation *op) {
  int cnt = 0;
  for (auto in : op->getOperands()) {
    if (isa<top::InputOp>(in.getDefiningOp()) ||
        isa<top::WeightOp>(in.getDefiningOp()) ||
        isa<top::NoneOp>(in.getDefiningOp()))
      continue;
    else {
      if (cnt != 0)
        return false;
      else
        cnt++;
    }
  }
  if (cnt == 0)
    return false;
  return (std::distance(op->getResult(0).user_begin(),
                        op->getResult(0).user_end()) == 1);
}

bool ConvertTopToTpu::convergence(Operation *from, Operation *to) {
  bool res = true;
  auto re = from->getResult(0);
  for (auto r : re.getUsers()) {
    if (isa<top::NoneOp>(r))
      return false;
    else if (r == to)
      return true;
    else if (isa<ReturnOp>(r)) {
      return false;
    }
    res &= convergence(r, to);
  }
  return res;
}

bool ConvertTopToTpu::convergence_with_sm_matmul_slice(
    Operation *from, Operation *to, int &sm_cnt, int &matmul_cnt,
    int &triple_matmul, int &triple_slice, int &six_slice) {
  bool res = true;
  auto re = from->getResult(0);
  int mm_cnt = 0;
  int slice_cnt = 0;
  for (auto r : re.getUsers()) {
    if (isa<top::NoneOp>(r))
      return false;
    else if (r == to)
      return true;
    else if (isa<ReturnOp>(r)) {
      return false;
    }
    if (isa<top::SoftmaxOp>(r))
      sm_cnt++;
    if (isa<top::MatMulOp>(r)) {
      if (std::distance(r->getUsers().begin(), r->getUsers().end()) > 0) {
        mm_cnt++;
        matmul_cnt++;
      }
    }
    if (isa<top::SliceOp>(r)) {
      slice_cnt++;
    }

    res &= convergence_with_sm_matmul_slice(
        r, to, sm_cnt, matmul_cnt, triple_matmul, triple_slice, six_slice);
  }
  if (mm_cnt == 3)
    triple_matmul++;
  if (slice_cnt == 3)
    triple_slice++;
  if (slice_cnt == 6)
    six_slice++;
  return res;
}

// return a list of end layernorms after ffn , the count would be the number of
// encoder ffn part
void ConvertTopToTpu::match_bert_ffn(std::vector<Operation *> &ffn) {
  mainFunc_.walk([&](Operation *op) {
    if (auto lnop = dyn_cast<top::LayerNormOp>(op)) {
      if (auto addop = dyn_cast<top::AddOp>(lnop.getInput().getDefiningOp())) {
        if (!addop.getOutput().hasOneUse())
          return;
        top::MatMulOp mmop = NULL;
        top::LayerNormOp lnop1 = NULL;
        for (auto in : addop.getOperands()) {
          if (isa<top::LayerNormOp>(in.getDefiningOp()))
            lnop1 = dyn_cast_or_null<top::LayerNormOp>(in.getDefiningOp());
          else if (isa<top::MatMulOp>(in.getDefiningOp()))
            mmop = dyn_cast_or_null<top::MatMulOp>(in.getDefiningOp());
          else
            return;
        }
        if (mmop == NULL || lnop1 == NULL || !isSISO(mmop.getOperation()))
          return;
        if (isa<top::GELUOp>(mmop.getInput().getDefiningOp())) {
          auto geluop =
              dyn_cast_or_null<top::GELUOp>(mmop.getInput().getDefiningOp());
          if (!isSISO(geluop.getOperation()))
            return;
          if (isa<top::MatMulOp>(geluop.getInput().getDefiningOp())) {
            auto mmop1 = dyn_cast_or_null<top::MatMulOp>(
                geluop.getInput().getDefiningOp());
            if (mmop1.getInput().getDefiningOp() != lnop1 ||
                !isSISO(mmop1.getOperation())) {
              return;
            }
            if (!addop.getOutput().hasOneUse() ||
                !mmop.getOutput().hasOneUse() ||
                !geluop.getOutput().hasOneUse() ||
                !mmop1.getOutput().hasOneUse())
              return;
            ffn.push_back(lnop);
            return;
          }
        }
      }
    }
  });
}

// return a set of end layernorms after ffn , the count would be the number of
// encoder ffn part
void ConvertTopToTpu::match_bert_mha(std::vector<Operation *> &mha) {
  mainFunc_.walk([&](Operation *op) {
    if (auto lnop = dyn_cast<top::LayerNormOp>(op)) {
      if (auto addop = dyn_cast<top::AddOp>(lnop.getInput().getDefiningOp())) {
        if (!addop.getOutput().hasOneUse()) {
          return;
        }
        top::MatMulOp mmop = NULL;
        top::LayerNormOp top_lnop = NULL;

        top::ReshapeOp reshapeop = NULL;
        top::PermuteOp pmop = NULL;
        top::MatMulOp mmop1 = NULL;

        for (auto in : addop.getOperands()) {
          if (isa<top::MatMulOp>(in.getDefiningOp())) {
            mmop = dyn_cast_or_null<top::MatMulOp>(in.getDefiningOp());
          } else if (isa<top::LayerNormOp>(in.getDefiningOp())) {
            top_lnop = dyn_cast_or_null<top::LayerNormOp>(in.getDefiningOp());
          } else
            return;
        }
        if (mmop == NULL || top_lnop == NULL || !isSISO(mmop.getOperation()))
          return;
        if (isa<top::ReshapeOp>(mmop.getInput().getDefiningOp())) {
          reshapeop =
              dyn_cast_or_null<top::ReshapeOp>(mmop.getInput().getDefiningOp());
        }
        if (reshapeop == NULL || !isSISO(reshapeop.getOperation()))
          return;
        if (isa<top::PermuteOp>(reshapeop.getInput().getDefiningOp())) {
          pmop = dyn_cast_or_null<top::PermuteOp>(
              reshapeop.getInput().getDefiningOp());
        }
        if (pmop == NULL || !isSISO(pmop.getOperation()))
          return;
        if (isa<top::MatMulOp>(pmop.getInput().getDefiningOp())) {
          mmop1 =
              dyn_cast_or_null<top::MatMulOp>(pmop.getInput().getDefiningOp());
        }
        if (mmop1 == NULL)
          return;

        top::PermuteOp pmv = NULL;
        top::ReshapeOp rsv = NULL;
        top::SoftmaxOp sm = NULL;

        for (auto in : mmop1.getOperands()) {
          if (isa<top::PermuteOp>(in.getDefiningOp()))
            pmv = dyn_cast_or_null<top::PermuteOp>(in.getDefiningOp());
          else if (isa<top::SoftmaxOp>(in.getDefiningOp()))
            sm = dyn_cast_or_null<top::SoftmaxOp>(in.getDefiningOp());
          else if (isa<top::NoneOp>(in.getDefiningOp()))
            continue;
          else
            return;
        }
        if (pmv == NULL || sm == NULL)
          return;

        // check value branch
        if (isa<top::ReshapeOp>(pmv.getInput().getDefiningOp()))
          rsv =
              dyn_cast_or_null<top::ReshapeOp>(pmv.getInput().getDefiningOp());
        if (rsv == NULL || !isSISO(rsv.getOperation()))
          return;
        if (isa<top::MatMulOp>(rsv.getInput().getDefiningOp())) {
          auto mm_ =
              dyn_cast_or_null<top::MatMulOp>(rsv.getInput().getDefiningOp());
          if (mm_ == NULL || !isSISO(mm_.getOperation()) ||
              mm_.getInput().getDefiningOp() != top_lnop)
            return;
        }

        // check k,v branches
        top::AddOp addop1 = NULL;
        top::MulConstOp mcop = NULL;
        if (isa<top::AddOp>(sm.getInput().getDefiningOp()))
          addop1 = dyn_cast_or_null<top::AddOp>(sm.getInput().getDefiningOp());
        if (!addop1 || !addop1.getOutput().hasOneUse())
          return;
        for (auto in : addop1.getOperands()) {
          top::MulConstOp mcop_ = NULL;
          if (isa<top::MulConstOp>(in.getDefiningOp()))
            mcop_ = dyn_cast_or_null<top::MulConstOp>(in.getDefiningOp());
          else
            return;
          if (mcop_.getConstVal().convertToDouble() == 0.125)
            mcop = mcop_;
          else if (mcop_.getConstVal().convertToDouble() == -10000)
            continue;
          else
            return;
        }
        if (mcop == NULL || !isSISO(mcop.getOperation()))
          return;

        top::MatMulOp mmop2 = NULL;
        if (isa<top::MatMulOp>(mcop.getInput().getDefiningOp()))
          mmop2 =
              dyn_cast_or_null<top::MatMulOp>(mcop.getInput().getDefiningOp());
        if (mmop2 == NULL)
          return;
        int inputs = 0;
        for (auto in : mmop2.getOperands()) {
          if (isa<top::WeightOp>(in.getDefiningOp()))
            continue;
          else if (isa<top::PermuteOp>(in.getDefiningOp())) {
            auto p_ = dyn_cast_or_null<top::PermuteOp>(in.getDefiningOp());
            if (p_ == NULL || !isSISO(p_.getOperation()))
              return;
            if (!isa<top::ReshapeOp>(p_.getInput().getDefiningOp()))
              return;
            auto r_ =
                dyn_cast_or_null<top::ReshapeOp>(p_.getInput().getDefiningOp());
            if (r_ == NULL || !isSISO(r_.getOperation()))
              return;
            if (!isa<top::MatMulOp>(r_.getInput().getDefiningOp()))
              return;
            auto m_ =
                dyn_cast_or_null<top::MatMulOp>(r_.getInput().getDefiningOp());
            if (m_ == NULL || !isSISO(m_.getOperation()))
              return;
            if (m_.getInput().getDefiningOp() != top_lnop)
              return;
            inputs++;
          }
        }
        if (inputs != 2)
          return;
        mha.push_back(lnop);
      }
    }
  });
}

void ConvertTopToTpu::match_attention(std::vector<Operation *> &attention) {
  mainFunc_.walk([&](Operation *op) {
    if (auto atop = dyn_cast<top::AttentionOp>(op)) {
      attention.push_back(op);
    }
  });
}

static int partial_float_bert_ffn = -1;
bool ConvertTopToTpu::bert_mix_precision() {
  std::vector<Operation *> ffn;
  std::vector<Operation *> mha;
  std::vector<Operation *> attention;

  match_bert_ffn(ffn);
  match_bert_mha(mha);
  match_attention(attention);

  if (ffn.size() > 0 && (mha.size() > 0 || attention.size() > 0)) {
    // now to set
    // 1. all add before layernorm to f16
    // 2. last matmul of mha output to f16
    // 3. add before softmax to f32
    for (auto op : mha) {
      auto addop = dyn_cast_or_null<top::AddOp>(
          dyn_cast<top::LayerNormOp>(op).getInput().getDefiningOp());
      if (addop == NULL)
        return false;
      if (LoweringConfig::quantize_map.find(
              module::getName(addop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(addop.getOperation()).str(), module::Mode::F16});
      }
    }
    int cnt = 0;
    for (auto op : ffn) {
      cnt++;
      auto addop = dyn_cast_or_null<top::AddOp>(
          dyn_cast<top::LayerNormOp>(op).getInput().getDefiningOp());
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
          if (cnt >= 5 && (partial_float_bert_ffn > 0))
            continue;
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

    for (auto op : attention) {
      auto atenop = dyn_cast_or_null<top::AttentionOp>(op);
      assert(atenop != NULL);
      auto lnop =
          dyn_cast_or_null<top::LayerNormOp>(atenop.getInput().getDefiningOp());
      if (lnop == NULL)
        return false;
      for (auto out : lnop.getResult().getUsers()) {
        if (auto addop = dyn_cast<top::AddOp>(*out)) {
          if (LoweringConfig::quantize_map.find(
                  module::getName(addop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(addop.getOperation()).str(),
                 module::Mode::F16});
          }
        }
      }
    }

    for (auto op : attention) {
      auto atenop = dyn_cast_or_null<top::AttentionOp>(op);
      assert(atenop != NULL);
      auto lnop =
          dyn_cast_or_null<top::LayerNormOp>(atenop.getInput().getDefiningOp());
      if (lnop == NULL)
        return false;
      for (auto out : lnop.getResult().getUsers()) {
        if (auto addop = dyn_cast<top::AddOp>(*out)) {
          if (LoweringConfig::quantize_map.find(
                  module::getName(addop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(addop.getOperation()).str(),
                 module::Mode::F16});
          }
        }
      }
    }

    return true;
  } else
    return false;
}

void ConvertTopToTpu::match_vit_mlp(std::vector<Operation *> &mlp) {
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
            if (lnop.getInput().getDefiningOp() != aop ||
                !isSISO(lnop.getOperation())) {
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

void ConvertTopToTpu::match_vit_mha(std::vector<Operation *> &mha) {
  mainFunc_.walk([&](Operation *op) {
    if (auto addop = dyn_cast<top::AddOp>(op)) {
      top::LayerNormOp lnop = NULL;
      top::AddOp aop = NULL;
      for (auto user : addop.getOutput().getUsers()) {
        if (isa<top::LayerNormOp>(user)) {
          lnop = dyn_cast<top::LayerNormOp>(user);
        } else if (isa<top::AddOp>(user)) {
          aop = dyn_cast<top::AddOp>(user);
        }
      }
      if (lnop == NULL || aop == NULL)
        return;
      if (!isSISO(lnop.getOperation()))
        return;
      if (!convergence(lnop, aop))
        return;
      if (isa<top::MatMulOp>(*(lnop.getResult().getUsers().begin()))) {
        auto mmop =
            dyn_cast<top::MatMulOp>(*(lnop.getResult().getUsers().begin()));
        if (isa<top::ReshapeOp>(*(mmop.getResult().getUsers().begin()))) {
          auto rsop =
              dyn_cast<top::ReshapeOp>(*(mmop.getResult().getUsers().begin()));
          if (isa<top::PermuteOp>(*(rsop.getResult().getUsers().begin()))) {
            auto permop = dyn_cast<top::PermuteOp>(
                *(rsop.getResult().getUsers().begin()));
            if (std::distance(permop.getResult().getUsers().begin(),
                              permop.getResult().getUsers().end()) != 3)
              return;
            top::SliceOp sop[3] = {NULL};
            top::ReshapeOp rsop_[3] = {NULL};
            top::MatMulOp matop = NULL;
            for (auto u : permop.getResult().getUsers()) {
              if (auto sliceop = dyn_cast<top::SliceOp>(u)) {
                if (module::getI64Array(sliceop.getOffsetAttr())->at(0) == 0)
                  sop[0] = sliceop;
                else if (module::getI64Array(sliceop.getOffsetAttr())->at(0) ==
                         1)
                  sop[1] = sliceop;
                else if (module::getI64Array(sliceop.getOffsetAttr())->at(0) ==
                         2)
                  sop[2] = sliceop;
                else
                  return;
              } else {
                return;
              }
            }
            if (!isa<top::ReshapeOp>(
                    *(sop[0]->getResult(0).getUsers().begin())))
              return;
            else {
              rsop_[0] = dyn_cast<top::ReshapeOp>(
                  *(sop[0]->getResult(0).getUsers().begin()));
            }
            if (!isa<top::ReshapeOp>(
                    *(sop[1]->getResult(0).getUsers().begin())))
              return;
            else {
              rsop_[1] = dyn_cast<top::ReshapeOp>(
                  *(sop[1]->getResult(0).getUsers().begin()));
            }
            if (!isa<top::ReshapeOp>(
                    *(sop[2]->getResult(0).getUsers().begin())))
              return;
            else {
              rsop_[2] = dyn_cast<top::ReshapeOp>(
                  *(sop[2]->getResult(0).getUsers().begin()));
              if (!isa<top::MatMulOp>(
                      *(rsop_[2]->getResult(0).getUsers().begin())))
                return;
              matop = dyn_cast<top::MatMulOp>(
                  *(rsop_[2]->getResult(0).getUsers().begin()));
            }

            if (!isa<top::MatMulOp>(
                    *(rsop_[0]->getResult(0).getUsers().begin())) ||
                !isa<top::MatMulOp>(
                    *(rsop_[1]->getResult(0).getUsers().begin())))
              return;
            if (*(rsop_[0]->getResult(0).getUsers().begin()) !=
                (*(rsop_[1]->getResult(0).getUsers().begin())))
              return;
            auto mmop_ = dyn_cast<top::MatMulOp>(
                *(rsop_[0]->getResult(0).getUsers().begin()));
            if (auto mcop = dyn_cast<top::MulConstOp>(
                    *(mmop_.getOutput().getUsers().begin()))) {
              if (auto smop = dyn_cast<top::SoftmaxOp>(
                      *(mcop.getOutput().getUsers().begin()))) {
                if (*(smop.getOutput().getUsers().begin()) != matop)
                  return;
              } else
                return;
            } else
              return;
            if (!isa<top::PermuteOp>(*(matop.getResult().getUsers().begin())))
              return;
            else {
              auto pop = dyn_cast<top::PermuteOp>(
                  *(matop.getResult().getUsers().begin()));
              if (!isa<top::ReshapeOp>(*(pop.getResult().getUsers().begin())))
                return;
              auto rop = dyn_cast<top::ReshapeOp>(
                  *(pop.getResult().getUsers().begin()));
              if (!isa<top::MatMulOp>(*(rop.getResult().getUsers().begin())))
                return;
              auto mop = dyn_cast<top::MatMulOp>(
                  *(rop.getResult().getUsers().begin()));
              if (!isa<top::AddOp>(*(mop.getResult().getUsers().begin())))
                return;
              if (*(mop.getResult().getUsers().begin()) != aop)
                return;
              mha.push_back(addop);
            }
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
  });
}

void ConvertTopToTpu::match_vit_mha1(std::vector<Operation *> &mha) {
  mainFunc_.walk([&](Operation *op) {
    if (auto addop = dyn_cast<top::AddOp>(op)) {
      top::LayerNormOp lnop = NULL;
      top::AddOp aop = NULL;
      for (auto user : addop.getOutput().getUsers()) {
        if (isa<top::LayerNormOp>(user)) {
          lnop = dyn_cast<top::LayerNormOp>(user);
        } else if (isa<top::AddOp>(user)) {
          aop = dyn_cast<top::AddOp>(user);
        }
      }
      if (lnop == NULL || aop == NULL)
        return;
      if (!convergence(lnop, aop))
        return;
      if ((std::distance(lnop.getOutput().user_begin(),
                         lnop.getOutput().user_end()) != 3) &&
          (std::distance(lnop.getOutput().user_begin(),
                         lnop.getOutput().user_end()) !=
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
      for (auto u : lnop.getResult().getUsers()) {
        if (auto mmop = dyn_cast<top::MatMulOp>(u)) {
          if (!isSISO(mmop.getOperation()))
            return;
          if (auto rsop =
                  dyn_cast<top::ReshapeOp>(*(mmop.getResult().user_begin()))) {
            if (rsop.getResult().getUsers().empty()) // the original matmul
              continue;
            if (!isSISO(rsop))
              return;
            if (auto pmop = dyn_cast<top::PermuteOp>(
                    *(rsop.getResult().user_begin()))) {
              if (!isSISO(pmop))
                return;
              if (isa<top::MulConstOp>(*(pmop.getResult().user_begin()))) {
                mcop_ =
                    dyn_cast<top::MulConstOp>(*(pmop.getResult().user_begin()));
                if (!isSISO(mcop_))
                  return;
                idx = 0;
              } else if (isa<top::MatMulOp>(*(pmop.getResult().user_begin()))) {
                auto mmop_tmp =
                    dyn_cast<top::MatMulOp>(*(pmop.getResult().user_begin()));
                if (isa<top::SoftmaxOp>(*(mmop_tmp.getResult().user_begin()))) {
                  idx = 1;
                } else if (isa<top::PermuteOp>(
                               *(mmop_tmp.getResult().user_begin()))) {
                  idx = 2;
                } else if (isa<top::MulConstOp>(
                               *(mmop_tmp.getResult().user_begin()))) {
                  // in vit_l, must const is after mmop1
                  if (mmop_[0] == NULL)
                    idx = 0;
                  else if (mmop_[1] != NULL)
                    return;
                  else
                    idx = 1;
                } else {
                  (*(mmop_tmp.getResult().user_begin()))->dump();
                  return;
                }
              } else {
                return;
              }
              mmop_[idx] = mmop;
              rsop_[idx] = rsop;
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
        if (!isa<top::MatMulOp>(*(pmop_[0].getResult().user_begin())) ||
            !isa<top::MatMulOp>(*(pmop_[1].getResult().user_begin())) ||
            (*pmop_[0].getResult().user_begin() !=
             *pmop_[1].getResult().user_begin()))
          return;
      }

      mmop1 = dyn_cast<top::MatMulOp>(*(pmop_[1].getResult().user_begin()));
      top::SoftmaxOp smop = NULL;
      if (mcop_ == NULL &&
          isa<top::MulConstOp>(*(mmop1.getOutput().getUsers().begin()))) {
        auto mcop =
            dyn_cast<top::MulConstOp>(*(mmop1.getOutput().getUsers().begin()));
        if (!isa<top::SoftmaxOp>(*mcop.getResult().user_begin()) ||
            !isSISO(mcop.getOperation()))
          return;
        smop = dyn_cast<top::SoftmaxOp>(*mcop.getResult().user_begin());
      } else
        smop = dyn_cast_or_null<top::SoftmaxOp>(
            *(mmop1.getOutput().getUsers().begin()));
      if (smop == NULL)
        return;
      if (*(smop.getOutput().getUsers().begin()) !=
          *(pmop_[2].getResult().user_begin()))
        return;
      mmop2 = dyn_cast_or_null<top::MatMulOp>(*(smop.getResult().user_begin()));
      if (mmop2 == NULL)
        return;
      if (auto pmop1 =
              dyn_cast<top::PermuteOp>(*(mmop2.getResult().user_begin()))) {
        if (auto rsop1 =
                dyn_cast<top::ReshapeOp>(*(pmop1.getResult().user_begin()))) {
          if (auto mmop3 =
                  dyn_cast<top::MatMulOp>(*(rsop1.getResult().user_begin()))) {
            if (*(mmop3.getResult().user_begin()) != aop)
              return;
            else
              mha.push_back(addop);
          }
        }
      }
    }
  });
}

bool ConvertTopToTpu::vit_mix_precision() {
  std::vector<Operation *> mlp;
  std::vector<Operation *> mha;

  match_vit_mlp(mlp); // ending add in mlp
  match_vit_mha(
      mha); // beginging add in mha, infact mostly same with those in mlp
  if (mha.size() == 0)
    match_vit_mha1(mha);

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
    int total_blk = mlp.size();
    int idx = 0;
    for (auto op : mlp) {
      idx++;
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
          if ((idx >= total_blk - 3) &&
              (total_blk >
               18)) { // base 224 has 12 block and large 384 has 24 block
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
    return true;
  } else
    return false;
}

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

void processLNUsers(top::LayerNormOp lnop) {
  for (auto user : lnop.getResult().getUsers()) {
    if (isa<top::SliceOp>(user) &&
        std::distance(user->getResult(0).getUsers().begin(),
                      user->getResult(0).getUsers().end()) == 0)
      continue;
    else if (isa<top::ReshapeOp>(user) || isa<top::SwapDimInnerOp>(user))
      continue;
    else
      return;
  }
}

void ConvertTopToTpu::match_swin_wmsa(std::vector<Operation *> &wmsa,
                                      std::vector<Operation *> &sm_ops,
                                      std::vector<Operation *> &qkmmops) {
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
        auto preOp = rsop->getOperands()[0].getDefiningOp(); //dino pattern layernorm->reshape
        if (isa<top::LayerNormOp>(preOp)) {
          auto ln_tmp = dyn_cast<top::LayerNormOp>(preOp);
          if (isa<top::AddOp>(ln_tmp->getOperands()[0].getDefiningOp())) { //pattern has to be Add->LayerNorm->Reshape
            lnop = dyn_cast<top::LayerNormOp>(preOp);
          }
        }
        if (lnop == NULL)
          return;
      } else if (mmop != NULL) {
        if (std::distance(mmop.getResult().user_begin(),
                          mmop.getResult().user_end()) == 1 &&
            isa<top::AddOp>(*(mmop.getResult().user_begin()))) {
          auto aop_ = dyn_cast<top::AddOp>(*(mmop.getResult().user_begin()));
          for (auto user : aop_.getOutput().getUsers()) {
            if (isa<top::LayerNormOp>(user)) {
              lnop = dyn_cast<top::LayerNormOp>(user);
            }
          }
        } else {
          for (auto user : mmop.getOutput().getUsers()) {
            if (isa<top::LayerNormOp>(user)) {
              lnop = dyn_cast<top::LayerNormOp>(user);
            } else if (isa<top::ReshapeOp>(user)) {
              auto reshapeop_ = dyn_cast<top::ReshapeOp>(user); // Deal with reshape pattern
              for (auto user1 : reshapeop_.getOutput().getUsers()) {
                if (isa<top::AddOp>(user1)) {
                  aop = dyn_cast<top::AddOp>(user1);
                }
              }
            } else if (isa<top::AddOp>(user)) {
              aop = dyn_cast<top::AddOp>(user);
            }
          }
        }
        if (lnop == NULL || aop == NULL)
          return;
      } else if (lnop_ != NULL) {
        for (auto user : lnop_.getOutput().getUsers()) {
          if (isa<top::LayerNormOp>(user)) {
            lnop = dyn_cast<top::LayerNormOp>(user);
          } else if (isa<top::ReshapeOp>(user)) {
              auto reshapeop_ = dyn_cast<top::ReshapeOp>(user); // Deal with reshape pattern
              for (auto user1 : reshapeop_.getOutput().getUsers()) {
                if (isa<top::AddOp>(user1)) {
                  aop = dyn_cast<top::AddOp>(user1);
                }
              }
          } else if (isa<top::AddOp>(user)) {
            aop = dyn_cast<top::AddOp>(user);
          }
        }
        if (lnop == NULL || aop == NULL)
          return;
      } else
        return;
#if 0
      if (!isSISO(lnop.getOperation()))
        return;
#else
      processLNUsers(lnop);
#endif
      // if (!convergence(lnop, aop))
      //   return;
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
        int dino_flag = 0; // add flag to distinguish swin series with dino
        if (rsop1 != NULL)
          rsop2 = rsop1;
        else if (swdiop != NULL)
          rsop2 = dyn_cast_or_null<top::ReshapeOp>(
              *(swdiop.getResult().getUsers().begin()));
        if (rsop2 == NULL)
          return;
        // Update reshape here(for dino).
        if (isa<top::PadOp>(*(rsop2.getResult().getUsers().begin()))) {
          auto padop_ = dyn_cast<top::PadOp>(*(rsop2.getResult().getUsers().begin()));
          dino_flag = 1;
          if (isa<top::ReshapeOp>(*(padop_.getResult().getUsers().begin()))) {
            rsop2 = dyn_cast<top::ReshapeOp>(*(padop_.getResult().getUsers().begin()));
          } else if (isa<top::SwapDimInnerOp>(*(padop_.getResult().getUsers().begin()))) {
            auto swdiop_tmp = dyn_cast<top::SwapDimInnerOp>(*(padop_.getResult().getUsers().begin()));
            for (auto swduser : swdiop_tmp.getResult().getUsers()) {
              if (isa<top::SwapDimInnerOp>(swduser)) {
                swdiop_tmp = dyn_cast<top::SwapDimInnerOp>(swduser);
              }
            }
            if (isa<top::ReshapeOp>(*(swdiop_tmp.getResult().getUsers().begin()))) {
              rsop2 = dyn_cast<top::ReshapeOp>(*(swdiop_tmp.getResult().getUsers().begin()));
            }
          }
        }
        // Update reshape for mask-rcnn.
        if (isa<top::ConcatOp>(*(rsop2.getResult().getUsers().begin()))) {
          auto concat_tmp = dyn_cast<top::ConcatOp>(*(rsop2.getResult().getUsers().begin()));
          if (isa<top::ConcatOp>(*(concat_tmp.getResult().getUsers().begin()))) {
            concat_tmp = dyn_cast<top::ConcatOp>(*(concat_tmp.getResult().getUsers().begin()));
            if (isa<top::ReshapeOp>(*(concat_tmp.getResult().getUsers().begin()))) {
              rsop2 = dyn_cast<top::ReshapeOp>(*(concat_tmp.getResult().getUsers().begin()));
            } else if (isa<top::SwapDimInnerOp>(*(concat_tmp.getResult().getUsers().begin()))) {
              auto swdiop_tmp = dyn_cast<top::SwapDimInnerOp>(*(concat_tmp.getResult().getUsers().begin()));
              for (auto user : swdiop_tmp.getResult().getUsers()) {
                if(isa<top::SwapDimInnerOp>(user)) {
                  swdiop_tmp = dyn_cast<top::SwapDimInnerOp>(user);
                  if (isa<top::ReshapeOp>(*(swdiop_tmp.getResult().getUsers().begin()))) {
                    rsop2 = dyn_cast<top::ReshapeOp>(*(swdiop_tmp.getResult().getUsers().begin()));
                  }
                }
              }
            }
          }
        }
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

        if (dino_flag != 1) { // dino split might be different
          if ((std::distance(rsop2.getOutput().user_begin(),
                           rsop2.getOutput().user_end()) != 3) &&
            (std::distance(rsop2.getOutput().user_begin(),
                           rsop2.getOutput().user_end()) !=
             4)) // chip opt may split matmul to 3, but left the original matmul
                 // not removed
          return;
        }

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
                //Update permute here(for dino)
                if (isa<top::PermuteOp>(*(pmop.getResult().user_begin()))) {
                  pmop = dyn_cast<top::PermuteOp>(*(pmop.getResult().user_begin()));
                  dino_flag = 1;
                  for (auto permuteuser : pmop.getResult().getUsers()) {
                    auto sliceop_ = dyn_cast<top::SliceOp>(permuteuser);
                    if (isa<top::MulConstOp>(*(sliceop_.getOutput().getUsers().begin()))) {
                      mcop_ = dyn_cast<top::MulConstOp>(
                          *(sliceop_.getResult().user_begin()));
                      if (!isSISO(mcop_))
                        return;
                      idx = 0;
                    } else if (isa<top::ReshapeOp>(*(sliceop_.getOutput().getUsers().begin()))) {
                      rsop2 = dyn_cast<top::ReshapeOp>(*(sliceop_.getOutput().getUsers().begin()));
                      if (isa<top::MatMulOp>(*(rsop2.getResult().user_begin()))) {
                        auto mmop_tmp1 = dyn_cast<top::MatMulOp>(*(rsop2.getResult().user_begin()));
                        if (isa<top::AddOp>(*(mmop_tmp1.getResult().user_begin()))) {
                          idx = 1;
                        } else if (isa<top::PermuteOp>(*(mmop_tmp1.getResult().user_begin()))) {
                          idx = 2;
                        }
                      }
                    }
                  }
                } else {
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
                }
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

        if (dino_flag != 1) {
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
        } else {
          auto reshape_tmp = dyn_cast<top::ReshapeOp>(*(mcop_.getResult().user_begin()));
          mmop1 = dyn_cast<top::MatMulOp>(*(reshape_tmp.getResult().user_begin()));
        }
        qkmmops.push_back(mmop1); // for dino only
        auto addop1 =
            dyn_cast_or_null<top::AddOp>(*(mmop1.getResult().user_begin()));
        if (addop1 == NULL)
          return;
        top::SoftmaxOp smop = dyn_cast_or_null<top::SoftmaxOp>(
            *(addop1.getOutput().getUsers().begin()));
        if (smop == NULL)
          return;
        sm_ops.push_back(smop.getOperation());
        if (dino_flag != 1) {
          if (*(smop.getOutput().getUsers().begin()) !=
            *(pmop_[2].getResult().user_begin()))
          return;
        }
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
                    for (auto user : rsop5.getResult().getUsers()) { // dino updated pattern(July)
                      if (isa<top::SliceOp>(user)) {
                        auto slice_tmp = dyn_cast<top::SliceOp>(user);
                        if (isa<top::ConcatOp>(*(slice_tmp.getResult().user_begin()))) {
                          auto cc_tmp = dyn_cast<top::ConcatOp>(*(slice_tmp.getResult().user_begin()));
                          if (isa<top::SliceOp>(*(cc_tmp.getResult().user_begin()))) {
                            slice_tmp = dyn_cast<top::SliceOp>(*(cc_tmp.getResult().user_begin()));
                            if (isa<top::ConcatOp>(*(slice_tmp.getResult().user_begin()))) {
                              cc_tmp = dyn_cast<top::ConcatOp>(*(slice_tmp.getResult().user_begin()));
                              if (isa<top::SliceOp>(*(cc_tmp.getResult().user_begin()))) {
                                slice_tmp = dyn_cast<top::SliceOp>(*(cc_tmp.getResult().user_begin()));
                                if (isa<top::ReshapeOp>(*(slice_tmp.getResult().user_begin()))) {
                                  rsop5 = dyn_cast<top::ReshapeOp>(*(slice_tmp.getResult().user_begin()));
                                }
                              }
                            }
                          }
                        } else if (isa<top::SliceOp>(*(rsop5.getResult().user_begin()))) {
                          auto slice_tmp = dyn_cast<top::SliceOp>(*(rsop5.getResult().user_begin()));
                          if (isa<top::ReshapeOp>(*(slice_tmp.getResult().user_begin()))) {
                            rsop5 = dyn_cast<top::ReshapeOp>(*(slice_tmp.getResult().user_begin()));
                          }
                        }
                      }
                    }
                    if (auto swdi2 = dyn_cast<top::SwapDimInnerOp>(
                            *(rsop5.getResult().user_begin()))) {
                      for (auto u : swdi2.getResult().getUsers()) {
                        if (isa<top::SwapDimInnerOp>(u)) {
                          swdi2 = dyn_cast_or_null<top::SwapDimInnerOp>(u);
                          if (isa<top::SliceOp>(*(swdi2.getResult().user_begin()))) {
                            auto slice_tmp = dyn_cast<top::SliceOp>(*(swdi2.getResult().user_begin()));
                            if (isa<top::ReshapeOp>(*(slice_tmp.getResult().user_begin()))) {
                              rsop5 = dyn_cast<top::ReshapeOp>(*(slice_tmp.getResult().user_begin()));
                            }
                          }
                          break;
                        } else {
                          ;
                        }
                      }
                      if (*(swdi2.getResult().user_begin()) == aop ||
                          isa<top::ReshapeOp>(*(swdi2.getResult().user_begin()))) { // additional reshape op
                        if (rsop != NULL) {
                          wmsa.push_back(rsop);
                        } else if (mmop != NULL) {
                          wmsa.push_back(mmop);
                        } else if (lnop_ != NULL) {
                          wmsa.push_back(lnop_);
                        } else
                          return;
                      } else if (*(rsop5.getResult().user_begin()) == aop ||
                                 isa<top::AddOp>(*(rsop5.getResult().user_begin()))) {
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
                               aop || isa<top::AddOp>(*(rsop5.getResult().user_begin()))) { // maybe no swdi2
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
                } else if (*(rsop4.getResult().user_begin()) == aop ||
                           isa<top::AddOp>(*(rsop4.getResult().user_begin()))) {
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
              } else if (isa<top::AddOp>(*(mmop3.getResult().user_begin()))) {
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
      }
    }
  });
}

bool tpu_mlir::ConvertTopToTpu::swin_mix_precision() {
  std::vector<Operation *> mlp;
  std::vector<Operation *> wmsa;
  std::vector<Operation *> qkmmops;
  std::vector<Operation *> dep2space;
  std::vector<Operation *> smops;
  bool patch_embed = false;
  bool dino_flag = false;

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
  match_swin_wmsa(wmsa, smops, qkmmops);
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
      top::AddOp addop_end = NULL;
      top::ReshapeOp reshapeop_ = NULL;
      top::PadOp padop = NULL;
      top::SwapDimInnerOp concatop = NULL;
      if (rsop != NULL) {
        auto preOp = rsop->getOperands()[0].getDefiningOp();
        if (isa<top::AddOp>(preOp)) {
          addop = dyn_cast_or_null<top::AddOp>(preOp);
          for (auto u : addop.getResult().getUsers()) {
            if (isa<top::AddOp>(u))
              addop_end = dyn_cast_or_null<top::AddOp>(u);
          }
        } else if (isa<top::LayerNormOp>(preOp)) {
          lnop_ = dyn_cast<top::LayerNormOp>(preOp);
          auto ln_prev = lnop_->getOperands()[0].getDefiningOp();
          if (isa<top::AddOp>(ln_prev)) {
            addop = dyn_cast_or_null<top::AddOp>(ln_prev);
            for (auto u : addop.getResult().getUsers()) {
              if (isa<top::AddOp>(u))
                addop_end = dyn_cast_or_null<top::AddOp>(u);
            }
          }
        }
        for (auto u : rsop.getResult().getUsers()) {
          if (isa<top::LayerNormOp>(u))
            lnop_ = dyn_cast_or_null<top::LayerNormOp>(u);
          if (isa<top::AddOp>(u))
            addop = dyn_cast_or_null<top::AddOp>(u);
          if (isa<top::PadOp>(u)) {
            padop = dyn_cast_or_null<top::PadOp>(u);
            if (isa<top::SwapDimInnerOp>(*(padop.getResult().getUsers().begin()))) { // Try set ConcatOp
              concatop = dyn_cast_or_null<top::SwapDimInnerOp>(*(padop.getResult().getUsers().begin()));
            }
          }
        }
      } else if (mmop != NULL) {
        for (auto u : mmop.getResult().getUsers()) {
          if (isa<top::ReshapeOp>(u)) {
            reshapeop_ = dyn_cast_or_null<top::ReshapeOp>(u);
            for (auto user1 : reshapeop_.getResult().getUsers()) {
              addop = dyn_cast_or_null<top::AddOp>(user1);
            }
          }
          if (isa<top::LayerNormOp>(u)) {
            lnop_ = dyn_cast_or_null<top::LayerNormOp>(u);
            if (isa<top::ReshapeOp>(*(lnop_.getResult().user_begin()))) {
              auto reshape_tmp = dyn_cast<top::ReshapeOp>(*(lnop_.getResult().user_begin()));
              if (isa<top::PadOp>(*(reshape_tmp.getResult().user_begin()))) {
                padop = dyn_cast_or_null<top::PadOp>(*(reshape_tmp.getResult().user_begin()));
              }
            }
          }
          if (isa<top::AddOp>(u))
            addop = dyn_cast_or_null<top::AddOp>(u);
        }
        if (LoweringConfig::quantize_map.find(
                module::getName(mmop.getOperation()).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(mmop.getOperation()).str(), module::Mode::F16});
        }
      } else if (lnop != NULL) {
        auto preOp = lnop->getOperands()[0].getDefiningOp();
        if (isa<top::AddOp>(preOp)) {
          addop = dyn_cast_or_null<top::AddOp>(preOp);
          for (auto u : addop.getResult().getUsers()) {
            if (isa<top::AddOp>(u))
              addop_end = dyn_cast_or_null<top::AddOp>(u);
          }
        }
        for (auto u : lnop.getResult().getUsers()) {
          if (isa<top::ReshapeOp>(u)) {
            reshapeop_ = dyn_cast_or_null<top::ReshapeOp>(u);
            for (auto user1 : reshapeop_.getResult().getUsers()) {
              if (isa<top::AddOp>(user1))
                addop = dyn_cast<top::AddOp>(user1);
            }
          }
          if (isa<top::LayerNormOp>(u)) {
            lnop_ = dyn_cast_or_null<top::LayerNormOp>(u);
            if (isa<top::ReshapeOp>(*(lnop_.getResult().user_begin()))) {
              auto reshape_tmp = dyn_cast_or_null<top::ReshapeOp>(*(lnop_.getResult().user_begin()));
              if (isa<top::PadOp>(*(reshape_tmp.getResult().user_begin())))
                padop = dyn_cast_or_null<top::PadOp>(*(reshape_tmp.getResult().user_begin()));
            }
          }
          if (isa<top::AddOp>(u))
            addop = dyn_cast_or_null<top::AddOp>(u);
        }
      }
      if (addop == NULL || lnop_ == NULL)
        return false;
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
      if (addop_end != NULL) {
        if (LoweringConfig::quantize_map.find(
                module::getName(addop_end.getOperation()).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(addop_end.getOperation()).str(), module::Mode::F16});
        }
        auto preadd = addop_end->getOperands()[0].getDefiningOp(); //set reshape before add to F16
        if (isa<top::ReshapeOp>(preadd)) {
          auto rs_add = dyn_cast<top::ReshapeOp>(preadd);
          if (LoweringConfig::quantize_map.find(
                  module::getName(rs_add.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(rs_add.getOperation()).str(), module::Mode::F16});
          }
        }
      }
      if (padop != NULL) { // set some pad to f16
        dino_flag = true;
        if (LoweringConfig::quantize_map.find(
                module::getName(padop.getOperation()).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(padop.getOperation()).str(), module::Mode::F16});
        }
      }
      if (concatop != NULL) { // set some Concat to f16
        if (LoweringConfig::quantize_map.find(
                module::getName(concatop.getOperation()).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(concatop.getOperation()).str(), module::Mode::F16});
        }
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
      if (!(module::isBM1688() || module::isSG2380() || module::isMARS3())){
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
      // Try More MLP nodes to F16(for dino only)
      if (dino_flag) {
        for (auto in : addop.getOperands()) {
          if (auto mmop = dyn_cast<top::MatMulOp>(in.getDefiningOp())) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(mmop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(mmop.getOperation()).str(),
                  module::Mode::F16});
            }
            if (auto geluop = dyn_cast<top::GELUOp>(
                    mmop.getOperands()[0].getDefiningOp())) {
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

        for (auto qkop : qkmmops) { // some q,k matmul might affect accuracy
          auto qkmmop = dyn_cast_or_null<top::MatMulOp>(qkop);
          if (LoweringConfig::quantize_map.find(
                  module::getName(qkmmop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(qkmmop.getOperation()).str(), module::Mode::F16});
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
      if (dino_flag) {
        // might try matmul after softmax
        top::MatMulOp vmmop = NULL;
        if (isa<top::MatMulOp>(*(smop.getResult().user_begin()))) {
          vmmop = dyn_cast<top::MatMulOp>(*(smop.getResult().user_begin()));
        }
        if (LoweringConfig::quantize_map.find(
                module::getName(vmmop.getOperation()).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(vmmop.getOperation()).str(), module::Mode::F16});
        }
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
    if (!(module::isBM1688() || module::isSG2380() || module::isMARS3())){
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
      set_block_fp16<top::MatMulOp>(aopu.getOperation(), aopl.getOperation());
      set_block_fp16<top::MatMulOp>(mmop.getOperation(), aopl.getOperation());
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

  spread_q_config();

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

template <typename opType>
bool ConvertTopToTpu::set_block_fp16(Operation *from, Operation *to) {
  bool res = true;
  if (isa<opType>(from)) {
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
    res &= set_block_fp16<opType>(r, to);
  }
  return res;
}

void ConvertTopToTpu::spread_q_config() {
  mainFunc_.walk([&](Operation *op) {
    if (isa<top::PermuteOp, top::ReshapeOp, top::SliceOp, top::SqueezeOp,
            top::UnsqueezeOp>(op)) {
      auto pre_op = op->getOperands()[0].getDefiningOp();
      if (LoweringConfig::quantize_map.find(module::getName(pre_op).str()) !=
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(op).str(),
             LoweringConfig::quantize_map.find(module::getName(pre_op).str())
                 ->second});
      }
    }
  });
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
      top::AddOp b_aop = NULL;
      top::LayerNormOp b_lnop = NULL;
      if (LoweringConfig::quantize_map.find(module::getName(op).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(op).str(), module::Mode::F16});
      }
      for (auto i : op->getOperands()) {
        if (auto aop = dyn_cast<top::AddOp>(i.getDefiningOp())) {
          b_aop = aop;
          if (LoweringConfig::quantize_map.find(
                  module::getName(aop.getOperation()).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(aop.getOperation()).str(), module::Mode::F16});
          }
        }
      }
      if (b_aop == NULL)
        return false;
      for (auto o : b_aop.getResult().getUsers()) {
        if (isa<top::LayerNormOp>(o))
          b_lnop = dyn_cast<top::LayerNormOp>(o);
      }
      if (b_lnop == NULL)
        return false;
      for (auto i : op->getOperands()) {
        if (auto mmop = dyn_cast<top::MatMulOp>(i.getDefiningOp())) {
          if (mlp.size() == 12) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(mmop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(mmop.getOperation()).str(),
                   module::Mode::F16});
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
          } else if (mlp.size() == 24 && cnt <= 22) {
            if (LoweringConfig::quantize_map.find(
                    module::getName(mmop.getOperation()).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(mmop.getOperation()).str(),
                   module::Mode::F16});
            }
            if (convergence(b_lnop.getOperation(), op)) {
              set_block_fp16<top::MulOp>(b_lnop.getOperation(), op);
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
      if ((cnt == 2 &&
           mlp.size() == 12)) { /// || (mlp.size() == 24 && cnt <= 7)) {
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
          set_block_fp16<top::MatMulOp>(lnop.getOperation(),
                                        aop.getOperation());
          set_block_fp16<top::MulOp>(lnop.getOperation(), aop.getOperation());
          set_block_fp16<top::MulConstOp>(lnop.getOperation(),
                                          aop.getOperation());
          set_block_fp16<top::ConcatOp>(lnop.getOperation(),
                                        aop.getOperation());
          set_block_fp16<top::AddOp>(lnop.getOperation(), aop.getOperation());
        }
      }
    }
    spread_q_config();
    return true;
  }
  return false;
}

} // namespace tpu_mlir
