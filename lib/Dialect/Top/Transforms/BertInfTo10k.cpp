//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"

using namespace llvm;
using namespace mlir;

namespace tpu_mlir {
namespace top {

class BertInfTo10kPass : public BertInfTo10kBase<BertInfTo10kPass> {
public:
  BertInfTo10kPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    auto getUserCnt = [] (Operation *op) {
      auto out = op->getResult(0);
      if (out.getUsers().empty())
        return 0;
      else {
        int cnt=0;
        auto x=out.user_begin();
        while(x!=out.user_end()) {
          cnt++;
          x=std::next(x);
        }
        return cnt;
      }
    };
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (op->getLoc().dyn_cast<NameLoc>() && !module::isOpInGroup(op)) {
          if (auto mulconstop = dyn_cast<top::MulConstOp>(op)){
            if (getUserCnt(mulconstop) == 12 || getUserCnt(mulconstop) == 24) {
              for (auto nopmc : mulconstop->getResult(0).getUsers()) {
                if (auto addop = dyn_cast<top::AddOp>(nopmc)){
                  if(getUserCnt(addop) != 1) {
                    return;
                  }
                  for (auto nopadd : addop->getResult(0).getUsers()) {
                    if (auto softmaxop = dyn_cast<top::SoftmaxOp>(nopadd)) {
                      continue;
                    }
                    else {
                      return;
                    }
                  }
                }
                else {
                  return;
                }
              }
              float const_v = float(mulconstop.getConstVal().convertToDouble());
              if ( std::isinf(const_v) != 0 || const_v < -1.0*1e-38) {
                mulconstop.setConstVal(llvm::APFloat(-10000.0));
              }
            }
          }
        }
  	  });
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createBertInfTo10kPass() {
  return std::make_unique<BertInfTo10kPass>();
}
} // namespace top
} // namespace tpu_mlir
