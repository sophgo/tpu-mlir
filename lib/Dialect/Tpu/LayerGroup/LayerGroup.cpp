#include "sophgo/Dialect/Tpu/Transforms/Passes.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Helper/Module.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <sstream>
#include <fstream>
#include <set>

using namespace llvm;
using namespace mlir;
using namespace sophgo::helper;
namespace sophgo {
namespace tpu {

class LayerGroupPass : public LayerGroupBase<LayerGroupPass> {
public:
  LayerGroupPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    func.walk([&](Operation *op) {
      // do nothing
    });
    llvm_unreachable("unsupport layer group");
  }
};

std::unique_ptr<OperationPass<FuncOp>> createLayerGroupPass() {
  return std::make_unique<LayerGroupPass>();
}
} // namespace tpu
} // namespace sophgo
