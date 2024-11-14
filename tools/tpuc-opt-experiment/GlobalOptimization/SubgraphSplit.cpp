#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include <functional>
#include <iostream>
#include <vector>
namespace mlir {
enum RunMode : int { STATIC = 0, DYNAMIC = 1 };

static llvm::StringRef getMode(RunMode type) {
  switch (type) {
  case RunMode::STATIC:
    return "static";
  case RunMode::DYNAMIC:
    return "dynamic";
  default:
    assert(0 && "error");
  }
}

static void getInputsOutputs(std::vector<Operation *> &ops,
                             std::vector<Value> &inputs,
                             std::vector<Value> &outputs) {
  std::vector<Value> allValues;
  for (auto op : ops) {
    for (auto v : op->getResults()) {
      allValues.push_back(v);
    }
  }
  for (auto op : ops) {
    for (auto v : op->getOperands()) {
      if (find(inputs.begin(), inputs.end(), v) != inputs.end()) {
        continue;
      }

      if (v.isa<BlockArgument>()) {
        inputs.push_back(v);
        continue;
      }

      if (find(allValues.begin(), allValues.end(), v) == allValues.end()) {
        inputs.push_back(v);
      }
    }
    for (auto v : op->getResults()) {
      if (find(outputs.begin(), outputs.end(), v) != outputs.end()) {
        continue;
      }
      for (auto use : v.getUsers()) {
        if (find(ops.begin(), ops.end(), use) == ops.end()) {
          outputs.push_back(v);
          break;
        }
      }
    }
  }

  /*inputs.erase(
      std::unique(inputs.begin(), inputs.end(), [](const Value &lhs, const Value
  &rhs) { return lhs.getImpl() < rhs.getImpl();
      }), inputs.end());

  outputs.erase(
      std::unique(outputs.begin(), outputs.end(), [](const Value &lhs, const
  Value &rhs) { return lhs.getImpl() < rhs.getImpl();
      }), outputs.end());*/
}

struct subnet_basic_info;
using InfoVec = llvm::SmallVector<subnet_basic_info *>;
struct subnet_basic_info {
  subnet_basic_info() : id(__next_id) {
    index = __next_id;
    __next_id++;
  }
  subnet_basic_info(int index_, std::vector<int> &&next_index_, RunMode type_,
                    const int id_ = 0)
      : index(index_), next_index(std::move(next_index_)), type(type_),
        id(id_) {
    __next_id++;
  }

  void clear_io() noexcept {
    ins.clear();
    outs.clear();
  }
  static void reset_id() { __next_id = 0; }
  RunMode type;
  std::vector<Operation *> ops;
  std::vector<Value> ins;
  std::vector<Value> outs;
  const int id;
  int index;
  static int __next_id;
  InfoVec next_subnets;
  InfoVec prev_subnets;
  std::vector<int> next_index;
};
int subnet_basic_info::__next_id = 0;

class SubgraphSplitPass
    : public PassWrapper<SubgraphSplitPass, OperationPass<ModuleOp>> {
private:
  bool mode; // 0: static 1: dynamic
public:
  SubgraphSplitPass(bool dynamic_mode) : mode(dynamic_mode) {}
  SubgraphSplitPass() : mode(false) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  bool isReadyRun(Operation *op, llvm::DenseSet<Value> &valid_values) {
    int valid_num = 0;
    for (int i = 0; i < op->getNumOperands(); i++) {
      if (valid_values.count(op->getOperand(i)))
        valid_num++;
    }
    return valid_num == op->getNumOperands();
  }

  bool dynamicRun(Operation *op) {
    auto isStaticShape = [&](Type type) {
      auto isDynamic = [&](Type type) {
        auto t = llvm::dyn_cast<RankedTensorType>(type);
        if (t && !t.hasStaticShape())
          return true;
        // Todo: VectorType
        return false;
      };

      if (isa<mlir::UnrankedTensorType>(type) ||
          (isa<mlir::RankedTensorType, mlir::VectorType>(type) &&
           isDynamic(type)))
        return false;
      return true;
    };

    if (llvm::all_of(op->getOperandTypes(), isStaticShape) &&
        llvm::all_of(op->getResultTypes(), isStaticShape))
      return false;
    else
      return true;
  }

  void Outliner(mlir::FunctionOpInterface &funcOp, InfoVec &subnets) {
    for (auto &subnet : subnets) {
      std::vector<Type> argType;
      std::vector<Type> resType;
      std::vector<Value> fnInputs;
      std::vector<Value> fnOutputs;
      getInputsOutputs(subnet->ops, fnInputs, fnOutputs);
      std::vector<Location> argLoc;
      OpBuilder builder(&getContext());
      OpBuilder::InsertionGuard insertGuard(builder);

      for (auto &input : fnInputs) {
        argType.push_back(input.getType());
        argLoc.push_back(input.getLoc());
      }

      for (auto &output : fnOutputs) {
        resType.push_back(output.getType());
      }

      auto moduleOp = SymbolTable::getNearestSymbolTable(funcOp);
      builder.setInsertionPointToStart(&moduleOp->getRegion(0).front());
      int64_t id = subnet->index;
      std::string func_name =
          funcOp.getName().str() + "subfunc_" + std::to_string(id);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          builder.getNamedAttr("id", builder.getI64IntegerAttr(id)));
      attrs.push_back(builder.getNamedAttr(
          "mode", builder.getStringAttr(getMode(subnet->type))));
      attrs.push_back(builder.getNamedAttr(
          "next_index", builder.getDenseI32ArrayAttr(subnet->next_index)));

      auto fnType =
          FunctionType::get(&getContext(), llvm::ArrayRef<Type>{argType},
                            llvm::ArrayRef<Type>{resType});
      auto fnOp =
          builder.create<func::FuncOp>(builder.getUnknownLoc(), func_name,
                                       fnType, ArrayRef<NamedAttribute>(attrs));

      auto block = fnOp.addEntryBlock();
      builder.setInsertionPoint(subnet->ops.back());
      func::CallOp callOp = builder.create<func::CallOp>(
          funcOp->getParentOfType<mlir::ModuleOp>().getLoc(), func_name,
          resType, fnInputs);
      for (auto it : llvm::enumerate(callOp.getResults())) {
        fnOutputs[it.index()].replaceUsesWithIf(
            it.value(), [&](OpOperand &operand) {
              Operation *user = operand.getOwner();
              return find(subnet->ops.begin(), subnet->ops.end(), user) ==
                     subnet->ops.end();
            });
      }

      builder.setInsertionPointToEnd(block);

      auto retOp = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                                        fnOutputs);
      for (auto &op : subnet->ops) {
        op->moveBefore(retOp);
      }

      for (auto it : llvm::enumerate(fnInputs)) {
        auto arg = block->getArgument(it.index());
        arg.setLoc(argLoc[it.index()]);
        it.value().replaceUsesWithIf(arg, [&](OpOperand &operand) {
          return fnOp->isProperAncestor(operand.getOwner());
        });
      }
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    auto funcOps =
        llvm::to_vector(moduleOp.getOps<mlir::FunctionOpInterface>());

    for (auto funcOp : funcOps) {
      if (funcOp.isDeclaration())
        continue;
      RewritePatternSet patterns(context);
      {
        // legalize the control flow op
        scf::IfOp::getCanonicalizationPatterns(patterns, context);
        scf::ExecuteRegionOp::getCanonicalizationPatterns(patterns, context);
        scf::ForOp::getCanonicalizationPatterns(patterns, context);
        scf::ForallOp::getCanonicalizationPatterns(patterns, context);
        scf::ParallelOp::getCanonicalizationPatterns(patterns, context);
        scf::WhileOp::getCanonicalizationPatterns(patterns, context);
        affine::AffineIfOp::getCanonicalizationPatterns(patterns, context);
        if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
          return signalPassFailure();
        }
      }

      std::vector<Operation *> ops;
      llvm::DenseSet<Value> valid_values;
      valid_values.insert(funcOp.args_begin(), funcOp.args_end());
      funcOp.walk<WalkOrder::PreOrder, ForwardDominanceIterator<true>>(
          [&](Operation *op) {
            if (isa<mlir::FunctionOpInterface>(op->getParentOp()) &&
                !op->hasTrait<OpTrait::ReturnLike>()) {
              ops.emplace_back(op);
              if (op->hasTrait<OpTrait::ConstantLike>())
                valid_values.insert(op->result_begin(), op->result_end());
            }

            return WalkResult::advance();
          });

      // step1: toposort
      bool result = mlir::computeTopologicalSorting(ops);
      assert(result && "unable to sort topologically");

      // step2: basic split
      subnet_basic_info::reset_id();
      InfoVec subnet_infos;
      auto dfs = [&]() noexcept -> void {
        while (!ops.empty()) {
          subnet_infos.emplace_back(new subnet_basic_info);
          auto &info = subnet_infos.back();
          info->type = mode ? RunMode::DYNAMIC : RunMode::STATIC;
          bool updated = false;
          do {
            updated = false;
            for (auto op : ops) {
              if (isReadyRun(op, valid_values)) {
                if (isa<RegionBranchOpInterface>(op) &&
                    op->hasTrait<OpTrait::NoRegionArguments>()) {
                  // Todo
                  assert(0 && "don; support now");
                } else if (info->type != RunMode::DYNAMIC && dynamicRun(op)) {
                  if (!info->ops.empty())
                    continue;
                  info->type = RunMode::DYNAMIC;
                  info->ops.emplace_back(op);
                  valid_values.insert(op->result_begin(), op->result_end());
                } else {
                  if (isa<mlir::FunctionOpInterface>(op->getParentOp())) {
                    info->ops.emplace_back(op);
                    valid_values.insert(op->result_begin(), op->result_end());
                    updated = true;
                  }
                }
              }
            }

            for (auto op : info->ops) {
              auto it = std::find(ops.begin(), ops.end(), op);
              if (it != ops.end())
                ops.erase(it);
            }
          } while (updated);
        }
      };

      std::invoke(dfs);
      // Todo:sort、merger、move op betweens subnets etc

      // stepN: outliner
      Outliner(funcOp, subnet_infos);

      for (auto info : subnet_infos)
        delete info;
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createSubgraphSplitPass(bool dynamic_node) {
  return std::make_unique<SubgraphSplitPass>(dynamic_node);
}
} // namespace mlir
