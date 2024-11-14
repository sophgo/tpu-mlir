#include "Passes.h"
#include "include/Utils.h"
namespace mlir {

// Todo: conversion stablehlo(from tensorflow), tosa(from tflite) etc to linalg
struct AutoInputConversionPipelinePass final
    : AutoInputConversionPipelineBase<AutoInputConversionPipelinePass> {
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;
};

// All the features seen that should be handled during input conversion.
struct InputFeatures {
  // HLO features.
  bool hasStableHLO = false;
  // - XLA import features.
  bool hasTuples = false;
  // TOSA features.
  bool hasTOSA = false;
};

static void populateHloFeatures(Operation *op, InputFeatures &features) {
  if (features.hasTuples) {
    return;
  }

  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    FunctionType type = dyn_cast<FunctionType>(funcOp.getFunctionType());
    for (auto t : type.getResults()) {
      if (isa<TupleType>(t)) {
        features.hasTuples = true;
        return;
      }
    }
    for (auto t : type.getInputs()) {
      if (isa<TupleType>(t)) {
        features.hasTuples = true;
        return;
      }
    }
  }

  // Check for tuple operands or results.
  for (auto t : op->getOperandTypes()) {
    if (isa<TupleType>(t)) {
      features.hasTuples = true;
      return;
    }
  }
  for (auto t : op->getResultTypes()) {
    if (isa<TupleType>(t)) {
      features.hasTuples = true;
      return;
    }
  }
}

static void populateFeatures(Operation *op, const Dialect *stablehloDialect,
                             const Dialect *tosaDialect,
                             InputFeatures &features) {
  Dialect *d = op->getDialect();
  if (d == stablehloDialect) {
    features.hasStableHLO = true;
    return populateHloFeatures(op, features);
  }
  if (d == tosaDialect) {
    features.hasTOSA = true;
    return;
  }
}

void AutoInputConversionPipelinePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctxt = &getContext();

  InputFeatures features;
  const Dialect *stablehloDialect = ctxt->getLoadedDialect("stablehlo");
  const Dialect *tosaDialect = ctxt->getLoadedDialect("tosa");
  if (!stablehloDialect && !tosaDialect) {
    return;
  }

  auto res = module.walk([&](Operation *op) {
    populateFeatures(op, stablehloDialect, tosaDialect, features);
    if (features.hasStableHLO && features.hasTOSA) {
      module.emitError("not yet implemented mixture of *HLO and TOSA");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) {
    return signalPassFailure();
  }
  if (!features.hasStableHLO && !features.hasTOSA) {
    return;
  }

  OpPassManager pm(ModuleOp::getOperationName(),
                   OpPassManager::Nesting::Explicit);
#ifdef HAVE_STABLEHLO_INPUT
  // Todo:
  if (features.hasStableHLO) {
    stablehlo::StableHloOptions options;
    options.demoteI64ToI32 = demoteI64ToI32;
    options.demoteF64ToF32 = demoteF64ToF32;
    options.promoteBF16ToF32 = promoteBF16ToF32;
    if (features.hasTuples) {
      stablehlo::buildStableHLOXLAInputConversionPassPipeline(pm, options);
    } else {
      stablehlo::buildStableHLOInputConversionPassPipeline(pm, options);
    }
  }
#endif
#ifdef HAVE_TOSA_INPUT
  if (features.hasTOSA) {
    buildTOSAInputConversionPassPipeline(pm);
  }
#endif

  if (failed(runPipeline(pm, module))) {
    signalPassFailure();
  }
}

void AutoInputConversionPipelinePass::getDependentDialects(
    DialectRegistry &registry) const {
  // Register dialects from all possible pipelines, as we do not statically know
  // which pipeline will be selected, while dialect registration happens before
  // we run any detection on the input.

#ifdef HAVE_STABLEHLO_INPUT
  auto appendStablehloPipelineDialects =
      [&registry](function_ref<void(OpPassManager &,
                                    const stablehlo::StableHloOptions &options)>
                      buildFn) {
        const stablehlo::StableHloOptions options;
        OpPassManager pm;
        buildFn(pm, options);
        pm.getDependentDialects(registry);
      };

  appendStablehloPipelineDialects(
      stablehlo::buildStableHLOInputConversionPassPipeline);
  appendStablehloPipelineDialects(
      stablehlo::buildStableHLOXLAInputConversionPassPipeline);
#endif

#ifdef HAVE_TOSA_INPUT
  auto appendPipelineDialects =
      [&registry](function_ref<void(OpPassManager &)> buildFn) {
        OpPassManager pm;
        buildFn(pm);
        pm.getDependentDialects(registry);
      };
  appendPipelineDialects(buildTOSAInputConversionPassPipeline);
#endif
}

std::unique_ptr<OperationPass<ModuleOp>>
createAutoInputConversionPipelinePass() {
  return std::make_unique<AutoInputConversionPipelinePass>();
}
} // namespace  mlir
