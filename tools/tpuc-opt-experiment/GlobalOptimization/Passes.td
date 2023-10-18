#ifndef GLOBAL_OPTIMIZATION_PASSES
#define GLOBAL_OPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"
def DetachElementwiseFromNamedOps :
    Pass<"detach-elementwise-from-named-ops", ""> {
  let summary = "Detaches elementwise ops from named Linalg ops";
  let constructor = "mlir::createDetachElementwiseFromNamedOpsPass()";
}
#endif
