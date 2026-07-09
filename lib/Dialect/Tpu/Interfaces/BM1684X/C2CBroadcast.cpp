//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

static void fill_c2c_broadcast_spec(tpu::C2CBroadcastOp op,
                                    c2c_broadcast_common_spec_t &spec) {
  memset(&spec, 0, sizeof(spec));
  spec.root = op.getRoot();
  spec.nranks = op.getNranks();
  spec.rank = op.getRank();
  spec.sccl_algo = op.getScclAlgo();
  auto chip_map = module::getI64Array(op.getChipMap());
  size_t map_size = std::min(chip_map->size(), (size_t)MAX_RANKS);
  for (size_t i = 0; i < map_size; i++) {
    spec.chip_map[i] = chip_map->at(i);
  }
}

void tpu::C2CBroadcastOp::codegen_global_bm1684x() {
  if (!module::isBM1684X2()) {
    llvm_unreachable("C2CBroadcast only supports BM1684X2");
  }
  c2c_broadcast_common_spec_t spec = {0};
  fill_c2c_broadcast_spec(*this, spec);
  auto op = getOperation();
  auto output_spec = BM168x::get_output_spec(op);
  auto input_spec =
      module::isNone(getInput()) ? output_spec : BM168x::get_input_spec(op);
  BM168x::call_global_func("backend_api_c2c_broadcast_global", &spec,
                           sizeof(spec), input_spec->data(),
                           output_spec->data());
}

int64_t tpu::C2CBroadcastOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(c2c_broadcast_common_spec_t);
  c2c_broadcast_common_spec_t spec = {0};
  fill_c2c_broadcast_spec(*this, spec);
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::C2CBroadcastOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
