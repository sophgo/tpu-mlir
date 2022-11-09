//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

// #include "tpu_mlir/Backend/BM168x/cv18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GloballGenInterface
// =========================================
template <typename T>
void broadcast_table(std::vector<T> &table, int n, std::vector<T> &table_out) {
  for (int i = 0; i < n; ++i) {
    table_out.insert(table_out.end(), table.begin(), table.end());
  }
}

void tpu::SoftmaxOp::weight_reorder_int8_cv18xx() {

}

void tpu::SoftmaxOp::weight_reorder_bf16_cv18xx() {
  auto op = getOperation();
  auto chip_name = Module::getChip(this->getOperation()).lower();
  CviBackendContext ctx(chip_name.c_str());
  auto table_data = (table().getDefiningOp<top::WeightOp>()).read<uint16_t>();
  auto slope_data = (slope_table().getDefiningOp<top::WeightOp>()).read<uint16_t>();
  auto reciprocal_data = (reciprocal_table().getDefiningOp<top::WeightOp>()).read<uint16_t>();
  auto reciprocal_mantissa_data = (reciprocal_mantissa_table().getDefiningOp<top::WeightOp>()).read<uint16_t>();
  std::vector<uint16_t> new_table;
  std::vector<uint16_t> new_slope_table;
  std::vector<uint16_t> new_reciprocal_table;
  std::vector<uint16_t> new_reciprocal_mantissa_table;
  broadcast_table(*table_data, NPU_NUM, new_table);
  broadcast_table(*slope_data, NPU_NUM, new_slope_table);
  broadcast_table(*reciprocal_data, NPU_NUM, new_reciprocal_table);
  broadcast_table(*reciprocal_mantissa_data, NPU_NUM, new_reciprocal_mantissa_table);
  OpBuilder builder(getContext());
  auto table_shape = Module::getShape(table()).vec();
  table_shape[table_shape.size() - 3] *= NPU_NUM;
  auto table_type = RankedTensorType::get(table_shape, builder.getBF16Type());
  auto new_table_op =
      top::WeightOp::create(op, "table_reordered", new_table, table_type);
  auto new_slope_op =
      top::WeightOp::create(op, "slope_table_reordered", new_slope_table, table_type);
  auto new_reciprocal_op =
      top::WeightOp::create(op, "reciprocal_table_reordered", new_reciprocal_table, table_type);
  auto new_reciprocal_mantissa_op =
      top::WeightOp::create(op, "reciprocal_mantissa_table_reordered", new_reciprocal_mantissa_table, table_type);
  op->setOperand(1, new_table_op);
  op->setOperand(2, new_slope_op);
  op->setOperand(3, new_reciprocal_op);
  op->setOperand(4, new_reciprocal_mantissa_op);
}

void tpu::SoftmaxOp::codegen_global_cv18xx(void* ctx, int64_t layer_id) {
   CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
   bool do_log = false;
   int axis = this->axis();
   gaddr_t ga_input = Module::getAddress(input());
   gaddr_t ga_output = Module::getAddress(output());
   gaddr_t exponential_table_data_lut_gaddr = Module::getAddress(table());
   gaddr_t exponential_slope_table_data_lut_gaddr = Module::getAddress(slope_table());
   gaddr_t reciprocal_table_data_lut_gaddr = Module::getAddress(reciprocal_table());
   gaddr_t reciprocal_mantissa_table_data_lut_gaddr = Module::getAddress(reciprocal_mantissa_table());
   std::vector<int64_t> shape;
   Module::getShapeVec(input(), shape);
   int dimension = shape.size();
   cvi_backend_tg_bf16_softmax_kernel(
      *backend_ctx, layer_id,
      ga_input,
      exponential_table_data_lut_gaddr, exponential_slope_table_data_lut_gaddr,
      reciprocal_table_data_lut_gaddr, reciprocal_mantissa_table_data_lut_gaddr,
      ga_output,
      shape.data(), axis, dimension, do_log);
}
