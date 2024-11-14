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

// =========================================
// LocalGenInterface
// =========================================

void move_slice(int64_t src_add, int64_t dest_add, int64_t size,
                DATA_TYPE_T dtype) {
  tensor_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.dtype = dtype;
  spec.shape[0] = 1;
  spec.shape[1] = Arch::NPU_NUM;
  spec.shape[2] = 1;
  spec.shape[3] =
      size / BM168x::getFmtBytes(dtype); // todo 搬移的参数具有不同的dtype

  std::vector<tensor_spec_t> input_spec, output_spec;
  spec.addr = src_add;
  input_spec.push_back(spec);
  spec.addr = dest_add;
  output_spec.push_back(spec);

  bdc_cpy_spec_t param = {0};
  BM168x::call_local_func("backend_api_bdc_cpy_local", &param, sizeof(param),
                          nullptr, input_spec.data(), output_spec.data());
}

SmallString<128> gen_op_id(Operation *op, int i) {
  SmallString<128> prefix = op->getName().getStringRef().substr(4);
  prefix.append({"_", std::to_string(i)});
  return prefix;
}

void tpu::MoveOp::codegen_only_for_moveOp(std::vector<int64_t> &move_src_add,
                                          std::vector<int64_t> &move_dest_add,
                                          std::vector<int64_t> &move_size) {
  auto op = getOperation();
  auto bm168x = BM168x::instance();
  DATA_TYPE_T dtype = BM168x::getDataType(op->getOperand(0));
  auto pid_node = (CMD_ID_NODE *)(*BM168x::instance())->bdc_node;
  int move_num = move_src_add.size(), src_addr = 0, dest_addr = 0;
  llvm::errs() << "codegen_only_for_moveOp:\n";
  for (int i = 0; i < move_num; i++) {
    if (move_dest_add[i] > move_src_add[i]) { //目标地址在后面
      if (move_dest_add[i] < move_src_add[i] + move_size[i]) { //重叠
        llvm::errs() << "move_src_add[i]:" << move_src_add[i]
                     << ", move_dest_add[i]:" << move_dest_add[i]
                     << ", move_size[i]:" << move_size[i] << "\n";
        int size_per_copy = move_dest_add[i] - move_src_add[i];
        int copy_num = move_size[i] / size_per_copy;
        src_addr = move_src_add[i] + move_size[i];
        dest_addr = move_dest_add[i] + move_size[i];
        for (int j = 0; j < copy_num; j++) {
          bm168x->divide_sync_id();
          BM168x::instance()->dl_set_cmd_id_prefix(pid_node,
                                                   gen_op_id(op, j).c_str());
          src_addr -= size_per_copy;
          dest_addr -= size_per_copy;
          llvm::errs() << "tensor" << i << ", " << j
                       << "th copy, src_addr:" << src_addr
                       << ", dest_addr:" << dest_addr
                       << ", size:" << size_per_copy << "\n";
          move_slice(src_addr, dest_addr, size_per_copy, dtype);
          bm168x->merge_sync_id();
        }
        int last_move_size = move_size[i] - copy_num * size_per_copy;
        if (last_move_size) {
          bm168x->divide_sync_id();
          BM168x::instance()->dl_set_cmd_id_prefix(
              pid_node, gen_op_id(op, copy_num).c_str());
          src_addr -= last_move_size;
          dest_addr -= last_move_size;
          llvm::errs() << "tensor" << i << ", copy tail, src_addr:" << src_addr
                       << ", dest_addr:" << dest_addr
                       << ", size:" << last_move_size << "\n";
          move_slice(move_src_add[i], move_dest_add[i], last_move_size, dtype);
          bm168x->merge_sync_id();
        }
        continue;
      }
    } else { //目标地址在前面
      if (move_dest_add[i] + move_size[i] > move_src_add[i]) { //重叠
        int size_per_copy = move_src_add[i] - move_dest_add[i];
        int copy_num = move_size[i] / size_per_copy;
        for (int j = 0; j < copy_num; j++) {
          bm168x->divide_sync_id();
          BM168x::instance()->dl_set_cmd_id_prefix(pid_node,
                                                   gen_op_id(op, j).c_str());
          src_addr = move_src_add[i] + j * size_per_copy;
          dest_addr = move_dest_add[i] + j * size_per_copy;
          move_slice(src_addr, dest_addr, size_per_copy, dtype);
          bm168x->merge_sync_id();
        }
        int last_move_size = move_size[i] - copy_num * size_per_copy;
        llvm::errs() << "codegen_only_for_moveOp, copy_num:" << copy_num
                     << ", last_move_size:" << last_move_size << "\n";
        if (last_move_size) {
          bm168x->divide_sync_id();
          BM168x::instance()->dl_set_cmd_id_prefix(
              pid_node, gen_op_id(op, copy_num).c_str());
          src_addr = move_src_add[i] + copy_num * size_per_copy;
          dest_addr = move_dest_add[i] + copy_num * size_per_copy;
          move_slice(src_addr, dest_addr, last_move_size, dtype);
          bm168x->merge_sync_id();
        }
        continue;
      }
    }

    //不重叠的情形
    bm168x->divide_sync_id();
    BM168x::instance()->dl_set_cmd_id_prefix(pid_node,
                                             gen_op_id(op, i).c_str());
    move_slice(move_src_add[i], move_dest_add[i], move_size[i], dtype);
    bm168x->merge_sync_id();
  }
}
