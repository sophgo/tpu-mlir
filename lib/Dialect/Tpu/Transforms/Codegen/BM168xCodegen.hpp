//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once
#include "tpu_mlir/Builder/BM168x/bmodel.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicNetIr.hpp"
#include "ProfileCtx.h"
#include "TensorLocation.hpp"

using namespace llvm;


using namespace flatbuffers;
namespace tpu_mlir {
namespace tpu {

class BMCodegen {
public:
  BMCodegen() {}
  void run(ModuleOp &module, std::string &filename);

private:
  u32 max_cpu_mem_size = 0;
  Offset<Vector<Offset<bmodel::Shape>>>
  CreateShapeVector(const ArrayRef<int64_t> &shape);
  Offset<Vector<Offset<bmodel::Tensor>>>
  CreateTensorVector(const std::vector<Value> &values);
  Offset<bmodel::SubNet> CreateSubNet(func::CallOp call);
  Offset<bmodel::SubNet> CreateSubNet(func::CallOp call,
                                      std::unique_ptr<SubnetIr> subnet_ir_,
                                      std::unique_ptr<Context> &context);
  Offset<bmodel::SubNet> CreateCPUSubNet(func::CallOp call);
  std::shared_ptr<std::vector<Offset<bmodel::CmdGroup>>> CreateCmdGroupVector();
  Offset<bmodel::CoeffMem> CreateCoeffMem(std::vector<top::WeightOp> &coeffs,
                                          uint64_t coeff_addr,
                                          uint64_t coeff_size);
  Offset<Vector<Offset<bmodel::StageIR>>>
  CreateStageIRVector(const vector<stage_param_t> &stage_param_v,
                      const vector<u32> &binary_ir_v, u32 ir_offset,
                      bmodel::Binary &binary_ir);
  void codegen(Operation *op);
  void codegen_for_group(GroupOp gOP, Operation *prev_op, Operation *next_op);
  void codegen_for_overlap_ops(
      std::map<int64_t, std::vector<Operation *>> cur_other_downs,
      std::map<int64_t, std::vector<Operation *>> cur_other_ups,
      Operation *prev_op, Operation *next_op, int64_t cur_ts,
      bool first_compute_loop, bool last_compute_loop);
  void codegen_ir(Operation *op, SubnetIr *subnet_ir_);
  SmallString<128> gen_op_id(Operation *op);

private:
  ModuleOp module;
  StringRef state;
  std::string chip;
  BM168x *bm168x;
  std::shared_ptr<bmodel::ModelGen> model_gen;
  std::shared_ptr<std::vector<Offset<bmodel::CmdGroup>>> cmd_group_all;
  TensorLocation tensor_loc;
  ProfileCtx profile_ctx;
  std::unordered_map<std::string, std::vector<bool>> tensor_is_cpu;
  AsmState::LocationMap opToLineCol;
};

} // namespace tpu
} // namespace tpu_mlir
