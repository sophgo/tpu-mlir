//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Builder/BM168x/bmodel.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::backend;
using namespace tpu_mlir::helper;
using namespace flatbuffers;
namespace tpu_mlir {
namespace tpu {

class CodegenPass : public CodegenBase<CodegenPass> {
public:
  CodegenPass() {}
  void runOnOperation() override {
    module = getOperation();
    assert(Module::isState(Module::State::TPU_ADDRESSED));
    chip = Module::getChip();
    std::string filename = this->model_file;
    if (filename.empty()) {
      llvm_unreachable("output filename is empty");
    }
    Arch::init();
    bm168x = BM168x::instance();
    bm168x->start_env();
    std::vector<top::WeightOp> weights;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) {
        // TODO: store all weight to gmem for compare
        // bm168x->value_s2d(op.output(), op.read_as_byte()->data());
        weights.push_back(op);
      });
    }
    std::vector<Value> inputs;
    std::vector<Value> outputs;
    Module::getInputsOutputs(inputs, outputs);

    auto coeff_addr = Module::getCoeffAddr();
    auto coeff_size = Module::getCoeffSize();
    auto neuron_addr = Module::getNeuronAddr();
    auto neuron_size = Module::getNeuronSize();
    model_gen = std::make_shared<bmodel::ModelGen>();
    // add chip name
    model_gen->AddChip(chip.str());
    auto &builder = model_gen->Builder();
    auto input_tensor = CreateTensorVector(inputs);
    auto output_tensor = CreateTensorVector(outputs);
    auto coeff_mem = CreateCoeffMem(weights, coeff_addr, coeff_size);
    std::vector<uint64_t> neuron_sizes = {(uint64_t)neuron_size};
    auto neuron_sizes_fb = builder.CreateVector(neuron_sizes);
    // codegen all subnet
    cmd_group_all = std::make_shared<std::vector<Offset<bmodel::CmdGroup>>>();
    auto main_func = Module::getMainFuncOp();
    std::vector<Offset<bmodel::SubNet>> subnet_v;
    main_func.walk([&](func::CallOp call) {
      auto subnet = CreateSubNet(call);
      subnet_v.push_back(subnet);
    });
    auto subnets = builder.CreateVector(subnet_v);
    auto cmd_group = model_gen->Builder().CreateVector(*cmd_group_all);
    bmodel::NetParameterBuilder npb(builder);
    npb.add_input_tensor(input_tensor);
    npb.add_output_tensor(output_tensor);
    npb.add_ctx_addr(neuron_addr);
    npb.add_ctx_size(neuron_size);
    npb.add_ctx_sizes(neuron_sizes_fb);
    npb.add_coeff_mem(coeff_mem);
    npb.add_is_dynamic(false);
    npb.add_n_dynamic(false);
    npb.add_h_w_dynamic(false);
    npb.add_cmd_group(cmd_group);
    // create subnet
    npb.add_sub_net(subnets);
    model_gen->AddNet(Module::getModuleName().str(), npb.Finish());
    model_gen->Finish();
    model_gen->Save(filename);
    bm168x->end_env();
  }

private:
  Offset<Vector<Offset<bmodel::Shape>>>
  CreateShapeVector(const ArrayRef<int64_t> &shape);
  Offset<Vector<Offset<bmodel::Tensor>>>
  CreateTensorVector(const std::vector<Value> &values);
  Offset<bmodel::SubNet> CreateSubNet(func::CallOp call);
  std::shared_ptr<std::vector<Offset<bmodel::CmdGroup>>> CreateCmdGroupVector();
  Offset<bmodel::CoeffMem> CreateCoeffMem(std::vector<top::WeightOp> &coeffs,
                                          uint64_t coeff_addr,
                                          uint64_t coeff_size);
  void codegen(Operation *op);
  void codegen_for_group(tpu::GroupOp gOP);

private:
  ModuleOp module;
  StringRef state;
  StringRef chip;
  BM168x *bm168x;
  std::shared_ptr<bmodel::ModelGen> model_gen;
  std::shared_ptr<std::vector<Offset<bmodel::CmdGroup>>> cmd_group_all;
};

std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass() {
  return std::make_unique<CodegenPass>();
}

Offset<Vector<Offset<bmodel::Shape>>>
CodegenPass::CreateShapeVector(const ArrayRef<int64_t> &shape) {
  auto &builder = model_gen->Builder();
  std::vector<Offset<bmodel::Shape>> stage_shape_v;
  std::vector<uint64_t> dims;
  for (auto dim : shape) {
    dims.push_back(dim);
  }
  auto dim_vector = builder.CreateVector(dims);
  auto shapes = bmodel::CreateShape(builder, dim_vector);
  stage_shape_v.push_back(shapes);
  return builder.CreateVector(stage_shape_v);
}

Offset<Vector<Offset<bmodel::Tensor>>>
CodegenPass::CreateTensorVector(const std::vector<Value> &values) {
  auto &builder = model_gen->Builder();
  std::vector<Offset<bmodel::Tensor>> tensor_v;
  for (auto v : values) {
    auto v_name = Module::getName(v).str();
    auto type = Module::getStorageType(v);
    auto shape = Module::getShape(v);
    auto typeBytes = type.getIntOrFloatBitWidth() / 8;
    auto data_type = BM168x::getDataType(type);
    auto gmem_stmode = STORE_MODE_1N;
    if (chip == Module::Chip::BM1684) {
      if (typeBytes == 1) {
        gmem_stmode = STORE_MODE_4N;
      } else if (typeBytes == 2) {
        gmem_stmode = STORE_MODE_2N;
      }
    }
    // shape info
    auto name = builder.CreateString(v_name);
    auto stage_shape = CreateShapeVector(shape);
    bmodel::TensorBuilder tb(builder);
    tb.add_name(name);
    tb.add_data_type(data_type);
    tb.add_gmem_stmode(gmem_stmode);
    tb.add_shape(stage_shape);
    tb.add_mem_type(MEM_TYPE_TPU);
    float scale = 1.0f;
    int zero_point = 0;
    if (Quant::isUniformQuantized(v)) {
      auto qtype = Quant::getUniformQuantizedType(v);
      scale = qtype.getScale();
      if (isa<top::InputOp>(v.getDefiningOp())) {
        scale = 1.0 / qtype.getScale();
      }
      zero_point = qtype.getZeroPoint();
      tb.add_scale(scale);
      tb.add_zero_point(zero_point);
    }
    tb.add_device_addr(Module::getAddress(v));
    tb.add_size(Module::getBytes(v));
    tensor_v.push_back(tb.Finish());
  }
  return builder.CreateVector(tensor_v);
}

Offset<bmodel::CoeffMem>
CodegenPass::CreateCoeffMem(std::vector<top::WeightOp> &coeffs,
                            uint64_t coeff_addr, uint64_t coeff_size) {
  if (coeff_size == 0) {
    return 0;
  }
  auto data_u8 = std::make_shared<std::vector<uint8_t>>(coeff_size, 0);
  uint64_t offset = 0;
  for (auto weight : coeffs) {
    auto data = weight.read_as_byte();
    memcpy(data_u8->data() + offset, data->data(), data->size());
    offset += align_up((int64_t)data->size(), BM168x::ALIGNMENT);
  }
  assert(offset == coeff_size);
  std::vector<uint8_t> sha256(bmodel::SHA256_LEN, 0);
  bmodel::CalcSha256(data_u8->data(), coeff_size, sha256.data());
  auto binary_coeff = model_gen->WriteBinary(coeff_size, data_u8->data());
  auto coeff_sha256 = model_gen->Builder().CreateVector(sha256);
  bmodel::CoeffMemBuilder cmb(model_gen->Builder());
  cmb.add_address(coeff_addr);
  cmb.add_check_code(coeff_sha256);
  cmb.add_binary_coeff(&binary_coeff);
  return cmb.Finish();
}

std::shared_ptr<std::vector<Offset<bmodel::CmdGroup>>>
CodegenPass::CreateCmdGroupVector() {
  auto cmd_group_v = std::make_shared<std::vector<Offset<bmodel::CmdGroup>>>();
  auto gdma_ptr = (uint8_t *)bm168x->gdma_buffer.data();
  auto bdc_ptr = (uint8_t *)bm168x->bdc_buffer.data();
  int bdc_offset = 0, gdma_offset = 0;
  for (int group_idx = 0; group_idx < bm168x->cmdid_groupnum; group_idx++) {
    auto bdc_num = bm168x->bdc_group_id[group_idx];
    auto gdma_num = bm168x->gdma_group_id[group_idx];
    bmodel::Binary binary_bdc;
    bmodel::Binary binary_gdma;
    auto bdc_len = bm168x->get_bdc_len(bdc_num, group_idx);
    if (bdc_num != 0) {
      binary_bdc = model_gen->WriteBinary(bdc_len, bdc_ptr + bdc_offset);
      bdc_offset += bdc_len;
    }
    auto gdma_len = bm168x->get_gdma_len(gdma_num, group_idx);
    if (gdma_num != 0) {
      binary_gdma = model_gen->WriteBinary(gdma_len, gdma_ptr + gdma_offset);
      gdma_offset += gdma_len;
    }
    bmodel::CmdGroupBuilder cgb(model_gen->Builder());
    cgb.add_bdc_num(bdc_num);
    cgb.add_gdma_num(gdma_num);
    cgb.add_bdc_cmd_byte(bdc_len);
    cgb.add_gdma_cmd_byte(gdma_len);
    if (bdc_num != 0) {
      cgb.add_binary_bdc(&binary_bdc);
    }
    if (gdma_num != 0) {
      cgb.add_binary_gdma(&binary_gdma);
    }
    cmd_group_v->push_back(cgb.Finish());
  }
  if (cmd_group_v->size() == 0) {
    return 0;
  }
  return std::move(cmd_group_v);
}

void CodegenPass::codegen_for_group(tpu::GroupOp gOp) {
  auto nsecs = gOp.nsecs();
  auto hsecs = gOp.hsecs();
  auto swpipl_stage_num = gOp.swpipl_stage_num();
  auto &body = gOp.body().front();
  auto flow = Module::getI64Array(gOp.flow());
  // 1. restore timestep_table from flow
  std::vector<std::vector<int64_t>> timestep_table;
  std::vector<int64_t> ts_row;
  int64_t max_id = 0;
  for (size_t i = 1; i < flow->size(); ++i) {
    if (flow->at(i) < 0) {
      timestep_table.push_back(ts_row);
      ts_row.clear();
      continue;
    }
    ts_row.push_back(flow->at(i));
    max_id = std::max(max_id, flow->at(i));
  }
  timestep_table.push_back(ts_row);
  int timestep_num = timestep_table.size();
  // 2. create a vector to map id to op
  std::vector<Operation *> group_ops;
  for (int64_t id = 0; id < max_id;) {
    body.walk([&](Operation *op) {
      if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
        auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0);
        if (ginfo.id == id) {
          group_ops.push_back(op);
          id++;
        }
      }
    });
  }
  // 3. codegen for group
  int64_t stage_idx = 0;
  int64_t draining_idx = 0;
  bool draining_period = false;
  SoftwarePipeline timestep_swpipl;
  local_sec_info_t sec_info;
  for (uint64_t nstep = 0, hstep = 0; nstep < nsecs || draining_period;) {
    /* add for software pipeline */
    timestep_swpipl.write_swloop_buffer(nstep, hstep, swpipl_stage_num);
    for (uint32_t ts = 0; ts < timestep_num; ++ts) {
      bm168x->divide_sync_id();

      auto cur_op_ids = timestep_table[ts];
      for (auto id : cur_op_ids) {
        auto lgOp = cast<LocalGenInterface>(group_ops[id]);
        auto ginfo = lgOp.getGroupInfo(nstep, hstep);
        if ((!draining_period && ginfo.stage > stage_idx) ||
            (draining_period &&
             (ginfo.stage < draining_idx || ginfo.stage > stage_idx))) {
          continue;
        }
        const tensor_step_t *tensor_step =
            timestep_swpipl.read_swloop_buffer(ginfo.stage);
        ginfo = lgOp.getGroupInfo(tensor_step->nstep, tensor_step->hstep);

        // add prefix to each cmd in profile.txt
        std::string prefix = Module::getName(group_ops[id]).str();
        if (ginfo.overstepped == false) {
          if (Module::isBM1684Family()) {
            lgOp.codegen_local_bm1684(tensor_step->nstep, tensor_step->hstep);
          } else if (Module::isBM1684XFamily()) {
            auto pid_node = (CMD_ID_NODE *)BM168x::instance()->bdc_node;
            if (isa<tpu::LoadOp, tpu::StoreOp>(*group_ops[id])) {
              pid_node = (CMD_ID_NODE *)BM168x::instance()->gdma_node;
            }
            BM168x::instance()->dl_set_cmd_id_prefix(pid_node, prefix.c_str());
            lgOp.assign_sec_info(tensor_step->nstep, tensor_step->hstep, &sec_info);
            lgOp.codegen_local_bm1684x(tensor_step->nstep, tensor_step->hstep, &sec_info);
          } else {
            llvm_unreachable("chip not support");
          }
        }
      } // ops, include Load/Store op

      bm168x->merge_sync_id();
    } // timestep

    if (!draining_period) {
      hstep++;
      if (hstep >= hsecs) {
        hstep = 0;
        nstep++;
        if (nstep >= nsecs) {
          draining_period = true;
        }
      }
    }
    if (draining_period) {
      draining_idx++;
      if (draining_idx >= swpipl_stage_num) {
        draining_period = false;
      }
    }
    stage_idx++;
  }
}

void CodegenPass::codegen(Operation *op) {
  if (auto castOp = dyn_cast<tpu::GroupOp>(op)) {
    codegen_for_group(castOp);
  } else if (Module::isOpInGroup(op)) {
    return;
  } else if (auto castOp = dyn_cast<GlobalGenInterface>(op)) {
    if (Module::isBM1684Family()) {
      castOp.codegen_global_bm1684();
    } else if (Module::isBM1684XFamily()) {
      castOp.codegen_global_bm1684x();
    } else {
      llvm_unreachable("chip not support");
    }
  }
}

Offset<bmodel::SubNet> CodegenPass::CreateSubNet(func::CallOp call) {
  bm168x->before_codegen();
  auto func = Module::getFuncOp(call.getCallee());
  func.walk([&](Operation *op) { codegen(op); });
  bm168x->after_codegen(Module::getFLOPs());
  int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  Module::getInputsOutputs(call, inputs, outputs);
  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);
  std::vector<int> next_id_v = {};
  for (auto v : call.getResults()) {
    for (auto user : v.getUsers()) {
      if (isa<func::ReturnOp>(user)) {
        next_id_v.push_back(-1);
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto func = Module::getFuncOp(call.getCallee());
        auto id = func->getAttrOfType<IntegerAttr>("id").getInt();
        next_id_v.push_back(id);
      } else {
        llvm_unreachable("next op is illegal");
      }
    }
  }
  auto &builder = model_gen->Builder();
  auto next_ids = builder.CreateVector(next_id_v);
  auto cmd_group_v = CreateCmdGroupVector();
  for (auto &c : *cmd_group_v) {
    cmd_group_all->push_back(c);
  }
  auto cmd_group = model_gen->Builder().CreateVector(*cmd_group_v);
  bmodel::SubNetBuilder snb(builder);
  snb.add_cmd_group(cmd_group);
  snb.add_is_dynamic(false);
  snb.add_subnet_mode(SUBNET_MODE_TPU);
  snb.add_input_tensor(input_tensor);
  snb.add_output_tensor(output_tensor);
  snb.add_id(subnet_id);
  snb.add_next_subnet_ids(next_ids);
  return snb.Finish();
}

} // namespace tpu
} // namespace tpu_mlir
