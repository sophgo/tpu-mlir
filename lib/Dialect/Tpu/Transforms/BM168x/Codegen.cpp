//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Builder/BM168x/bmodel.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/SwPipeline.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynamicNetIr.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynamicLayer.hpp"
#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::backend;

using namespace flatbuffers;
namespace tpu_mlir {
namespace tpu {

class CodegenPass : public CodegenBase<CodegenPass> {
public:
  CodegenPass() {}
  void runOnOperation() override {
    module = getOperation();
    assert(module::isState(module::State::TPU_ADDRESSED));
    auto chip_ = module::getChip();
    chip = module::stringifyChip(chip_).upper();
    std::string filename = this->model_file;
    if (filename.empty()) {
      llvm_unreachable("output filename is empty");
    }
    bm168x = BM168x::instance();
    DynCodegenInit();
    std::vector<top::WeightOp> weights;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) {
        // TODO: store all weight to gmem for compare
        // bm168x->value_s2d(op.getOutput(), op.read_as_byte()->data());
        weights.push_back(op);
      });
    }
    std::vector<Value> inputs;
    std::vector<Value> outputs;
    module::getInputsOutputs(inputs, outputs);
    SetNetIO(inputs, outputs);

    auto coeff_addr = module::getCoeffAddr();
    auto coeff_size = module::getCoeffSize();
    auto neuron_addr = module::getNeuronAddr();
    auto neuron_size = module::getNeuronSize();
    model_gen = std::make_shared<bmodel::ModelGen>();
    // add chip name
    model_gen->AddChip(chip);
    auto &builder = model_gen->Builder();
    auto input_tensor = CreateTensorVector(inputs);
    auto output_tensor = CreateTensorVector(outputs);
    auto coeff_mem = CreateCoeffMem(weights, coeff_addr, coeff_size);
    std::vector<uint64_t> neuron_sizes = {(uint64_t)neuron_size};
    auto neuron_sizes_fb = builder.CreateVector(neuron_sizes);
    // codegen all subnet
    cmd_group_all = std::make_shared<std::vector<Offset<bmodel::CmdGroup>>>();
    auto main_func = module::getMainFuncOp();
    std::vector<Offset<bmodel::SubNet>> subnet_v;
    auto context = std::make_unique<Context>();
    bool first_dynamic = false;
    bool is_first = true;
    int dynamic_mode = module::isBM1684XFamily() ? 2 : 1;
    main_func.walk([&](func::CallOp call) {
      auto func = module::getFuncOp(call.getCallee());
      auto mode = getRunMode(func);
      switch (mode) {
      case RunMode::TPU_STATIC: {
        auto subnet = CreateSubNet(call);
        subnet_v.push_back(subnet);
      } break;
      case RunMode::TPU_DYNAMIC: {
        auto subnet_ir_ = std::make_unique<SubnetIr>(dynamic_mode);
        auto subnet = CreateSubNet(call, std::move(subnet_ir_), context);
        subnet_v.push_back(subnet);
      } break;
      default:
        llvm_unreachable("Not Implemented");
        break;
      }
      if (is_first) {
        is_first = false;
        if (mode == RunMode::TPU_DYNAMIC) {
          first_dynamic = true;
        }
      }
    });
    auto subnets = builder.CreateVector(subnet_v);
    auto cmd_group = model_gen->Builder().CreateVector(*cmd_group_all);
    bmodel::Binary binary_ir;
    uint32_t ir_info_len = context->get_cur_net_ir_len();
    uint32_t ir_info_len_word =
        (ir_info_len + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    stage_param_t stage_param = {ir_info_len_word, 0, 0, 0, 0};
    context->set_stage_param(stage_param);
    auto stage_ir = CreateStageIRVector(
        context->get_stage_param(context->get_cur_net_idx()),
        context->get_binary_ir(),
        context->get_cur_net_offset() * sizeof(uint32_t), binary_ir);
    bmodel::NetParameterBuilder npb(builder);
    npb.add_input_tensor(input_tensor);
    npb.add_output_tensor(output_tensor);
    npb.add_ctx_addr(neuron_addr);
    npb.add_ctx_size(neuron_size);
    npb.add_ctx_sizes(std::move(neuron_sizes_fb));
    npb.add_coeff_mem(coeff_mem);
    npb.add_is_dynamic(first_dynamic);
    npb.add_n_dynamic(first_dynamic);
    npb.add_h_w_dynamic(first_dynamic);
    npb.add_cmd_group(cmd_group);
    npb.add_stage_ir(stage_ir);
    npb.add_binary_ir(&binary_ir);
    // create subnet
    npb.add_sub_net(subnets);
    model_gen->AddNet(module::getModuleName().str(), npb.Finish());
    model_gen->Finish();
    model_gen->Save(filename);
  }

private:
  Offset<Vector<Offset<bmodel::Shape>>>
  CreateShapeVector(const ArrayRef<int64_t> &shape);
  Offset<Vector<Offset<bmodel::Tensor>>>
  CreateTensorVector(const std::vector<Value> &values);
  Offset<bmodel::SubNet> CreateSubNet(func::CallOp call);
  Offset<bmodel::SubNet> CreateSubNet(func::CallOp call,
                                      std::unique_ptr<SubnetIr> subnet_ir_,
                                      std::unique_ptr<Context> &context);
  std::shared_ptr<std::vector<Offset<bmodel::CmdGroup>>> CreateCmdGroupVector();
  Offset<bmodel::CoeffMem> CreateCoeffMem(std::vector<top::WeightOp> &coeffs,
                                          uint64_t coeff_addr,
                                          uint64_t coeff_size);
  Offset<Vector<Offset<bmodel::StageIR>>>
  CreateStageIRVector(const vector<stage_param_t> &stage_param_v,
                      const vector<u32> &binary_ir_v, u32 ir_offset,
                      bmodel::Binary &binary_ir);
  void codegen(Operation *op);
  void codegen_for_group(GroupOp gOP);
  void codegen_ir(Operation *op, SubnetIr *subnet_ir_);

private:
  ModuleOp module;
  StringRef state;
  std::string chip;
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
    auto v_name = module::getName(v).str();
    auto type = module::getStorageType(v);
    auto shape = module::getShape(v);
    auto data_type = BM168x::getDataType(type);
    auto gmem_stmode = STORE_MODE_1N;
    if (module::isBM1684Family()) {
      auto typeBytes = type.getIntOrFloatBitWidth() / 8;
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
    if (module::isUniformQuantized(v)) {
      auto qtype = module::getUniformQuantizedType(v);
      scale = qtype.getScale();
      if (isa<top::InputOp>(v.getDefiningOp())) {
        scale = 1.0 / qtype.getScale();
      }
      zero_point = qtype.getZeroPoint();
      tb.add_scale(scale);
      tb.add_zero_point(zero_point);
    }
    tb.add_device_addr(module::getAddress(v));
    tb.add_size(Arch::get_gmem_bytes(v));
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
  if(offset != coeff_size) {
    llvm::errs() << "Warning: coeff size is not correct\n";
  }
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

void CodegenPass::codegen_for_group(GroupOp gOp) {
  auto nsecs = gOp.getNsecs();
  auto hsecs = gOp.getHsecs();
  auto swpipl_stage_num = gOp.getSwpiplStageNum();
  auto &body = gOp.getBody().front();
  auto flow = module::getI64Array(gOp.getFlow());
  auto group_type = static_cast<group_type_t>(gOp.getGroupType());
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
        std::string prefix = group_ops[id]->getName().getStringRef().str();
        if (ginfo.overstepped == false) {
          auto pid_node = (CMD_ID_NODE *)BM168x::instance()->bdc_node;
          if (isa<LoadOp, StoreOp>(*group_ops[id])) {
            pid_node = (CMD_ID_NODE *)BM168x::instance()->gdma_node;
          }
          BM168x::instance()->dl_set_cmd_id_prefix(pid_node, prefix.c_str());
          lgOp.assign_sec_info(tensor_step->nstep, tensor_step->hstep,
                               group_type, sec_info);
          lgOp.codegen_local_bm168x(tensor_step->nstep, tensor_step->hstep,
                                    group_type, sec_info);
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
  if (auto castOp = dyn_cast<GroupOp>(op)) {
    codegen_for_group(castOp);
  } else if (module::isOpInGroup(op)) {
    return;
  } else if (auto castOp = dyn_cast<GlobalGenInterface>(op)) {
    std::string prefix = op->getName().getStringRef().str();
    auto pid_node = (CMD_ID_NODE *)BM168x::instance()->cmdid_node;
    BM168x::instance()->dl_set_cmd_id_prefix(pid_node, prefix.c_str());
    castOp.codegen_global_bm168x();
  }
}

Offset<bmodel::SubNet> CodegenPass::CreateSubNet(func::CallOp call) {
  bm168x->before_codegen();
  auto func = module::getFuncOp(call.getCallee());
  func.walk([&](Operation *op) { codegen(op); });
  bm168x->after_codegen(module::getFLOPs());
  int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  module::getInputsOutputs(call, inputs, outputs);
  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);
  std::vector<int> next_id_v = {};
  for (auto v : call.getResults()) {
    for (auto user : v.getUsers()) {
      if (isa<ReturnOp>(user)) {
        next_id_v.push_back(-1);
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto func = module::getFuncOp(call.getCallee());
        auto id = func->getAttrOfType<IntegerAttr>("id").getInt();
        //callOp's result maybe have more than two users
        next_id_v.insert(next_id_v.begin(), id);
      } else {
        llvm_unreachable("next op is illegal");
      }
    }
  }

  std::sort(next_id_v.begin(), next_id_v.end(), std::greater<int>());
  next_id_v.erase(std::unique(next_id_v.begin(), next_id_v.end()), next_id_v.end());
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

void CodegenPass::codegen_ir(Operation *op, SubnetIr *subnet_ir_) {
  if (module::isOpInGroup(op) || isa_and_nonnull<top::WeightOp>(op)) {
    return;
  } else if (dyn_cast<GroupOp>(op) || dyn_cast<DynGlobalGenInterface>(op)) {
    subnet_ir_->generate_crop_layer_shape_tensor_record();
    subnet_ir_->generate_group_time_step_ir(op);
  }
}

Offset<bmodel::SubNet>
CodegenPass::CreateSubNet(func::CallOp call,
                          std::unique_ptr<SubnetIr> subnet_ir_,
                          std::unique_ptr<Context> &context) {
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  module::getInputsOutputs(call, inputs, outputs);
  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);
  std::function<void(Operation *, SubnetIr *)> task =
      std::bind(&CodegenPass::codegen_ir, this, std::placeholders::_1,
                std::placeholders::_2);
  subnet_ir_->generate_compiler_ir(module, call, task);
  subnet_ir_->write_binary_ir_to_buffer(context);
  auto func = module::getFuncOp(call.getCallee());
  int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();

  std::vector<int> next_id_v = {};
  for (auto v : call.getResults()) {
    for (auto user : v.getUsers()) {
      if (isa<func::ReturnOp>(user)) {
        next_id_v.push_back(-1);
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto func = module::getFuncOp(call.getCallee());
        auto id = func->getAttrOfType<IntegerAttr>("id").getInt();
        next_id_v.insert(next_id_v.begin(), id);
      } else {
        llvm_unreachable("next op is illegal");
      }
    }
  }

  std::sort(next_id_v.begin(), next_id_v.end(), std::greater<int>());
  next_id_v.erase(std::unique(next_id_v.begin(), next_id_v.end()), next_id_v.end());
  auto &builder = model_gen->Builder();
  auto next_ids = builder.CreateVector(next_id_v);

  bmodel::SubNetBuilder snb(builder);
  snb.add_is_dynamic(true);
  snb.add_subnet_mode(SUBNET_MODE_TPU);
  snb.add_input_tensor(input_tensor);
  snb.add_output_tensor(output_tensor);
  snb.add_ir_offset(subnet_ir_->get_ir_offset());
  snb.add_ir_len(subnet_ir_->get_ir_len());
  snb.add_id(subnet_id);
  snb.add_next_subnet_ids(next_ids);
  return snb.Finish();
}

Offset<Vector<Offset<bmodel::StageIR>>>
CodegenPass::CreateStageIRVector(const vector<stage_param_t> &stage_param_v,
                                 const vector<uint32_t> &binary_ir_v,
                                 uint32_t ir_offset,
                                 bmodel::Binary &binary_ir) {
  auto &builder = model_gen->Builder();
  vector<Offset<bmodel::StageIR>> stage_ir_v;
  u32 ir_len = 0;
  for (auto &stage_param : stage_param_v) {
    ir_len += stage_param.ir_info_len;
    bmodel::StageIRBuilder sirb(builder);
    sirb.add_ir_info_len(stage_param.ir_info_len);
    sirb.add_height_high(stage_param.height_high);
    sirb.add_height_low(stage_param.height_low);
    sirb.add_width_high(stage_param.width_high);
    sirb.add_width_low(stage_param.width_low);
    stage_ir_v.push_back(sirb.Finish());
  }
  if (stage_ir_v.size() == 0) {
    return 0;
  }
  auto stage_ir = builder.CreateVector(stage_ir_v);

  // ir binary
  u32 ir_size = ir_len * sizeof(u32);
  assert((ir_offset + ir_size) <= (binary_ir_v.size() * sizeof(u32)));
  u8 *buffer = (u8 *)binary_ir_v.data();
  binary_ir = model_gen->WriteBinary(ir_size, buffer + ir_offset);
  return stage_ir;
}
} // namespace tpu
} // namespace tpu_mlir
