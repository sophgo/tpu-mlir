//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "BM168xCodegen.hpp"

#include "tpu_mlir/Backend/BM168x/BM1686.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>

#define DEBUG_TYPE "bm_codegen"

using namespace llvm;

using namespace tpu_mlir::backend;

using namespace flatbuffers;
namespace tpu_mlir {
namespace tpu {

static bmodel::Binary CreateBinaryFromFile(bmodel::ModelGen *model_gen,
                                           FILE *fp) {
  std::vector<u8> data;
  fseek(fp, 0, SEEK_END);
  uint32_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  data.resize(size);
  fread(data.data(), 1, size, fp);
  fclose(fp);
  auto binary = model_gen->WriteBinary(data.size(), data.data());
  return binary;
}

static bmodel::Binary CreateBinaryFromFile(bmodel::ModelGen *model_gen,
                                           const std::string &filename) {
  auto fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    llvm_unreachable((std::string("can't find file: ") + filename).c_str());
    return bmodel::Binary();
  }
  return CreateBinaryFromFile(model_gen, fp);
}

void BMCodegen::init(ModuleOp m, const std::string &filename) {
  this->filename = filename;
  llvm::raw_null_ostream os;
  AsmState state(m, OpPrintingFlags(), &opToLineCol);
  m->print(os, state);
  auto chip_ = module::getChip();
  auto num_device = module::getDeviceNum();
  chip = module::stringifyChip(chip_).upper();
  tensor_loc = TensorLocation(&opToLineCol, filename + ".json");
  profile_ctx = ProfileCtx(&opToLineCol, true);
  bm168x = BM168x::instance();
  model_gen = std::make_shared<bmodel::ModelGen>();
  // add chip name
  model_gen->AddChip(chip);
  model_gen->AddNumDevice(num_device);
  if (module::isBM1684X()) {
    std::string kernel_name = backend::BM1684X::LIB_KERNEL_NAME.str();
    std::string root_path = getenv("TPUC_ROOT");
    std::string kernel_path = root_path + std::string("/lib/") + kernel_name;
    bmodel::Binary kernel_module =
        CreateBinaryFromFile(&(*model_gen), kernel_path);
    model_gen->AddKernelModule(kernel_name, kernel_module);
  }
  input_names = module::getInputs();
  output_names = module::getOutputs();
  hidden_names.clear();
  current_step = 0;
  current_device = 0;
}

void BMCodegen::run(ModuleOp s, bool embed_debug_info) {
  // record the line number of operation in module.
  DynCodegenInit();
  std::vector<top::WeightOp> weights;
  for (auto func : s.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) { weights.push_back(op); });
  }
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  module::getInputsOutputs(s, inputs, outputs);
  SetNetIO(inputs, outputs);
  bmodel::ModelGen::CASCADE_INFO_T cascade = {0, 0, ""};
  if (module::getNumSubModule() > 1) {
    module::getSubModuleId(s, cascade.device_id, cascade.step);
  }
  checkAndUpdateHidden(inputs, outputs);

  auto coeff_addr = module::getCoeffAddr(s);
  auto coeff_size = module::getCoeffSize(s);
  auto neuron_addr = module::getNeuronAddr(s);
  auto neuron_size = module::getNeuronSize(s);

  auto &builder = model_gen->Builder();
  // if tensor not in device 0, will be hidden
  auto input_tensor = CreateTensorVector(inputs, cascade.device_id != 0);
  auto output_tensor = CreateTensorVector(outputs, cascade.device_id != 0);
  auto coeff_mem = CreateCoeffMem(weights, coeff_addr, coeff_size);
  std::vector<uint64_t> neuron_sizes = {(uint64_t)neuron_size};
  auto neuron_sizes_fb = builder.CreateVector(neuron_sizes);
  // codegen all subnet
  cmd_group_all = std::make_shared<std::vector<Offset<bmodel::CmdGroup>>>();
  std::vector<Offset<bmodel::SubNet>> subnet_v;
  auto context = std::make_unique<Context>();
  int dynamic_mode = module::isBM1684XFamily() ? 2 : 1;
  bool first_dynamic = false;

  s.walk<WalkOrder::PreOrder>([&](func::FuncOp func) {
    if (func == module::getMainFuncOp(s)) {
      return WalkResult::advance();
    }
    if (auto call = module::getCallOp(func)) {
      auto mode = getRunMode(func);
      int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
      if (subnet_id == 0 && mode == RunMode::TPU_DYNAMIC) {
        first_dynamic = true;
      }

      switch (mode) {
      case RunMode::TPU_STATIC: {
        profile_ctx.set_profile_start();
        auto subnet = CreateSubNet(s, call);
        subnet_v.push_back(subnet);
        profile_ctx.set_profile_end();
      } break;
      case RunMode::TPU_DYNAMIC: {
        auto subnet_ir_ = std::make_unique<SubnetIr>(dynamic_mode);
        auto subnet = CreateSubNet(s, call, std::move(subnet_ir_), context);
        subnet_v.push_back(subnet);
      } break;
      case RunMode::CPU: {
        auto subnet = CreateCPUSubNet(s, call);
        subnet_v.push_back(subnet);
      } break;
      // actually use switch subnet
      case RunMode::LOOP:
      case RunMode::SWITCH: {
        auto subnet = CreateSwitchSubNet(s, call);
        subnet_v.push_back(subnet);
      } break;
      case RunMode::MERGE: {
        auto subnet = CreateMergeSubNet(s, call);
        subnet_v.push_back(subnet);
      } break;
      default:
        llvm_unreachable("Not Implemented");
        break;
      }
    }

    return WalkResult::advance();
  });

  auto subnets = builder.CreateVector(subnet_v);
  auto cmd_group = builder.CreateVector(*cmd_group_all);
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
  npb.add_cpu_mem_size(0);
  // multi-core
  if (auto bm1686 = dyn_cast<BM1686>(bm168x)) {
    auto bufferNum = bm1686->getCodebuffer().size();
    if (module::getCoreNum() > 1 && bufferNum > 1) {
      assert(module::getCoreNum() == bufferNum &&
             "The code buffer size does not match the core number defined in "
             "Module.");
      npb.add_core_num(module::getCoreNum());
    }
  }

  if (embed_debug_info) {
    auto save_profile_info = [&](StringRef pfname, auto fun) -> bool {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
          llvm::MemoryBuffer::getFileOrSTDIN(pfname);
      if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open file: '" << pfname << "'"
                     << ec.message() << "\n";
        return false;
      }
      auto profile_data =
          model_gen->WriteBinary((*fileOrErr)->getBufferSize(),
                                 (uint8_t *)(*fileOrErr)->getBufferStart());
      fun(&profile_data);
      return true;
    };
    using namespace std::placeholders;
    save_profile_info(
        "net_0.profile",
        std::bind(&bmodel::NetParameterBuilder::add_net_profile, &npb, _1));
    if (!save_profile_info(
            "compiler_profile_0.txt",
            std::bind(&bmodel::NetParameterBuilder::add_net_stat, &npb, _1))) {
      save_profile_info(
          "compiler_profile_0.dat",
          std::bind(&bmodel::NetParameterBuilder::add_net_stat, &npb, _1));
    };
  }
  if (module::getNumSubModule() > 1) {
    // make sure the order is by step and device id
    if (cascade.step == current_step) {
      assert(cascade.device_id == current_device);
      current_device++;
    } else {
      assert(cascade.step == current_step + 1 && cascade.device_id == 0);
      current_step++;
      current_device = 1;
    }
    cascade.main_name = module::getName(module::getModuleOp());
  }
  model_gen->AddNet(module::getName(s).str(), cascade, npb.Finish());
}

void BMCodegen::store() {
  model_gen->Finish();
  model_gen->Save(filename);
}

Offset<Vector<Offset<bmodel::Shape>>>
BMCodegen::CreateShapeVector(const ArrayRef<int64_t> &shape) {
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
BMCodegen::CreateTensorVector(const std::vector<Value> &values,
                              bool hidden_all) {
  auto &builder = model_gen->Builder();
  std::vector<Offset<bmodel::Tensor>> tensor_v;
  int index = 0;
  for (auto v : values) {
    auto s_name = module::getName(v);
    auto v_name = s_name.str();
    auto type = module::getStorageType(v);
    auto shape = module::getShape(v);
    auto data_type = BM168x::getDataType(type);
    auto gmem_stmode = BM168x::getStoreMode(v);
    // shape info
    auto name = builder.CreateString(v_name);
    auto stage_shape = CreateShapeVector(shape);
    bmodel::TensorBuilder tb(builder);
    tb.add_name(name);
    tb.add_data_type(data_type);
    tb.add_gmem_stmode(gmem_stmode);
    tb.add_shape(stage_shape);
    if (hidden_all || isHiddenTensor(s_name)) {
      tb.add_hidden(0);
    } else {
      auto in_iter =
          std::find(input_names->begin(), input_names->end(), s_name);
      auto out_iter =
          std::find(output_names->begin(), output_names->end(), s_name);
      if (in_iter != input_names->end()) {
        tb.add_hidden(1); // input
        tb.add_index(std::distance(input_names->begin(), in_iter));
      } else if (out_iter != output_names->end()) {
        tb.add_hidden(2); // output
        tb.add_index(std::distance(output_names->begin(), out_iter));
      } else {
        tb.add_hidden(0);
      }
    }

    /*
+--------------------------+     +-----------------------------------------+
| TPU_LAYER (MEM_TYPE_ALL) | --> | (MEM_TYPE_ALL) CPU_LAYER (MEM_TYPE_ALL) |
+--------------------------+     +-----------------------------------------+
                                                  |
                                                  |
                                                  |
                                                  |
                                                  |
                                                  |
                                                  |
                                                  v
                                 +-----------------------------------------+
                                 |        (MEM_TYPE_ALL) TPU_LAYER)        |
                                 +-----------------------------------------+
     */
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
    } else {
      tb.add_scale(scale);
    }
    if (!tensor_is_cpu.count(v_name) || tensor_is_cpu[v_name].empty()) {
      tb.add_mem_type(MEM_TYPE_TPU);
    } else {
      int mem_type = 0;
      for (int i = 0; i < tensor_is_cpu[v_name].size(); i++) {
        mem_type |= tensor_is_cpu[v_name][i] ? MEM_TYPE_ALL : MEM_TYPE_TPU;
      }
      tb.add_mem_type(mem_type);
    }
    tb.add_device_addr(module::getAddress(v));
    tb.add_size(Arch::get_gmem_bytes(v));
    tensor_v.push_back(tb.Finish());
    ++index;
  }
  return builder.CreateVector(tensor_v);
}

Offset<bmodel::CoeffMem>
BMCodegen::CreateCoeffMem(std::vector<top::WeightOp> &coeffs,
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
  if (offset != coeff_size) {
    llvm::errs() << "Warning: coeff size is not correct\n";
  }
  auto sha256 = llvm::SHA256::hash(llvm::ArrayRef(data_u8->data(), coeff_size));
  auto binary_coeff = model_gen->WriteBinary(coeff_size, data_u8->data());
  auto coeff_sha256 =
      model_gen->Builder().CreateVector(sha256.data(), sha256.size());
  bmodel::CoeffMemBuilder cmb(model_gen->Builder());
  cmb.add_address(coeff_addr);
  cmb.add_check_code(coeff_sha256);
  cmb.add_binary_coeff(&binary_coeff);
  return cmb.Finish();
}

std::shared_ptr<std::vector<Offset<bmodel::CmdGroup>>>
BMCodegen::CreateCmdGroupVector() {
  auto cmd_group_v = std::make_shared<std::vector<Offset<bmodel::CmdGroup>>>();
  auto gdma_ptr = (uint8_t *)(*bm168x)->gdma_buffer.data();
  auto bdc_ptr = (uint8_t *)(*bm168x)->bdc_buffer.data();
  int bdc_offset = 0, gdma_offset = 0;
  for (int group_idx = 0; group_idx < (*bm168x)->cmdid_groupnum; group_idx++) {
    auto bdc_num = (*bm168x)->bdc_group_id[group_idx];
    auto gdma_num = (*bm168x)->gdma_group_id[group_idx];
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

Offset<bmodel::SwitchParam>
BMCodegen::CreateSwitchParamVector(vector<int> &output_from,
                                   vector<int> &output_branch) {
  auto &builder = model_gen->Builder();
  auto out_from = builder.CreateVector(output_from);
  auto out_branch = builder.CreateVector(output_branch);
  bmodel::SwitchParamBuilder mpb(builder);
  mpb.add_output_from(out_from);
  mpb.add_output_branch(out_branch);
  return mpb.Finish();
}

Offset<bmodel::MergeParam>
BMCodegen::CreateMergeParamVector(vector<vector<int>> &output_from) {
  auto &builder = model_gen->Builder();
  vector<Offset<Vector<int>>> indice_v;
  for (auto &indice : output_from) {
    indice_v.push_back(builder.CreateVector(indice));
  }
  vector<Offset<bmodel::OutputFrom>> output_from_v;
  for (auto idx : indice_v) {
    bmodel::OutputFromBuilder ofb(builder);
    ofb.add_indice(idx);
    output_from_v.push_back(ofb.Finish());
  }
  auto output_froms = builder.CreateVector(output_from_v);
  bmodel::MergeParamBuilder mpb(builder);
  mpb.add_output_from(output_froms);
  return mpb.Finish();
}

void BMCodegen::codegen_for_overlap_ops(
    std::map<int64_t, std::vector<Operation *>> cur_other_downs,
    std::map<int64_t, std::vector<Operation *>> cur_other_ups,
    Operation *prev_op, Operation *next_op, int64_t cur_ts,
    bool first_compute_loop, bool last_compute_loop) {

  local_sec_info_t sec_info;
  if (last_compute_loop) {
    auto iter = cur_other_ups.find(cur_ts);
    if (iter != cur_other_ups.end()) {
      auto castOp = cast<GroupOp>(next_op);
      auto next_group_type = static_cast<group_type_t>(castOp.getGroupType());
      auto &cur_ops = iter->second;
      for (auto op : cur_ops) {
        auto lgOp = cast<LocalGenInterfaceDecorator>(op);
        auto pid_node = (CMD_ID_NODE *)(*BM168x::instance())->bdc_node;
        if (isa<LoadOp, StoreOp>(op)) {
          pid_node = (CMD_ID_NODE *)(*BM168x::instance())->gdma_node;
        }
        BM168x::instance()->dl_set_cmd_id_prefix(pid_node,
                                                 gen_op_id(op).c_str());
        lgOp.assign_sec_info(0l, 0l, 0l, 0l, 0l, next_group_type, sec_info);
        LLVM_DEBUG(llvm::dbgs()
                   << "codegen op: '" << module::getName(lgOp) << "'\n");
        profile_ctx.log_local_layer(op, 0, 0);
        lgOp.codegen_local_bm168x(0l, 0l, 0l, 0l, 0l, next_group_type,
                                  sec_info);
      }
    }
  }

  if (first_compute_loop) {
    auto iter = cur_other_downs.find(cur_ts);
    if (iter != cur_other_downs.end()) {
      auto castOp = cast<GroupOp>(prev_op);
      auto prev_group_type = static_cast<group_type_t>(castOp.getGroupType());
      auto nsecs = castOp.getNsecs();
      auto hsecs = castOp.getHsecs();
      auto dsecs = castOp.getDsecs();
      auto wsecs = castOp.getWsecs();
      auto csecs = castOp.getCsecs();
      auto &cur_ops = iter->second;
      for (auto op : cur_ops) {
        auto lgOp = cast<LocalGenInterfaceDecorator>(op);
        auto pid_node = (CMD_ID_NODE *)(*BM168x::instance())->bdc_node;
        if (isa<LoadOp, StoreOp>(op)) {
          pid_node = (CMD_ID_NODE *)(*BM168x::instance())->gdma_node;
        }
        BM168x::instance()->dl_set_cmd_id_prefix(pid_node,
                                                 gen_op_id(op).c_str());
        lgOp.assign_sec_info(nsecs - 1, csecs - 1, hsecs - 1, dsecs - 1,
                             wsecs - 1, prev_group_type, sec_info);
        LLVM_DEBUG(llvm::dbgs()
                   << "codegen op: '" << module::getName(lgOp) << "'\n");
        profile_ctx.log_local_layer(op, nsecs - 1, hsecs - 1);
        lgOp.codegen_local_bm168x(nsecs - 1, csecs - 1, hsecs - 1, dsecs - 1,
                                  wsecs - 1, prev_group_type, sec_info);
      }
    }
  }
}

void BMCodegen::codegen_for_group(GroupOp gOp, Operation *prev_op,
                                  Operation *next_op) {
  auto nsecs = gOp.getNsecs();
  auto hsecs = gOp.getHsecs();
  auto dsecs = gOp.getDsecs();
  auto wsecs = gOp.getWsecs();
  auto csecs = gOp.getCsecs();
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
  // 2. create a vector to map id to op
  std::vector<Operation *> group_ops;
  for (int64_t id = 0; id < max_id;) {
    body.walk([&](Operation *op) {
      if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
        auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0,
                                       (int64_t)0, (int64_t)0);
        if (ginfo.id == id) {
          group_ops.push_back(op);
          id++;
        }
      }
    });
  }
  // 3. recover overlap ops that will be executed in this group
  int64_t tmp_ts = 0;
  // <timestep_idx, prev_group_op>
  std::map<int64_t, std::vector<Operation *>> cur_other_downs;
  if (auto castOp = dyn_cast_or_null<GroupOp>(prev_op)) {
    auto other_down_overlap_op =
        module::getI64Array(gOp.getOtherDownOverlapOp());

    auto &prev_body = castOp.getBody().front();
    for (size_t i = 0; i < other_down_overlap_op->size(); ++i) {
      if (other_down_overlap_op->at(i) < 0) {
        tmp_ts = -other_down_overlap_op->at(i) - 1;
        cur_other_downs[tmp_ts] = std::vector<Operation *>();
      } else {
        int64_t id = other_down_overlap_op->at(i);
        prev_body.walk([&](Operation *op) {
          if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
            auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0,
                                           (int64_t)0, (int64_t)0);
            if (ginfo.id == id) {
              cur_other_downs[tmp_ts].push_back(op);
            }
          }
        });
      }
    }
  }
  // <timestep_idx, next_group_op>
  std::map<int64_t, std::vector<Operation *>> cur_other_ups;
  if (auto castOp = dyn_cast_or_null<GroupOp>(next_op)) {
    auto other_up_overlap_op = module::getI64Array(gOp.getOtherUpOverlapOp());
    auto &next_body = castOp.getBody().front();
    for (size_t i = 0; i < other_up_overlap_op->size(); ++i) {
      if (other_up_overlap_op->at(i) < 0) {
        tmp_ts = -other_up_overlap_op->at(i) - 1;
        cur_other_ups[tmp_ts] = std::vector<Operation *>();
      } else {
        int64_t id = other_up_overlap_op->at(i);
        next_body.walk([&](Operation *op) {
          if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
            auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0,
                                           (int64_t)0, (int64_t)0);
            if (ginfo.id == id) {
              cur_other_ups[tmp_ts].push_back(op);
            }
          }
        });
      }
    }
  }

  auto self_up_overlap_op = module::getI64Array(gOp.getSelfUpOverlapOp());
  auto self_down_overlap_op = module::getI64Array(gOp.getSelfDownOverlapOp());

  // 4. codegen for group
  int64_t stage_idx = 0;
  int64_t draining_idx = 0;
  bool draining_period = false;
  SoftwarePipeline timestep_swpipl;
  local_sec_info_t sec_info;
  int64_t timestep_num = timestep_table.size();
  // multi-core-setup
  auto core_num = module::getCoreNum();
  bool useMuliCore = core_num > 1;
  auto secs = nsecs * csecs * dsecs * hsecs * wsecs;
  auto max_task_per_core = (secs + core_num - 1) / core_num;
  int coreId = 0;
  auto bm1686 = dyn_cast<BM1686>(bm168x);
  useMuliCore &= (secs > 1) && bm1686;
  if (useMuliCore && (bm1686->getCoreNum() != core_num)) {
    assert(bm1686->getCoreNum() == 1 &&
           "The core_num should be set only once, and can not be changed.");
    bm1686->dl_tpu_core_context_setup(0, core_num, 0);
    bm1686->setCoreNum(module::getCoreNum());
  }
  // multi-core-setup END
  for (uint64_t nstep = 0, cstep = 0, hstep = 0, dstep = 0, wstep = 0;
       nstep < nsecs || draining_period;) {
    if (useMuliCore && stage_idx == 0) {
      bm1686->useCore(coreId++);
      bm1686->sync_all();
    }
    /* add for software pipeline */
    timestep_swpipl.write_swloop_buffer(nstep, cstep, hstep, dstep, wstep,
                                        swpipl_stage_num);
    for (int64_t ts = 0; ts < timestep_num; ++ts) {
      bm168x->divide_sync_id();
      auto cur_op_ids = timestep_table[ts];
      for (auto id : cur_op_ids) {

        auto lgOp = cast<LocalGenInterfaceDecorator>(group_ops[id]);
        auto ginfo = lgOp.getGroupInfo(nstep, hstep, dstep, wstep, cstep);
        if ((!draining_period && ginfo.stage > stage_idx) ||
            (draining_period &&
             (ginfo.stage < draining_idx || ginfo.stage > stage_idx))) {
          continue;
        }
        const tensor_step_t *tensor_step =
            timestep_swpipl.read_swloop_buffer(ginfo.stage);

        // only consider first loop load
        if (stage_idx == 0 &&
            std::find(self_up_overlap_op->begin(), self_up_overlap_op->end(),
                      id) != self_up_overlap_op->end()) {
          continue;
        }
        // only consider last loop store
        if (draining_period && draining_idx == 2 &&
            std::find(self_down_overlap_op->begin(),
                      self_down_overlap_op->end(),
                      id) != self_down_overlap_op->end()) {
          continue;
        }

        ginfo = lgOp.getGroupInfo(tensor_step->nstep, tensor_step->hstep,
                                  tensor_step->dstep, tensor_step->wstep,
                                  tensor_step->cstep);
        if (ginfo.overstepped == false || stage_idx == ginfo.stage) {
          ginfo.overstepped = true;
          auto pid_node = (CMD_ID_NODE *)(*BM168x::instance())->bdc_node;
          if (isa<LoadOp, StoreOp>(*group_ops[id])) {
            pid_node = (CMD_ID_NODE *)(*BM168x::instance())->gdma_node;
          }
          BM168x::instance()->dl_set_cmd_id_prefix(
              pid_node, gen_op_id(group_ops[id]).c_str());
          lgOp.assign_sec_info(tensor_step->nstep, tensor_step->cstep,
                               tensor_step->hstep, tensor_step->dstep,
                               tensor_step->wstep, group_type, sec_info);
          LLVM_DEBUG(llvm::dbgs()
                     << "codegen op: '" << module::getName(lgOp) << "'\n");
          auto op = group_ops[id];
          profile_ctx.log_local_layer(op, tensor_step->nstep,
                                      tensor_step->hstep);
          lgOp.codegen_local_bm168x(tensor_step->nstep, tensor_step->cstep,
                                    tensor_step->hstep, tensor_step->dstep,
                                    tensor_step->wstep, group_type, sec_info);
        }
      } // ops, include Load/Store op

      // process overlap ops
      bool first_compute_loop = stage_idx == 1;
      bool last_compute_loop = (draining_period && draining_idx == 1);
      codegen_for_overlap_ops(cur_other_downs, cur_other_ups, prev_op, next_op,
                              ts, first_compute_loop, last_compute_loop);

      bm168x->merge_sync_id();
    } // timestep

    if (!draining_period) {
      cstep++;
      if (cstep >= csecs) {
        cstep = 0;
        wstep++;
      }
      if (wstep >= wsecs) {
        wstep = 0;
        hstep++;
      }
      if (hstep >= hsecs) {
        hstep = 0;
        dstep++;
      }
      if (dstep >= dsecs) {
        dstep = 0;
        nstep++;
        if (nstep >= nsecs) { // && swpipl_stage_num > 1
          draining_period = true;
        }
      }
      if (useMuliCore && ((stage_idx + 1) % max_task_per_core) == 0 &&
          nstep < nsecs) {
        draining_period = true;
        draining_idx = 0;
      }
    }
    stage_idx++;
    if (draining_period) {
      draining_idx++;
      if (draining_idx >= swpipl_stage_num) {
        draining_period = false;
        stage_idx = 0;
        draining_idx = 0;
        if (useMuliCore) {
          bm1686->sync_all();
        }
      }
    }
  }
  { // consume all the MSG send/wait.
    for (; useMuliCore && coreId < core_num; coreId++) {
      bm1686->useCore(coreId);
      bm1686->sync_all();
      bm1686->sync_all();
    }
    if (useMuliCore)
      bm1686->useCore(0);
  }
}

void BMCodegen::codegen(Operation *op) {
  if (module::isOpInGroup(op) || module::isOpInParallel(op))
    return;
  if (auto castOp = dyn_cast<GroupOp>(op)) {
    Operation *prev_op = op->getPrevNode();
    while (prev_op && !isa<GroupOp, GlobalGenInterfaceDecorator>(prev_op)) {
      prev_op = prev_op->getPrevNode();
    }
    Operation *next_op = op->getNextNode();
    while (next_op && !isa<GroupOp, GlobalGenInterfaceDecorator>(next_op)) {
      next_op = next_op->getNextNode();
    }
    codegen_for_group(castOp, prev_op, next_op);
    return;
  }
  if (auto parallelOp = dyn_cast<ParallelOp>(op)) {
    if (auto bm1686 = dyn_cast<BM1686>(bm168x)) {
      // For the sync-all method, we can use two message IDs to represent all
      // the dependencies in a single run. We can try the following sequence:
      // send0, wait0, send1, wait1. If any of the wait0 operations succeed,
      // it confirms that all the send0 operations have finished. Similarly,
      // if any of the wait1 operations succeed, it confirms that all the
      // send1 operations have finished, which also implies that all the wait0
      // operations have been completed. After this, it is safe to reuse
      // message0.
      auto core_num = module::getCoreNum();
      if (bm1686->getCoreNum() != core_num) {
        assert(bm1686->getCoreNum() == 1 &&
               "The core_num should be set only once, and can not be changed.");
        bm1686->dl_tpu_core_context_setup(0, core_num, 0);
        bm1686->setCoreNum(module::getCoreNum());
      }
      int id = 0;
      for (auto globalOp : parallelOp.getOps<GlobalGenInterfaceDecorator>()) {
        bm1686->useCore(id++);
        bm1686->sync_all(); // begin compute sync-all
        auto pid_node = (CMD_ID_NODE *)(*bm1686)->cmdid_node;
        BM168x::instance()->dl_set_cmd_id_prefix(pid_node,
                                                 gen_op_id(op).c_str());
        globalOp.codegen_global_bm168x();
        bm1686->sync_all(); // end compute sync-all
      }
      for (; id < core_num; id++) { // consume all the MSG send/wait.
        bm1686->useCore(id);
        bm1686->sync_all();
        bm1686->sync_all();
      }
      bm1686->useCore(0); // reset the command buffer to 0
    } else {
      llvm_unreachable("The backend is missing configuration.");
    }
    return;
  }
  if (auto castOp = dyn_cast<GlobalGenInterfaceDecorator>(op)) {
    auto pid_node = (CMD_ID_NODE *)(*BM168x::instance())->cmdid_node;
    BM168x::instance()->dl_set_cmd_id_prefix(pid_node, gen_op_id(op).c_str());
    LLVM_DEBUG(llvm::dbgs() << "codegen op: '" << module::getName(op) << "'\n");
    profile_ctx.log_global_layer(op);
    castOp.codegen_global_bm168x();
  }
}

Offset<bmodel::SubNet> BMCodegen::CreateSubNet(ModuleOp s, func::CallOp call) {
  bm168x->before_codegen();
  auto func = module::getFuncOp(s, call.getCallee());
  func.walk([&](Operation *op) { codegen(op); });
  bm168x->after_codegen(module::getFLOPs());
  int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
  LLVM_DEBUG(llvm::dbgs() << "subnet id: '" << subnet_id << "'\n");
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  module::getInputsOutputs(call, inputs, outputs);
  auto next_index = func->getAttrOfType<DenseI32ArrayAttr>("next_index");
  std::vector<int> next_id_v(next_index.asArrayRef());
  // for net input directly link to cpu subnet
  for (auto v : llvm::enumerate(inputs)) {
    auto v_name = module::getName(v.value()).str();
    // if v_name do not in tensor_is_cpu, means it is a net input
    if (tensor_is_cpu.count(v_name) && !tensor_is_cpu[v_name].empty()) {
      continue;
    }
    std::vector<bool> user_is_cpu;
    tensor_is_cpu[v_name] = user_is_cpu;
    for (auto user : v.value().getUsers()) {
      if (isa<tpu::YieldOp, ReturnOp>(user)) {
        tensor_is_cpu[v_name].push_back(false);
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto func = module::getFuncOp(s, call.getCallee());
        auto mode = getRunMode(func);
        tensor_is_cpu[v_name].push_back(mode == RunMode::CPU);
      }
    }
  }
  for (auto v : llvm::enumerate(call.getResults())) {
    std::string v_name = module::getName(outputs[v.index()]).str();
    std::vector<bool> user_is_cpu;
    tensor_is_cpu[v_name] = user_is_cpu;
    for (auto user : v.value().getUsers()) {
      if (isa<tpu::YieldOp, ReturnOp>(user)) {
        tensor_is_cpu[v_name].push_back(false);
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto func = module::getFuncOp(s, call.getCallee());
        auto mode = getRunMode(func);
        tensor_is_cpu[v_name].push_back(mode == RunMode::CPU);
      }
    }
  }

  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);
  auto &builder = model_gen->Builder();
  auto next_ids = builder.CreateVector(next_id_v);

  std::vector<Offset<bmodel::CmdGroup>> cmd_group_vs;
  std::vector<Offset<bmodel::CoreCommands>> core_commands;
  auto bm1686 = dyn_cast<BM1686>(bm168x);
  if (bm1686 && bm1686->getCodebuffer().size() > 1) {
    auto code_buffers = bm1686->getCodebuffer();
    for (int i = 0, n = code_buffers.size(); i < n; i++) {
      bm1686->useCore(i);
      auto cmd_group_v = CreateCmdGroupVector();
      auto cmd_group = builder.CreateVector(*cmd_group_v);
      bmodel::CoreCommandsBuilder ccb(builder);
      ccb.add_gdma_tiu_commands(cmd_group);
      core_commands.push_back(ccb.Finish());
    }
  } else {
    auto cmd_group_v = CreateCmdGroupVector();
    cmd_group_vs.insert(cmd_group_vs.end(), cmd_group_v->begin(),
                        cmd_group_v->end());
  }

  cmd_group_all->insert(cmd_group_all->end(), cmd_group_vs.begin(),
                        cmd_group_vs.end());
  auto cmd_group = builder.CreateVector(cmd_group_vs);
  auto core_cmds = builder.CreateVector(core_commands);
  bmodel::SubNetBuilder snb(builder);
  if (cmd_group_vs.size() > 0)
    snb.add_cmd_group(cmd_group);
  if (core_commands.size() > 0)
    snb.add_core_commands(core_cmds);
  snb.add_is_dynamic(false);
  snb.add_subnet_mode(SUBNET_MODE_TPU);
  snb.add_input_tensor(input_tensor);
  snb.add_output_tensor(output_tensor);
  snb.add_id(subnet_id);
  snb.add_next_subnet_ids(next_ids);
  return snb.Finish();
}

Offset<bmodel::SubNet> BMCodegen::CreateCPUSubNet(ModuleOp s,
                                                  func::CallOp call) {
  bm168x->before_codegen();
  auto func = module::getFuncOp(s, call.getCallee());
  bm168x->after_codegen(module::getFLOPs());
  int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
  LLVM_DEBUG(llvm::dbgs() << "subnet id: '" << subnet_id << "'\n");
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  module::getInputsOutputs(call, inputs, outputs);
  inputs.clear();
  func.walk([&](tpu::GenericCpuOp op) {
    for (auto opd : op.getOperands()) {
      if (!module::isNone(opd))
        inputs.push_back(opd);
    }
  });

  auto next_index = func->getAttrOfType<DenseI32ArrayAttr>("next_index");
  std::vector<int> next_id_v(next_index.asArrayRef());

  for (auto input : inputs) {
    auto input_name = module::getName(input).str();
    if (!tensor_is_cpu.count(input_name) || tensor_is_cpu[input_name].empty()) {
      tensor_is_cpu[input_name] = std::vector<bool>(1, true);
    }
  }

  for (auto v : llvm::enumerate(call.getResults())) {
    std::string v_name = module::getName(outputs[v.index()]).str();
    std::vector<bool> user_is_cpu;
    tensor_is_cpu[v_name] = user_is_cpu;
    for (auto user : v.value().getUsers()) {
      (void)user;
      tensor_is_cpu[v_name].push_back(true);
    }
  }
  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);
  auto &builder = model_gen->Builder();
  auto next_ids = builder.CreateVector(next_id_v);

  vector<Offset<bmodel::CpuParam>> cpu_param_v;

  /* currently only 1 cpu layer per subnet, but it should to able to support
   * multiple cpu layers in 1 subnet soon.
   */
  vector<Offset<bmodel::CpuConst>> tmp;
  auto cpu_const_v = builder.CreateVector(tmp);
  bmodel::CpuParamBuilder cpb(builder);
  void *param = nullptr;
  int op_type;
  int param_size;
  func.walk([&](tpu::GenericCpuOp op) {
    BMCpuOp cpuOp(op);
    param = malloc(cpuOp.param_size);
    memcpy(param, cpuOp.param, cpuOp.param_size);
    uint32_t io_size = 0;
    for (int i = 0; i < op.getInputs().size(); ++i) {
      io_size += module::getNumElements(op.getInputs()[i]) * sizeof(float);
    }
    for (int i = 0; i < op.getOutputs().size(); ++i) {
      io_size += module::getNumElements(op.getOutputs()[i]) * sizeof(float);
    }
    op_type = cpuOp.op_type;
    param_size = cpuOp.param_size;
  });
  cpb.add_op_type(op_type);
  bmodel::Binary binary_cpu_param =
      model_gen->WriteBinary(param_size, (u8 *)param);
  cpb.add_binary_param(&binary_cpu_param);
  cpb.add_cpu_const(cpu_const_v);
  cpu_param_v.push_back(cpb.Finish());
  auto cpu_param = builder.CreateVector(cpu_param_v);
  bmodel::SubNetBuilder snb(builder);
  snb.add_cpu_param(cpu_param);
  snb.add_subnet_mode(SUBNET_MODE_CPU);
  snb.add_input_tensor(input_tensor);
  snb.add_output_tensor(output_tensor);
  snb.add_id(subnet_id);
  snb.add_next_subnet_ids(next_ids);
  free(param);
  return snb.Finish();
}

Offset<bmodel::SubNet> BMCodegen::CreateSwitchSubNet(ModuleOp s,
                                                     func::CallOp call) {
  auto func = module::getFuncOp(s, call.getCallee());
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  auto next_index = func->getAttrOfType<DenseI32ArrayAttr>("next_index");
  std::vector<int> next_id_v(next_index.asArrayRef());
  std::reverse(next_id_v.begin(), next_id_v.end());
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<tpu::IfOp>(op)) {
      inputs.emplace_back(module::getOriValue(op->getOperand(0)));
    } else if (isa<tpu::LoopOp>(op)) {
      // the last operand is the condition Operation
      inputs.emplace_back(
          module::getOriValue(op->getOperand(op->getNumOperands() - 1)));
    }
  });

  int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
  LLVM_DEBUG(llvm::dbgs() << "subnet id: '" << subnet_id << "'\n");

  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);

  auto &builder = model_gen->Builder();
  auto next_ids = builder.CreateVector(next_id_v);
  vector<int> output_from;
  vector<int> output_branch;
  Offset<bmodel::SwitchParam> switch_param =
      CreateSwitchParamVector(output_from, output_branch);

  bmodel::SubNetBuilder snb(builder);
  snb.add_switch_param(switch_param);
  snb.add_is_dynamic(false);
  snb.add_subnet_mode(SUBNET_MODE_SWITCH);
  snb.add_input_tensor(input_tensor);
  snb.add_output_tensor(output_tensor);
  snb.add_id(subnet_id);
  snb.add_next_subnet_ids(next_ids);
  return snb.Finish();
}

Offset<bmodel::SubNet> BMCodegen::CreateMergeSubNet(ModuleOp s,
                                                    func::CallOp call) {
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  auto func = module::getFuncOp(s, call.getCallee());
  int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
  LLVM_DEBUG(llvm::dbgs() << "subnet id: '" << subnet_id << "'\n");
  auto next_index = func->getAttrOfType<DenseI32ArrayAttr>("next_index");
  std::vector<int> next_id_v(next_index.asArrayRef());

  module::getInputsOutputs(call, inputs, outputs);
  for (auto v : llvm::enumerate(call.getResults())) {
    std::string v_name = module::getName(outputs[v.index()]).str();
    std::vector<bool> user_is_cpu;
    tensor_is_cpu[v_name] = user_is_cpu;
    for (auto user : v.value().getUsers()) {
      if (isa<tpu::YieldOp, ReturnOp>(user)) {
        tensor_is_cpu[v_name].push_back(false);
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto func = module::getFuncOp(s, call.getCallee());
        auto mode = getRunMode(func);
        tensor_is_cpu[v_name].push_back(mode == RunMode::CPU);
      }
    }
  }

  if (isa<tpu::LoopOp>(call->getParentOp())) {
    std::vector<Value> tmp;
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(tmp));
    std::vector<Value>().swap(inputs);
    for (int i = 0; i < tmp.size(); i++) {
      inputs.insert(inputs.begin() + 2 * i, {tmp[i], tmp[i]});
    }

    int32_t out_num = outputs.size();
    std::vector<Value>().swap(outputs);
    Operation *loopop = cast<tpu::LoopOp>(call->getParentOp());
    // replace the actual output for next iteration
    for (int i = 0; i < out_num; i++) {
      outputs.emplace_back(module::getOriValue(loopop->getOperand(i + 1)));
    }
  } else {
    if (isa<tpu::IfOp>(module::getOriValue(inputs[0]).getDefiningOp())) {
      auto ifOp =
          dyn_cast<tpu::IfOp>(module::getOriValue(inputs[0]).getDefiningOp());
      inputs.clear();
      for (int k = 0; k < ifOp.getNumResults(); k++) {
        for (int i = 0; i < ifOp.getNumRegions(); i++) {
          Region &region = ifOp.getRegion(i);
          Operation *yieldOp = region.back().getTerminator();
          inputs.emplace_back(module::getOriValue(yieldOp->getOperand(k)));
        }
      }
    } else {
      // loopOp
      auto loopOp =
          dyn_cast<tpu::LoopOp>(module::getOriValue(inputs[0]).getDefiningOp());
      inputs.clear();
      Operation *yieldOp = loopOp.getBody().back().getTerminator();
      for (int k = 0; k < loopOp.getNumResults(); k++) {
        inputs.emplace_back(module::getOriValue(yieldOp->getOperand(k + 1)));
        inputs.emplace_back(module::getOriValue(loopOp.getOperand(k + 2)));
      }
    }
  }

  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);
  auto &builder = model_gen->Builder();
  auto next_ids = builder.CreateVector(next_id_v);
  vector<vector<int>> output_from;
  int index = 0;
  for (int i = 0; i < outputs.size(); i++) {
    output_from.emplace_back(vector{index++, index++});
  }
  Offset<bmodel::MergeParam> merge_param = CreateMergeParamVector(output_from);

  bmodel::SubNetBuilder snb(builder);
  snb.add_merge_param(merge_param);
  snb.add_is_dynamic(false);
  snb.add_subnet_mode(SUBNET_MODE_MERGE);
  snb.add_input_tensor(input_tensor);
  snb.add_output_tensor(output_tensor);
  snb.add_id(subnet_id);
  snb.add_next_subnet_ids(next_ids);
  return snb.Finish();
}

void BMCodegen::codegen_ir(Operation *op, SubnetIr *subnet_ir_) {
  if (module::isOpInGroup(op) || isa_and_nonnull<top::WeightOp>(op)) {
    return;
  } else if (dyn_cast<GroupOp>(op) || dyn_cast<DynGlobalGenInterface>(op)) {
    subnet_ir_->generate_crop_layer_shape_tensor_record();
    subnet_ir_->generate_group_time_step_ir(op);
  }
}

Offset<bmodel::SubNet>
BMCodegen::CreateSubNet(ModuleOp s, func::CallOp call,
                        std::unique_ptr<SubnetIr> subnet_ir_,
                        std::unique_ptr<Context> &context) {
  auto func = module::getFuncOp(s, call.getCallee());
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  module::getInputsOutputs(call, inputs, outputs);
  auto next_index = func->getAttrOfType<DenseI32ArrayAttr>("next_index");
  std::vector<int> next_id_v(next_index.asArrayRef());
  // for net input directly link to cpu subnet
  for (auto v : llvm::enumerate(inputs)) {
    auto v_name = module::getName(v.value()).str();
    // if v_name do not in tensor_is_cpu, means it is a net input
    if (tensor_is_cpu.count(v_name) && !tensor_is_cpu[v_name].empty()) {
      continue;
    }
    std::vector<bool> user_is_cpu;
    tensor_is_cpu[v_name] = user_is_cpu;
    for (auto user : v.value().getUsers()) {
      if (isa<tpu::YieldOp, ReturnOp>(user)) {
        tensor_is_cpu[v_name].push_back(false);
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto func = module::getFuncOp(s, call.getCallee());
        auto mode = getRunMode(func);
        tensor_is_cpu[v_name].push_back(mode == RunMode::CPU);
      }
    }
  }
  for (auto v : llvm::enumerate(call.getResults())) {
    std::string v_name = module::getName(outputs[v.index()]).str();
    std::vector<bool> user_is_cpu;
    tensor_is_cpu[v_name] = user_is_cpu;
    for (auto user : v.value().getUsers()) {
      if (isa<tpu::YieldOp, ReturnOp>(user)) {
        tensor_is_cpu[v_name].push_back(false);
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto func = module::getFuncOp(s, call.getCallee());
        auto mode = getRunMode(func);
        tensor_is_cpu[v_name].push_back(mode == RunMode::CPU);
      }
    }
  }

  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);
  std::function<void(Operation *, SubnetIr *)> task =
      std::bind(&BMCodegen::codegen_ir, this, std::placeholders::_1,
                std::placeholders::_2);
  subnet_ir_->generate_compiler_ir(s, call, task);
  subnet_ir_->write_binary_ir_to_buffer(context);
  int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();

  LLVM_DEBUG(llvm::dbgs() << "subnet id: '" << subnet_id << "'\n");

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
BMCodegen::CreateStageIRVector(const vector<stage_param_t> &stage_param_v,
                               const vector<uint32_t> &binary_ir_v,
                               uint32_t ir_offset, bmodel::Binary &binary_ir) {
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

SmallString<128> BMCodegen::gen_op_id(Operation *op) {
  int64_t line_num = -1; // unknown location
  auto it = opToLineCol.find(op);
  if (it != opToLineCol.end()) {
    line_num = it->second.first;
  }
  SmallString<128> prefix = op->getName().getStringRef().substr(4);
  prefix.append({"_", std::to_string(line_num)});
  return prefix;
}

bool BMCodegen::isHiddenTensor(StringRef name) {
  return std::find(hidden_names.begin(), hidden_names.end(), name) !=
         hidden_names.end();
}

void BMCodegen::checkAndUpdateHidden(const std::vector<Value> &inputs,
                                     const std::vector<Value> &outputs) {
  for (auto in : inputs) {
    auto name = module::getName(in);
    if (std::find(input_names->begin(), input_names->end(), name) !=
        input_names->end()) {
      continue;
    }
    assert(std::find(hidden_names.begin(), hidden_names.end(), name) !=
           hidden_names.end());
  }
  for (auto out : outputs) {
    auto name = module::getName(out);
    if (std::find(output_names->begin(), output_names->end(), name) !=
        output_names->end()) {
      continue;
    }
    hidden_names.push_back(name);
  }
}

} // namespace tpu
} // namespace tpu_mlir
