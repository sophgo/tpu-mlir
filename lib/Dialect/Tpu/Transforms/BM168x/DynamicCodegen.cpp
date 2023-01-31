//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <set>
#include <sstream>
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Builder/BM168x/bmodel.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynamicNetIr.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynamicLayer.hpp"

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::backend;
using namespace flatbuffers;
namespace tpu_mlir {
namespace tpu {

class DynCodegenPass : public DynCodegenBase<DynCodegenPass> {
public:
  DynCodegenPass() {}
  ~DynCodegenPass() {}
  void runOnOperation() override {
    module = getOperation();
    assert(module::isState(module::State::TPU_ADDRESSED));
    chip = module::getChip();
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
        // bm168x->value_s2d(op.output(), op.read_as_byte()->data());
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
    model_gen->AddChip(chip.str());
    auto &builder = model_gen->Builder();
    auto input_tensor = CreateTensorVector(inputs);
    auto output_tensor = CreateTensorVector(outputs);
    auto coeff_mem = CreateCoeffMem(weights, coeff_addr, coeff_size);
    std::vector<uint64_t> neuron_sizes = {(uint64_t)neuron_size};
    auto neuron_sizes_fb = builder.CreateVector(neuron_sizes);
    // codegen all subnet
    auto main_func = module::getMainFuncOp();
    std::vector<Offset<bmodel::SubNet>> subnet_v;
    std::vector<SubnetIr*> subnet_ir_v;
    main_func.walk([&](func::CallOp call) {
      auto subnet_ir_ = new SubnetIr(chip, module::isBM1684XFamily() ? 2 : 1);
      subnet_ir_v.push_back(subnet_ir_);
      auto subnet = CreateSubNet(call, subnet_ir_);
      subnet_v.push_back(subnet);
    });
    auto subnets = builder.CreateVector(subnet_v);

    bmodel::Binary binary_ir;
    auto stage_ir = CreateStageIRVector(subnet_ir_v[0]->stage_param_vv,
                                      subnet_ir_v[0]->m_ir_buffer, 0, binary_ir);

    bmodel::NetParameterBuilder npb(builder);
    npb.add_input_tensor(input_tensor);
    npb.add_output_tensor(output_tensor);
    npb.add_ctx_addr(neuron_addr);
    npb.add_ctx_size(neuron_size);
    npb.add_ctx_sizes(neuron_sizes_fb);
    npb.add_coeff_mem(coeff_mem);

    npb.add_is_dynamic(true);
    npb.add_n_dynamic(true);
    npb.add_h_w_dynamic(true);
    npb.add_stage_ir(stage_ir);
    npb.add_binary_ir(&binary_ir);
    // create subnet
    npb.add_sub_net(subnets);
    model_gen->AddNet(module::getModuleName().str(), npb.Finish());
    model_gen->Finish();
    model_gen->Save(filename);
    for (auto v: subnet_ir_v)
      delete v;
  }

private:
  Offset<Vector<Offset<bmodel::Shape>>>
  CreateShapeVector(const ArrayRef<int64_t> &shape);
  Offset<Vector<Offset<bmodel::Tensor>>>
  CreateTensorVector(const std::vector<Value> &values);
  Offset<bmodel::SubNet> CreateSubNet(func::CallOp call, SubnetIr* subnet_ir_);
  Offset<bmodel::CoeffMem> CreateCoeffMem(std::vector<top::WeightOp> &coeffs,
                                          uint64_t coeff_addr,
                                          uint64_t coeff_size);
  Offset<Vector<Offset<bmodel::StageIR>>> CreateStageIRVector(
                                            const vector<stage_param_t> &stage_param_v,
                                            const vector<u32> &binary_ir_v, u32 ir_offset,
                                            bmodel::Binary &binary_ir);
  void codegen(Operation *op, SubnetIr* subnet_ir_);
private:
  ModuleOp module;
  StringRef state;
  StringRef chip;
  BM168x *bm168x;
  std::shared_ptr<bmodel::ModelGen> model_gen;
};

std::unique_ptr<OperationPass<ModuleOp>> createDynCodegenPass() {
  return std::make_unique<DynCodegenPass>();
}

Offset<Vector<Offset<bmodel::Shape>>>
DynCodegenPass::CreateShapeVector(const ArrayRef<int64_t> &shape) {
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
DynCodegenPass::CreateTensorVector(const std::vector<Value> &values) {
  auto &builder = model_gen->Builder();
  std::vector<Offset<bmodel::Tensor>> tensor_v;
  for (auto v : values) {
    auto v_name = module::getName(v).str();
    auto type = module::getStorageType(v);
    auto shape = module::getShape(v);
    auto typeBytes = type.getIntOrFloatBitWidth() / 8;
    auto data_type = BM168x::getDataType(type);
    auto gmem_stmode = STORE_MODE_1N;
    if (chip == module::Chip::BM1684) {
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
DynCodegenPass::CreateCoeffMem(std::vector<top::WeightOp> &coeffs,
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

void DynCodegenPass::codegen(Operation *op, SubnetIr* subnet_ir_) {
  if (module::isOpInGroup(op) || isa_and_nonnull<top::WeightOp>(op)) {
    return;
  } else if (dyn_cast<tpu::GroupOp>(op) || dyn_cast<DynGlobalGenInterface>(op)) {
    subnet_ir_->generate_crop_layer_shape_tensor_record();
    subnet_ir_->generate_group_time_step_ir(op);
  }
}

Offset<bmodel::SubNet> DynCodegenPass::CreateSubNet(func::CallOp call, SubnetIr* subnet_ir_) {
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  module::getInputsOutputs(call, inputs, outputs);
  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);
  std::function<void(Operation *, SubnetIr*)> task = std::bind(&DynCodegenPass::codegen, this,
                      std::placeholders::_1, std::placeholders::_2);
  subnet_ir_->generate_compiler_ir(module, call, task);
  int subnet_ir_len = subnet_ir_->write_binary_ir_to_buffer();
  stage_param_t stage;
  memset(&stage, 0, sizeof(stage_param_t));
  stage.ir_info_len = (subnet_ir_len + sizeof(u32) - 1) / sizeof(u32);
  subnet_ir_->stage_param_vv.push_back(stage);
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
        next_id_v.push_back(id);
      } else {
        llvm_unreachable("next op is illegal");
      }
    }
  }
  auto &builder = model_gen->Builder();
  auto next_ids = builder.CreateVector(next_id_v);

  bmodel::SubNetBuilder snb(builder);
  snb.add_is_dynamic(true);
  snb.add_subnet_mode(SUBNET_MODE_TPU);
  snb.add_input_tensor(input_tensor);
  snb.add_output_tensor(output_tensor);
  snb.add_ir_offset(0);
  snb.add_ir_len(subnet_ir_len);
  snb.add_id(subnet_id);
  snb.add_next_subnet_ids(next_ids);
  return snb.Finish();
}

Offset<Vector<Offset<bmodel::StageIR>>> DynCodegenPass::CreateStageIRVector(
    const vector<stage_param_t> &stage_param_v,
    const vector<u32> &binary_ir_v, u32 ir_offset,
    bmodel::Binary &binary_ir)
{
  auto &builder = model_gen->Builder();
  vector<Offset<bmodel::StageIR>> stage_ir_v;
  u32 ir_len = 0;
  for (auto& stage_param : stage_param_v) {
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
