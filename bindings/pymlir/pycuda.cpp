//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "pycuda.h"
#include "cuda/cuda_helper.h"

py_cuda::py_cuda() { CHECK_CUDNN(cudnnCreate(&cudnn_)); }

py_cuda::~py_cuda() { CHECK_CUDNN(cudnnDestroy(cudnn_)); }

void py_cuda::gpu_load(ModuleOp m) {
  // load weight to cuda
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp weight) {
      cuda_malloc(weight_map_, weight);
      auto data = weight.read_as_byte();
      auto name = module::getName(weight.getOutput()).str();
      auto mem = weight_map_[name].get();
      CHECK_CUDA(
          cudaMemcpy(mem, data->data(), data->size(), cudaMemcpyHostToDevice));
    });
  }
  // perpare input and output
  std::vector<mlir::Value> inputs;
  std::vector<mlir::Value> outputs;
  module::getInputsOutputs(m, inputs, outputs);

  for (auto v : inputs) {
    auto name = module::getName(v).str();
    input_names.append(name);
    input_names_.push_back(name);
    cuda_malloc(activation_map_, v);
  }
  for (auto &v : outputs) {
    auto name = module::getName(v).str();
    output_names.append(name);
    output_names_.push_back(name);
    cuda_malloc(activation_map_, v);
    auto buffer =
        std::make_shared<std::vector<float>>(module::getNumElements(v));
    buffer_map_[name] = std::move(buffer);
    // std::cout << "66 output name: " << name << " size " << module::getNumElements(v)*sizeof(float) << " addr " << buffer_map_[name] << std::endl;
  }
}

void py_cuda::load(std::string filename) {
  if (context_) {
    llvm_unreachable("mlir can only load once");
  }
  DialectRegistry registry;
  registry.insert<func::FuncDialect, top::TopDialect, tpu::TpuDialect,
                  quant::QuantizationDialect>();
  context_ = std::make_unique<MLIRContext>(registry);

  module_ = parseSourceFile<ModuleOp>(filename, context_.get());
  assert(module_);
  auto m = module_.get();
  module::init(m);

  mix_mode_ = false;
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<ReturnOp, top::InputOp, top::WeightOp, top::NoneOp, func::FuncOp>(op))
        return;
      else if (!is_cuda_support_op(op)) {
        mix_mode_ = true;
        return;
      }
    });
  }
  if (!mix_mode_)
    return gpu_load(m);
  else
    return mix_load(m);
}

bool py_cuda::is_cuda_support_op(Operation *op) {
  if (isa<tpu::AddOp, tpu::Conv2DOp, tpu::CastOp, tpu::ConcatOp, tpu::DeconvOp,
          tpu::GatherOp, tpu::GenericCpuOp, tpu::LutOp, tpu::MatMulOp, tpu::MulOp,
          tpu::MulShiftOp, tpu::ReshapeOp, tpu::RequantIntAxisOp, tpu::Pool2DOp,
          tpu::PReluOp, tpu::PermuteOp, tpu::SliceOp, tpu::SoftmaxOp, tpu::SqueezeOp,
          tpu::TileOp, tpu::UpsampleOp, tpu::UnsqueezeOp, tpu::ActiveOp, tpu::SubOp,
          tpu::MulConstOp, tpu::LayerNormOp, tpu::Depth2SpaceOp, tpu::ReduceOp,
          tpu::SwapDimInnerOp, tpu::SubConstOp, tpu::RequantFpOp>(op))
    return true;
  else if (isa<top::AddOp, top::ConvOp, top::ScaleOp, top::MaxPoolOp, top::AvgPoolOp,
              top::MatMulOp, top::ReshapeOp, top::SiLUOp, top::ConcatOp, top::UpsampleOp,
              top::PermuteOp, top::SliceOp, top::SoftmaxOp, top::SubOp, top::MulConstOp,
              top::MulOp, top::SigmoidOp, top::LayerNormOp, top::SqueezeOp, top::GELUOp,
              top::Depth2SpaceOp, top::ReduceOp, top::SwapDimInnerOp, top::UnsqueezeOp,
              top::SubConstOp, top::GatherOp, top::RequantFpOp>(op)) {
    return true;
  }
  return false;
}

bool py_cuda::is_no_mem_op(Operation *op) {
  if (isa<top::ReshapeOp, tpu::ReshapeOp, top::UnsqueezeOp, top::SqueezeOp, tpu::UnsqueezeOp, tpu::SqueezeOp>(
      op))
    return true;
  return false;
}

Operation * py_cuda::get_weight_target_op(Operation *weight_op) {
  assert(!weight_op->getUsers().empty());
  assert(std::distance(weight_op->getUsers().begin(), weight_op->getUsers().end())== 1 &&
         "weight has multiple target ops, not support yet");
  return *weight_op->getUsers().begin();
}

// load module in cpu and gpu
void py_cuda::mix_load(ModuleOp m) {
  // load weight to cuda
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (op == func.getOperation() || isa<top::NoneOp>(op)) {
        // self
      } else if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          auto name = module::getName(v).str();
          output_names.append(name);
          output_names_.push_back(name);
          auto count = module::getNumElements(v);
          if (count != 0){
            cuda_malloc(activation_map_, v);
            auto buffer =
                std::make_shared<std::vector<float>>(module::getNumElements(v));
            buffer_map_[name] = std::move(buffer);
            auto buffer_infer =
                std::make_shared<std::vector<float>>(module::getNumElements(v));
            infer_map_[name] = std::move(buffer_infer);
            value_map_[name] = v;
          }
        }
      } else if (auto in_op = dyn_cast<top::InputOp>(op)) {
        auto v = in_op.getOutput();
        auto name = module::getName(v).str();
        input_names.append(name);
        input_names_.push_back(name);
        auto count = module::getNumElements(v);
        if (count != 0){
          buffer_map_[name] = std::make_shared<std::vector<float>>(count);
          infer_map_[name] = std::make_shared<std::vector<float>>(count);
          cuda_malloc(activation_map_, v);
          value_map_[name] = v;
        }
      } else if (auto wOp = dyn_cast<top::WeightOp>(op)) {
        auto v = wOp.getOutput();
        auto name = module::getName(v).str();
        if (is_cuda_support_op(get_weight_target_op(wOp))) {
          cuda_malloc(weight_map_, wOp);
          auto data = wOp.read_as_byte();
          auto mem = weight_map_[name].get();
          CHECK_CUDA(
              cudaMemcpy(mem, data->data(), data->size(), cudaMemcpyHostToDevice));
        } else {
          infer_map_[name] = wOp.read_as_float();
        }
        value_map_[name] = v;
      // } else if (is_no_mem_op(op)) { // all cuda supported
      //   auto v = op->getResult(0);
      //   auto name = module::getName(v).str();
      //   auto in = module::getName(op->getOperand(0)).str();
      //   buffer_map_[name] = buffer_map_[in];
      //   value_map_[name] = v;
      } else {
        if (is_cuda_support_op(op)) {
          for (auto r : op->getResults()) {
            auto count = module::getNumElements(r);
            auto name = module::getName(r).str();
            bool to_cpu = false;
            for (auto u:r.getUsers()){
              if (!is_cuda_support_op(u)){
                to_cpu = true;
                break;
              }
            }
            if (count != 0){
              if (to_cpu){
                buffer_map_[name] = std::make_shared<std::vector<float>>(count);
                infer_map_[name] = std::make_shared<std::vector<float>>(count);
              }
              value_map_[name] = r;
            }
          }
        }
      }
    });
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        if (is_cuda_support_op(op)) {
          return;
        }
        auto name = module::getName(op).str();
        auto param = std::make_shared<InferenceParameter>();
        for (auto result : op->getResults()) {
          if (result.getType().isa<NoneType>()) {
            param->outputs.push_back(nullptr);
          } else if (0 == module::getNumElements(result)) {
            param->outputs.push_back(nullptr);
          } else {
            auto o_name = module::getName(result).str();
            if (infer_map_.find(o_name) == infer_map_.end()){
              auto buffer =
                  std::make_shared<std::vector<float>>(module::getNumElements(result));
              infer_map_[o_name] = std::move(buffer);
              if (buffer_map_.find(o_name) == buffer_map_.end())
                buffer_map_[o_name] = infer_map_[o_name];
            }
            param->outputs.push_back(infer_map_[o_name]->data());
            value_map_[o_name] = result;
          }
        }
        for (auto input : op->getOperands()) {
          std::string input_name;
          if (module::isNone(input)) {
            param->inputs.push_back(nullptr);
            continue;
          } else if (input.isa<BlockArgument>()) {
            /* op support nested ops,
                can transfer operands by blockargument */
            std::size_t index = input.cast<BlockArgument>().getArgNumber();
            auto vv = input.cast<BlockArgument>()
                            .getOwner()
                            ->getParentOp()
                            ->getOperands()[index];
            input_name = module::getName(vv).str();
          } else {
            input_name = module::getName(input).str();
          }

          if (infer_map_.find(input_name) == infer_map_.end()) {
            if (module::getNumElements(input) == 0) {
              param->inputs.push_back(nullptr);
            } else {
              auto buffer =
                  std::make_shared<std::vector<float>>(module::getNumElements(input));
              infer_map_[input_name] = std::move(buffer);
              if (buffer_map_.find(input_name) == buffer_map_.end())
                buffer_map_[input_name] = infer_map_[input_name];
              value_map_[input_name] = input;
              param->inputs.push_back(infer_map_[input_name]->data());
            }
          } else {
            param->inputs.push_back(infer_map_[input_name]->data());
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "init: '" << name << "'\n");
        if (failed(infer_op.init(*param))) {
          op->dump();
          llvm_unreachable("op inferece init failed");
        }
        inference_map[name] = param;
      }
    });
  }
  module::detachWeightFile(); // to free weight memory
  // now to construct cpu inference params
}

void py_cuda::cuda_malloc(std::map<std::string, cuda_ptr> &map, mlir::Value v) {
  auto name = module::getName(v).str();
  void *cuda_mem;
  CHECK_CUDA(cudaMalloc(&cuda_mem, module::getBytes(v)));
  cuda_ptr wrapper(cuda_mem);
  map[name] = std::move(wrapper);
  value_map_[name] = v;
  // std::cout << "cudamalloc value name: " << name << " size " << module::getBytes(v) << " addr " << cuda_mem << std::endl;
}

cuda_ptr py_cuda::cuda_malloc(size_t bytes) {
  void *cuda_mem;
  CHECK_CUDA(cudaMalloc(&cuda_mem, bytes));
  cuda_ptr wrapper(cuda_mem);
  return std::move(wrapper);
}

// data in activation_map_ copy to buffer_map_; if it not float, convert to
// float first
void py_cuda::cuda_to_host(const std::string &name, bool for_infer=false) {
  auto it_act = activation_map_.find(name);
  if (it_act == activation_map_.end()) {
    llvm_unreachable("name not find in activation");
  }
  auto cudaData = it_act->second.get();
  auto it_value = value_map_.find(name);
  if (it_value == value_map_.end()) {
    llvm_unreachable("name not find in value");
  }
  auto v = it_value->second;
  auto it_buffer = buffer_map_.find(name);
  if (buffer_map_.find(name) == buffer_map_.end()) {
    auto buffer =
        std::make_shared<std::vector<float>>(module::getNumElements(v));
    buffer_map_[name] = std::move(buffer);
    it_buffer = buffer_map_.find(name);
    // std::cout << "99 output name: " << name << " size " << module::getNumElements(v)*sizeof(float) << " addr " << buffer_map_[name] << std::endl;
  }
  auto buffer = it_buffer->second->data();

  // std::cout << "cuda_to_host name: " << name << " size " << module::getNumElements(v)*sizeof(float) << " addr " << cudaData << " infer " << for_infer<< std::endl;
  auto stype = module::getStorageType(v);
  // stype.dump();
  if (for_infer) {
    // convert type with out scaling
    assert(infer_map_.find(name) != infer_map_.end() && "infer map not found");
    auto infer_buffer = infer_map_[name]->data();
    if (getCudaType(v) == cuda::DT_F32) {
      CHECK_CUDA(cudaMemcpy(infer_buffer, cudaData, module::getNumElements(v)*sizeof(float), cudaMemcpyDeviceToHost));
    } else {
      auto tmp = cuda_malloc(module::getNumElements(v)*sizeof(float));
      CHECK_CUDA(cuda::convertType(cudaData, tmp.get(), module::getNumElements(v), getCudaType(v), cuda::DT_F32));
      CHECK_CUDA(cudaMemcpy(infer_buffer, tmp.get(), module::getNumElements(v)*sizeof(float), cudaMemcpyDeviceToHost));
      tmp.reset();
    }
    return;
  }
  if (stype.isF32()) {
    // std::cout << "cuda_to_host F32 name: " << name << " size " << module::getNumElements(v)*sizeof(float) << " addr " << cudaData << std::endl;
    auto bytes = it_buffer->second->size() * sizeof(float);
    CHECK_CUDA(cudaMemcpy(buffer, cudaData, bytes, cudaMemcpyDeviceToHost));
  } else if (module::isUniformQuantized(v) && stype.isInteger(8)) {
    auto qtype = module::getUniformQuantizedType(v);
    auto num = module::getNumElements(v);
    uint8_t *temp = new uint8_t[num];
    CHECK_CUDA(cudaMemcpy(temp, cudaData, num, cudaMemcpyDeviceToHost));
    if (stype.isUnsignedInteger(8)) {
      for (int i = 0; i < num; i++) {
        buffer[i] = ((float)temp[i] - (float)qtype.getZeroPoint()) *
                    (float)qtype.getScale();
      }
    } else {
      int8_t *temp_i8 = (int8_t *)temp;
      for (int i = 0; i < num; i++) {
        buffer[i] = ((float)temp_i8[i] - (float)qtype.getZeroPoint()) *
                    (float)qtype.getScale();
      }
    }
    delete[] temp;
  } else if (stype.isInteger(32)) {
    auto num = module::getNumElements(v);
    int32_t *temp = new int32_t[num];
    CHECK_CUDA(cudaMemcpy(temp, cudaData, num * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < num; i++) {
      buffer[i] = static_cast<float>(temp[i]);
    }
    delete[] temp;
  } else if (stype.isBF16()) {
    auto num = module::getNumElements(v);
    uint16_t *temp = new uint16_t[num];
    CHECK_CUDA(cudaMemcpy(temp, cudaData, num * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < num; i++) {
      buffer[i] = bf16_to_f32(temp[i]);
    }
    delete[] temp;
  } else if (stype.isF16()) {
    auto num = module::getNumElements(v);
    uint16_t *temp = new uint16_t[num];
    CHECK_CUDA(cudaMemcpy(temp, cudaData, num * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < num; i++) {
      buffer[i] = f16_to_f32(temp[i]);
    }
    delete[] temp;
  } else if (stype.isInteger(16)) {
    auto num = module::getNumElements(v);
    auto qtype = module::getUniformQuantizedType(v);
    int16_t * temp = new int16_t[num];
    auto scale = qtype.getScale();
    CHECK_CUDA(cudaMemcpy(temp, cudaData, num * sizeof(int16_t),
                          cudaMemcpyDeviceToHost));
    for (size_t i=0; i<num; i++) {
      buffer[i] = temp[i] * scale;
    }
  } else if (stype.isFloat8E4M3FN()) {
    auto num = module::getNumElements(v);
    auto ctype = module::getCalibratedType(v);
    float scale = ctype.getMax() / get_f8e4m3_max();
    uint8_t *temp = new uint8_t[num];
    CHECK_CUDA(cudaMemcpy(temp, cudaData, num * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < num; i++) {
      buffer[i] = f8e4m3_to_f32(temp[i]) * scale;
    }
    delete[] temp;
  } else {
    v.dump();
    llvm_unreachable("Not Implemented");
  }
}

void py_cuda::set_tensor(
    std::string name,
    py::array_t<float, py::array::c_style | py::array::forcecast> data) {
  auto it_value = value_map_.find(name);
  if (it_value == value_map_.end()) {
    llvm_unreachable("set_tensor name is not exist");
  }
  auto v = it_value->second;
  auto num = module::getNumElements(v);
  if (data.size() != num) {
    llvm_unreachable("set_tensor data size is uncorrect");
  }
  auto bytes = module::getBytes(v);
  auto v_type = getCudaType(v);
  void *dst = getCudaData(v);
  if (v_type != cuda::DT_F32) {
    int src_bytes = data.size() * sizeof(float);
    auto tmp = cuda_malloc(src_bytes);
    CHECK_CUDA(
        cudaMemcpy(tmp.get(), data.data(), src_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cuda::convertType(tmp.get(), dst, num, cuda::DT_F32, v_type));
  } else {
    CHECK_CUDA(cudaMemcpy(dst, data.data(), bytes, cudaMemcpyHostToDevice));
  }
  // FIXME consider the datatype and scale
  auto it_buffer = buffer_map_.find(name);
  if (it_buffer != buffer_map_.end()) {
    it_buffer->second->resize(data.size());
    auto buffer = it_buffer->second->data();
    std::copy(data.data(), data.data() + data.size(), buffer);
  }
  it_buffer = infer_map_.find(name);
  if (it_buffer != infer_map_.end()) {
    it_buffer->second->resize(data.size());
    auto buffer = it_buffer->second->data();
    std::copy(data.data(), data.data() + data.size(), buffer);
  }
}

void py_cuda::gpu_invoke(bool dump_all, const std::vector<std::string>& extra_outputs) {
  auto m = module_.get();
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<top::NoneOp, top::InputOp, top::WeightOp, func::FuncOp,
              func::ReturnOp>(op)) {
        if (dump_all && isa<top::InputOp>(op)) {
          auto name = module::getName(op).str();
          cudaDeviceSynchronize();
          cuda_to_host(name);
        }
      } else {
        // 1. alloc output mem in cuda
        for (auto out : op->getResults()) {
          if (module::isNone(out)) {
            continue;
          }
          auto name = module::getName(out).str();
          if (activation_map_.find(name) != activation_map_.end()) {
            continue;
          }
          cuda_malloc(activation_map_, out);
        }
        // 2. inference
        if (auto tpuOp = dyn_cast<tpu::AddOp>(op)) {
          cudaAddOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::AddOp>(op)) {
          cudaAddOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::Conv2DOp>(op)) {
          cudaConv2DOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::ConvOp>(op)) {
          cudaConvOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::ConcatOp>(op)) {
          cudaConcatOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::ConcatOp>(op)) {
          cudaConcatOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::CastOp>(op)) {
          cudaCastOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::DeconvOp>(op)) {
          cudaDeconvOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::Depth2SpaceOp>(op)) {
          cudaDepth2SpaceOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::GenericCpuOp>(op)) {
          cudaGenericCpuOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::GatherOp>(op)) {
          cudaGatherOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::GatherOp>(op)) {
          cudaGatherOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::LutOp>(op)) {
          cudaLutOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::MatMulOp>(op)) {
          cudaMatMulOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::MatMulOp>(op)) {
          cudaMatMulOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::MulOp>(op)) {
          cudaMulOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::MulOp>(op)) {
          cudaMulOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::MulShiftOp>(op)) {
          cudaMulShiftOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::ReshapeOp>(op)) {
          cudaReshapeOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::ReshapeOp>(op)) {
          cudaReshapeOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::RequantIntAxisOp>(op)) {
          cudaRequantIntAxisOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::Pool2DOp>(op)) {
          cudaPool2DOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::MaxPoolOp>(op)) {
          cudaMaxPoolOp(topOp);
        } else if (auto topOp = dyn_cast<top::AvgPoolOp>(op)) {
          cudaAvgPoolOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::PReluOp>(op)) {
          cudaPReluOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::PermuteOp>(op)) {
          cudaPermuteOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::PermuteOp>(op)) {
          cudaPermuteOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SliceOp>(op)) {
          cudaSliceOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SliceOp>(op)) {
          cudaSliceOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SoftmaxOp>(op)) {
          cudaSoftmaxOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SoftmaxOp>(op)) {
          cudaSoftmaxOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SqueezeOp>(op)) {
          cudaSqueezeOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SqueezeOp>(op)) {
          cudaSqueezeOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::TileOp>(op)) {
          cudaTileOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::UpsampleOp>(op)) {
          cudaUpsampleOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::UpsampleOp>(op)) {
          cudaUpsampleOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::UnsqueezeOp>(op)) {
          cudaUnsqueezeOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::UnsqueezeOp>(op)) {
          cudaUnsqueezeOp(topOp);
        } else if (auto topOp = dyn_cast<top::ScaleOp>(op)) {
          cudaScaleOp(topOp);
        } else if (auto topOp = dyn_cast<top::SiLUOp>(op)) {
          cudaSiLUOp(topOp);
        } else if (auto topOp = dyn_cast<top::GELUOp>(op)) {
          cudaGELUOp(topOp);
        } else if (auto topOp = dyn_cast<top::SigmoidOp>(op)) {
          cudaSigmoidOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::ActiveOp>(op)) {
          cudaActiveOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SubOp>(op)) {
          cudaSubOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SubOp>(op)) {
          cudaSubOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SubConstOp>(op)) {
          cudaSubConstOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SubConstOp>(op)) {
          cudaSubConstOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::MulConstOp>(op)) {
          cudaMulConstOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::MulConstOp>(op)) {
          cudaMulConstOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::LayerNormOp>(op)) {
          cudaLayerNormOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::LayerNormOp>(op)) {
          cudaLayerNormOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::Depth2SpaceOp>(op)) {
          cudaDepth2SpaceOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::Depth2SpaceOp>(op)) {
          cudaDepth2SpaceOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::ReduceOp>(op)) {
          cudaReduceOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::ReduceOp>(op)) {
          cudaReduceOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SwapDimInnerOp>(op)) {
          cudaSwapDimInnerOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SwapDimInnerOp>(op)) {
          cudaSwapDimInnerOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::RequantFpOp>(op)) {
          cudaRequantFpOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::RequantFpOp>(op)) {
          cudaRequantFpOp(tpuOp);
        } else {
          op->dump();
          __asm__("int3");
          UNREACHABLE_OP("Not Implemented", op);
        }
        // 3. after inference, check input is still need, or remove from cuda
        for (auto in : op->getOperands()) {
          if (!module::isActive(in)) {
            continue;
          }
          auto name = module::getName(in).str();
          if (isa<top::InputOp>(in.getDefiningOp())) {
            continue;
          }
          bool need = false;
          for (auto u : in.getUsers()) {
            if (op->isBeforeInBlock(u)) {
              need = true;
              break;
            }
          }
          if (!need) {
            if (dump_all || std::find(extra_outputs.begin(), extra_outputs.end(), name) != extra_outputs.end()) {
              cudaDeviceSynchronize();
              cuda_to_host(name);
            }
            activation_map_.erase(name);
          }
        }
      }
    });
  }
  // after inference, copy result to buffer
  cudaDeviceSynchronize();
  for (auto &name : output_names_) {
    cuda_to_host(name);
  }
}

void py_cuda::mix_invoke(bool dump_all, const std::vector<std::string>& extra_outputs) {
  auto m = module_.get();
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<top::NoneOp, top::InputOp, top::WeightOp, func::FuncOp,
              func::ReturnOp>(op)) {
        if (dump_all && isa<top::InputOp>(op)) {
          auto name = module::getName(op).str();
          cudaDeviceSynchronize();
          cuda_to_host(name, false);
          cuda_to_host(name, true);
        }
      } else if (!is_cuda_support_op(op)) {
        auto infer_op = dyn_cast<InferenceInterface>(op);
        auto name = module::getName(op).str();
        if (failed(infer_op.inference(*inference_map[name]))) {
          infer_op.dump();
          llvm_unreachable("invoke failed!!");
        }
        // output tensor locates in cpu, and if it is needed to gpu op, it will be
        // copied to gpu in getCudaData and newCudaData
        // std::cout << "invoke cpu op: " << op->getName().getStringRef().str() << std::endl;
        // if (buffer_map_.find(name) == buffer_map_.end()) {
        //   std::cout << "invoke cpu op result not in buffer: " << op->getName().getStringRef().str() << std::endl;
        // }
        // op->dump();
      } else {
        // 1. alloc output mem in cuda
        for (auto out : op->getResults()) {
          if (module::isNone(out)) {
            continue;
          }
          auto name = module::getName(out).str();
          if (activation_map_.find(name) != activation_map_.end()) {
            continue;
          }
          cuda_malloc(activation_map_, out);
          // std::cout << "alloc cuda mem for: " << name << std::endl;
        }
        // std::cout << "invoke cuda op: " << op->getName().getStringRef().str() << std::endl;
        // op->dump();
        // 2. inference
        if (auto tpuOp = dyn_cast<tpu::AddOp>(op)) {
          cudaAddOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::AddOp>(op)) {
          cudaAddOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::Conv2DOp>(op)) {
          cudaConv2DOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::ConvOp>(op)) {
          cudaConvOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::ConcatOp>(op)) {
          cudaConcatOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::ConcatOp>(op)) {
          cudaConcatOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::CastOp>(op)) {
          cudaCastOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::DeconvOp>(op)) {
          cudaDeconvOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::Depth2SpaceOp>(op)) {
          cudaDepth2SpaceOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::GenericCpuOp>(op)) {
          cudaGenericCpuOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::GatherOp>(op)) {
          cudaGatherOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::GatherOp>(op)) {
          cudaGatherOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::LutOp>(op)) {
          cudaLutOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::MatMulOp>(op)) {
          cudaMatMulOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::MatMulOp>(op)) {
          cudaMatMulOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::MulOp>(op)) {
          cudaMulOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::MulOp>(op)) {
          cudaMulOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::MulShiftOp>(op)) {
          cudaMulShiftOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::ReshapeOp>(op)) {
          cudaReshapeOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::ReshapeOp>(op)) {
          cudaReshapeOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::RequantIntAxisOp>(op)) {
          cudaRequantIntAxisOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::Pool2DOp>(op)) {
          cudaPool2DOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::MaxPoolOp>(op)) {
          cudaMaxPoolOp(topOp);
        } else if (auto topOp = dyn_cast<top::AvgPoolOp>(op)) {
          cudaAvgPoolOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::PReluOp>(op)) {
          cudaPReluOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::PermuteOp>(op)) {
          cudaPermuteOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::PermuteOp>(op)) {
          cudaPermuteOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SliceOp>(op)) {
          cudaSliceOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SliceOp>(op)) {
          cudaSliceOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SoftmaxOp>(op)) {
          cudaSoftmaxOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SoftmaxOp>(op)) {
          cudaSoftmaxOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SqueezeOp>(op)) {
          cudaSqueezeOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SqueezeOp>(op)) {
          cudaSqueezeOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::TileOp>(op)) {
          cudaTileOp(tpuOp);
        } else if (auto tpuOp = dyn_cast<tpu::UpsampleOp>(op)) {
          cudaUpsampleOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::UpsampleOp>(op)) {
          cudaUpsampleOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::UnsqueezeOp>(op)) {
          cudaUnsqueezeOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::UnsqueezeOp>(op)) {
          cudaUnsqueezeOp(topOp);
        } else if (auto topOp = dyn_cast<top::ScaleOp>(op)) {
          cudaScaleOp(topOp);
        } else if (auto topOp = dyn_cast<top::SiLUOp>(op)) {
          cudaSiLUOp(topOp);
        } else if (auto topOp = dyn_cast<top::GELUOp>(op)) {
          cudaGELUOp(topOp);
        } else if (auto topOp = dyn_cast<top::SigmoidOp>(op)) {
          cudaSigmoidOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::ActiveOp>(op)) {
          cudaActiveOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SubOp>(op)) {
          cudaSubOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SubOp>(op)) {
          cudaSubOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SubConstOp>(op)) {
          cudaSubConstOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SubConstOp>(op)) {
          cudaSubConstOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::MulConstOp>(op)) {
          cudaMulConstOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::MulConstOp>(op)) {
          cudaMulConstOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::LayerNormOp>(op)) {
          cudaLayerNormOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::LayerNormOp>(op)) {
          cudaLayerNormOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::Depth2SpaceOp>(op)) {
          cudaDepth2SpaceOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::Depth2SpaceOp>(op)) {
          cudaDepth2SpaceOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::ReduceOp>(op)) {
          cudaReduceOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::ReduceOp>(op)) {
          cudaReduceOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::SwapDimInnerOp>(op)) {
          cudaSwapDimInnerOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::SwapDimInnerOp>(op)) {
          cudaSwapDimInnerOp(tpuOp);
        } else if (auto topOp = dyn_cast<top::RequantFpOp>(op)) {
          cudaRequantFpOp(topOp);
        } else if (auto tpuOp = dyn_cast<tpu::RequantFpOp>(op)) {
          cudaRequantFpOp(tpuOp);
        } else {
          op->dump();
          __asm__("int3");
          UNREACHABLE_OP("Not Implemented", op);
        }
        // 3. after inference, check input is still need, or remove from cuda
        for (auto in : op->getOperands()) {
          if (!module::isActive(in)) {
            continue;
          }
          auto name = module::getName(in).str();
          if (isa<top::InputOp>(in.getDefiningOp())) {
            continue;
          }
          bool need = false;
          for (auto u : in.getUsers()) {
            if (op->isBeforeInBlock(u)) {
              need = true;
              break;
            }
          }
          if (!need) {
            if (dump_all || std::find(extra_outputs.begin(), extra_outputs.end(), name) != extra_outputs.end()) {
              cudaDeviceSynchronize();
              cuda_to_host(name, false);
            }
            activation_map_.erase(name);
            // std::cout << "release cuda activation: " << name << std::endl;
          }
        }
        for (auto out : op->getResults()) {
          if (!module::isActive(out)) {
            continue;
          }
          auto name = module::getName(out).str();
          if (buffer_map_.find(name) != buffer_map_.end()) {
            cudaDeviceSynchronize();
            cuda_to_host(name,false);
            // std::cout << "copy output to buffer: " << name << std::endl;
          }
          if (infer_map_.find(name) != infer_map_.end()) {
            cudaDeviceSynchronize();
            cuda_to_host(name, true);
            // std::cout << "copy output to infer buffer: " << name << std::endl;
          }
        }
      }
    });
  }
  // after inference, copy result to buffer
  cudaDeviceSynchronize();
  for (auto &name : output_names_) {
    cuda_to_host(name, false);
  }
}

void py_cuda::invoke(bool dump_all, const std::vector<std::string>& extra_outputs) {
  if (mix_mode_)
    return mix_invoke(dump_all, extra_outputs);
  else
    return gpu_invoke(dump_all, extra_outputs);
}

py::array py_cuda::get_tensor(std::string name) {
  auto it_buffer = buffer_map_.find(name);
  if (it_buffer == buffer_map_.end()) {
    std::cout << name << std::endl;
    llvm_unreachable("can't get_tensor name in buffer");
  }
  auto data = it_buffer->second;
  auto it_value = value_map_.find(name);
  if (it_value == value_map_.end()) {
    llvm_unreachable("can't get_tensor name in value");
  }
  auto shape = module::getShape(it_value->second);
  return getPyArray(std::move(data), shape);
}

py::dict py_cuda::get_all_tensor() {
  py::dict py_ret;
  for (auto &it : buffer_map_) {
    auto v = value_map_[it.first];
    auto shape = module::getShape(v);
    py::str py_s(it.first);
    py_ret[py_s] = getPyArray(it.second, shape);
  }
  return py_ret;
}
