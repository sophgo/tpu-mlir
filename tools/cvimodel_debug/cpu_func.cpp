#include "cpu_func.hpp"
#include "tpu_mlir/Builder/CV18xx/parameter_generated.h"
using namespace tpu_mlir;
namespace cvi_debug {

void copyShape(std::vector<int64_t> &ori, std::vector<int64_t> &dst,
               int num = 4) {
  dst.resize(num);
  for (int i = 0; i < num; i++) {
    // dst.push_back(ori[i]);
    dst[i] = ori[i];
  }
}

void handleFuncArgs(const uint8_t *args, OpParam &param) {
  auto packed_param = cvi::cpu_op::GetParameter(args);
  auto &attributes = *packed_param->attributes();
  for (auto attr : attributes) {
    if (attr->int_attr()) {
      auto _int = attr->int_attr();
      param.put<int32_t>(_int->key()->str(), _int->value());
    } else if (attr->float_attr()) {
      auto _float = attr->float_attr();
      param.put<float>(_float->key()->str(), _float->value());
    } else if (attr->bool_attr()) {
      auto _bool = attr->bool_attr();
      param.put<bool>(_bool->key()->str(), _bool->value());
    } else if (attr->str_attr()) {
      auto _str = attr->str_attr();
      param.put<std::string>(_str->key()->str(), _str->value()->str());
    } else if (attr->int_array_attr()) {
      auto _int_array = attr->int_array_attr();
      std::vector<int32_t> vec;
      auto &value = *_int_array->value();
      for (auto v : value) {
        vec.push_back(v);
      }
      param.put<std::vector<int32_t>>(_int_array->key()->str(), vec);
    } else if (attr->float_array_attr()) {
      auto _float_array = attr->float_array_attr();
      std::vector<float> vec;
      auto &value = *_float_array->value();
      for (auto v : value) {
        vec.push_back(v);
        llvm::errs() << v << " ";
      }
      param.put<std::vector<float>>(_float_array->key()->str(), vec);
    } else {
      assert(0);
    }
  }
}

void getCpuInput(std::vector<std::vector<float>> &inputs, uint8_t *vaddr,
                 int64_t &offset, io_mem_info &info) {
  std::vector<float> temp_data(info.count);
  if (info.type == "int8") {
    ConvertInt8ToFp32NoScale((int8_t *)(vaddr + offset), temp_data.data(),
                             info.count);
  } else if (info.type == "uint16") {
    ConvertUint16ToFp32((uint16_t *)(vaddr + offset), temp_data.data(),
                        info.count);
  } else if (info.type == "bf16") {
    ConvertBF16ToFp32((uint16_t *)(vaddr + offset), temp_data.data(),
                      info.count);
  } else if (info.type == "int32") {
    ConvertInt32ToFp32((int32_t *)(vaddr + offset), temp_data.data(),
                       info.count);
  } else if (info.type == "fp32") {
    memcpy(temp_data.data(), (float *)(vaddr + offset),
           info.count * sizeof(float));
  } else {
    std::string err_msg = info.type + " not support in cpu function input";
    llvm_unreachable(err_msg.c_str());
  }
  inputs.emplace_back(temp_data);
}

void getAndSaveCpuOutput(std::vector<float> &output,
                         std::vector<float> &save_data, uint8_t *vaddr,
                         io_mem_info &info) {
  gaddr_t addr = info.gaddr;
  auto memTypeIndx = (addr >> 40) & 0x07;
  int64_t offset = addr - (memTypeIndx << 40);
  if (info.type == "int8") {
    // map to cvimodel vaddr
    ConvertFp32ToInt8NoScale(output.data(), (int8_t *)(vaddr + offset),
                             info.count);
    // save it
    ConvertInt8ToFp32((int8_t *)(vaddr + offset), save_data.data(), info.count,
                      info.qscale);
  } else if (info.type == "bf16") {
    // map to cvimodel vaddr
    ConvertFp32ToBF16(output.data(), (uint16_t *)(vaddr + offset), info.count);
    // save it
    memcpy(save_data.data(), output.data(), info.count * sizeof(float));
  } else if (info.type == "uint16") {
    // map to cvimodel vaddr
    ConvertFp32ToUint16(output.data(), (uint16_t *)(vaddr + offset),
                        info.count);
    // save it
    memcpy(save_data.data(), output.data(), info.count * sizeof(float));
  } else if (info.type == "int32") {
    // map to cvimodel vaddr
    ConvertFp32ToInt32(output.data(), (int32_t *)(vaddr + offset), info.count);
    // save it
    memcpy(save_data.data(), output.data(), info.count * sizeof(float));
  } else if (info.type == "fp32") {
    // map to cvimodel vaddr
    memcpy((float *)(vaddr + offset), output.data(),
           info.count * sizeof(float));
    // save it
    memcpy(save_data.data(), output.data(), info.count * sizeof(float));
  } else {
    std::string err_msg = info.type + " not support in cpu function output";
    llvm_unreachable(err_msg.c_str());
  }
}

void invoke(cpu_func_info &func_info, std::vector<std::vector<float>> &inputs,
            std::vector<std::vector<float>> &outputs) {
  std::string func_name = func_info.func_name;
  if (func_name == "quant") {
    assert(func_info.inputs.size() == 1);
    auto num_elem = func_info.outputs[0].count;
    auto in_type = func_info.inputs[0].type;
    auto out_type = func_info.outputs[0].type;
    if (in_type == "fp32" && out_type == "int8") {
      OpParam param = func_info.params;
      float scale = param.get<float>("scale");
      quantizeToInt8(inputs[0].data(), outputs[0].data(), num_elem, scale);
    } else {
      llvm_unreachable("not supported!\n");
    }
  } else if (func_name == "interp") {
    InterpParam interp_param;
    OpParam param = func_info.params;
    interp_param.height = param.get<int>("height");
    interp_param.width = param.get<int>("width");
    interp_param.pad_beg = param.get<int>("pad_beg");
    interp_param.pad_end = param.get<int>("pad_end");
    interp_param.shrink_factor = param.get<int>("shrink_factor");
    interp_param.zoom_factor = param.get<int>("zoom_factor");
    interp_param.coordinate_transformation_mode =
        param.get<std::string>("coordinate_transformation_mode");
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = inputs[i].data();
      tensor_list.size = func_info.inputs[i].count;
      // tensor_list.shape = getPartShape(func_info.inputs[i].g_shape, 4);
      copyShape(func_info.inputs[i].g_shape, tensor_list.shape);
      interp_param.inputs.emplace_back(std::move(tensor_list));
    }
    interp_param.output.ptr = outputs[0].data();
    interp_param.output.size = func_info.outputs[0].count;
    auto output_shape = func_info.outputs[0].g_shape;
    interp_param.output.shape = {
        (int64_t)output_shape[0], (int64_t)output_shape[1],
        (int64_t)output_shape[2], (int64_t)output_shape[3]};
    assert(interp_param.inputs[0].shape.size() == 4);
    assert(interp_param.output.shape.size() == 4);
    InterpFunc interp_func(interp_param);
    interp_func.invoke();
  } else if (func_name == "embedding") {
    EmbeddingParam embedding_param;
    assert(inputs.size() == 2);
    tensor_list_t input0, input1, output;
    input0.ptr = inputs[0].data();
    input0.size = func_info.inputs[0].count;
    input1.ptr = inputs[1].data();
    input1.size = func_info.inputs[1].count;
    output.ptr = outputs[0].data();
    output.size = func_info.outputs[0].count;
    // infer dim and set shape
    auto input_shape0 = func_info.inputs[0].g_shape;
    auto input_shape1 = func_info.inputs[1].g_shape;
    auto output_shape = func_info.outputs[0].g_shape;
    assert(input_shape1[1] > 1); // embedding dim
    if (output_shape[3] == 1) {
      // output real dim = 3
      assert(output_shape[2] == input_shape1[1]);
      input0.shape = {input_shape0[0], input_shape0[1]};
      input1.shape = {input_shape1[0], input_shape1[1]};
      output.shape = {output_shape[0], output_shape[1], output_shape[2]};
    } else {
      // output real dim = 4
      assert(output_shape[3] == input_shape1[1]);
      input0.shape = {input_shape0[0], input_shape0[1], input_shape0[2]};
      input1.shape = {input_shape1[0], input_shape1[1]};
      output.shape = output_shape;
    }
    embedding_param.inputs.emplace_back(std::move(input0));
    embedding_param.inputs.emplace_back(std::move(input1));
    embedding_param.output = output;
    EmbeddingFunc embedding_func(embedding_param);
    embedding_func.invoke();
  } else if (func_name == "detectionoutput") {
    DetParam det_param;
    copyShape(func_info.inputs[0].g_shape, det_param.loc_shape);
    copyShape(func_info.inputs[1].g_shape, det_param.conf_shape);
    det_param.onnx_nms = func_info.inputs.size() >= 3 ? 0 : 1;
    if (func_info.inputs.size() >= 3) {
      copyShape(func_info.inputs[2].g_shape, det_param.prior_shape);
    }
    det_param.loc_data = inputs[0].data();
    det_param.conf_data = inputs[1].data();
    det_param.prior_data = det_param.onnx_nms ? nullptr : inputs[2].data();
    det_param.output_data = outputs[0].data();
    OpParam param = func_info.params;
    det_param.keep_top_k = param.get<int>("keep_top_k");
    det_param.top_k = param.get<int>("top_k");
    det_param.num_classes = param.get<int>("num_classes");
    det_param.background_label_id = param.get<int>("background_label_id");
    det_param.share_location = param.get<bool>("share_location");
    det_param.confidence_threshold = param.get<float>("confidence_threshold");
    det_param.nms_threshold = param.get<float>("nms_threshold");
    std::string str_code_type = param.get<std::string>("code_type");
    if (str_code_type == "CORNER") {
      det_param.code_type = PriorBoxParameter_CodeType_CORNER;
    } else if (str_code_type == "CENTER_SIZE") {
      det_param.code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
    } else if (str_code_type == "CORNER_SIZE") {
      det_param.code_type = PriorBoxParameter_CodeType_CORNER_SIZE;
    } else {
      llvm_unreachable("code type wrong");
    }
    DetectionOutputFunc det_func(det_param);
    det_func.invoke();
  } else if (func_name == "yolo_detection") {
    YoloDetParam yolo_param;
    OpParam param = func_info.params;
    yolo_param.keep_topk = param.get<int>("keep_topk");
    yolo_param.class_num = param.get<int>("class_num");
    yolo_param.net_input_h = param.get<int>("net_input_h");
    yolo_param.net_input_w = param.get<int>("net_input_w");
    yolo_param.nms_threshold = param.get<float>("nms_threshold");
    yolo_param.obj_threshold = param.get<float>("obj_threshold");
    auto anchors = param.get<std::string>("anchors");
    std::istringstream iss(anchors);
    std::string s;
    while (std::getline(iss, s, ',')) {
      yolo_param.anchors.push_back(atoi(s.c_str()));
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = inputs[i].data();
      tensor_list.size = func_info.inputs[i].count;
      copyShape(func_info.inputs[i].g_shape, tensor_list.shape);
      yolo_param.inputs.emplace_back(std::move(tensor_list));
    }
    yolo_param.output.ptr = outputs[0].data();
    yolo_param.output.size = func_info.outputs[0].count;
    copyShape(func_info.outputs[0].g_shape, yolo_param.output.shape);
    YoloDetectionFunc yolo_func(yolo_param);
    yolo_func.invoke();
  } else if (func_name == "proposal") {
    ProposalParam proposal_param;
    OpParam param = func_info.params;
    proposal_param.anchor_base_size = param.get<int>("anchor_base_size");
    proposal_param.feat_stride = param.get<int>("feat_stride");
    proposal_param.net_input_h = param.get<int>("net_input_h");
    proposal_param.net_input_w = param.get<int>("net_input_w");
    proposal_param.rpn_nms_post_top_n = param.get<int>("rpn_nms_post_top_n");
    proposal_param.rpn_obj_threshold = param.get<float>("rpn_obj_threshold");
    proposal_param.rpn_nms_threshold = param.get<float>("rpn_nms_threshold");
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = inputs[i].data();
      tensor_list.size = func_info.inputs[i].count;
      copyShape(func_info.inputs[i].g_shape, tensor_list.shape);
      proposal_param.inputs.emplace_back(std::move(tensor_list));
    }
    proposal_param.output.ptr = outputs[0].data();
    proposal_param.output.size = func_info.outputs[0].count;
    copyShape(func_info.outputs[0].g_shape, proposal_param.output.shape);
    ProposalFunc proposal_func(proposal_param);
    proposal_func.invoke();
  } else if (func_name == "roi_pooling") {
    ROIPoolingParam roip_param;
    OpParam param = func_info.params;
    roip_param.pooled_h = param.get<int>("pooled_h");
    roip_param.pooled_w = param.get<int>("pooled_w");
    roip_param.spatial_scale = param.get<float>("spatial_scale");
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = inputs[i].data();
      tensor_list.size = func_info.inputs[i].count;
      copyShape(func_info.inputs[i].g_shape, tensor_list.shape);
      roip_param.inputs.emplace_back(std::move(tensor_list));
    }
    roip_param.output.ptr = outputs[0].data();
    roip_param.output.size = func_info.outputs[0].count;
    copyShape(func_info.outputs[0].g_shape, roip_param.output.shape);
    ROIPoolingFunc roip_func(roip_param);
    roip_func.invoke();
  } else if (func_name == "frcn_detection") {
    FrcnDetParam frcn_param;
    OpParam param = func_info.params;
    frcn_param.class_num = param.get<int>("class_num");
    frcn_param.keep_topk = param.get<int>("keep_topk");
    frcn_param.nms_threshold = param.get<float>("nms_threshold");
    frcn_param.obj_threshold = param.get<float>("obj_threshold");
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = inputs[i].data();
      tensor_list.size = func_info.inputs[i].count;
      copyShape(func_info.inputs[i].g_shape, tensor_list.shape);
      frcn_param.inputs.emplace_back(std::move(tensor_list));
    }
    frcn_param.output.ptr = outputs[0].data();
    frcn_param.output.size = func_info.outputs[0].count;
    copyShape(func_info.outputs[0].g_shape, frcn_param.output.shape);
    FrcnDetctionFunc frcn_func(frcn_param);
    frcn_func.invoke();
  } else if (func_name == "retinaface_detection") {
    RetinaFaceDetectionParam retina_param;
    OpParam param = func_info.params;
    retina_param.keep_topk = param.get<int>("keep_topk");
    retina_param.confidence_threshold =
        param.get<float>("confidence_threshold");
    retina_param.nms_threshold = param.get<float>("nms_threshold");
    RetinaFaceDetectionFunc func;
    std::vector<tensor_list_t> tensor_inputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      tensor_list_t tensor;
      tensor.ptr = inputs[i].data();
      tensor.size = func_info.inputs[i].count;
      copyShape(func_info.inputs[i].g_shape, tensor.shape);
      tensor_inputs.emplace_back(std::move(tensor));
    }
    tensor_list_t output;
    output.ptr = outputs[0].data();
    output.size = func_info.outputs[0].count;
    copyShape(func_info.outputs[0].g_shape, output.shape);
    func.setup(tensor_inputs, output, retina_param);
    func.invoke();
  } else if (func_name == "instance_norm") {
    InstanceNormParam inst_param;
    OpParam param = func_info.params;
    inst_param.eps = param.get<float>("variance_epsilon");
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = inputs[i].data();
      tensor_list.size = func_info.inputs[i].count;
      copyShape(func_info.inputs[i].g_shape, tensor_list.shape);
      inst_param.inputs.emplace_back(std::move(tensor_list));
    }
    inst_param.output.ptr = outputs[0].data();
    inst_param.output.size = func_info.outputs[0].count;
    copyShape(func_info.outputs[0].g_shape, inst_param.output.shape);
    InstanceNormFunc inst_func(inst_param);
    inst_func.invoke();
  } else if (func_name == "argmax_v3") {
    ArgMaxParam argmax_param;
    OpParam param = func_info.params;
    argmax_param.axis = param.get<int>("axis");
    argmax_param.fmt_i8 = false;
    argmax_param.scale = 1.0;
    auto in_type = func_info.inputs[0].type;
    if (in_type == "int8") {
      argmax_param.fmt_i8 = true;
    }
    argmax_param.scale = param.get<float>("scale");
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = inputs[i].data();
      tensor_list.size = func_info.inputs[i].count;
      copyShape(func_info.inputs[i].g_shape, tensor_list.shape);
      argmax_param.inputs.emplace_back(std::move(tensor_list));
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = outputs[i].data();
      tensor_list.size = func_info.outputs[i].count;
      copyShape(func_info.outputs[i].g_shape, tensor_list.shape);
      argmax_param.outputs.emplace_back(std::move(tensor_list));
    }
    ArgMaxFunc argmax_func(argmax_param);
    argmax_func.invoke();
  } else if (func_name == "onnx_nms") {
    llvm_unreachable("cv18xx not support onnx_nms function");
  } else if (func_name == "topk") {
    llvm_unreachable("cv18xx not support topk function");
  } else if (func_name == "gathernd_tf") {
    OpParam dic_param = func_info.params;
    GatherNDParam param;
    param.batch_dims = dic_param.get<int>("batch_dims");
    int indice_dims = dic_param.get<int>("indice_dims");
    for (int i = 0; i < inputs.size(); ++i) {
      tensor_list_t input;
      input.ptr = inputs[i].data();
      input.size = func_info.inputs[i].count;
      if (i == 0) {
        copyShape(func_info.inputs[i].g_shape, input.shape);
      } else {
        copyShape(func_info.inputs[i].g_shape, input.shape, indice_dims);
      }
      param.inputs.push_back(input);
    }
    tensor_list_t output;
    output.ptr = outputs[0].data();
    output.size = func_info.outputs[0].count;
    copyShape(func_info.outputs[0].g_shape, output.shape);
    param.output = output;
    GatherndFunc func(param);
    func.invoke();
  } else if (func_name == "tensor_scatter") {
    llvm_unreachable("cv18xx not support tensor_scatter cpu function");
  } else if (func_name == "grid_sampler") {
    OpParam dic_param = func_info.params;
    GridSamplerParam param;
    param.mode = dic_param.get<int>("mode");
    param.padding_mode = dic_param.get<int>("padding_mode");
    param.align_corners = dic_param.get<bool>("align_corners");
    tensor_list_t input;
    tensor_list_t grid;
    input.ptr = inputs[0].data();
    input.size = func_info.inputs[0].count;
    copyShape(func_info.inputs[0].g_shape, input.shape);
    grid.ptr = inputs[1].data();
    grid.size = func_info.inputs[1].count;
    copyShape(func_info.inputs[1].g_shape, grid.shape);
    param.inputs.push_back(input);
    param.inputs.push_back(grid);

    tensor_list_t output;
    output.size = func_info.outputs[0].count;
    copyShape(func_info.outputs[0].g_shape, output.shape);
    output.ptr = outputs[0].data();
    param.output = output;

    GridSamplerFunc func(param);
    func.invoke();
  } else if (func_name == "deform_gather") {
    llvm_unreachable("cv18xx not support deform_gather cpu function");
  } else {
    const std::string error_info =
        func_name + "generic cpu func not supported!\n";
    llvm_unreachable(error_info.c_str());
  }
}
} // namespace cvi_debug
