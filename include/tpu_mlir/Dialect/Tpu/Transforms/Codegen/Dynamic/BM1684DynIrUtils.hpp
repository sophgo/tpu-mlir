#ifndef __STATIC_IR_UTILS_HPP__
#define __STATIC_IR_UTILS_HPP__

#ifdef IR_PACKING
#define IR_USE_TYPE_DATA(buf, type, data)                                      \
  do {                                                                         \
    *(type *)buf = (type)(data);                                               \
    buf = ((type *)buf) + 1;                                                   \
  } while (0)
#define IR_USE_MEM_DATA(buf, data, dlen)                                       \
  do {                                                                         \
    memcpy(buf, data, dlen);                                                   \
    buf = ((char *)buf) + dlen;                                                \
  } while (0)
#endif

#ifdef IR_UNPACKING
#define IR_USE_TYPE_DATA(buf, type, data)                                      \
  do {                                                                         \
    data = *(type *)(buf);                                                     \
    buf = ((type *)buf) + 1;                                                   \
  } while (0)
#define IR_USE_MEM_DATA(buf, data, dlen)                                       \
  do {                                                                         \
    memcpy(data, buf, dlen);                                                   \
    buf = ((char *)buf) + dlen;                                                \
  } while (0)
#endif

#ifdef IR_CALC_LENGTH
#define IR_DECLARE_LENGTH_VAR() u32 __ir_buf_len = 0;

#define IR_USE_TYPE_DATA(buf, type, data)                                      \
  do {                                                                         \
    (void)(buf);                                                               \
    (void)(data);                                                              \
    __ir_buf_len += sizeof(type);                                              \
  } while (0)

#define IR_USE_MEM_DATA(buf, data, dlen)                                       \
  do {                                                                         \
    (void)(buf);                                                               \
    (void)(data);                                                              \
    (void)(dlen);                                                              \
    __ir_buf_len += dlen;                                                      \
  } while (0)

#define IR_GET_LENGTH() __ir_buf_len

#endif

#define IR_USE_F32_DATA(buf, data) IR_USE_TYPE_DATA(buf, float, data)
#define IR_USE_U64_DATA(buf, data) IR_USE_TYPE_DATA(buf, uint64_t, data)
#define IR_USE_U32_DATA(buf, data) IR_USE_TYPE_DATA(buf, uint32_t, data)
#define IR_USE_U16_DATA(buf, data) IR_USE_TYPE_DATA(buf, unsigned short, data)

#define IR_USE_U16x2_DATA(buf, high16, low16)                                  \
  IR_USE_U32_DATA(buf, (((high16) << 16) | ((low16)&0xFFFF)))

#define IR_USE_STRUCT_DATA(buf, data) IR_USE_MEM_DATA(buf, &data, sizeof(data))

#define IR_USE_GLB_TENSOR(buf, tensor)                                         \
  do {                                                                         \
    if (tensor.tensor_type == IR_TENSOR_TYPE_NEURON) {                         \
      IR_USE_U32_DATA(buf, tensor.is_io_tensor);                               \
      IR_USE_U32_DATA(buf, tensor.tensor_id);                                  \
      if (!tensor.is_io_tensor) {                                              \
        IR_USE_U64_DATA(buf, tensor.global_mem_offset);                        \
      }                                                                        \
    } else if (tensor.tensor_type == IR_TENSOR_TYPE_COEFF) {                   \
      IR_USE_U64_DATA(buf, tensor.global_mem_offset);                          \
    } else if (tensor.tensor_type == IR_TENSOR_TYPE_SHAPE) {                   \
      IR_USE_U32_DATA(buf, tensor.tensor_id);                                  \
    } else if (tensor.tensor_type == IR_TENSOR_TYPE_ARRAY) {                   \
      IR_USE_U32_DATA(buf, tensor.tensor_id);                                  \
      IR_USE_U64_DATA(buf, tensor.global_mem_offset);                          \
    } else if (tensor.tensor_type == IR_TENSOR_TYPE_FLOW) {                    \
      IR_USE_U32_DATA(buf, tensor.tensor_id);                                  \
    } else {                                                                   \
      std::cout << "UNKNOWN IR_TENSOR_TYPE=" << tensor.tensor_type;            \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define IR_USE_LOC_TENSOR(buf, tensor)                                         \
  do {                                                                         \
    if (tensor.is_io_tensor) {                                                 \
      IR_USE_U32_DATA(buf, tensor.tensor_id);                                  \
    }                                                                          \
    IR_USE_U32_DATA(buf, tensor.local_mem_offset);                             \
    if (tensor.is_io_tensor == 2) {                                            \
      IR_USE_U32_DATA(buf, tensor.consumer_number);                            \
    }                                                                          \
  } while (0)

#endif

#define IR_PARAM_COMMON(name)                                                  \
  ir_layer_info_t *layer_info = (ir_layer_info_t *)ir_layer_info;              \
  dynamic_common_ir_layer_info(layer_info, getInput(), getOutput());           \
  assign_fw_param(                                                             \
      (void *)&layer_info->fw_layer_param_u.fw_##name##_layer_param);          \
  fw_ir_length += sizeof(fw_##name##_layer_param_t);

#define GLOBAL_IR_COMMON(name)                                                 \
  uint32_t fw_ir_length = 0;                                                   \
  IR_PARAM_COMMON(name)                                                        \
  return fw_ir_length;

#define IR_PARAM_BROADCAST_BINARY(op_code)                                     \
  fw_broadcast_binary_layer_param_t *fw_broadcast_binary_layer_param =         \
      (fw_broadcast_binary_layer_param_t *)param;                              \
  fw_broadcast_binary_layer_param->binary_op = op_code;                        \
  fw_broadcast_binary_layer_param->a_is_coeff =                                \
      module::isWeight(getInputs()[0]);                                        \
  fw_broadcast_binary_layer_param->b_is_coeff =                                \
      module::isWeight(getInputs()[1]);                                        \
  fw_broadcast_binary_layer_param->if_relu = getDoRelu() ? 1 : 0;              \
  fw_broadcast_binary_layer_param->relu_upper_limit =                          \
      getReluLimit().convertToDouble();                                        \
  fw_broadcast_binary_layer_param->buffer_addr = 0;                            \
  auto input_num = getInputs().size();                                         \
  assert(input_num == 2);                                                      \
  auto muls = module::getI32Array(getMultipliers(), input_num, 1);             \
  auto rs = module::getI32Array(getRshifts(), input_num, 0);                   \
  for (int i = 0; i < input_num; ++i) {                                        \
    fw_broadcast_binary_layer_param->scale[i] = (uint8_t)muls->at(i);          \
    fw_broadcast_binary_layer_param->rshift_num[i] = (uint8_t)rs->at(i);       \
    auto dtype = BM1684::getDataType(getInputs()[i]);                          \
    if (dtype == DTYPE_INT8)                                                   \
      fw_broadcast_binary_layer_param->opd_sign[i] = 1;                        \
    else if (dtype == DTYPE_UINT8)                                             \
      fw_broadcast_binary_layer_param->opd_sign[i] = 0;                        \
    else if (dtype == DTYPE_INT16)                                             \
      fw_broadcast_binary_layer_param->opd_sign[i] = 2;                        \
    else if (dtype == DTYPE_UINT16)                                            \
      fw_broadcast_binary_layer_param->opd_sign[i] = 3;                        \
  }                                                                            \
  fw_broadcast_binary_layer_param->opd_sign[2] = module::isSign(getOutput());  \
  int a_shape[MAX_SHAPE_DIMS], b_shape[MAX_SHAPE_DIMS];                        \
  module::getGlobalShape(getInputs()[0], a_shape);                             \
  module::getGlobalShape(getInputs()[1], b_shape);                             \
  fw_broadcast_binary_layer_param->a_dims =                                    \
      module::getShape(getInputs()[0]).size();                                 \
  memcpy(&(fw_broadcast_binary_layer_param->a_shape[0]), &a_shape[0],          \
         MAX_SHAPE_DIMS * sizeof(int));                                        \
  fw_broadcast_binary_layer_param->b_dims =                                    \
      module::getShape(getInputs()[1]).size();                                 \
  memcpy(&(fw_broadcast_binary_layer_param->b_shape[0]), &b_shape[0],          \
         MAX_SHAPE_DIMS * sizeof(int));

#define IR_PARAM_CONST_BINARY(op_code)                                         \
  fw_const_binary_layer_param_t fw_const_binary_layer_param = {0};             \
  fw_const_binary_layer_param.binary_op = op_code;                             \
  fw_const_binary_layer_param.b_value = getConstVal().convertToDouble();       \
  fw_const_binary_layer_param.inversed = 0;                                    \
  int out_sign = module::isSign(getOutput());                                  \
  auto data_size = get_dynamic_compiler_tensor_datasize(getInput());           \
  if (getDoRelu() || (DSIZE_8 == data_size && !out_sign)) {                    \
    fw_const_binary_layer_param.if_relu = 1;                                   \
  } else {                                                                     \
    fw_const_binary_layer_param.if_relu = 0;                                   \
  }                                                                            \
  fw_const_binary_layer_param.relu_upper_limit =                               \
      getReluLimit().convertToDouble();                                        \
  for (int idx = 0; idx < 2; ++idx) {                                          \
    fw_const_binary_layer_param.scale[idx] = getMultiplier();                  \
    fw_const_binary_layer_param.rshift_num[idx] = getRshift();                 \
  }                                                                            \
  fw_const_binary_layer_param.opd_sign[0] = module::isSign(getInput());        \
  memcpy(param, &fw_const_binary_layer_param,                                  \
         sizeof(fw_const_binary_layer_param_t));
