from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper

get_size = symbolic_helper._get_tensor_sizes
def _fake_quantize_learnable_per_tensor_affine(g, x, scale, zero_point, quant_min, quant_max, grad_factor):
    output = g.op("Sophgo_custom::LearnablePerTensorAffine", x, scale, zero_point, quant_min, quant_max)
    input_shape = symbolic_helper._get_tensor_sizes(x)

    if input_shape is not None and hasattr(x.type(), 'with_sizes'):
        output_type = x.type().with_sizes(input_shape)
        output.setType(output_type)

    return output

def fake_quantize_per_channel_affine(g, x, scale, zero_point, ch_axis, quant_min, quant_max):
    output = g.op("Sophgo_custom::FixedPerChannelAffine", x, scale, zero_point, ch_axis, quant_min, quant_max)
    input_shape = symbolic_helper._get_tensor_sizes(x)

    if input_shape is not None and hasattr(x.type(), 'with_sizes'):
        output_type = x.type().with_sizes(input_shape)
        output.setType(output_type)

    return output

def fake_quantize_per_tensor_affine(g, x, scale, zero_point, quant_min, quant_max):
    output = g.op("Sophgo_custom::FixedPerTensorAffine", x, scale, zero_point, quant_min, quant_max)
    input_shape = symbolic_helper._get_tensor_sizes(x)

    if input_shape is not None and hasattr(x.type(), 'with_sizes'):
        output_type = x.type().with_sizes(input_shape)
        output.setType(output_type)

    return output

for i in range(1, 16):
    register_custom_op_symbolic('::_fake_quantize_learnable_per_tensor_affine', _fake_quantize_learnable_per_tensor_affine, i)
    register_custom_op_symbolic('::fake_quantize_per_channel_affine', fake_quantize_per_channel_affine, i)
    register_custom_op_symbolic('::fake_quantize_per_tensor_affine', fake_quantize_per_tensor_affine, i)