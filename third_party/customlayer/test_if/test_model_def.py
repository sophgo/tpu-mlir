import os
import transform.TpuLang as tpul
import numpy as np
import my_tpulang_layer

def rand_data(shape, dtype):
    if dtype == 'float32':
        return np.random.random(shape).astype(np.float32)
    if dtype == 'int32' or 'uint32' or 'int16' or 'uint16' or 'int8' or 'uint8':
        return np.random.randint(0, 256, size=shape).astype(dtype)
    raise Exception("Not supported data type: {}!".format(dtype))

def coeff_tensor(shape, dtype, scale=1.0):
    data = rand_data(shape, dtype)
    data = data * scale if dtype == 'float32' else data
    return tpul.Tensor(dtype=dtype, shape=shape, data=data)

def conv_op(x,
            kshape,
            stride,
            pad=None,
            group=1,
            dilation=[1, 1],
            zp=[None, None],
            dtype="float32"):
    oc = kshape[0]
    weight = coeff_tensor(kshape, dtype)
    out_dtype = dtype if dtype == 'float32' else 'int32'
    bias = coeff_tensor(oc, out_dtype)
    conv = tpul.conv_v2(x,
                        weight,
                        bias=bias,
                        stride=stride,
                        pad=pad,
                        dilation=dilation,
                        group=group,
                        input_zp=zp[0],
                        weight_zp=zp[1],
                        out_dtype=out_dtype)
    return conv

def abs_add_op(x, b, dtype="float32"):
    return my_tpulang_layer.absAdd.tpulang([x], b, dtype)[0]

def model_def(x, flag: int):
    if flag & 0b1:
        custom = abs_add_op(x, 1.2)
    conv1 = conv_op(custom if flag & 0b1 else x, [4, 32, 3, 3], [1, 1], [1, 1, 1, 1])
    relu1 = tpul.relu(conv1)
    conv2 = conv_op(relu1, [4, 4, 3, 3], [2, 2], [2, 2, 2, 2])
    if flag & 0b10:
        conv2 = abs_add_op(conv2, -1.5)
    relu2 = tpul.relu(conv2)
    conv3 = conv_op(relu2, [4, 4, 3, 3], [1, 1], [1, 1, 1, 1])
    relu3 = tpul.relu(conv3)
    if flag & 0b100:
        relu3 = abs_add_op(relu3, 2.1)
    return relu3

def gen_name(flag: int):
    return "front" if flag == 0b1 else "middle" if flag == 0b10 else "back" if flag == 0b100 else "mix"

def compile_model(flag: int):
    shape = [4, 32, 64, 48]
    x_data = np.random.random(shape).astype(np.float32)
    x = tpul.Tensor(dtype='float32', shape=shape, data=x_data)
    y = model_def(x, flag)
    tpul.compile("model_def_{}".format(gen_name(flag)), [x], [y], has_custom=True)
    deploy_cmd = "model_deploy.py --mlir model_def_{}.mlir --model model_{}.bmodel " \
                 "--quantize f32 --chip BM1684X".format(gen_name(flag), gen_name(flag))
    assert(os.system(deploy_cmd) == 0)

if __name__ == '__main__':
    os.makedirs("tmp", exist_ok=True)
    os.chdir("tmp")
    for flag in (0b1, 0b10, 0b100, 0b111):
        tpul.init("BM1684X")
        print(">>> flag = {}\n".format(flag))
        compile_model(flag)
        tpul.deinit()
