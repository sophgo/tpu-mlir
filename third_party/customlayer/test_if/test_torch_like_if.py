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
    conv = tpul.conv(x,
                    weight,
                    bias=bias,
                    stride=stride,
                    pad=pad,
                    dilation=dilation,
                    group=group,
                    out_dtype=out_dtype)
    return conv

def abs_add_op(x, b, dtype="float32"):
    return my_tpulang_layer.absAdd.tpulang([x], b, dtype)[0]

class Model:
  def __init__(self, flag:int, chip:str, dynamic:bool):
    self.flag = flag
    self.chip = chip
    self.dynamic = dynamic

  def forward(self, x):
    if self.flag & 0b1:
        custom = abs_add_op(x, 1.2)
    conv1 = conv_op(custom if self.flag & 0b1 else x, [4, 32, 3, 3], [1, 1], [1, 1, 1, 1])
    relu1 = tpul.relu(conv1)
    conv2 = conv_op(relu1, [4, 4, 3, 3], [2, 2], [2, 2, 2, 2])
    if self.flag & 0b10:
        conv2 = abs_add_op(conv2, -1.5)
    relu2 = tpul.relu(conv2)
    conv3 = conv_op(relu2, [4, 4, 3, 3], [1, 1], [1, 1, 1, 1])
    relu3 = tpul.relu(conv3)
    if self.flag & 0b100:
        relu3 = abs_add_op(relu3, 2.1)
    postfix = "dyn" if self.dynamic else "static"
    tpul.compile('torch_like_model_{}_{}'.format(gen_name(self.flag), postfix), [x], [relu3], dynamic=self.dynamic)

def gen_name(flag: int):
    return "front" if flag == 0b1 else "middle" if flag == 0b10 else "back" if flag == 0b100 else "mix"

def model_main(shape, flag:int, chip:str, dynamic:bool):
    tpul.init(chip)
    x_data = np.random.random(shape).astype(np.float32)
    x = tpul.Tensor(dtype='float32', shape=shape, data=x_data)
    model = Model(flag, chip, dynamic)
    model.forward(x)
    tpul.deinit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684x', 'bm1688'],
                        help="chip platform name")
    parser.add_argument("--dynamic", action="store_true", help='do dynamic compile')
    args = parser.parse_args()
    os.makedirs("tmp", exist_ok=True)
    os.chdir("tmp")
    for flag in (0b1, 0b10, 0b100, 0b111):
        print("--- torch_like_model_{} ---".format(gen_name(flag)))
        model_main([4,32,36,36], flag, args.chip, args.dynamic)
    print('Test succeed.')
