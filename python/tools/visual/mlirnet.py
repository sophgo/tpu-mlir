from calibration.data_selector import DataSelector
from utils.preprocess import get_preprocess_parser, preprocess
from utils.mlir_parser import MlirParser
from threading import Thread
import pandas as pd
import numpy as np
import pymlir
pymlir.set_mem_mode("value_mem")
import sys
import copy
sys.path.append('../..')


def find_mlir_net(folder):
    import os

    # https://docs.python.org/3/library/itertools.html#itertools.zip_longest
    def zip_longest(*args, fillvalue=None):
        from itertools import repeat
        iterators = [iter(it) for it in args]
        if fillvalue is None:
            fillvalue = [it[-1] for it in args]
        num_active = len(iterators)
        if not num_active:
            return
        while True:
            values = []
            for i, it in enumerate(iterators):
                try:
                    value = next(it)
                except StopIteration:
                    num_active -= 1
                    if not num_active:
                        return
                    iterators[i] = repeat(fillvalue[i])
                    value = fillvalue[i]
                values.append(value)
            yield tuple(values)

    def gen_path(fp, f_list):
        fp_len = len(fp)
        find = [x[-fp_len:] for x in f_list]
        if find.count(fp) == 1:
            return os.path.join(folder, f_list[find.index(fp)])
        else:
            return None

    file_fp = (
        ('f32_tpu.mlir', ),
        ('int8_sym_tpu.mlir', ),
        ('in_f32.npz',)
    )

    f_list = os.listdir(folder)
    for args_ in zip_longest(*file_fp):
        out_path = [gen_path(fp, f_list) for fp in args_]
        id_t = [x is None for x in out_path]
        if not any(id_t):
            return out_path
    index = id_t.index(True)
    raise Exception('folder does not contain one and only one {}.'.format(
        file_fp[index]))


class mlir_net:
    def __init__(self, mlir_file, input):
        self.mlir_file = mlir_file
        self.mlir_parser = MlirParser(self.mlir_file)
        self.mlir_module = None
        self.input = input
        self.inputs = {}
        self.tensor_att = {
            '_dtype': lambda x: self.tensor_qtype(x),
            '_shape': lambda x: self.tensor_shape(x)
        }
        self.preprocess = []

    def forward(self, start=None, end=None):
        self.mlir_module = pymlir.module()
        self.mlir_module.load(self.mlir_file)
        # simple input param handle, if npz, contain all inputs, if image, ends with same ext
        if self.input.lower().endswith(".npz"):
            self.inputs = np.load(self.input)
            for i in self.mlir_module.input_names:
                if not i in self.inputs.files:
                    print("{}th input in npz not match mlir file {}".format(
                        i, self.mlir_file))
            for name in self.mlir_module.input_names:
                if self.inputs[name].dtype == np.int8 or self.inputs[name].dtype == np.uint8:
                    self.mlir_module.set_tensor_from_int(
                        name, self.inputs[name].astype(np.float32))
                else:
                    self.mlir_module.set_tensor(
                        name, self.inputs[name].astype(np.float32))
        else:
            from utils.preprocess import preprocess
            for i in np.arange(self.mlir_parser.get_input_num()):
                pre_ = preprocess()
                pre_.load_config(self.mlir_parser.get_input_op_by_idx(i))
                self.preprocess.append(pre_)
            inputs_ = self.input.split(",")
            if len(inputs_) != self.mlir_parser.get_input_num() or len(inputs_) != len(self.preprocess):
                print("input number not match net input number!")
                sys.exit(1)
            for i in np.arange(self.mlir_parser.get_input_num()):
                i_name = self.mlir_module.input_names[i]
                self.inputs[i_name] = self.preprocess[i].run(inputs_[i])
                if self.inputs[i_name].dtype == np.int8 or self.inputs[i_name].dtype == np.uint8:
                    self.mlir_module.set_tensor_from_int(
                        i_name, self.inputs[i_name].astype(np.float32))
                else:
                    self.mlir_module.set_tensor(
                        i_name, self.inputs[self.mlir_module.input_names[i]].astype(np.float32))

        self.mlir_module.invoke()

    def layer_by_idx(self, idx):
        return self.mlir_parser.ops[idx]

    def tensor(self, name):
        return self.mlir_module.get_tensor(name)

    def tensor_shape(self, name):
        q_info = self.mlir_module.get_tensor_qinfo(name)
        return q_info.shape

    def tensor_qtype(self, name):
        q_info = self.mlir_module.get_tensor_qinfo(name)
        return q_info.dtype

    def tensor_scale(self, name):
        q_info = self.mlir_module.get_tensor_qinfo(name)
        return q_info.scale

    def output_names(self):
        return self.mlir_module.output_names

    def all_tensor_names(self):
        return self.mlir_module.all_tensor_names

    def all_weight_names(self):
        return self.mlir_module.all_weight_names

class analysis_data():
    def __init__(self, path, param_f32mlir, param_qmlir, param_input):
        if self.check_valid(param_f32mlir, param_qmlir, param_input):
            f32_mlir = param_f32mlir
            quant_mlir = param_qmlir
            mlir_input = param_input
        else:
            f32_mlir, quant_mlir, mlir_input = find_mlir_net(path)

        print("---------------------------------------------")
        print("f32 mlir is  :{}".format(f32_mlir, quant_mlir))
        print("quant mlir is:{}".format(quant_mlir))
        print("input is     :{}".format(mlir_input))
        print("---------------------------------------------")

        self.f32_mlir = f32_mlir
        self.quant_mlir = quant_mlir
        self.f32_net = mlir_net(f32_mlir, mlir_input)
        self.quant_net = mlir_net(quant_mlir, mlir_input)
        self.layer_dict = self.quant_net.mlir_parser.get_op_name_list()
        self.layer_names = list(self.layer_dict)
        self.metric_state = 0
        self.forwarding = False
        self.progress = 0

        self.work = []
        self.forward_count = 0
        layer_num = len(self.layer_names)
        self.tensor_info = None

    def check_valid(self, f32, quant, input):
        import os

        def nval1(x): return True if x == "" or not os.path.isfile(
            x) else False
        if nval1(f32) or nval1(quant):
            return False

        def nval2(x): return True if x.lower().split(
            ".")[-1] not in ['jpg', 'jpeg', 'png', 'npz'] else False
        if input == "" or nval2(input):
            return False
        if input.lower().endswith(".npz"):
            if not os.path.isfile(input):
                return False
        elif not nval2(input):
            inputs_ = input.split(",")
            for i in np.arange(len(inputs_)):
                if not os.path.isfile(inputs_[i]):
                    return False
        return True

    def build_blob_info(self, graph):
        # using graph edge as index
        blob_info = []
        index = []
        for s, t in graph.graph.edges:
            layer = graph.node(s)
            blob_info.append({
                'tensor': graph.edge((s, t)),
                'type': layer.type,
                'layer': layer.name,
            })
            index.append(str((s, t)))

        self.tensor_base = pd.DataFrame(blob_info, index=index)
        self.tensor_info = pd.DataFrame()

    def clean_unused(self):
        for t in self.tensor_base['tensor']:
            if t not in self.quant_net.all_tensor_names():
                row = self.tensor_base[self.tensor_base.tensor ==
                                       t].index.tolist()
                self.tensor_base = self.tensor_base.drop(row)

    def forward(self):
        self.forwarding = True
        self.progress = 0
        self.f32_net.forward()
        self.progress = 50
        self.quant_net.forward()
        self.progress = 100
        self.forwarding = False
        self.clean_unused()

    def tensor(self, name):
        if name in self.quant_net.all_tensor_names():
            quant = self.quant_net.tensor(name)
        else:
            return None, None
        if name in self.f32_net.all_tensor_names():
            f32 = self.f32_net.tensor(name)
        else:
            f32 = quant
        return f32, quant

    def weight(self, name, name1):
        if name in self.f32_net.all_weight_names():
            weight = self.f32_net.tensor(name).copy()
        elif name in self.quant_net.all_weight_names():
            weight = self.quant_net.tensor(name).copy()
        else:
            return None, None
        if not name1 is None:
            if name1 in self.f32_net.all_weight_names():
                bias = self.f32_net.tensor(name1).copy()
            elif name1 in self.quant_net.all_weight_names():
                bias = self.quant_net.tensor(name1).copy()
        else:
            bias = None
        if bias is not None:
            return weight, weight.shape, bias, bias.shape
        else:
            return weight, weight.shape, bias, None

    def run_metrics(self):
        m_work = Thread(target=self.__metrics)
        m_work.start()
        self.work.append(m_work)

    def __metrics(self):
        if len(self.tensor_info) > 0 and (
                self.forward_count in self.tensor_info.forward_index.values):
            return
        from .metrics import metrics
        metrics_ = []

        def fun(v, a, b):
            if a is None or b is None:
                return 0.0
            elif a.size == b.size:
                return v(a, b)
            return None

        all_m = len(self.tensor_base['tensor'])
        for index, name in enumerate(self.tensor_base['tensor']):
            new_items = {'forward_index': self.forward_count}
            f32_tensor = None
            quant_tensor = None
            f32_qinfo = None
            quant_qinfo = None
            if name in self.quant_net.all_tensor_names():
                quant_tensor = self.quant_net.tensor(name)
            else:
                continue
            if name in self.f32_net.all_tensor_names():
                f32_tensor = self.f32_net.tensor(name)
            else:
                f32_tensor = quant_tensor

            new_items.update({
                'fp32net' + k: f(name)
                for k, f in self.f32_net.tensor_att.items()
            })
            new_items.update({
                'int8net' + k: f(name)
                for k, f in self.quant_net.tensor_att.items()
            })

            #fp32_t, int8_t = self.tensor(name)
            new_items.update({
                k.upper(): fun(v, f32_tensor, quant_tensor)
                for k, v in metrics.items()
            })
            metrics_.append(new_items)
            self.metric_state = (index + 1) / all_m
        df = pd.DataFrame(metrics_, index=self.tensor_base.index)
        df = self.tensor_base.join(df)
        self.tensor_info = pd.concat([self.tensor_info, df])
