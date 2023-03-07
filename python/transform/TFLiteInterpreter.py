# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import warnings

try:
    import tensorflow.lite as TFLiteItp
except ModuleNotFoundError:
    from tflite_runtime import interpreter as TFLiteItp


class TFLiteInterpreter(TFLiteItp.Interpreter):

    def __init__(self, model_path=None, **args):
        __args = {
            "experimental_op_resolver_type": TFLiteItp.experimental.OpResolverType.BUILTIN_REF,
        }
        __args.update(args)
        super().__init__(
            model_path=model_path,
            **__args,
        )
        self.allocate_tensors()
        self.name2index = {t["name"]: t["index"] for t in self.get_tensor_details() if t["name"]}

    @property
    def inputs(self):
        return self.get_input_details()

    @property
    def outputs(self):
        return self.get_output_details()

    def reshape(self, **inputs_shape):
        need_reallocate = False
        for name, shape in inputs_shape.items():
            index = self.name2index[name]
            if self.tensor(index)().shape != shape:
                need_reallocate = True
                self.resize_tensor_input(index, shape, strict=False)
                # if the shape change, we need to reallocate the buffer
        if need_reallocate:
            self.allocate_tensors()

    def run(self, input_is_nchw=False, **inputs):

        def nchw2nhwc(x):
            if x.ndim == 4:
                return x.transpose([0, 2, 3, 1])
            return x

        input_data = inputs
        if input_is_nchw:
            input_data = {k: nchw2nhwc(v) for k, v in inputs.items()}

        self.reshape(**{k: v.shape for k, v in input_data.items()})
        for k, v in input_data.items():
            index = self.name2index[k]
            self.set_tensor(index, v)

        self.invoke()
        outs = []
        for out in self.get_output_details():
            outs.append((out, self.get_tensor(out["index"])))
        return outs

    def get_all_tensors(self):
        for t in self.get_tensor_details():
            try:
                yield (t, self.tensor(t["index"])())
            except ValueError:
                warnings.warn("Can not get tensor '{name}'(index: {index}) value.".format(
                    name=t["name"], index=t["index"]))
                yield (t, None)

    def to_expressed_dat(self, tensor_with_desc):
        import numpy as np

        desc, v = tensor_with_desc
        quantization_param = desc["quantization_parameters"]
        scales = quantization_param["scales"]
        zeros = quantization_param["zero_points"]
        if not np.issubdtype(v.dtype.type, np.integer):
            return v
        try:
            if len(zeros) == 1:
                return (v.astype(np.int32) - zeros[0]) * scales[0]
            if len(zeros) == 0:
                return v
            if len(zeros) > 1:
                shape = np.ones(v.ndim, dtype=np.int64)
                shape[quantization_param["quantized_dimension"]] = len(zeros)
                zeros = np.reshape(zeros, shape)
                scales = np.reshape(scales, shape)
                return (v.astype(np.int32) - zeros) * scales
        except:
            print(desc)
            raise ValueError()
