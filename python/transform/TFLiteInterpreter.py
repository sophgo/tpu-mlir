import warnings


try:
    import tensorflow.lite as TFLiteItp
except ModuleNotFoundError:
    from tflite_runtime import interpreter as TFLiteItp


class TFLiteInterpreter(TFLiteItp.Interpreter):
    def __init__(self, model_path=None):
        super().__init__(model_path=model_path)
        self.allocate_tensors()
        self.name2index = {
            t["name"]: t["index"] for t in self.get_tensor_details() if t["name"]
        }

    @property
    def inputs(self):
        return self.get_input_details()

    @property
    def outputs(self):
        return self.get_output_details()

    def run(self, **inputs):

        for k, v in inputs.items():
            index = self.name2index[k]
            if self.tensor(index)().shape != v.shape:
                self.resize_tensor_input(index, v.shape, strict=False)
            self.tensor(index)()[...] = v[...]

        self.invoke()
        output = {}
        for out in self.get_output_details():
            name = out["name"]
            if name in output:
                warnings.warn("Duplicate tensor name '{}'.".format(name))
                continue
            output[name] = self.get_tensor(out["index"])
        return output

    def get_all_tensors(self):
        for t in self.get_tensor_details():
            try:
                yield (t, self.tensor(t["index"])())
            except ValueError:
                warnings.warn(
                    "Can not get tensor '{name}'(index: {index}) value.".format(
                        name=t["name"], index=t["index"]
                    )
                )
                yield (t, None)
