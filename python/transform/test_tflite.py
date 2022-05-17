from .TFLiteConverter import TFLiteReader
from .TFLiteConverter import TFLiteConverter
import numpy as np

file = "../../regression/resnet50_quant_int8.tflite"


class TestTFLiteReader:
    def test_reader(self):

        tfl = TFLiteReader(file)
        with open("test_resnet50_reader_out.ss", "w") as f:
            f.write(str(tfl))


class TestTFLiteConverter:
    def test_init_no_shape(self):
        tfc = TFLiteConverter("resnt50", file)

    def test_init(self):
        tfc = TFLiteConverter("resnt50", file, [(1, 3, 224, 224)])

    def test_add_constant(self):

        tfc = TFLiteConverter("resnt50", file, [(1, 3, 224, 224)])
        tensor = tfc.graph.inputs[0]
        tfc._TFLiteConverter__create_weight_op(tensor)
        assert len(tfc.constant) == 1

    def test_add_oprations(self):

        tfc = TFLiteConverter("resnt50", file, [(1, 3, 224, 224)])
        tfc.generate_mlir("test_resnet50_out.mlir")


from .TFLiteInterpreter import TFLiteInterpreter


class TestTFLiteInterpreter:
    def test_resnet50(self):

        interpreter = TFLiteInterpreter(file)
        inputs = {
            t["name"]: np.random.random(t["shape"]).astype(t["dtype"])
            for t in interpreter.inputs
        }
        out = interpreter.run(**inputs)
        all_tensor_len = len(interpreter.get_tensor_details())
        assert all_tensor_len == len([t for t in interpreter.get_all_tensors()])

    def test_resnet50_reshape(self):

        interpreter = TFLiteInterpreter(file)
        inputs = {t["name"]: (4, 224, 224, 3) for t in interpreter.inputs}
        interpreter.reshape(**inputs)
        all_tensor_len = len(interpreter.get_tensor_details())
        output_index = interpreter.outputs[0]["index"]
        output_shape = interpreter.tensor(output_index)().shape
        assert output_shape == (4, 1000)

    def test_resnet50_reshape_with_forward(self):

        interpreter = TFLiteInterpreter(file)
        inputs = {
            t["name"]: np.random.random([4, 224, 224, 3]).astype(t["dtype"])
            for t in interpreter.inputs
        }
        out = interpreter.run(**inputs)
        all_tensor_len = len(interpreter.get_tensor_details())
        assert all_tensor_len == len([t for t in interpreter.get_all_tensors()])

    def test_resnet50_data(self):

        interpreter = TFLiteInterpreter(file, experimental_preserve_all_tensors=True)
        inputs = {
            t["name"]: np.zeros(t["shape"]).astype(t["dtype"])
            for t in interpreter.inputs
        }
        out = interpreter.run(**inputs)
        for desc, v in interpreter.get_all_tensors():
            if desc["name"] == "input_1_int8":
                break
        assert (v == -13).all()
