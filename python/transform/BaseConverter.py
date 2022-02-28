import numpy as np
class BaseConverter(object):
    def __init__(self):
        self.operands = dict()
        self.tensors = dict()
        self.shapes = dict()

    def run(self):
        raise NotImplementedError('run')

    def addShape(self, name, shape):
        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            raise KeyError("{}:{} unknown shape".format(name, shape))
        if name in self.shapes:
            if self.shapes[name] != shape:
                raise KeyError("shape {} conflict".format(name))
        self.shapes[name] = shape

    def getShape(self, name):
        if name not in self.shapes:
            raise KeyError("shape {} not found".format(name))
        return self.shapes[name]

    def addOperand(self, name, op):
        if name in self.operands:
            if self.operands[name] != op:
                raise KeyError("operand {} conflict".format(name))
            return
        self.operands[name] = op

    def getOperand(self, name):
        if name not in self.operands:
            raise KeyError("operand {} not found".format(name))
        return self.operands[name]

    def addTensor(self, name, data):
        if name in self.tensors:
            raise KeyError("tensor {} conflict".format(name))
        if not isinstance(data, np.ndarray):
            raise KeyError("tensor data must be numpy array")
        self.tensors[name] = data
        self.addShape(name, data.shape)

    def isTensor(self, name):
        if name in self.tensors:
            return True
        return False

    def getTensor(self, name):
        if name not in self.tensors:
            raise KeyError("No {} tensor in model".format(name))
        return self.tensors[name]

    def getWeightOp(self, name):
        if name not in self.tensors:
            raise KeyError("Should addTensor first:{}!!!".format(name))
        op = self.mlir.create_weight_op(name, self.getShape(name))
        self.addOperand(name, op)
        return op

    def WeightToNpz(self, weight_file):
        tensor_npz = {}
        for name in self.tensors:
            if name in self.operands:
                tensor_npz[name] = self.tensors[name]
        np.savez(weight_file, **tensor_npz)
