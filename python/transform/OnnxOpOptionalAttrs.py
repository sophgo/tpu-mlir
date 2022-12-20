class OnnxOpOptionalAttrGetter(object):
  def __init__(self):
    self._optional_attrs = {
      "ArgMax": {
        "axis": 0,
        "keepdims": 1,
        "select_last_index": 0,
      },
      "ArgMin": {
        "axis": 0,
        "keepdims": 1,
        "select_last_index": 0,
      },
      "AveragePool": {
        "auto_pad": "NOTSET",
        "ceil_mode": 0,
        "count_include_pad": 0,
        # "pads": [0],
        # "strides": [1],
      },
      "BatchNormalization": {
        "epsilon": 1e-05,
        "momentum": 0.9,
        "training_mode": 0,
      },
      "Celu": {
        "alpha": 1.0,
      },
      # "CenterCropPad": {
      #   "axes": [],
      # },
      "ConcatFromSequence": {
        "new_axis": 0,
      },
      "Conv": {
        "auto_pad": "NOTSET",
        # "dilations": [1],
        "group": 1,
        # "pads": [0],
        # "strides": [1],
      },
      "ConvInteger": {
        "auto_pad": "NOTSET",
        # "dilations": [1],
        "group": 1,
        # "pads": [0],
        # "strides": [1],
      },
      "ConvTranspose": {
        "auto_pad": "NOTSET",
        # "dilations": [1],
        "group": 1,
        # "output_padding": [0],
        # "pads": [0],
        # "strides": [1],
      },
      "CumSum": {
        "exclusive": 0,
        "reverse": 0,
      },
      "DepthToSpace": {
        "mode": "DCR",
      },
      "DequantizeLinear": {
        "axis": 1,
      },
      "Elu": {
        "alpha": 1.0,
      },
      "EyeLike": {
        "k": 0,
      },
      "Flatten": {
        "axis": 1,
      },
      "GRU": {
        "direction": "forward",
        "layout": 0,
        "linear_before_reset": 0,
      },
      "Gather": {
        "axis": 0,
      },
      "GatherElements": {
        "axis": 0,
      },
      "GatherND": {
        "batch_dims": 0,
      },
      "Gemm": {
        "alpha": 1.0,
        "beta" : 1.0,
        "transA" : 0,
        "transB" : 0,
      },
      "GlobalLpPool": {
        "p": 2,
      },
      "GridSample": {
        "align_corners": 0,
        "mode": "bilinear",
        "padding_mode": "zeros",
      },
      "GroupNormalization": {
        "epsilon": 1e-05,
      },
      "HardSigmoid": {
        "alpha": 0.2,
        "beta": 0.5,
      },
      "Hardmax": {
        "axis": -1,
      },
      "InstanceNormalization": {
        "epsilon": 1e-05,
      },
      "IsInf": {
        "detect_negative": 1,
        "detect_positive": 1,
      },
      "LRN": {
        "alpha": 0.0001,
        "beta": 0.75,
        "bias": 1.0,
      },
      "LSTM": {
        "direction": "forward",
        "input_forget": 0,
        "layout": 0,
      },
      "LayerNormalization": {
        "axis": -1,
        "epsilon": 1e-05,
        "stash_type": -1,
      },
      "LeakyRelu": {
        "alpha": 0.01,
      },
      "LogSoftmax": {
        "axis": -1,
      },
      "LpNormalization": {
        "axis": -1,
        "p": 2,
      },
      "LpPool": {
        "auto_pad": "NOTSET",
        "ceil_mode": 0,
        "p": 2,
        # "dilations": [1],
        # "pads": [0],
        # "strides": [1],
      },
      "MaxPool": {
        "auto_pad": "NOTSET",
        "ceil_mode": 0,
        "storage_order": 0,
        # "dilations": [1],
        # "pads": [0],
        # "strides": [1],
      },
      "MaxRoiPool": {
        "spatial_scale": 1.0,
      },
      "MaxUnpool": {
        # "pads": [0],
        # "strides": [1],
      },
      "MeanVarianceNormalization": {
        "axes": [0, 2, 3],
      },
      "NonMaxSuppression": {
        "center_point_box": 0,
      },
      "Pad": {
        "mode": "constant",
      },
      "QLinearConv": {
        "auto_pad": "NOTSET",
        # "dilations": [1],
        "group": 1,
        # "pads": [0],
        # "strides": [1],
      },
      "QuantizeLinear": {
        "axis": 1,
      },
      "RNN": {
        "direction": "forward",
        "layout": 0,
      },
      "ReduceL1": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "ReduceL2": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "ReduceLogSum": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "ReduceLogSumExp": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "ReduceMax": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "ReduceMean": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "ReduceMin": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "ReduceProd": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "ReduceSum": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "ReduceSumSquare": {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
      },
      "Reshape": {
        "allowzero": 0,
      },
      "Resize": {
        "antialias": 0,
        # "axes": [],
        "coordinate_transformation_mode": "half_pixel",
        "cubic_coeff_a": -0.75,
        "exclude_outside": 0,
        "extrapolation_value": 0.0,
        "keep_aspect_ratio_policy": "stretch",
        "mode": "nearest",
        "nearest_mode": "round_prefer_floor",
      },
      "ReverseSequence": {
        "batch_axis": 1,
        "time_axis": 0,
      },
      "RoiAlign": {
        "coordinate_transformation_mode": "half_pixel",
        "mode": "avg",
        "output_height": 1,
        "output_width": 1,
        "sampling_ratio": 0,
        "spatial_scale": 1.0,
      },
      "ScatterElements": {
        "axis": 0,
        "reduction": "none",
      },
      "ScatterND": {
        "reduction": "none",
      },
      "Selu": {
        "alpha": 1.67326,
        "gamma": 1.0507,
      },
      "Shrink": {
        "bias": 0.0,
        "lambd": 0.5,
      },
      "Softmax": {
        "axis": -1,
      },
      "Split": {
        "axis": 0,
      },
      "SplitToSequence": {
        "axis": 0,
        "keepdims": 1,
      },
      "ThresholdedRelu": {
        "alpha": 1.0,
      },
      "Topk": {
        "axis": -1,
        "largest": 1,
        "sorted": 1,
      },
      "Trilu": {
        "upper": 1,
      },
      "Unique": {
        # "axis": [],
        "sorted": 1,
      },
    }

  def get(self, op_type: str) -> dict:
    return self._optional_attrs.get(op_type, {})
