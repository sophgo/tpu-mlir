class OnnxOpOptionalAttrGetter(object):
  def __init__(self):
    self._optional_attrs = {
      "HardSigmoid": {
        "alpha": 0.2,
        "beta": 0.5,
      },
      "LayerNormalization": {
        "axis": -1,
        "epsilon": 1e-05,
        "stash_type": -1,
      },
    }

  def get(self, op_type: str) -> dict:
    return self._optional_attrs.get(op_type, {})
