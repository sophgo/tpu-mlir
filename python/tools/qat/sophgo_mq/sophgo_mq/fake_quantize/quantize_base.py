import torch
from torch.quantization import FakeQuantizeBase
from torch.quantization.observer import MovingAverageMinMaxObserver
from torch.quantization.fake_quantize import _is_per_channel, _is_per_tensor

from sophgo_mq.utils import is_symmetric_quant

_version_under_1100 = (int(torch.__version__.split('.')[1]) < 10) and (int(torch.__version__.split('.')[0]) == 1)

class QuantizeBase(FakeQuantizeBase):
    r""" This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and support learning of the scale
    and zero point parameters through backpropagation. For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.
    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.
    """
    def __init__(self, observer=MovingAverageMinMaxObserver, **observer_kwargs):
        super().__init__()
        self.only_enable_observer = False
        self.run_fquant_time = 0
        if observer is None:
            return
        self.activation_post_process = observer(**observer_kwargs)
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        assert self.quant_min <= self.quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.pot_scale = self.activation_post_process.pot_scale
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert _is_per_channel(self.qscheme) or \
            _is_per_tensor(self.qscheme), \
            'Only per channel and per tensor quantization are supported in fake quantize' + \
            ' got qscheme: ' + str(self.qscheme)
        self.is_per_channel = _is_per_channel(self.qscheme)
        bitrange = torch.tensor(self.quant_max - self.quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        self.is_symmetric_quant = is_symmetric_quant(self.qscheme)
    
    def enable_only_observer(self, enable = True):
        self.only_enable_observer = enable
        self.run_fquant_time = 1

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, '.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis)
