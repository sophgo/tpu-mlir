from .linear_fused import LinearBn1d
from .deconv_fused import ConvTransposeBnReLU2d, ConvTransposeBn2d, ConvTransposeReLU2d
from .conv_fused import ConvBnReLU2d, ConvBn2d, ConvReLU2d
from .freezebn import ConvFreezebn2d, ConvFreezebnReLU2d, ConvTransposeFreezebn2d, ConvTransposeFreezebnReLU2d

from .conv_fused_sophgo_tpu import ConvBnReLU2d_sophgo, ConvBn2d_sophgo, ConvReLU2d_sophgo
from .linear_fused_sophgo_tpu import LinearBn1d_sophgo, LinearReLU_sophgo, Linear_sophgo
from .deconv_fused_sophgo_tpu import ConvTransposeBnReLU2d_sophgo, ConvTransposeBn2d_sophgo, ConvTransposeReLU2d_sophgo
