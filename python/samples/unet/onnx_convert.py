import torch
import torch.onnx
import os
import onnx
from unet_model import UNet


def pth_to_onnx():
    inputs = torch.randn(1, 3, 200, 400)
    net = UNet(3, 2)
    net.load_state_dict(torch.load("../model/unet_carvana_scale0.5_epoch2.pth", map_location="cpu"))
    torch.onnx.export(
        net,
        inputs,
        f"../model/float_scale_0.5_200_400.onnx",
        verbose=True,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    pth_to_onnx()
