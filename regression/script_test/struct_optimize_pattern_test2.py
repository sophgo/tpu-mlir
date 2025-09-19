#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import onnx


def build_and_save(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "struct_optimize_pattern_test2.onnx")

    def groupwise_correlation(fea1, fea2, num_groups):
        B, C, H, W = fea1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
        assert cost.shape == (B, num_groups, H, W)
        return cost

    def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
        for i in range(maxdisp):
            if i > 0:
                volume[:, :, i, :,
                       i:] = groupwise_correlation(refimg_fea[:, :, :, i:],
                                                   targetimg_fea[:, :, :, :-i], num_groups)
            else:
                volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
        volume = volume.contiguous()
        return volume

    class Model(nn.Module):

        def __init__(self, maxdisp, num_groups):
            super(Model, self).__init__()
            self.maxdisp = maxdisp
            self.num_groups = num_groups

        def forward(self, refimg_fea, targetimg_fea):
            return build_gwc_volume(refimg_fea, targetimg_fea, self.maxdisp, self.num_groups)

    # Use realistic stereo vision parameters
    left_fea = torch.randn(1, 96, 104, 160).float()
    right_fea = torch.randn(1, 96, 104, 160).float()
    maxdisp = 48
    num_groups = 8

    # Create model and export to ONNX
    model = Model(maxdisp, num_groups)
    model.eval()

    # Export to ONNX
    torch.onnx.export(model, (left_fea, right_fea),
                      onnx_path,
                      export_params=True,
                      opset_version=14,
                      do_constant_folding=True,
                      input_names=["in_0", "in_1"],
                      output_names=["output"],
                      dynamic_axes=None)

    print(f"[OK] Exported ONNX: {onnx_path}")
    return onnx_path


def main():
    out_dir = os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
    model_path = build_and_save(out_dir)

    # Save input for regression test
    input_npz = os.path.join(out_dir, "struct_optimize_pattern_test2_input.npz")

    # Use same seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)

    left_fea = torch.randn(1, 96, 104, 160).float()
    right_fea = torch.randn(1, 96, 104, 160).float()

    input_data = {"in_0": left_fea.numpy(), "in_1": right_fea.numpy()}

    np.savez(input_npz, **input_data)
    print(f"[OK] Saved input npz: {input_npz}")


if __name__ == "__main__":
    main()
