import transform.TpuLang as tpul
import os

local_tools_path = os.getenv("PROJECT_ROOT", None)
if not local_tools_path:
    local_tools_path = os.getenv("TPUC_ROOT")
    assert local_tools_path

tpul.bmodel_inference_combine(
    os.path.join(local_tools_path, "python/tools/soc_infer/demo/Yuv2rgb_0_int8/compilation.bmodel"),
    os.path.join(local_tools_path, "python/tools/soc_infer/demo/Yuv2rgb_0_int8/final.mlir"),
    os.path.join(local_tools_path,
                 "python/tools/soc_infer/demo/Yuv2rgb_0_int8/Yuv2rgb_0_in_f32.npz"),
    os.path.join(local_tools_path,
                 "python/tools/soc_infer/demo/Yuv2rgb_0_int8/tensor_location.json"),
    os.path.join(local_tools_path,
                 "python/tools/soc_infer/demo/Yuv2rgb_0_int8/Yuv2rgb_0_int8_tpu_out.npz"),
    dump_file=True,
    save_path=os.path.join(local_tools_path, "soc_infer"),
    out_fixed=False,
    is_soc=True,
    soc_tmp_path="/tmp/soc_tmp",
)
print()

cmp_command1 = f"npz_tool.py compare {os.path.join(local_tools_path, 'soc_infer/soc_infer_Yuv2rgb_0_int8_tpu_out.npz')} {os.path.join(local_tools_path, 'python/tools/soc_infer/demo/Yuv2rgb_0_int8/Yuv2rgb_0_int8_tpu_out.npz')}"
print(f"[Comparing soc_infer and tpu_infer.npz]: {cmp_command1}")
os.system(cmp_command1)
print()

cmp_command2 = f"npz_tool.py compare {os.path.join(local_tools_path, 'soc_infer/soc_infer_Yuv2rgb_0_int8_tpu_out.npz')} {os.path.join(local_tools_path, 'python/tools/soc_infer/demo/Yuv2rgb_0_int8/Yuv2rgb_0_int8_model_out.npz')}"
print(f"[Comparing soc_infer and bmodel_infer.npz]: {cmp_command2}")
os.system(cmp_command2)

os.system(f"rm {os.path.join(os.getcwd(), 'failed_bmodel_outputs.npz')}")
