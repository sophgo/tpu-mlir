import transform.TpuLang as tpul
import os

files_path = os.getcwd()

tpul.bmodel_inference_combine(
    f"{files_path}/yolov5s_1684x_f16/compilation.bmodel",
    f"{files_path}/yolov5s_1684x_f16/final.mlir",
    f"{files_path}/yolov5s_in_f32.npz",
    f"{files_path}/yolov5s_1684x_f16/tensor_location.json",
    f"{files_path}/yolov5s_bm1684x_f16_tpu_outputs.npz",
    dump_file=True,
    save_path=os.path.join(f"{files_path}", "soc_infer"),
    out_fixed=False,
    dump_cmd_info=False,
    skip_check=False,
    run_by_op=False,
)
