python3 onnx_convert.py;

# python3 ../../../tpu-mlir/python/tools/model_transform.py \
model_transform.py \
--model_name  unet \
--model_def ../model/float_scale_0.5_200_400.onnx \
--input_shapes [[1,3,200,400]] \
--pixel_format rgb \
--keep_aspect_ratio \
--mean 0.0,0.0,0.0 \
--scale 0.0039216,0.0039216,0.0039216 \
--test_input ../test_hq/0cdf5b5d0ce1_01.jpg \
--test_result ./top_result_200_400.npz \
--mlir ../model/unet_0.5_200_400.mlir;

# python3 ../../../tpu-mlir/python/tools/model_deploy.py \
model_deploy.py \
--mlir ../model/unet_0.5_200_400.mlir \
--quantize INT8 \
--calibration_table ./unet_scale0.5_cali_table \
--chip bm1684x \
--tolerance 0.99,0.85 \
--model ../model/unet_scale0.5_int8.bmodel;

start=`date +%s`;
python3 mlir_tester.py --img_dir ../data/test_hq --out_dir ../result --height 200 --width 400 --model  unet_bm1684x_int8_sym_tpu.mlir;
end=`date +%s`;
runtime=$((end-start));
echo "Mlir Tester Time:${runtime} s"

mv ../model/unet_scale0.5_int8.bmodel.compiler_profile_0.txt profile.txt;
