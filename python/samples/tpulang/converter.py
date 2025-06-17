import os
import argparse
import qwen2

# Usage: python converter.py --case Qwen2_7B_Block --mode f16 --quant_bits 4 --seq_len 1272
#                             --file_path /workspace/Qwen2-VL-7B-Instruct-GPTQ-Int4/weights/qwen2vl_decoder_layer_0_quantized.pth
#                             --cos_sin_path /workspace/Qwen2-VL-7B-Instruct-GPTQ-Int4/weights/cos_sin_tensor.pth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, required=True, help='case name')
    parser.add_argument("--chip",
                        default="bm1684x",
                        type=str,
                        choices=['bm1684x', 'bm1688'],
                        help="chip platform name")
    parser.add_argument("--mode",
                        default="f16",
                        type=str,
                        choices=['f16', 'bf16'],
                        help="quantize modes")
    parser.add_argument("--path",
                        default="",
                        type=str,
                        help="the path to store intermediate file, accept "
                        " or absolute path.")
    parser.add_argument("--quant_bits", default=4, type=int, help="quantized weight bits")
    parser.add_argument("--seq_len", default=1272, type=int, help="input sequence length")
    parser.add_argument(
        "--file_path",
        default=
        "/workspace/Qwen2-VL-7B-Instruct-GPTQ-Int4/weights/qwen2vl_decoder_layer_0_quantized.pth",
        type=str,
        help="path for quantized weights")
    parser.add_argument(
        "--cos_sin_path",
        default="/workspace/Qwen2-VL-7B-Instruct-GPTQ-Int4/weights/cos_sin_tensor.pth",
        type=str,
        help="path for rotary_emb: output as {'cos': cos, 'sin': sin}")
    args = parser.parse_args()

    tester = qwen2.utils.TPULANG_CONVERTER(args.chip, args.mode, args.file_path, args.cos_sin_path,
                                           args.quant_bits, args.seq_len)

    if args.path:
        dir = args.path
    else:
        dir = "samples_tpulang_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)

    run_case = {
        "Qwen2_7B_Block": qwen2.block.test_qwen2_7b_block,
        "Qwen2_7B_Block_Cache": qwen2.block_cache.test_qwen2_7b_block_cache
    }

    func = run_case[args.case]
    func(tester)
    print("====== TEST {} Success ======".format(args.case))
