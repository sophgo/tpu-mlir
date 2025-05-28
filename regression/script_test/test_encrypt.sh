#!/bin/bash
# test case: test torch input int32 or int16 situation
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p test_encrypt
pushd test_encrypt
#
python3 $DIR/test_encrypt.py

model_transform.py \
  --model_name test_encrypt_a \
  --model_def test_encrypt_a.onnx \
  --mlir test_encrypt_a.mlir

model_deploy.py \
  --mlir test_encrypt_a.mlir \
  --quantize W4F16 \
  --q_group_size 32 \
  --chip bm1684x \
  --model test_encrypt_a.bmodel

model_transform.py \
  --model_name test_encrypt_b \
  --model_def test_encrypt_b.onnx \
  --mlir test_encrypt_b.mlir

model_deploy.py \
  --mlir test_encrypt_b.mlir \
  --quantize W4F16 \
  --q_group_size 32 \
  --chip bm1684x \
  --model test_encrypt_b.bmodel

# combine
model_tool --combine test_encrypt_a.bmodel test_encrypt_b.bmodel -o test_encrypt_combine.bmodel

model_runner.py --input test_encrypt_input.npz --model test_encrypt_a.bmodel --output a_out.npz
model_runner.py --input test_encrypt_input.npz --model test_encrypt_b.bmodel --output b_out.npz

# encrypt
model_tool --encrypt -model test_encrypt_combine.bmodel -net test_encrypt_b -lib $DIR/libcipher.so -o encrypted.bmodel

# decrypt
model_tool --decrypt -model encrypted.bmodel -lib $DIR/libcipher.so -o decrypted.bmodel

# extract
model_tool --extract decrypted.bmodel

model_runner.py --input test_encrypt_input.npz --model bm_net0_stage0.bmodel --output a_dec_out.npz
model_runner.py --input test_encrypt_input.npz --model bm_net1_stage0.bmodel --output b_dec_out.npz

npz_tool.py compare a_out.npz a_dec_out.npz --tolerance 1.0,1.0
npz_tool.py compare b_out.npz b_dec_out.npz --tolerance 1.0,1.0

# encrypt a
# model_tool --encrypt -model test_encrypt_a.bmodel -net test_encrypt_a -lib $DIR/libcipher.so -o encrypted_a.bmodel
# model_runner.py --input test_encrypt_input.npz --model encrypted_a.bmodel --output a_encrypt_out.npz --decrypt_lib $DIR/libcipher.so
# npz_tool.py compare a_out.npz a_encrypt_out.npz --tolerance 1.0,1.0


rm -rf *.npz

popd
