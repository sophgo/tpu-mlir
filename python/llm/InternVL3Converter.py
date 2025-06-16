# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override

int64_max = np.iinfo(np.int64).max


class InternVL3Converter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.do_vit = True
        # modify weight info
        for i in ["LAYERS", "EMBEDING", "NORM", "LMHEAD"]:
            self.model_info.weights[i] = "language_model." + self.model_info.weights[i]
        # vision config
        self.vision_config = config.vision_config
        self.downsample_ratio = config.downsample_ratio
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size
        self.num_image_token = int(
            (self.image_size // self.patch_size)**2 * (self.downsample_ratio**2))
        self.depth = self.vision_config.num_hidden_layers
        self.vit_hidden_size = self.vision_config.hidden_size
        self.vit_num_heads = self.vision_config.num_attention_heads
        self.vit_head_dim = self.vit_hidden_size // self.vit_num_heads
        self.vit_intermediate_size = self.vision_config.intermediate_size
        self.vit_ln_eps = self.vision_config.layer_norm_eps

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.llm_config = config.llm_config
        self.llm_type = self.llm_config.model_type

    def vision_block(self, vit_mlir, idx: int, in_op):
        attn_qkv = f"vision_model.encoder.layers.{idx}.attn.qkv"
        attn_proj = f"vision_model.encoder.layers.{idx}.attn.proj"
        norm1 = f"vision_model.encoder.layers.{idx}.norm1"
        norm2 = f"vision_model.encoder.layers.{idx}.norm2"
        mlp = f"vision_model.encoder.layers.{idx}.mlp.fc"
        ls1 = f"vision_model.encoder.layers.{idx}.ls1"
        ls2 = f"vision_model.encoder.layers.{idx}.ls2"
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        def vision_attention(in_op):
            weight_op0 = vit_mlir.create_weight_op(norm1 + ".weight", [1, 1, self.vit_hidden_size])
            weight_op1 = vit_mlir.create_weight_op(norm1 + ".bias", [1, 1, self.vit_hidden_size])
            norm1_op = top.LayerNormOp(T([1, 1025, self.vit_hidden_size]),
                                       in_op,
                                       weight_op0,
                                       weight_op1,
                                       axis=2,
                                       eps=self.vit_ln_eps,
                                       normalized_shape=[self.vit_hidden_size],
                                       loc=L(norm1),
                                       ip=ip).output
            weight_op2 = vit_mlir.create_weight_op(attn_qkv + ".weight",
                                                   [self.vit_hidden_size, self.vit_hidden_size * 3])
            weight_op3 = vit_mlir.create_weight_op(attn_qkv + ".bias",
                                                   [1, 1, self.vit_hidden_size * 3])
            qkv_mm_op = top.MatMulOp(T([1, 1025, self.vit_hidden_size * 3]),
                                     norm1_op,
                                     weight_op2,
                                     weight_op3,
                                     loc=L(attn_qkv),
                                     ip=ip).output
            reshape_op = top.ReshapeOp(T([1, 1025, 3, self.vit_num_heads, self.vit_head_dim]),
                                       qkv_mm_op,
                                       shape=[1, 1025, 3, self.vit_num_heads, self.vit_head_dim],
                                       loc=L(attn_qkv + ".reshape"),
                                       ip=ip).output
            q_slice_op = top.SliceOp(T([1, 1025, 1, self.vit_num_heads, self.vit_head_dim]),
                                     reshape_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     offset=[0, 0, 0, 0, 0],
                                     steps=[1, 1, 1, 1, 1],
                                     ends=[int64_max, int64_max, 1, int64_max, int64_max],
                                     hasparamConvert_axes=[2],
                                     loc=L(attn_qkv + ".q.slice"),
                                     ip=ip).output
            k_slice_op = top.SliceOp(T([1, 1025, 1, self.vit_num_heads, self.vit_head_dim]),
                                     reshape_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     offset=[0, 0, 1, 0, 0],
                                     steps=[1, 1, 1, 1, 1],
                                     ends=[int64_max, int64_max, 2, int64_max, int64_max],
                                     hasparamConvert_axes=[2],
                                     loc=L(attn_qkv + ".k.slice"),
                                     ip=ip).output
            v_slice_op = top.SliceOp(T([1, 1025, 1, self.vit_num_heads, self.vit_head_dim]),
                                     reshape_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     offset=[0, 0, 2, 0, 0],
                                     steps=[1, 1, 1, 1, 1],
                                     ends=[int64_max, int64_max, 3, int64_max, int64_max],
                                     hasparamConvert_axes=[2],
                                     loc=L(attn_qkv + ".v.slice"),
                                     ip=ip).output
            q_squeeze_op = top.SqueezeOp(T([1, 1025, self.vit_num_heads, self.vit_head_dim]),
                                         q_slice_op,
                                         axes=[2],
                                         loc=L(attn_qkv + ".q.squeeze"),
                                         ip=ip).output
            k_squeeze_op = top.SqueezeOp(T([1, 1025, self.vit_num_heads, self.vit_head_dim]),
                                         k_slice_op,
                                         axes=[2],
                                         loc=L(attn_qkv + ".k.squeeze"),
                                         ip=ip).output
            v_squeeze_op = top.SqueezeOp(T([1, 1025, self.vit_num_heads, self.vit_head_dim]),
                                         v_slice_op,
                                         axes=[2],
                                         loc=L(attn_qkv + ".v.squeeze"),
                                         ip=ip).output
            q_permute_op = top.PermuteOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                         q_squeeze_op,
                                         order=[0, 2, 1, 3],
                                         loc=L(attn_qkv + ".q.permute"),
                                         ip=ip).output
            q_mulconst_op = top.MulConstOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                           q_permute_op,
                                           const_val=0.125,
                                           loc=L(attn_qkv + ".q.mulconst"),
                                           ip=ip).output
            k_permute_op0 = top.PermuteOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                          k_squeeze_op,
                                          order=[0, 2, 1, 3],
                                          loc=L(attn_qkv + ".k.permute0"),
                                          ip=ip).output
            k_permute_op1 = top.PermuteOp(T([1, self.vit_num_heads, self.vit_head_dim, 1025]),
                                          k_permute_op0,
                                          order=[0, 1, 3, 2],
                                          loc=L(attn_qkv + ".k.permute1"),
                                          ip=ip).output

            qk_mm_op = top.MatMulOp(T([1, self.vit_num_heads, 1025, 1025]),
                                    q_mulconst_op,
                                    k_permute_op1,
                                    vit_mlir.none_op,
                                    loc=L(attn_qkv + ".qk.matmul"),
                                    ip=ip).output
            qk_softmax_op = top.SoftmaxOp(T([1, self.vit_num_heads, 1025, 1025]),
                                          qk_mm_op,
                                          axis=3,
                                          loc=L(attn_qkv + ".qk.softmax"),
                                          ip=ip).output
            v_permute_op = top.PermuteOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                         v_squeeze_op,
                                         order=[0, 2, 1, 3],
                                         loc=L(attn_qkv + ".v.permute"),
                                         ip=ip).output
            qkv_mm_op = top.MatMulOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                     qk_softmax_op,
                                     v_permute_op,
                                     vit_mlir.none_op,
                                     loc=L(attn_qkv + ".qkv.matmul"),
                                     ip=ip).output
            o_permute_op = top.PermuteOp(T([1, 1025, self.vit_num_heads, self.vit_head_dim]),
                                         qkv_mm_op,
                                         order=[0, 2, 1, 3],
                                         loc=L(attn_proj + ".o.permute"),
                                         ip=ip).output
            o_reshape_op = top.ReshapeOp(T([1, 1025, self.vit_hidden_size]),
                                         o_permute_op,
                                         shape=[1, 1025, self.vit_hidden_size],
                                         loc=L(attn_proj + ".o.reshape"),
                                         ip=ip).output
            weight_op4 = vit_mlir.create_weight_op(attn_proj + ".weight",
                                                   [self.vit_hidden_size, self.vit_hidden_size])
            weight_op5 = vit_mlir.create_weight_op(attn_proj + ".bias",
                                                   [1, 1, self.vit_hidden_size])
            o_mm_op = top.MatMulOp(T([1, 1025, self.vit_hidden_size]),
                                   o_reshape_op,
                                   weight_op4,
                                   weight_op5,
                                   loc=L(attn_qkv + ".matmul"),
                                   ip=ip).output
            weight_op6 = vit_mlir.create_weight_op(ls1, [1, 1, self.vit_hidden_size])
            o_mul_op = top.MulOp(T([1, 1025, self.vit_hidden_size]), [o_mm_op, weight_op6],
                                 loc=L(attn_proj + ".o.mul"),
                                 ip=ip).output
            new_op = top.AddOp(T([1, 1025, self.vit_hidden_size]), [in_op, o_mul_op],
                               loc=L(attn_proj + ".add"),
                               ip=ip).output
            return new_op

        def vision_mlp(in_op):
            weight_op0 = vit_mlir.create_weight_op(norm2 + ".weight", [1, 1, self.vit_hidden_size])
            weight_op1 = vit_mlir.create_weight_op(norm2 + ".bias", [1, 1, self.vit_hidden_size])
            norm2_op = top.LayerNormOp(T([1, 1025, self.vit_hidden_size]),
                                       in_op,
                                       weight_op0,
                                       weight_op1,
                                       axis=2,
                                       eps=self.vit_ln_eps,
                                       normalized_shape=[self.vit_hidden_size],
                                       loc=L(norm2),
                                       ip=ip).output
            weight_op2 = vit_mlir.create_weight_op(
                mlp + "1.weight", [self.vit_hidden_size, self.vit_intermediate_size])
            weight_op3 = vit_mlir.create_weight_op(mlp + "1.bias",
                                                   [1, 1, self.vit_intermediate_size])
            mlp_up_op = top.MatMulOp(T([1, 1025, self.vit_intermediate_size]),
                                     norm2_op,
                                     weight_op2,
                                     weight_op3,
                                     loc=L(mlp + "1"),
                                     ip=ip).output
            active_op = self.activate(vit_mlir, mlp_up_op, self.vision_config.hidden_act, mlp)
            weight_op4 = vit_mlir.create_weight_op(
                mlp + "2.weight", [self.vit_intermediate_size, self.vit_hidden_size])
            weight_op5 = vit_mlir.create_weight_op(mlp + "2.bias", [1, 1, self.vit_hidden_size])
            mlp_down_op = top.MatMulOp(T([1, 1025, self.vit_hidden_size]),
                                       active_op,
                                       weight_op4,
                                       weight_op5,
                                       loc=L(mlp + "2"),
                                       ip=ip).output
            weight_op6 = vit_mlir.create_weight_op(ls2, [1, 1, self.vit_hidden_size])
            mul_op = top.MulOp(T([1, 1025, self.vit_hidden_size]), [mlp_down_op, weight_op6],
                               loc=L(mlp + ".mul"),
                               ip=ip).output
            new_op = top.AddOp(T([1, 1025, self.vit_hidden_size]), [in_op, mul_op],
                               loc=L(mlp + ".add"),
                               ip=ip).output
            return new_op

        in_op = vision_attention(in_op)
        in_op = vision_mlp(in_op)
        return in_op

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        # create weights file
        vit_npz = "vit_top_weights.npz"
        patch_embed = "vision_model.embeddings"
        layers = "vision_model.encoder.layers"
        merger = "mlp1"

        def save_weights():
            weights_dict = {}
            self.set_common_weight(patch_embed + ".class_embedding", weights_dict)
            self.set_common_weight(patch_embed + ".patch_embedding", weights_dict)
            self.set_common_weight(patch_embed + ".position_embedding", weights_dict)
            self.set_common_weight(merger + ".0", weights_dict)
            self.set_linear_weight(merger + ".1", weights_dict)
            self.set_linear_weight(merger + ".3", weights_dict)
            for i in range(self.depth):
                self.set_common_weight(layers + f".{i}.norm1", weights_dict)
                self.set_common_weight(layers + f".{i}.norm2", weights_dict)
                self.set_common_weight(layers + f".{i}.ls1", weights_dict)
                self.set_common_weight(layers + f".{i}.ls2", weights_dict)
                self.set_linear_weight(layers + f".{i}.attn.proj", weights_dict)
                self.set_linear_weight(layers + f".{i}.attn.qkv", weights_dict)
                self.set_linear_weight(layers + f".{i}.mlp.fc1", weights_dict)
                self.set_linear_weight(layers + f".{i}.mlp.fc2", weights_dict)
            # save weights
            np.savez(vit_npz, **weights_dict)

        # create mlir file
        vit_mlir = MLIRImporter([[1, 3, self.image_size, self.image_size]],
                                [[self.num_image_token, self.hidden_size]],
                                "vit",
                                Platform.LLM, ['F32'],
                                weight_file=vit_npz)
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        in_op = vit_mlir.create_input_op(L('pixel_value'), 0)
        # embedding
        weight_op0 = vit_mlir.create_weight_op(
            patch_embed + ".patch_embedding.weight",
            [self.vit_hidden_size, 3, self.patch_size, self.patch_size])
        weight_op1 = vit_mlir.create_weight_op(patch_embed + ".patch_embedding.bias",
                                               [self.vit_hidden_size])
        conv_op = top.ConvOp(T([1, self.vit_hidden_size, 32, 32]),
                             in_op,
                             weight_op0,
                             weight_op1,
                             kernel_shape=[self.patch_size, self.patch_size],
                             strides=[self.patch_size, self.patch_size],
                             pads=[0, 0, 0, 0],
                             dilations=[1, 1],
                             loc=L(patch_embed),
                             ip=ip).output
        reshape_op = top.ReshapeOp(T([1, 1024, self.vit_hidden_size]),
                                   conv_op,
                                   shape=[1, self.vit_hidden_size, -1],
                                   loc=L(patch_embed + ".reshape"),
                                   ip=ip).output
        permute_op = top.PermuteOp(T([1, self.vit_hidden_size, 1024]),
                                   reshape_op,
                                   order=[0, 2, 1],
                                   loc=L(patch_embed + ".permute"),
                                   ip=ip).output
        weight_op2 = vit_mlir.create_weight_op(patch_embed + ".class_embedding",
                                               [1, 1, self.vit_hidden_size])
        concat_op = top.ConcatOp(T([1, 1025, self.vit_hidden_size]), [weight_op2, permute_op],
                                 axis=1,
                                 loc=L(patch_embed + ".concat"),
                                 ip=ip).output
        weight_op3 = vit_mlir.create_weight_op(patch_embed + ".position_embedding",
                                               [1, 1025, self.vit_hidden_size])
        add_op = top.AddOp(T([1, 1025, self.vit_hidden_size]), [concat_op, weight_op3],
                           loc=L(patch_embed + ".add"),
                           ip=ip).output
        # block
        new_op = add_op
        for idx in range(self.depth):
            new_op = self.vision_block(vit_mlir, idx, new_op)
        # merge
        slice_op = top.SliceOp(T([1, 1024, 1024]),
                               new_op,
                               vit_mlir.none_op,
                               vit_mlir.none_op,
                               vit_mlir.none_op,
                               offset=[0, 1, 0],
                               steps=[1, 1, 1],
                               ends=[1, int64_max, 1024],
                               hasparamConvert_axes=[1],
                               loc=L(merger + ".slice"),
                               ip=ip).output
        reshape_op0 = top.ReshapeOp(T([1, 32, 16, 2048]),
                                    slice_op,
                                    shape=[1, 32, 16, 2048],
                                    loc=L(merger + ".reshape0"),
                                    ip=ip).output
        permute_op0 = top.PermuteOp(T([1, 16, 32, 2048]),
                                    reshape_op0,
                                    order=[0, 2, 1, 3],
                                    loc=L(merger + ".permute0"),
                                    ip=ip).output
        reshape_op1 = top.ReshapeOp(T([1, 16, 16, self.vit_intermediate_size]),
                                    permute_op0,
                                    shape=[1, 16, 16, self.vit_intermediate_size],
                                    loc=L(merger + ".reshape1"),
                                    ip=ip).output
        permute_op1 = top.PermuteOp(T([1, 16, 16, self.vit_intermediate_size]),
                                    reshape_op1,
                                    order=[0, 2, 1, 3],
                                    loc=L(merger + ".permute1"),
                                    ip=ip).output
        reshape_op2 = top.ReshapeOp(T([1, self.num_image_token, self.vit_intermediate_size]),
                                    permute_op1,
                                    shape=[1, -1, self.vit_intermediate_size],
                                    loc=L(merger + ".reshape2"),
                                    ip=ip).output
        weight_op4 = vit_mlir.create_weight_op(merger + ".0.weight",
                                               [1, 1, self.vit_intermediate_size])
        weight_op5 = vit_mlir.create_weight_op(merger + ".0.bias",
                                               [1, 1, self.vit_intermediate_size])
        norm_op = top.LayerNormOp(T([1, self.num_image_token, self.vit_intermediate_size]),
                                  reshape_op2,
                                  weight_op4,
                                  weight_op5,
                                  axis=2,
                                  eps=self.vit_ln_eps,
                                  normalized_shape=[self.vit_intermediate_size],
                                  loc=L(merger + ".norm"),
                                  ip=ip).output
        weight_op6 = vit_mlir.create_weight_op(merger + ".1.weight",
                                               [self.vit_intermediate_size, self.hidden_size])
        weight_op7 = vit_mlir.create_weight_op(merger + ".1.bias", [1, 1, self.hidden_size])
        mm_op0 = top.MatMulOp(T([1, self.num_image_token, self.hidden_size]),
                              norm_op,
                              weight_op6,
                              weight_op7,
                              loc=L(merger + ".1"),
                              ip=ip).output
        active_op = self.activate(vit_mlir, mm_op0, ActType.GELU, merger)
        weight_op8 = vit_mlir.create_weight_op(merger + ".3.weight",
                                               [self.hidden_size, self.hidden_size])
        weight_op9 = vit_mlir.create_weight_op(merger + ".3.bias", [1, 1, self.hidden_size])
        mm_op1 = top.MatMulOp(T([1, self.num_image_token, self.hidden_size]),
                              active_op,
                              weight_op8,
                              weight_op9,
                              loc=L(merger + ".3"),
                              ip=ip).output
        reshape_op3 = top.ReshapeOp(T([self.num_image_token, self.hidden_size]),
                                    mm_op1,
                                    shape=[self.num_image_token, self.hidden_size],
                                    loc=L(merger + ".reshape3"),
                                    ip=ip).output
        vit_mlir.create_return_op([reshape_op3])
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)
        save_weights()
