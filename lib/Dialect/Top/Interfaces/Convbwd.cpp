#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

convbwd_attr_t top::ConvbwdOp::parseParam() {
  convbwd_attr_t p = {0};
  auto input_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto gradout_s = getGradOut().getType().cast<RankedTensorType>().getShape();
  p.n = input_s[0];
  p.ic = input_s[1];
  p.ih = input_s.size() > 2 ? input_s[2] : 1;
  p.iw = input_s.size() > 3 ? input_s[3] : 1;
  p.oc = gradout_s[1];
  p.oh = gradout_s.size() > 2 ? gradout_s[2] : 1;
  p.ow = gradout_s.size() > 3 ? gradout_s[3] : 1;
  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(2);
  p.kw = kernel->at(3);
  auto pads_v = module::getI64Array(getPadding());
  p.pht = pads_v->at(0);
  p.pwl = pads_v->at(1);
  p.phb = pads_v->at(2);
  p.pwr = pads_v->at(3);
  auto strides_v = module::getI64Array(getStride());
  p.sh = strides_v->at(0);
  p.sw = strides_v->at(1);
  auto dhdw = module::getI64Array(getDilations(), 2, 1);
  p.dh = dhdw->at(0);
  p.dw = dhdw->at(1);
  auto ins = module::getI64Array(getInserts());
  p.insh = ins->at(0);
  p.insw = ins->at(1);
  p.groups = getGroups();
  return p;
}

inline static int calc_offset(const int *shape, const int *offset) {
    return ((offset[0] * shape[1] + offset[1]) * shape[2] + offset[2]) * shape[3] + offset[3];
}

template <typename T1>
void Tensor_nc_transpose(T1* in_data, T1* out_data, const int in_shape[4]) {
  int N = in_shape[0], C = in_shape[1], H= in_shape[2], W = in_shape[3];
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int nIndex = n * C * H * W + c * H * W + h * W + w;
                int cIndex = c * N * H * W + n * H * W + h * W + w;
                out_data[cIndex] = in_data[nIndex];
                }
            }
        }
    }
}
template <typename T1>
void tensor_interpolation(T1* input, T1* output, const int input_shape[4],int stride_h,int stride_w){
    int N = input_shape[0], C = input_shape[1], H = input_shape[2] ,W = input_shape[3];
    int H_out = H +(H-1)*(stride_h-1);
    int W_out = W +(W-1)*(stride_w-1);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0;h < H;h++){
                for (int w = 0;w < W;w++){
                    int pos_input =  w + h*W + c*(H*W) + n *(C*H*W);
                    int pos_output = n * C * H_out * W_out + c * H_out * W_out + h * stride_h * W_out + w * stride_w;
                    output[pos_output] = input[pos_input];
                }
            }
        }
    }
}

template <typename T1>
void kernel_flip(T1* input, T1* output, const int kernel_shape[4]){
    int oc = kernel_shape[0], ic = kernel_shape[1], kh = kernel_shape[2] ,kw = kernel_shape[3];
    for (int o_c = 0; o_c < oc; ++o_c) {
        for (int i_c = 0; i_c < ic; ++i_c) {
            for (int k_h = 0;k_h < kh;k_h++){
                for (int k_w =0;k_w < kw;k_w++){
                    int index = o_c * ic * kh * kw + i_c * kh * kw + k_h * kw + k_w;
                    int flippedIndex = o_c * ic * kh * kw + i_c * kh * kw + (kh - 1 - k_h) * kw + (kw - 1 - k_w);
                    output[flippedIndex] = input[index];
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4>
void conv_native_core(
        T2 *ofmap, const T1 *ifmap, const T4 *weight,
        const T3 *bias, const T1 *pad_ins,
        int input_n, int input_c, int input_h, int input_w,
        int output_c, int groups,
        int kh, int kw, int dh, int dw, int ins_h, int ins_w,
        int pht, int phb, int pwl, int pwr,
        int stride_h, int stride_w, int kernel_rotate,
        bool with_bias, bool nc_trans=false,
        int output_padding_h = 0,int output_padding_w = 0) {
    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;
    int ih_ext = (input_h - 1) * (ins_h + 1) + pht + phb + 1;
    int iw_ext = (input_w - 1) * (ins_w + 1) + pwr + pwl + 1;
    int oh = (ih_ext - kh_ext) / stride_h + 1 + output_padding_h;// output_padding for convtranspose
    int ow = (iw_ext - kw_ext) / stride_w + 1 + output_padding_w;

    int i_shape[4];
    i_shape[0] = input_n;
    i_shape[1] = input_c;
    i_shape[2] = input_h;
    i_shape[3] = input_w;
    int o_shape[4];
    o_shape[0] = input_n;
    o_shape[1] = output_c;
    o_shape[2] = oh;
    o_shape[3] = ow;
    int k_shape[4];
    k_shape[0] = output_c;
    k_shape[1] = input_c / groups;
    k_shape[2] = kh;
    k_shape[3] = kw;

    int o_g = output_c / groups;
    int k_g = input_c / groups;
    int o_head, k_head;

    memset(ofmap, 0, input_n * output_c * oh * ow * sizeof(T2));
    for (int n = 0; n < input_n; n++) {
        for (int g = 0; g < groups; g++) {
            for (int o = 0; o < o_g; o++) {
// #ifdef _OPENMP
// #pragma omp parallel for schedule(static, (oh * ow + omp_get_num_threads() - 1) / omp_get_num_threads()) collapse(2)
// #endif
                for (int y = 0; y < oh; y++) {
                    for (int x = 0; x < ow; x++) {
                        o_head = o_g * g;
                        k_head = k_g * g;
                        int weight_offset[4];
                        int in_offset[4];
                        int out_offset[4];
                        int out_idx = 0;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        out_offset[2] = y;
                        out_offset[3] = x;
                        for (int k = 0; k < k_g; k++) {
                            for (int p = 0; p < kh; p++) {
                                for (int q = 0; q < kw; q++) {
                                    int ih_pos = y * stride_h + p * dh - pht;
                                    int iw_pos = x * stride_w + q * dw - pwl;
                                    bool pos_ins = ih_pos % (ins_h + 1) > 0 ||
                                                   iw_pos % (ins_w + 1) > 0;
                                    bool pos_pad = ih_pos < 0 || ih_pos >= (ih_ext - phb - pht) ||
                                                   iw_pos < 0 || iw_pos >= (iw_ext - pwl - pwr);
                                    in_offset[0] = n;
                                    in_offset[1] = k + k_head;
                                    in_offset[2] = ih_pos / (ins_h + 1);
                                    in_offset[3] = iw_pos / (ins_w + 1);
                                    int src_idx = calc_offset(i_shape, in_offset);
                                    T1 ival;
                                    memset(&ival, 0, sizeof(T1));
                                    if (ih_pos >= 0 && ih_pos / (ins_h + 1) < input_h &&
                                        iw_pos >= 0 && iw_pos / (ins_w + 1) < input_w)
                                        ival = ifmap[src_idx];
                                    if (pos_ins) {
                                        ival = pad_ins[nc_trans ? (n * 2 + 1) : ((k_head + k) * 2 + 1)];
                                    }
                                    if (pos_pad) {
                                        ival = pad_ins[nc_trans ? n * 2 : (k_head + k) * 2];
                                    }
                                    weight_offset[0] = o + o_head;
                                    weight_offset[1] = k;
                                    weight_offset[2] = p;
                                    weight_offset[3] = q;

                                    out_idx = calc_offset(o_shape, out_offset);
                                    int widx = calc_offset(k_shape, weight_offset);
                                    ofmap[out_idx] = ofmap[out_idx] + (ival * weight[widx]);

                                }
                            }
                        }
                        if (with_bias) {
                          ofmap[out_idx] += bias[g * o_g + o];
                        }
                    }
                }
            }
        }
    }
}

int64_t top::ConvbwdOp::getFLOPs() {
  return module::getNumElements(getGradInput()) * 3;
}

LogicalResult top::ConvbwdOp::init(InferenceParameter &p) {
  return success();
}
void top::ConvbwdOp::deinit(InferenceParameter &p) {}

LogicalResult top::ConvbwdOp::inference(InferenceParameter &p) {
  auto attr = parseParam();
  float *gradout = p.inputs[0];
  float *input = p.inputs[1];
  float *kernel = p.inputs[2];
  float *gradinput = p.outputs[0];
  float *gradweight = p.outputs[1];
  float *gradbias = p.outputs[2];
  auto cal_grad_input = getGradInputEnable();
  auto cal_grad_weight = getGradWeightEnable();
  auto cal_grad_bias = getGradBiasEnable();

  int input_size = attr.n*attr.ic*attr.ih*attr.iw;
  float *input_nc = new float[input_size];
  int gradout_size = attr.n*attr.oc*attr.oh*attr.ow;
  float *gradout_nc = new float[gradout_size];
  int weight_size = attr.oc*attr.ic*attr.kh*attr.kw;
  float *grad_weight_nc = new float [weight_size];
  float *pad_ins_weight = new float [attr.n * 2];
  float *pad_ins_input = new float [attr.oc*2];
  memset(pad_ins_weight, 0, attr.n * 2 * sizeof(float));
  memset(pad_ins_input, 0, attr.oc * 2 * sizeof(float));
  float *flip_kernel = new float [weight_size];
  float *flip_kernel_nc = new float [weight_size];
  int interp_h = attr.oh+(attr.oh-1)*(attr.sh-1);
  int interp_w = attr.ow+(attr.ow-1)*(attr.sw-1);
  int grad_out_interp_size = attr.n * attr.oc*interp_h*interp_w;
  float *gradout_interp = new float [grad_out_interp_size];
  int kernel_shape[4] = {(int)attr.oc,(int)attr.ic,(int)attr.kh,(int)attr.kw};
  int gradout_shape[4] = {(int)attr.n,(int)attr.oc,(int)attr.oh,(int)attr.ow};
  int input_shape[4] = {(int)attr.n,(int)attr.ic,(int)attr.ih,(int)attr.iw};
  int kernel_grad_nc_origin_shape[4] = {(int)attr.ic, (int)attr.oc, (int)attr.kh, (int)attr.kw};


  if (cal_grad_bias){
    int n = attr.n, oc = attr.oc, oh = attr.oh ,ow = attr.ow;
    for (int o_c = 0; o_c < oc; o_c++) {
        float tmp = 0;
        for (int i_n = 0; i_n < n; i_n++) {
            for (int o_h = 0; o_h < oh; o_h++) {
                for (int o_w = 0; o_w < ow; o_w++) {
                    int pos = i_n * oc * oh * ow + o_c * oh * ow + o_h *  ow + o_w;
                    tmp += gradout[pos];
                }
            }
        }
        gradbias[o_c] = tmp;
    }
  }
  if (cal_grad_weight){
    Tensor_nc_transpose(gradout, gradout_nc, gradout_shape);
    Tensor_nc_transpose(input, input_nc, input_shape);
    int new_stride = 1;
    float bias = 0;
    int pad_h_b_new = attr.kh-(attr.pht+attr.ih-attr.sh*(attr.oh-1));
    int pad_w_r_new = attr.kw-(attr.pwl+attr.iw-attr.sw*(attr.ow-1));
    conv_native_core(grad_weight_nc, input_nc, gradout_nc, &bias, pad_ins_weight,
                attr.ic, attr.n, attr.ih, attr.iw, attr.oc, 1,/*group_num*/ attr.oh, attr.ow,
                attr.sh, attr.sw, attr.insh, attr.insw,
                attr.pht, pad_h_b_new, attr.pwl, pad_w_r_new, new_stride, new_stride,
                false, false);
    Tensor_nc_transpose(grad_weight_nc, gradweight, kernel_grad_nc_origin_shape);
  }
  if (cal_grad_input){
    kernel_flip(kernel,flip_kernel,kernel_shape);
    Tensor_nc_transpose(flip_kernel, flip_kernel_nc, kernel_shape);
    tensor_interpolation(gradout,gradout_interp,gradout_shape,attr.sh,attr.sw);
    float bias = 0;
    int pad_new_h_b = attr.kh-attr.phb-1;
    int pad_new_h_t = attr.kh-attr.pht-1;
    int pad_new_w_l = attr.kw-attr.pwl-1;
    int pad_new_w_r = attr.kw-attr.pwr-1;
    int stride_new = 1;
    int output_padding_h = attr.ih -(interp_h+pad_new_h_b+pad_new_h_t-attr.kh+1);
    int output_padding_w = attr.iw -(interp_w+pad_new_w_l+pad_new_w_r-attr.kw+1);
    conv_native_core(gradinput, gradout_interp, flip_kernel_nc, &bias, pad_ins_input,
                  attr.n, attr.oc, interp_h, interp_w, attr.ic, 1,/*group_num*/ attr.kh, attr.kw,
                  attr.dh, attr.dw, attr.insh, attr.insw,
                  pad_new_h_b, pad_new_h_t, pad_new_w_l, pad_new_w_r, stride_new, stride_new,
                  false, false, output_padding_h,output_padding_w);
  }
  return success();
}

void top::ConvbwdOp::shape_inference() {

}
