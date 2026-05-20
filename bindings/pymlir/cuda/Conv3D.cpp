#include "../pycuda.h"    
#include "cuda_helper.h"   
#include "iostream"


using namespace std;

void py_cuda::cudaConv3DOp(top::ConvOp op) {
  cout << "ffffffffffffffffffffffffffffffffffffff" << endl;
  auto p = op.parseParam();
  auto num_out = module::getNumElements(op.getOutput());


  bool need_pad = false;
  if (p.dims == 3) {
    if (p.pdf != p.pdb || p.pht != p.phb || p.pwl != p.pwr) need_pad = true;
  } else if (p.pht != p.phb || p.pwl != p.pwr) {
    need_pad = true;
  }

  auto in_f32 = getCudaData(op.getInput());
  int id = (int)p.id, ih = (int)p.ih, iw = (int)p.iw;
  int pad_d = (int)p.pdf, pad_h = (int)p.pht, pad_w = (int)p.pwl;

  cuda_ptr in_f;
  if (need_pad) {
    int pd = (int)(p.id + p.pdf + p.pdb);
    int ph = (int)(p.ih + p.pht + p.phb);
    int pw = (int)(p.iw + p.pwl + p.pwr);
    pad_d = 0; pad_h = 0; pad_w = 0;

    int64_t num = (int64_t)p.n * p.ic * pd * ph * pw;
    in_f = cuda_malloc(num * sizeof(float));
    in_f32 = in_f.get();

    if (p.dims == 3) {
      cuda::pad5D(getCudaData(op.getInput()), in_f32, p.n, p.ic, p.id, p.ih, p.iw,
                  p.pdf, p.pdb, p.pht, p.phb, p.pwl, p.pwr, 0.0f);
    } else {
      cuda::pad4D(getCudaData(op.getInput()), in_f32, p.n, p.ic, p.id, p.ih,
                  p.pht, p.phb, p.pwl, p.pwr, 0.0f);
    }
    id = pd; ih = ph; iw = pw;
  }

  // 1. Descriptors
  cudnnTensorDescriptor_t input_desc, outf32_desc;
  cudnnFilterDescriptor_t kernel_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  cudnnCreateTensorDescriptor(&input_desc);
  cudnnCreateFilterDescriptor(&kernel_desc);
  cudnnCreateTensorDescriptor(&outf32_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);

  int in_dims[5] = {(int)p.n, (int)p.ic, id, ih, iw};
  int in_strides[5] = {in_dims[1]*in_dims[2]*in_dims[3]*in_dims[4], in_dims[2]*in_dims[3]*in_dims[4], in_dims[3]*in_dims[4], in_dims[4], 1};
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 5, in_dims, in_strides));

  int filter_dims[5] = {(int)p.oc, (int)(p.ic / p.groups), (int)p.kd, (int)p.kh, (int)p.kw};
  CHECK_CUDNN(cudnnSetFilterNdDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filter_dims));

  int out_dims[5] = {(int)p.n, (int)p.oc, (int)p.od, (int)p.oh, (int)p.ow};
  int out_strides[5] = {out_dims[1]*out_dims[2]*out_dims[3]*out_dims[4], out_dims[2]*out_dims[3]*out_dims[4], out_dims[3]*out_dims[4], out_dims[4], 1};
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(outf32_desc, CUDNN_DATA_FLOAT, 5, out_dims, out_strides));

  std::vector<int> pad = {(int)pad_d, (int)pad_h, (int)pad_w};
  std::vector<int> str = {(int)p.sd, (int)p.sh, (int)p.sw};
  std::vector<int> dil = {(int)p.dd, (int)p.dh, (int)p.dw};
  CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(conv_desc, 3, pad.data(), str.data(), dil.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  if (p.groups > 1) CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, (int)p.groups));

  // 2. Execution
  auto kernel_f32 = getCudaData(op.getFilter());
  auto out_f32 = getCudaData(op.getOutput());
  size_t worksize = 0;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_, input_desc, kernel_desc, conv_desc, outf32_desc, algo, &worksize));
  auto conv_buffer = cuda_malloc(worksize);

  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionForward(cudnn_, &alpha, input_desc, in_f32, kernel_desc, kernel_f32, conv_desc, algo, conv_buffer.get(), worksize, &beta, outf32_desc, out_f32));

  // 3. Bias
  if (p.has_bias) {
    cudnnTensorDescriptor_t bias_desc;
    cudnnCreateTensorDescriptor(&bias_desc);
    int b_dims[5] = {1, (int)p.oc, 1, 1, 1};
    int b_strs[5] = {b_dims[1], 1, 1, 1, 1};
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, 5, b_dims, b_strs));
    float b_a = 1.0f, b_b = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(cudnn_, &b_a, bias_desc, getCudaData(op.getBias()), &b_b, outf32_desc, out_f32));
    cudnnDestroyTensorDescriptor(bias_desc);
  }

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyFilterDescriptor(kernel_desc);
  cudnnDestroyTensorDescriptor(outf32_desc); 
  cudnnDestroyConvolutionDescriptor(conv_desc);

  if (p.do_relu) doRelu(out_f32, num_out, cuda::DT_F32);
}





void py_cuda::cudaConv3DOp(tpu::Conv3DOp op) {
  cout << "===== Enter cudaConv3DOp(tpu::Conv3DOp) =====" << endl; 
  auto p = op.parseParam();
  auto num_out = module::getNumElements(op.getOutput());
  auto out_stype = module::getStorageType(op.getOutput());
  auto in_stype = module::getStorageType(op.getInput());

  bool is_quant_output = module::isUniformQuantized(op.getOutput());


  cout << "=== 111111111111111111111 ===" << endl;

  cout <<"p.dims="<<p.dims<<endl;

  bool need_pad = p.pht != p.phb || p.pwl != p.pwr || p.pdf != p.pdb;
  cout << "p.pht=" << p.pht << " p.phb=" << p.phb 
             << " p.pwl=" << p.pwl << " p.pwr=" << p.pwr 
             << " p.pdf=" << p.pdf << " p.pdb=" << p.pdb << endl;
  cout << "need_pad = " << (need_pad ? "true" : "false") << endl;





  int id = (int)p.id, ih = (int)p.ih, iw = (int)p.iw;
  int pad_d = (int)p.pdf, pad_h = (int)p.pht, pad_w = (int)p.pwl;

  cuda_ptr in_f32;
  if (need_pad) {
    cout << "=== Enter need_pad=true ===" << endl; 
    int pd = p.id + p.pdf + p.pdb;
    int ph = p.ih + p.pht + p.phb;
    int pw = p.iw + p.pwl + p.pwr;
    int64_t num = (int64_t)p.n * p.ic * pd * ph * pw;

    
    if(module::isUniformQuantized(op.getOutput())){
      cout << "=== isUniformQuantized=true ===" << endl;

      auto pad_in = cuda_malloc(num);
      auto input = getCudaData(op.getInput());

      cout << "=== 222222222222222222222222222222222222 ===" << endl;
      cuda::pad5D(input, pad_in.get(), p.n, p.ic, p.id, p.ih, p.iw,
                  p.pdf, p.pdb, p.pht, p.phb, p.pwl, p.pwr, 1);

      
      in_f32 = cuda_malloc(num * sizeof(float));

      const int print_num = 10; 
      int8_t* host_int8 = new int8_t[print_num];

      CHECK_CUDA(cudaMemcpy(host_int8, pad_in.get(), print_num * sizeof(int8_t), cudaMemcpyDeviceToHost));

      cout << "===== 转换前 INT8 数据（前" << print_num << "个）=====" << endl;
      for(int i=0; i<print_num; i++){
        cout << "index " << i << ": " << (int)host_int8[i] << " (int8_t)" << endl;
      }

      cuda::convertType(pad_in.get(), in_f32.get(), num, getCudaType(op.getInput()), cuda::DT_F32);


      float* host_float32 = new float[print_num];

      CHECK_CUDA(cudaMemcpy(host_float32, in_f32.get(), print_num * sizeof(float), cudaMemcpyDeviceToHost));

      cout << "===== 转换后 F32 数据（前" << print_num << "个）=====" << endl;
      for(int i=0; i<print_num; i++){
        cout << "index " << i << ": " << host_float32[i] << " (float)" << endl;
      }


    } else {

      in_f32 = cuda_malloc(num * sizeof(float));
      if(in_stype.isF32()){
        auto input = getCudaData(op.getInput());

        cuda::pad5D(input, in_f32.get(), p.n, p.ic, p.id, p.ih, p.iw,
                  p.pdf, p.pdb, p.pht, p.phb, p.pwl, p.pwr, sizeof(float));
      } else {

        auto input = newCudaData(op.getInput(), cuda::DT_F32);
        cuda::pad5D(input.get(), in_f32.get(), p.n, p.ic, p.id, p.ih, p.iw,
                  p.pdf, p.pdb, p.pht, p.phb, p.pwl, p.pwr, sizeof(float));
      }
    }


    id = pd; ih = ph; iw = pw;
    pad_d = 0; pad_h = 0; pad_w = 0;

  } else {

    if (module::isUniformQuantized(op.getOutput())) {
      cout << "=== need_pad=false && isUniformQuantized=true ===" << endl;

      auto input = getCudaData(op.getInput());     
      auto num_in = module::getNumElements(op.getInput());

      in_f32 = cuda_malloc(num_in * sizeof(float));   

      const int print_num = 10; 


      int8_t* host_int8 = new int8_t[print_num];

      CHECK_CUDA(cudaMemcpy(host_int8, input,
                      print_num * sizeof(int8_t),
                      cudaMemcpyDeviceToHost));

      cout << "===== BEFORE convert (INT8) =====" << endl;
      for (int i = 0; i < print_num; ++i) {
        cout << "index " << i << ": " << (int)host_int8[i] << endl;
      }

      cuda::convertType(input, in_f32.get(), num_in,
                      getCudaType(op.getInput()), cuda::DT_F32);


      float* host_f32 = new float[print_num];

      CHECK_CUDA(cudaMemcpy(host_f32, in_f32.get(),
                      print_num * sizeof(float),
                      cudaMemcpyDeviceToHost));

      cout << "===== AFTER convert (F32) =====" << endl;
      for (int i = 0; i < print_num; ++i) {
        cout << "index " << i << ": " << host_f32[i] << endl;
      }


      delete[] host_int8;
      delete[] host_f32;

    } else if (in_stype.isF32()) {

      in_f32 = cuda_ptr(getCudaData(op.getInput()));
    } else {

      in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    }
}


  // 1. Descriptors
  cudnnTensorDescriptor_t input_desc, outf32_desc;
  cudnnFilterDescriptor_t kernel_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  cudnnCreateTensorDescriptor(&input_desc);
  cudnnCreateFilterDescriptor(&kernel_desc);
  cudnnCreateTensorDescriptor(&outf32_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);

  int in_dims[5] = {(int)p.n, (int)p.ic, id, ih, iw};
  int in_strides[5] = {in_dims[1]*in_dims[2]*in_dims[3]*in_dims[4], in_dims[2]*in_dims[3]*in_dims[4], in_dims[3]*in_dims[4], in_dims[4], 1};
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 5, in_dims, in_strides));

  int filter_dims[5] = {(int)p.oc, (int)(p.ic / p.groups), (int)p.kd, (int)p.kh, (int)p.kw};
  CHECK_CUDNN(cudnnSetFilterNdDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filter_dims));

  int out_dims[5] = {(int)p.n, (int)p.oc, (int)p.od, (int)p.oh, (int)p.ow};
  int out_strides[5] = {out_dims[1]*out_dims[2]*out_dims[3]*out_dims[4], out_dims[2]*out_dims[3]*out_dims[4], out_dims[3]*out_dims[4], out_dims[4], 1};
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(outf32_desc, CUDNN_DATA_FLOAT, 5, out_dims, out_strides));

  std::vector<int> pad = {pad_d, pad_h, pad_w};
  std::vector<int> str = {(int)p.sd, (int)p.sh, (int)p.sw};
  std::vector<int> dil = {(int)p.dd, (int)p.dh, (int)p.dw};
  CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(conv_desc, 3, pad.data(), str.data(), dil.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  if (p.groups > 1) CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, (int)p.groups));


  auto wt_stype = module::getStorageType(op.getFilter());
  cuda_ptr kernel_f32;
  if (wt_stype.isF32()) {
    kernel_f32 = cuda_ptr(getCudaData(op.getFilter())); 
  } else {
    kernel_f32 = newCudaData(op.getFilter(), cuda::DT_F32); 
  }



  cuda_ptr out_f32 = cuda_malloc(num_out * sizeof(float));
  size_t worksize = 0;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_, input_desc, kernel_desc, conv_desc, outf32_desc, algo, &worksize));
  auto conv_buffer = cuda_malloc(worksize);

  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionForward(cudnn_, &alpha, input_desc, in_f32.get(),
                                      kernel_desc, kernel_f32.get(), conv_desc,
                                      algo, conv_buffer.get(), worksize, &beta,
                                      outf32_desc, out_f32.get()));

  if (p.has_bias && !is_quant_output) {
    cudnnTensorDescriptor_t bias_desc;
    cudnnCreateTensorDescriptor(&bias_desc);
    int b_dims[5] = {1, (int)p.oc, 1, 1, 1};
    int b_strs[5] = {b_dims[1], 1, 1, 1, 1};
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, 5, b_dims, b_strs));
    float b_a = 1.0f, b_b = 1.0f;
   
    if (module::getStorageType(op.getBias()).isF32()) {
      CHECK_CUDNN(cudnnAddTensor(cudnn_, &b_a, bias_desc, getCudaData(op.getBias()),
                                 &b_b, outf32_desc, out_f32.get()));
    } else {
      auto bias_f32 = newCudaData(op.getBias(), cuda::DT_F32);
      CHECK_CUDNN(cudnnAddTensor(cudnn_, &b_a, bias_desc, bias_f32.get(),
                                 &b_b, outf32_desc, out_f32.get()));
    }
    cudnnDestroyTensorDescriptor(bias_desc);
  } else if (p.has_bias) {
    

  // 4. ReLU
  if (p.do_relu && !is_quant_output) {
    doRelu(out_f32.get(), num_out, cuda::DT_F32);
  } else if (p.do_relu) {
    
  }

  // 5. Output processing
  if(is_quant_output){
    
    auto out_i32 = newCudaData(out_f32.get(), num_out, cuda::DT_F32, cuda::DT_INT32);

    
    if (p.has_bias) {
      cuda::add4DInt32((int32_t *)out_i32.get(), (int32_t *)getCudaData(op.getBias()),
                       (int32_t *)out_i32.get(),
                       p.n, p.oc, p.od * p.oh, p.ow,
                       1, p.oc, 1, 1,
                       p.n, p.oc, p.od * p.oh, p.ow);
    }

    // auto output = getCudaData(op.getOutput());
    // auto rshift_v = module::getI64Array(op.getRshift().value());
    // auto multiplier_v = module::getI64Array(op.getMultiplier(), rshift_v->size(), 1);
    // std::vector<int32_t> m(multiplier_v->begin(), multiplier_v->end());
    // std::vector<int32_t> rs(rshift_v->begin(), rshift_v->end());

    // auto cudaMults = cuda_malloc(p.oc * sizeof(int32_t));
    // auto cudaShifts = cuda_malloc(p.oc * sizeof(int32_t));
    // CHECK_CUDA(cudaMemcpy(cudaMults.get(), m.data(), m.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(cudaShifts.get(), rs.data(), rs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    auto output = getCudaData(op.getOutput());
    auto cudaMults = cuda_malloc(p.oc * sizeof(int32_t));
    auto cudaShifts = cuda_malloc(p.oc * sizeof(int32_t));
    auto rshift_v = module::getI64Array(op.getRshift().value());
    auto multiplier_v =
        module::getI64Array(op.getMultiplier(), rshift_v->size(), 1);
    std::vector<int32_t> m(multiplier_v->begin(), multiplier_v->end());
    std::vector<int32_t> rs(rshift_v->begin(), rshift_v->end());
    CHECK_CUDA(cudaMemcpy(cudaMults.get(), m.data(), m.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cudaShifts.get(), rs.data(),
                          rs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    if(out_stype.isInteger(16)){
      cuda::requantInt16Perchannel_3d(out_i32.get(), output, cudaMults.get(), cudaShifts.get(),
                                      p.n, p.oc, p.od, p.oh, p.ow, p.do_relu);
    } else {
      bool sign = !out_stype.isUnsignedInteger(8);
      bool qdm = op.getQuantMode() == tpu::RequantMode::QDM;
      bool relu = sign && p.do_relu;
      cuda::requantInt8Perchannel_3d(out_i32.get(), output, cudaMults.get(), cudaShifts.get(),
                                     p.n, p.oc, p.od, p.oh, p.ow, sign, qdm, relu);
    }
  } else {
  
    auto output = getCudaData(op.getOutput());

    if (out_stype.isFloat8E4M3FN()){
      bool relu = p.do_relu;
      auto cudaMults = cuda_malloc(p.oc * sizeof(float));
      float f8_scale = 1.0f;

      if (auto scale_opt = op.getOutF8Scale(); scale_opt.has_value()) {
        f8_scale = scale_opt.value().convertToFloat();
      }
      std::vector<float> scale_v(p.oc, f8_scale);
      CHECK_CUDA(cudaMemcpy(cudaMults.get(), scale_v.data(), scale_v.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      cuda::requantF8Perchannel_3d(out_f32.get(), output, cudaMults.get(),
                                   p.n, p.oc, p.od, p.oh, p.ow, relu, true);
    } else if (out_stype.isF32()) {
      cudaMemcpy(output, out_f32.get(), num_out * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
      cuda::convertType(out_f32.get(), output, num_out, cuda::DT_F32, getCudaType(op.getOutput()));
    }
  }

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyFilterDescriptor(kernel_desc);
  cudnnDestroyTensorDescriptor(outf32_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}


