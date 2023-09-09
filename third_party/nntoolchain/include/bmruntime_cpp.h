/*****************************************************************************
 *
 *    Copyright (c) 2016-2026 by Sophgo Technologies Inc. All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Sophgo Technologies Inc. This is proprietary information owned by
 *    Sophgo Technologies Inc. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Sophgo Technologies Inc.
 *
 *****************************************************************************/

#ifndef __BMRUNTIME_CPP_H__
#define __BMRUNTIME_CPP_H__

#include "bmdef.h"
#include <vector>

namespace bmruntime {
// class defined in this file.
class Context;
class Network;
class Tensor;

// --------------------------------------------------------------------------
// funcitons for basic type
size_t ByteSize(bm_data_type_t type);                // byte size of data type
size_t ByteSize(const bm_device_mem_t &device_mem);  // byte size of device memory
size_t ByteSize(const bm_tensor_t &tensor);          // byte size of origin tensor
size_t ByteSize(const Tensor &tensor);               // byte size of tensor

// element number, dims[0] * dims[1] * dims[...] * dims[num_dims-1]
uint64_t Count(const bm_shape_t &shape);
uint64_t Count(const bm_tensor_t &tensor);
uint64_t Count(const Tensor &tensor);

// compare whether shape dims is the same
bool IsSameShape(const bm_shape_t &left, const bm_shape_t &right);

// Context --------------------------------------------------------------------------
class Context {
 public:
  explicit Context(bm_handle_t bm_handle);
  explicit Context(int devid = 0);
  virtual ~Context();

  // load bmodel by file
  bm_status_t load_bmodel(const void *bmodel_data, size_t size);
  // load bmodel by buffer
  bm_status_t load_bmodel(const char *bmodel_file);

  // get network info
  int get_network_number() const;
  // get network names, vector will clear first ,and then push_back all net name
  void get_network_names(std::vector<const char *> *names) const;
  // get network info ; if net_name is not exist, it will return NULL
  const bm_net_info_t *get_network_info(const char *net_name) const;

  // access bm handle
  bm_handle_t handle() const;

  // trace error
  void trace() const;

 protected:
  void *body_;
  bm_handle_t bm_handle_;
  friend class Network;
};

// Network --------------------------------------------------------------------------
class Network {
 public:
  // if stage_id == -1, each input tensors can be reshaped by user, and must reshape at first time;
  // or each input tensors will hold the shape in the stage, and can't be reshaped
  Network(const Context &ctx, const char *net_name, int stage_id = -1);
  virtual ~Network();

  // do inference by input tensors
  // if sync == true, it will block util inference finish
  // if sync == false, bm_thread_sync should be called to make sure inference finished
  bm_status_t Forward(bool sync = true) const;

  // get input and output tensors
  const std::vector<Tensor *> &Inputs();
  const std::vector<Tensor *> &Outputs();

  // get input and output tensor by tensor name
  Tensor *Input(const char *tensor_name);
  Tensor *Output(const char *tensor_name);

  // get net information
  const bm_net_info_t *info() const;

 protected:
  const Context *ctx_;
  int net_id_;
  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;
  bm_tensor_t *input_tensors_;
  bm_tensor_t *output_tensors_;
};

// Tensor --------------------------------------------------------------------------
class Tensor {
 public:
  // copy data from device to system, copy size is ByteSize()
  bm_status_t CopyTo(void *data) const;
  // copy data from device to system, size+offset should <= ByteSize()
  bm_status_t CopyTo(void *data, size_t size, uint64_t offset = 0) const;
  // copy data from system to device, copy size is ByteSize()
  bm_status_t CopyFrom(const void *data);
  // copy data from system to device, size+offset should <= ByteSize()
  bm_status_t CopyFrom(const void *data, size_t size, uint64_t offset = 0);
  // set shape
  bm_status_t Reshape(const bm_shape_t &shape);
  // size in byte
  size_t ByteSize() const;
  // number of elements
  uint64_t num_elements() const;
  // get origin bm_tensor_t
  const bm_tensor_t *tensor() const;
  // set tensor store mode, if not set, BM_STORE_1N as default.
  void set_store_mode(bm_store_mode_t mode) const;
  // set device mem, update device mem by a new device mem
  bm_status_t set_device_mem(const bm_device_mem_t &device_mem);

 protected:
  // tensor only create by Network
  Tensor(bm_handle_t handle, bm_tensor_t *tensor, bool can_reshape = false);
  virtual ~Tensor();
  bool ready();
  bool user_mem_;
  bm_handle_t handle_;
  bm_tensor_t *tensor_;
  bool can_reshape_;
  bool have_reshaped_;
  friend class Network;
};

struct api_info_t {
  /// @brief api_id to be sent to driver
  int32_t api_id;
  /// @brief api data to be sent to driver, {core_idx, core_api_data}
  std::vector<std::vector<uint8_t>> api_data;
  /// @brief offset of input tensors' addr in api_data
  std::vector<uint32_t> input_addr_offset;
  /// @brief offset of output tensors' addr in api_data
  std::vector<uint32_t> output_addr_offset;
};
/**
 * @name    get_bmodel_api_info
 * @brief   To get the api info setting input tensors
 * @ingroup bmruntime
 *
 * This API only supports the neuron nework that is static-compiled.
 * After calling this API, api info will be setted and return,
 * and then you can call `bm_send_api` to start TPU inference.
 *
 * @param [in]    p_bmrt            Bmruntime that had been created
 * @param [in]    net_name          The name of the neuron network
 * @param [in]    input_tensors     Array of input tensor, defined like bm_tensor_t input_tensors[input_num],
 *                                  User should initialize each input tensor.
 * @param [in]    input_num         Input number
 * @param [out]   output_tensors    Array of output tensor, defined like bm_tensor_t output_tensors[output_num].
 *                                  User can set device_mem or stmode of output tensors. If user_mem is true, this interface
 *                                  will use device mem of output_tensors to store output data, and not alloc device mem;
 *                                  Or it will alloc device mem to store output. If user_stmode is true, it will use stmode in
 *                                  each output tensor; Or stmode will be BM_STORE_1N as default.
 * @param [in]    output_num        Output number
 * @param [in]    user_mem          whether device_mem of output tensors are set
 * @param [in]    user_stmode       whether stmode of output tensors are set
 *
 */
api_info_t get_bmodel_api_info(void *p_bmrt, const char *net_name,
                               const bm_tensor_t *input_tensors, int input_num,
                               bm_tensor_t *output_tensors, int output_num,
                               bool user_mem, bool user_stmode);
}  // namespace bmruntime

#endif /* __BMRUNTIME_CPP_H__ */
