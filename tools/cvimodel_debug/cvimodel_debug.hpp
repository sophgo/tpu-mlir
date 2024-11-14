#include "cnpy.h"
#include "cpu_func.hpp"
#include "cviruntime_context.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_profiling.hpp"
#include "tpu_mlir/Builder/CV18xx/cvimodel_generated.h"
#include "tpu_mlir/Builder/CV18xx/parameter_generated.h"
#include "utils.hpp"
using namespace tpu_mlir::backend;

namespace cvi_debug {

struct debug_info_t {
  int layer_id;
  std::string name;
  bool inGroup = false;
  bool ignore = false;
  uint64_t gaddr;
  uint32_t laddr;
  std::vector<int64_t> g_shape;
  std::vector<int64_t> lg_idx_slice;
  double qscale;
  std::string qtype;
};

struct MODEL_HEADER {
  char magic[8];
  uint32_t body_size;
  char major;
  char minor;
  char md5[16];
  char chip[16];
  char padding[2];
};

struct model_info {
  std::string chip;
  uint32_t shared_gmem_size;
  uint32_t private_gmem_size;
  uint32_t io_mem_num;
  uint32_t io_sizes[5] = {0, 0, 0, 0, 0};
  std::vector<io_mem_info> inputs_gmem_info;
  std::vector<io_mem_info> outs_gmem_info;
  std::vector<bool> cpu_tpu_order; // cpu:false, tpu:true
  std::vector<cpu_func_info> cpu_func_infos;
};

class FileStream {
public:
  FileStream(const std::string &file_name) {
    _fstream = new std::ifstream(file_name, std::ifstream::binary);
    if (!_fstream->good()) {
      printf("Error, Failed to open %s\n", file_name.c_str());
      return;
    }
    _fstream->seekg(0, _fstream->end);
    _length = _fstream->tellg();
    _fstream->seekg(0, _fstream->beg);
  }
  ~FileStream() {
    if (_fstream) {
      delete _fstream;
    }
  }
  size_t length() { return _length; }
  size_t read(uint8_t *buf, size_t offset, size_t size) {
    assert(offset + size <= _length && "model is incomplete or incorrect!");
    _fstream->seekg(offset);
    _fstream->read((char *)buf, size);
    return size;
  }

private:
  std::ifstream *_fstream;
  size_t _length;
};

uint32_t getFirstLayerId(std::vector<uint8_t> &cmdbuf) {
  uint32_t first_layer_id = 0;
  tiu_reg_t tiuReg;
  tdma_reg_t tdmaReg;
  unsigned char *pContent = nullptr;
  unsigned char *pCurrent = (unsigned char *)cmdbuf.data();
  cmd_hdr_t *pHeader = (cmd_hdr_t *)pCurrent;
  pContent = pCurrent + sizeof(cmd_hdr_t);
  if (pHeader->engine_id == 0) {
    TiuReg::parse_tiu_reg(&tiuReg, (uint32_t *)pContent, pHeader->magic);
    first_layer_id = tiuReg.layer_info;
  } else if (pHeader->engine_id == 2) {
    TdmaReg::parse_tdma_reg(&tdmaReg, (uint32_t *)pContent, pHeader->magic);
    first_layer_id = tdmaReg.layer_ID;
  } else {
    assert(0 && "Cmdbuf get no tdma and no tiu part");
  }
  return first_layer_id;
}

int splitCmdbuf(std::vector<uint8_t> &cmdbuf,
                std::vector<std::vector<uint8_t>> &cmdbufs) {
  uint32_t offset = 0;
  uint32_t last_offset = 0;
  uint32_t last_layer_id = getFirstLayerId(cmdbuf);
  tiu_reg_t tiuReg;
  tdma_reg_t tdmaReg;
  unsigned char *pContent = nullptr;
  unsigned char *pCurrent = (unsigned char *)cmdbuf.data();
  cmd_hdr_t *pHeader = nullptr;
  int cmdBufSize = cmdbuf.size();
  uint32_t tiuCnt = 0;
  uint32_t tdmaCnt = 0;
  uint32_t offset_tiuCnt = 0;
  uint32_t offset_tdmaCnt = 0;
  while (offset < cmdBufSize) {
    pHeader = (cmd_hdr_t *)pCurrent;
    pContent = pCurrent + sizeof(cmd_hdr_t);
    uint32_t cur_layer_id = last_layer_id;
    if (pHeader->engine_id == 0) {
      TiuReg::parse_tiu_reg(&tiuReg, (uint32_t *)pContent, pHeader->magic);
      cur_layer_id = tiuReg.layer_info;
      if (cur_layer_id != last_layer_id) {
        offset_tiuCnt = tiuCnt;
        offset_tdmaCnt = tdmaCnt;
      }
      tiuCnt++;
      reset_tiu_info((uint32_t *)pContent, offset_tiuCnt, offset_tdmaCnt,
                     pHeader->magic);
    } else if (pHeader->engine_id == 2) {
      TdmaReg::parse_tdma_reg(&tdmaReg, (uint32_t *)pContent, pHeader->magic);
      cur_layer_id = tdmaReg.layer_ID;
      if (cur_layer_id != last_layer_id) {
        offset_tiuCnt = tiuCnt;
        offset_tdmaCnt = tdmaCnt;
      }
      tdmaCnt++;
      reset_tdma_info((uint32_t *)pContent, offset_tiuCnt, offset_tdmaCnt,
                      pHeader->magic);
    } else {
      assert(0 && "Cmdbuf get no tdma and no tiu part");
    }
    if (cur_layer_id != last_layer_id) {
      // copy this slice
      uint32_t size = offset - last_offset;
      std::vector<uint8_t> slice_cmdbuf(size);
      memcpy(slice_cmdbuf.data(), cmdbuf.data() + last_offset, size);
      cmdbufs.emplace_back(std::move(slice_cmdbuf));
      // update
      last_layer_id = cur_layer_id;
      last_offset = offset;
    }
    pCurrent = pContent + pHeader->len;
    offset += sizeof(cmd_hdr_t) + pHeader->len;
  }

  // copy last slice
  uint32_t size = offset - last_offset;
  std::vector<uint8_t> slice_cmdbuf(size);
  memcpy(slice_cmdbuf.data(), cmdbuf.data() + last_offset, size);
  cmdbufs.emplace_back(std::move(slice_cmdbuf));

  // verify
  int count = 0;
  for (int i = 0; i < cmdbufs.size(); i++) {
    std::vector<uint8_t> slice = cmdbufs[i];
    for (int j = 0; j < slice.size(); j++) {
      assert(cmdbuf[count] == slice[j]);
      count++;
    }
  }
  assert(count == cmdbuf.size());
  return 0;
}

CVI_RC run_fun(CVI_RT_HANDLE ctx, std::vector<uint8_t> &cmdbuf,
               CVI_RT_ARRAYBASE &addrs, bool enable_pmu) {
  CVI_RT_MEM cmdbuf_mem;
  int ret = CVI_RT_LoadCmdbuf(ctx, cmdbuf.data(), cmdbuf.size(), 0, 0,
                              enable_pmu, &cmdbuf_mem);
  ret = CVI_RT_RunCmdbufEx(ctx, cmdbuf_mem, &addrs);
  if (ret != 0) {
    printf("runtimeExecuteKernelFunction failed ret[%d]\n", ret);
  }
  if (enable_pmu) {
    uint8_t *pmubuf = nullptr;
    uint32_t buf_len = 0;
    CVI_RT_ParsePmuBuf(cmdbuf_mem, &pmubuf, &buf_len);
  }
  CVI_RT_MemFree(ctx, cmdbuf_mem);
  return ret;
}

void printInfos(std::vector<std::vector<debug_info_t>> &tpu_debug_infos) {
  int num = tpu_debug_infos.size();
  llvm::errs() << "cmdbuf_num = " << num << "\n";
  for (int i = 0; i < num; i++) {
    std::vector<debug_info_t> debug_infos = tpu_debug_infos[i];
    int size = debug_infos.size();
    llvm::errs() << "layer_num = " << size << "\n";
    for (int j = 0; j < size; j++) {
      debug_info_t dinfo = debug_infos[j];
      llvm::errs() << dinfo.layer_id << " " << dinfo.name << " "
                   << dinfo.inGroup << " " << dinfo.ignore << " " << dinfo.gaddr
                   << " ";
      if (dinfo.g_shape.size() > 0) {
        for (int k = 0; k < dinfo.g_shape.size(); k++) {
          llvm::errs() << dinfo.g_shape[k] << ",";
        }
        llvm::errs() << " ";
      } else {
        llvm::errs() << "x ";
      }
      if (dinfo.g_shape.size() > 0) {
        for (int k = 0; k < dinfo.lg_idx_slice.size(); k++) {
          llvm::errs() << dinfo.lg_idx_slice[k] << ",";
        }
        llvm::errs() << " ";
      } else {
        llvm::errs() << "x ";
      }
      llvm::errs() << dinfo.laddr << " " << dinfo.qtype << " " << dinfo.qscale
                   << "\n";
    }
  }
}

void parseTensorInfo(char *fileName,
                     std::vector<std::vector<debug_info_t>> &tpu_debug_infos,
                     uint32_t &max_lg_gsize) {
  std::vector<debug_info_t> single_infos;
  std::ifstream istream;
  istream.open(fileName, std::ios::in);
  if (!istream.is_open()) {
    llvm_unreachable("cant't open tensor_info file");
  }
  std::string strLine;
  while (getline(istream, strLine)) {
    // llvm::errs()<<strLine<<"\n";
    if (strLine.substr(0, 6) == "Cmdbuf") {
      if (single_infos.size() > 0) {
        tpu_debug_infos.emplace_back(single_infos);
        single_infos.clear();
      }
      continue;
    }
    std::vector<std::string> first_infos;
    strSplit(strLine, " ", first_infos);
    assert(first_infos.size() == 10);
    debug_info_t dinfo;
    dinfo.layer_id = std::atoi(first_infos[0].c_str());
    dinfo.name = first_infos[1];
    dinfo.inGroup = (first_infos[2] == "true") ? true : false;
    dinfo.ignore = (first_infos[3] == "true") ? true : false;
    if (dinfo.ignore) {
      single_infos.emplace_back(dinfo);
      continue;
    }
    // parse global and local common attr
    std::vector<std::string> g_shapes;
    strSplit(first_infos[5], ",", g_shapes);
    uint32_t g_total_size = 1;
    for (int i = 0; i < g_shapes.size(); i++) {
      auto s = std::atol(g_shapes[i].c_str());
      dinfo.g_shape.emplace_back(s);
      g_total_size *= (uint32_t)s;
    }
    dinfo.qtype = first_infos[8];
    dinfo.qscale = std::atof(first_infos[9].c_str());
    if (dinfo.inGroup) {
      g_total_size = dinfo.qtype == "int8" ? g_total_size : g_total_size * 2;
      g_total_size = align_up(g_total_size, 16);
      if (g_total_size > max_lg_gsize) {
        max_lg_gsize = g_total_size;
      }
      std::vector<std::string> lg_slices;
      strSplit(first_infos[6], ",", lg_slices);
      for (int i = 0; i < lg_slices.size(); i++) {
        dinfo.lg_idx_slice.emplace_back(std::atol(lg_slices[i].c_str()));
      }
      dinfo.laddr = std::atol(first_infos[7].c_str());
    } else {
      dinfo.gaddr = std::atol(first_infos[4].c_str());
    }
    single_infos.emplace_back(dinfo);
  }
  tpu_debug_infos.emplace_back(single_infos);
  single_infos.clear();
  istream.close();
}

void parseCvimodel(char *fileName, std::vector<std::vector<uint8_t>> &cmdbuf,
                   std::vector<uint8_t> &weight, model_info &minfo) {
  // 1.parse cvimodel
  FileStream stream(fileName);
  flatbuffers::FlatBufferBuilder fbb(0);
  MODEL_HEADER header;
  if (stream.length() <= sizeof(header)) {
    printf("Error, invalid cvimodel file\n");
  }
  stream.read((uint8_t *)&header, 0, sizeof(header));
  size_t header_size;
  /* before version 1.1, heder size is 32 bytes */
  if (header.major == 1 && header.minor == 0)
    header_size = 0x20;
  else
    header_size = sizeof(MODEL_HEADER);
  size_t binary_offset = header_size + header.body_size;
  uint8_t *fb_model_buffer = new uint8_t[header.body_size];
  if (!fb_model_buffer) {
    printf("Failed to allocate memory\n");
  }
  stream.read(fb_model_buffer, header_size, header.body_size);
  const cvi::model::Model *fb_model;
  fb_model = cvi::model::GetModel(fb_model_buffer);
  minfo.chip = fb_model->target()->str();
  for (auto p : *fb_model->programs()) {
    // 2.get model inputs/outputs info
    minfo.private_gmem_size = p->private_gmem();
    minfo.shared_gmem_size = p->shared_gmem();
    auto &input_tensors = *p->input_tensors();
    auto &output_tensors = *p->output_tensors();
    auto getTensorByName = [&](std::string name) {
      for (const auto &t : *p->tensor_map()) {
        if (t->name()->str() == name) {
          return t;
        }
      }
      assert(0);
    };
    uint32_t io_mem_num = 0;
    for (int i = 0; i < input_tensors.size(); i++) {
      auto in_tensor = input_tensors[i];
      auto t = getTensorByName(in_tensor->str());
      auto gaddr = t->offset();
      auto memTypeIndx = (gaddr >> 40) & 0x07;
      auto &shape = *t->shape()->dim();
      uint32_t type_size = dtypeSize(t->dtype());
      uint32_t num_vals = (uint32_t)shape[0] * (uint32_t)shape[1] *
                          (uint32_t)shape[2] * (uint32_t)shape[3];
      uint32_t real_size = num_vals * type_size;
      uint32_t gmem_size = align_up(real_size, 16);
      if (memTypeIndx > 2) {
        io_mem_num += 1;
        minfo.io_sizes[memTypeIndx - 3] = gmem_size;
      }
      io_mem_info io_info;
      io_info.gaddr = gaddr;
      io_info.qscale = t->quant() ? t->quant()->qscale() : 1.0;
      // input_scale > 0, thus get its reciprocal
      io_info.qscale = 1.0 / io_info.qscale;
      io_info.type = dtypeToStr(t->dtype());
      io_info.name = in_tensor->str();
      io_info.g_shape = {shape[0], shape[1], shape[2], shape[3]};
      io_info.count = num_vals;
      io_info.size = real_size;
      if (real_size != t->size()) {
        // input vpss aligned
        llvm::errs() << "input_name:" << io_info.name << " aligned.\n";
        assert(io_info.type == "uint8");
        io_info.size = t->size();
        io_info.count = t->size();
        io_info.g_shape = {1, 1, 1, (int64_t)(io_info.size)};
      }
      minfo.inputs_gmem_info.emplace_back(io_info);
    }
    for (int i = 0; i < output_tensors.size(); i++) {
      auto out_tensor = output_tensors[i];
      auto t = getTensorByName(out_tensor->str());
      auto gaddr = t->offset();
      auto memTypeIndx = (gaddr >> 40) & 0x07;
      auto &shape = *t->shape()->dim();
      uint32_t type_size = dtypeSize(t->dtype());
      uint32_t num_vals = (uint32_t)shape[0] * (uint32_t)shape[1] *
                          (uint32_t)shape[2] * (uint32_t)shape[3];
      uint32_t real_size = num_vals * type_size;
      uint32_t gmem_size = align_up(real_size, 16);
      if (memTypeIndx > 2) {
        io_mem_num += 1;
        minfo.io_sizes[memTypeIndx - 3] = gmem_size;
      }
      io_mem_info io_info;
      io_info.gaddr = gaddr;
      io_info.qscale = t->quant() ? t->quant()->qscale() : 1.0;
      io_info.type = dtypeToStr(t->dtype());
      io_info.name = out_tensor->str();
      io_info.g_shape = {shape[0], shape[1], shape[2], shape[3]};
      io_info.count = num_vals;
      io_info.size = real_size;
      minfo.outs_gmem_info.emplace_back(io_info);
    }
    minfo.io_mem_num = io_mem_num;

    // 3.get cmdbuf and cpu_func_infos
    for (auto r : *p->routines()) {
      if (r->type() == cvi::model::RoutineType_CPU) {
        minfo.cpu_tpu_order.emplace_back(false);
        cpu_func_info func_info;
        func_info.func_name = r->cpu_routine()->function_section()->str();
        OpParam param;
        auto func_args = r->cpu_routine()->function_args();
        if (func_args) {
          handleFuncArgs(func_args->data(), param);
        }
        func_info.params = param;
        for (auto i : *r->in_tensors()) {
          auto name = i->str();
          io_mem_info io_info;
          io_info.name = "";
          for (const auto &t : *p->tensor_map()) {
            if (t->name()->str() == name) {
              io_info.name = name;
              io_info.gaddr = t->offset();
              io_info.qscale = t->quant() ? t->quant()->qscale() : 1.0;
              io_info.type = dtypeToStr(t->dtype());
              uint32_t type_size = dtypeSize(t->dtype());
              auto &shape = *t->shape()->dim();
              io_info.g_shape = {shape[0], shape[1], shape[2], shape[3]};
              io_info.count = (uint32_t)shape[0] * (uint32_t)shape[1] *
                              (uint32_t)shape[2] * (uint32_t)shape[3];
              io_info.size = io_info.count * type_size;
              break;
            }
          }
          if (io_info.name == "") {
            // input is weight
            for (auto w : *fb_model->weight_map()) {
              if (w->name()->str() == name) {
                io_info.name = w->name()->str();
                io_info.gaddr = w->offset();
                io_info.qscale = 1.0;
                uint32_t type_size = dtypeSize(w->type());
                io_info.type = dtypeToStr(w->type());
                auto &shape = *w->shape()->dim();
                io_info.g_shape = {shape[0], shape[1], shape[2], shape[3]};
                io_info.count = (uint32_t)shape[0] * (uint32_t)shape[1] *
                                (uint32_t)shape[2] * (uint32_t)shape[3];
                io_info.size = io_info.count * type_size;
                break;
              }
            }
          }
          assert(io_info.name != "");
          func_info.inputs.emplace_back(io_info);
        }
        for (auto out : *r->out_tensors()) {
          auto name = out->str();
          auto t = getTensorByName(name);
          io_mem_info io_info;
          io_info.name = name;
          io_info.gaddr = t->offset();
          io_info.qscale = t->quant() ? t->quant()->qscale() : 1.0;
          io_info.type = dtypeToStr(t->dtype());
          uint32_t type_size = dtypeSize(t->dtype());
          auto &shape = *t->shape()->dim();
          io_info.g_shape = {shape[0], shape[1], shape[2], shape[3]};
          io_info.count = (uint32_t)shape[0] * (uint32_t)shape[1] *
                          (uint32_t)shape[2] * (uint32_t)shape[3];
          io_info.size = io_info.count * type_size;
          func_info.outputs.emplace_back(io_info);
        }
        minfo.cpu_func_infos.emplace_back(func_info);
        continue;
      }
      minfo.cpu_tpu_order.emplace_back(true);
      std::string buf_name;
      std::string buf_type_name;
      if (r->tpu_routine()->cmdbuf_section()) {
        buf_name = r->tpu_routine()->cmdbuf_section()->str();
        buf_type_name = "cmdbuf";
      } else if (r->tpu_routine()->dmabuf_section()) {
        buf_name = r->tpu_routine()->dmabuf_section()->str();
        buf_type_name = "dmabuf";
      } else {
        assert(0 && "model has not cmdbuf and dmabuf");
      }
      for (auto s : *fb_model->sections()) {
        if (s->name()->str() == buf_name) {
          auto offset = s->offset();
          auto size = s->size();
          std::vector<uint8_t> one_cmdbuf(size);
          stream.read(one_cmdbuf.data(), binary_offset + offset, size);
          cmdbuf.emplace_back(one_cmdbuf);
          break;
        }
      }
    }

    // 4.get weight
    for (auto s : *fb_model->sections()) {
      if (s->type() != cvi::model::SectionType_WEIGHT) {
        continue;
      }
      auto offset = s->offset();
      auto size = s->size();
      weight.resize(size);
      stream.read(weight.data(), binary_offset + offset, size);
    }
  }
}

void setIOArrayBase(CVI_RT_ARRAYBASE &addrs, std::vector<uint64_t> io_paddrs,
                    int io_mem_num) {
  switch (io_mem_num) {
  case 5: {
    addrs.gaddr_base3 = io_paddrs[0];
    addrs.gaddr_base4 = io_paddrs[1];
    addrs.gaddr_base5 = io_paddrs[2];
    addrs.gaddr_base6 = io_paddrs[3];
    addrs.gaddr_base7 = io_paddrs[4];
    break;
  }
  case 4: {
    addrs.gaddr_base3 = io_paddrs[0];
    addrs.gaddr_base4 = io_paddrs[1];
    addrs.gaddr_base5 = io_paddrs[2];
    addrs.gaddr_base6 = io_paddrs[3];
    break;
  }
  case 3: {
    addrs.gaddr_base3 = io_paddrs[0];
    addrs.gaddr_base4 = io_paddrs[1];
    addrs.gaddr_base5 = io_paddrs[2];
    break;
  }
  case 2: {
    addrs.gaddr_base3 = io_paddrs[0];
    addrs.gaddr_base4 = io_paddrs[1];
    break;
  }
  case 1: {
    addrs.gaddr_base3 = io_paddrs[0];
    break;
  }
  default:
    assert(0);
    break;
  }
}

void copyLgData(std::vector<float> &dst_data, uint8_t *src_data,
                debug_info_t &dinfo) {
  auto g_shape = dinfo.g_shape;
  int64_t gc = g_shape[1], gh = g_shape[2], gw = g_shape[3];
  auto lg_info = dinfo.lg_idx_slice;
  int64_t n_idx = lg_info[0], n_slice = lg_info[1];
  int64_t h_idx = lg_info[2], h_slice = lg_info[3];
  std::string type = dinfo.qtype;
  if (n_slice > 1) {
    // just cut n_dim
    int start_idx = n_idx * gc * gh * gw;
    int total_size = n_slice * gc * gh * gw;
    if (type == "int8") {
      ConvertInt8ToFp32((int8_t *)src_data, dst_data.data() + start_idx,
                        total_size, dinfo.qscale);
    } else if (type == "bf16") {
      ConvertBF16ToFp32((uint16_t *)src_data, dst_data.data() + start_idx,
                        total_size);
    } else {
      assert(0 && "local op only support int8/bf16 data");
    }
  } else {
    // cut h_dim, n_dim maybe cut(each n_slice is 1)
    int start_idx = n_idx * gc * gh * gw + h_idx * gw;
    int stride = gh * gw;
    int block_size = h_slice * gw;
    for (int i = 0; i < gc; i++) {
      if (type == "int8") {
        ConvertInt8ToFp32((int8_t *)(src_data + i * stride),
                          dst_data.data() + start_idx + i * stride, block_size,
                          dinfo.qscale);
      } else if (type == "bf16") {
        ConvertBF16ToFp32(
            (uint16_t *)(src_data + i * stride * sizeof(uint16_t)),
            dst_data.data() + start_idx + i * stride, block_size);
      } else {
        assert(0 && "local op only support int8/bf16 data");
      }
    }
  }
}

void storeGlobalData(std::vector<float> &data, uint8_t *vaddr, int64_t offset,
                     std::string &type, uint32_t count, double qscale = 1.0) {
  if (type == "int8") {
    ConvertInt8ToFp32((int8_t *)(vaddr + offset), data.data(), count, qscale);
  } else if (type == "uint8") {
    ConvertUint8ToFp32((uint8_t *)(vaddr + offset), data.data(), count, qscale);
  } else if (type == "bf16") {
    ConvertBF16ToFp32((uint16_t *)(vaddr + offset), data.data(), count);
  } else if (type == "fp32") {
    memcpy(data.data(), (float *)(vaddr + offset), count * sizeof(float));
  } else {
    std::string err_msg = type + " not support in global op";
    llvm_unreachable(err_msg.c_str());
  }
}

void loadInput(std::string npzFile, std::vector<uint8_t *> io_vaddrs,
               model_info &minfo,
               std::map<std::string, std::vector<float>> &tensor_data,
               std::map<std::string, std::vector<size_t>> &tensor_shapes) {
  assert(npzFile.substr(npzFile.size() - 4) == ".npz" &&
         "input file should be npz format");
  llvm::errs() << "load_input\n";
  cnpy::npz_t input_npz = cnpy::npz_load(npzFile);
  int num_inputs = minfo.inputs_gmem_info.size();
  assert(num_inputs == (int)input_npz.size());
  for (auto &npy : input_npz) {
    auto &arr = npy.second;
    auto name = npy.first.c_str();
    int idx = 0;
    for (int i = 0; i < minfo.inputs_gmem_info.size(); i++) {
      if (minfo.inputs_gmem_info[i].name == std::string(name)) {
        idx = i;
        break;
      }
    }
    auto tensor = minfo.inputs_gmem_info[idx];
    auto memIdx = (tensor.gaddr >> 40) & 0x07;
    int io_idx = (int)memIdx - 3;
    std::vector<float> to_save_data(tensor.count);
    tensor_shapes[name] = {(size_t)tensor.g_shape[0], (size_t)tensor.g_shape[1],
                           (size_t)tensor.g_shape[2],
                           (size_t)tensor.g_shape[3]};
    if (arr.type == 'f') {
      assert(tensor.count == arr.num_vals);
      if (tensor.type == "fp32") {
        memcpy((float *)io_vaddrs[io_idx], arr.data<float>(), tensor.size);
        memcpy(to_save_data.data(), arr.data<float>(), tensor.size);
      } else if (tensor.type == "int8") {
        ConvertFp32ToInt8(arr.data<float>(), (int8_t *)io_vaddrs[io_idx],
                          tensor.count, tensor.qscale);
        memcpy(to_save_data.data(), arr.data<float>(),
               tensor.count * sizeof(float));
      } else if (tensor.type == "uint16") {
        ConvertFp32ToUint16(arr.data<float>(), (uint16_t *)io_vaddrs[io_idx],
                            tensor.count);
        memcpy(to_save_data.data(), arr.data<float>(),
               tensor.count * sizeof(uint16_t));
      } else if (tensor.type == "bf16") {
        ConvertFp32ToBF16(arr.data<float>(), (uint16_t *)io_vaddrs[io_idx],
                          tensor.count);
        memcpy(to_save_data.data(), arr.data<float>(),
               tensor.count * sizeof(uint16_t));
      } else if (tensor.type == "int32") {
        ConvertFp32ToInt32(arr.data<float>(), (int32_t *)io_vaddrs[io_idx],
                           tensor.count);
        memcpy(to_save_data.data(), arr.data<float>(),
               tensor.count * sizeof(int32_t));
      } else {
        assert(0 && "unsupport input type");
      }
    } else {
      assert(arr.num_bytes() == tensor.size);
      if (tensor.type == "uint16") {
        memcpy(io_vaddrs[io_idx], arr.data<uint16_t>(), tensor.size);
        ConvertUint16ToFp32(arr.data<uint16_t>(), to_save_data.data(),
                            tensor.count);
      } else if (tensor.type == "int32") {
        memcpy(io_vaddrs[io_idx], arr.data<int32_t>(), tensor.size);
        ConvertInt32ToFp32(arr.data<int32_t>(), to_save_data.data(),
                           tensor.count);
      } else if (tensor.type == "uint8") {
        // fuse preprocess
        memcpy(io_vaddrs[io_idx], arr.data<uint8_t>(), tensor.size);
        ConvertUint8ToFp32(arr.data<uint8_t>(), to_save_data.data(),
                           tensor.count, tensor.qscale);
      } else {
        assert(0 && "unsupport input type");
      }
    }
    tensor_data[name] = to_save_data;
  }
}

void saveResult(std::string npzFile,
                std::map<std::string, std::vector<float>> &tensor_data,
                std::map<std::string, std::vector<size_t>> &tensor_shapes) {
  assert(npzFile.substr(npzFile.size() - 4) == ".npz" &&
         "output file should be npz format");
  assert(tensor_data.size() == tensor_shapes.size());
  cnpy::npz_t npz;
  std::map<std::string, std::vector<float>>::iterator it;
  for (it = tensor_data.begin(); it != tensor_data.end(); it++) {
    const std::string name = it->first;
    const std::vector<float> data = it->second;
    cnpy::npz_add_array<float>(npz, name, data.data(), tensor_shapes[name]);
  }
  cnpy::npz_save_all(npzFile, npz);
}
} // namespace cvi_debug
