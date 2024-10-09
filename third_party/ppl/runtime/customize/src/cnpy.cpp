//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php
#define _FILE_OFFSET_BITS 64
#define __USE_FILE_OFFSET64
#define __USE_LARGEFILE64
#define _LARGEFILE64_SOURCE

#include"cnpy.h"
#include<complex>
#include<cstdlib>
#include<algorithm>
#include<cstring>
#include<iomanip>
#include<stdint.h>
#include<stdexcept>
#include <regex>

#define ZIP64_LIMIT  ((((size_t)1) << 31) - 1)

namespace cnpy {

static char BigEndianTest() {
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
}

template <typename T>
struct mapType;

#define DEFMAPTYPE(type, typeIdentifier)                                       \
  template <>                                                                  \
  struct mapType<type> {                                                       \
    static constexpr char value = typeIdentifier;                              \
  };

DEFMAPTYPE(float, 'f')
DEFMAPTYPE(double, 'f')
DEFMAPTYPE(long double, 'f')
DEFMAPTYPE(int, 'i')
DEFMAPTYPE(char, 'i')
DEFMAPTYPE(signed char, 'i')
DEFMAPTYPE(short, 'i')
DEFMAPTYPE(long, 'i')
DEFMAPTYPE(long long, 'i')
DEFMAPTYPE(unsigned char, 'u')
DEFMAPTYPE(unsigned short, 'u')
DEFMAPTYPE(unsigned long, 'u')
DEFMAPTYPE(unsigned long long, 'u')
DEFMAPTYPE(unsigned int, 'u')
DEFMAPTYPE(bool, 'b')
DEFMAPTYPE(std::complex<float>, 'c')
DEFMAPTYPE(std::complex<double>, 'c')
DEFMAPTYPE(std::complex<long double>, 'c')

template<typename T>
std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
    //write in little endian
    for(size_t byte = 0; byte < sizeof(T); byte++) {
        char val = *((const char*)&rhs+byte);
        lhs.push_back(val);
    }
    return lhs;
}

template<>
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs) {
    lhs.insert(lhs.end(),rhs.begin(),rhs.end());
    return lhs;
}

template<>
std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs) {
    //write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for(size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

std::vector<char> create_npy_header(const std::vector<size_t>& shape,
        size_t word_size, char type) {
    std::vector<char> dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += type;
    dict += std::to_string(word_size);
    dict += "', 'fortran_order': False, 'shape': (";
    dict += std::to_string(shape[0]);
    for(size_t i = 1;i < shape.size();i++) {
        dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if(shape.size() == 1) dict += ",";
    dict += "), }";
    //pad with spaces so that preamble+dict is modulo 16 bytes.
    //preamble is 10 bytes. dict needs to end with \n
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(),remainder,' ');
    dict.back() = '\n';

    std::vector<char> header;
    header += (char) 0x93;
    header += "NUMPY";
    header += (char) 0x01; //major version of numpy format
    header += (char) 0x00; //minor version of numpy format
    header += (uint16_t) dict.size();
    header.insert(header.end(),dict.begin(),dict.end());

    return header;
}

void parse_npy_header(unsigned char* buffer, size_t& word_size,  char& type,
        std::vector<size_t>& shape, bool& fortran_order) {
    //std::string magic_string(buffer,6);
    //uint8_t major_version = *reinterpret_cast<uint8_t*>(buffer+6);
    //uint8_t minor_version = *reinterpret_cast<uint8_t*>(buffer+7);
    uint16_t header_len = *reinterpret_cast<uint16_t*>(buffer+8);
    std::string header(reinterpret_cast<char*>(buffer+9),header_len);

    size_t loc1, loc2;

    //fortran order
    loc1 = header.find("fortran_order")+16;
    fortran_order = (header.substr(loc1,4) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1+1,loc2-loc1-1);
    while(std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    //endian, word size, data type
    //byte order code | stands for not applicable.
    //not sure when this applies except for byte array
    loc1 = header.find("descr")+9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    type = header[loc1+1];
    //assert(type == mapType(T)::value);

    std::string str_ws = header.substr(loc1+2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0,loc2).c_str());
}

void parse_npy_header(FILE* fp, size_t& word_size, char& type,
        std::vector<size_t>& shape, bool& fortran_order) {
    char buffer[256];
    size_t res = fread(buffer,sizeof(char),11,fp);
    if(res != 11)
        throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer,256,fp);
    assert(header[header.size()-1] == '\n');

    size_t loc1, loc2;

    //fortran order
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: "
                "failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1,4) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: "
                "failed to find header keyword: '(' or ')'");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1+1,loc2-loc1-1);
    while(std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    //endian, word size, data type
    //byte order code | stands for not applicable.
    //not sure when this applies except for byte array
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: "
                "failed to find header keyword: 'descr'");
    loc1 += 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    type = header[loc1+1];
    //assert(type == mapType(T));

    std::string str_ws = header.substr(loc1+2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0,loc2).c_str());
}

void parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size,
        size_t& global_header_offset) {
    std::vector<char> footer(22);
    fseek(fp,-22,SEEK_END);
    size_t res = fread(&footer[0],sizeof(char),22,fp);
    if(res != 22)
        throw std::runtime_error("parse_zip_footer: failed fread");

    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no = *(uint16_t*) &footer[4];
    disk_start = *(uint16_t*) &footer[6];
    nrecs_on_disk = *(uint16_t*) &footer[8];
    nrecs = *(uint16_t*) &footer[10];
    global_header_size = *(uint32_t*) &footer[12];
    global_header_offset = *(uint32_t*) &footer[16];
    comment_len = *(uint16_t*) &footer[20];

    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
    if (global_header_offset >= 0xFFFFFFFF) {
      //get global header offset from extra data
      std::vector<char> zip64endrec_header(56);
      fseek(fp,-98,SEEK_END);
      size_t result = fread(&zip64endrec_header[0], sizeof(char), 56, fp);
      if (result != 56) {
        fprintf(stderr,
                "Error reading file or unexpected end of file encountered.\n");
      }
      global_header_offset = *(uint64_t*) &zip64endrec_header[48];
    }
}

template<typename T>
void npy_save(std::string fname, const T* data,
        const std::vector<size_t> shape, std::string mode) {
    FILE* fp = NULL;
    //if appending, the shape of existing + new data
    std::vector<size_t> true_data_shape;

    if(mode == "a") fp = fopen(fname.c_str(),"r+b");

    if(fp) {
        //file exists. we need to append to it. read the header, modify the array size
        size_t word_size;
        char type;
        bool fortran_order;
        parse_npy_header(fp,word_size,type,true_data_shape,fortran_order);
        assert(!fortran_order);

        if(word_size != sizeof(T)) {
            std::cout << "libnpy error: " << fname << " has word size "
                      << word_size << " but npy_save appending data sized "
                      << sizeof(T) << "\n";
            assert( word_size == sizeof(T) );
        }
        if(true_data_shape.size() != shape.size()) {
            std::cout << "libnpy error: npy_save attempting to append "
                      << "misdimensioned data to " << fname << "\n";
            assert(true_data_shape.size() != shape.size());
        }

        for(size_t i = 1; i < shape.size(); i++) {
            if(shape[i] != true_data_shape[i]) {
                std::cout << "libnpy error: npy_save attempting to append "
                          << "misshaped data to " << fname << "\n";
                assert(shape[i] == true_data_shape[i]);
            }
        }
        true_data_shape[0] += shape[0];
    }
    else {
        fp = fopen(fname.c_str(),"wb");
        true_data_shape = shape;
    }

    size_t word_size = sizeof(T);
    char type = mapType<T>::value;
    std::vector<char> header = create_npy_header(true_data_shape, word_size, type);
    size_t nels = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<size_t>());

    fseek(fp,0,SEEK_SET);
    fwrite(&header[0],sizeof(char),header.size(),fp);
    fseek(fp,0,SEEK_END);
    fwrite(data,sizeof(T),nels,fp);
    fclose(fp);
}

template void npy_save<std::complex<double> >(std::string,
        const std::complex<double>*,
        const std::vector<size_t>, std::string);
template void npy_save<double>(std::string, const double*,
        const std::vector<size_t>, std::string);
template void npy_save<char>(std::string, const char*,
        const std::vector<size_t>, std::string);

template<typename T>
void npy_save(std::string fname, const std::vector<T> data,
        std::string mode) {
    std::vector<size_t> shape;
    shape.push_back(data.size());
    npy_save<T>(fname, &data[0], shape, mode);
}

template<typename T>
void npz_save(std::string zipname, std::string fname,
        const T* data, const std::vector<size_t>& shape,
        std::string mode) {
    //first, append a .npy to the fname
    fname += ".npy";

    //now, on with the show
    FILE* fp = NULL;
    uint16_t nrecs = 0;
    size_t global_header_offset = 0;
    std::vector<char> global_header;

    if(mode == "a") fp = fopen(zipname.c_str(),"r+b");

    if(fp) {
        //zip file exists. we need to add a new npy file to it.
        //first read the footer.
        //this gives us the offset and size of the global header
        //then read and store the global header.
        //below, we will write the the new data at the start of the global
        //header then append the global header and footer below it
        size_t global_header_size;
        parse_zip_footer(fp,nrecs,global_header_size,global_header_offset);
        fseek(fp,global_header_offset,SEEK_SET);
        global_header.resize(global_header_size);
        size_t res = fread(&global_header[0],sizeof(char),global_header_size,fp);
        if(res != global_header_size){
            throw std::runtime_error("npz_save: "
                    "header read error while adding to existing zip");
        }
        fseek(fp,global_header_offset,SEEK_SET);
    }
    else {
        fp = fopen(zipname.c_str(),"wb");
    }

    size_t word_size = sizeof(T);
    char type = mapType<T>::value;
    std::vector<char> npy_header;
    if(shape.size() != 0){
        npy_header = create_npy_header(shape, word_size, type);
    }else{
        std::cerr << "[Warning] zip name: " << fname <<" npz shape size is 0, skip it\n";
        fclose(fp);
        return;
    }

    size_t nels = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<size_t>());
    size_t nbytes = nels*sizeof(T) + npy_header.size();

    //get the CRC of the data to be added
    uint32_t crc = crc32(0L,(uint8_t*)&npy_header[0],npy_header.size());
    crc = crc32(crc,(const uint8_t*)data,nels*sizeof(T));

    //build the local header
    std::vector<char> local_header;
    local_header += "PK"; //first part of sig
    local_header += (uint16_t) 0x0403; //second part of sig
    local_header += (uint16_t) 20; //min version to extract
    local_header += (uint16_t) 0; //general purpose bit flag
    local_header += (uint16_t) 0; //compression method
    local_header += (uint16_t) 0; //file last mod time
    local_header += (uint16_t) 0;     //file last mod date
    local_header += (uint32_t) crc; //crc
    local_header += (uint32_t) nbytes; //compressed size
    local_header += (uint32_t) nbytes; //uncompressed size
    local_header += (uint16_t) fname.size(); //fname length
    local_header += (uint16_t) 0; //extra field length
    local_header += fname;

    fwrite(&local_header[0],sizeof(char),local_header.size(),fp);
    fwrite(&npy_header[0],sizeof(char),npy_header.size(),fp);
    fwrite(data,sizeof(T),nels,fp);
    /*
      Only support global_header_offset is larger than ZIP64_LIMIT.
      Not support size is larger than ZIP64_LIMIT now.
    */
    if (global_header_offset + nbytes + local_header.size() >= ZIP64_LIMIT) {
      //structCentralDir = "<4s4B4HL2L5H2L"
      //centdir = struct.pack(structCentralDir,
      //stringCentralDir, create_version,
      //zinfo.create_system, extract_version, zinfo.reserved,
      //flag_bits, zinfo.compress_type, dostime, dosdate,
      //zinfo.CRC, compress_size, file_size,
      //len(filename), len(extra_data), len(zinfo.comment),
      //0, zinfo.internal_attr, zinfo.external_attr,
      //header_offset)

      //build global header
      global_header += "PK"; //first part of sig
      global_header += (uint16_t) 0x0201; //second part of sig
      global_header += (uint8_t) 45; //create_version
      global_header += (uint8_t) 3; //zinfo.create_system
      global_header += (uint8_t) 45; //extract_version
      global_header += (uint8_t) 0; //zinfo.reserved
      global_header.insert(global_header.end(),local_header.begin()+6,
                           local_header.begin()+28);
      global_header += (uint16_t) 12; //extran data length
      global_header += (uint16_t) 0; //file comment length
      global_header += (uint16_t) 0; //disk number where file starts
      global_header += (uint16_t) 0; //internal file attributes
      global_header += (uint32_t) 0; //external file attributes
      //relative offset of local file header
      //since it begins where the global header used to begin
      global_header += (uint32_t) 0xFFFFFFFF ; //global_header_offset;
      global_header += fname;
      // Append a ZIP64 field to the extra's
      // extra_data = struct.pack(
      //         '<HH' + 'Q'*len(extra),
      //         1, 8*len(extra), *extra) + extra_data
      // extract_version = max(45, zinfo.extract_version)
      // create_version = max(45, zinfo.create_version)
      global_header += (uint16_t) 0x01;
      global_header += (uint16_t) 0x08;
      global_header += (uint64_t) global_header_offset;
    } else {
      //build global header
      global_header += "PK"; //first part of sig
      global_header += (uint16_t) 0x0201; //second part of sig
      global_header += (uint16_t) 20; //version made by
      global_header.insert(global_header.end(),local_header.begin()+4,
                           local_header.begin()+30);
      global_header += (uint16_t) 0; //file comment length
      global_header += (uint16_t) 0; //disk number where file starts
      global_header += (uint16_t) 0; //internal file attributes
      global_header += (uint32_t) 0; //external file attributes
      //relative offset of local file header
      //since it begins where the global header used to begin
      global_header += (uint32_t) global_header_offset;
      global_header += fname;
    }

    fwrite(&global_header[0],sizeof(char),global_header.size(),fp);

    if (global_header_offset >= ZIP64_LIMIT) {
      //structEndArchive64 = "<4sQ2H2L4Q"
      //zip64endrec = struct.pack(
      //        structEndArchive64, stringEndArchive64,
      //        44, 45, 45, 0, 0, centDirCount, centDirCount,
      //        centDirSize, centDirOffset)
      //self.fp.write(zip64endrec)
      std::vector<char> zip64endrec_header;
      zip64endrec_header += "PK";
      zip64endrec_header += (uint16_t) 0x0606;
      zip64endrec_header += (uint64_t) 0x44;
      zip64endrec_header += (uint16_t) 0x45;
      zip64endrec_header += (uint16_t) 0x45;
      zip64endrec_header += (uint32_t) 0x0;
      zip64endrec_header += (uint32_t) 0x0;
      zip64endrec_header += (uint64_t) (nrecs+1); //centDirCount
      zip64endrec_header += (uint64_t) (nrecs+1); //centDirCount
      zip64endrec_header += (uint64_t) global_header.size(); //centDirSize
      zip64endrec_header += (uint64_t) global_header_offset + nbytes + local_header.size(); //centDirOffset
      fwrite(&zip64endrec_header[0],sizeof(char),zip64endrec_header.size(),fp);

      //structEndArchive64Locator = "<4sLQL"
      //zip64locrec = struct.pack(
      //        structEndArchive64Locator,
      //        stringEndArchive64Locator, 0, pos2, 1)
      //self.fp.write(zip64locrec)
      std::vector<char> zip64locrec_header;
      zip64locrec_header += "PK";
      zip64locrec_header += (uint16_t) 0x0706;
      zip64locrec_header += (uint32_t) 0x0;
      zip64locrec_header += (uint64_t) global_header_offset + nbytes + local_header.size() +
                             zip64endrec_header.size(); // zip64endrec_header offset
      zip64locrec_header += (uint32_t) 0x1;
      fwrite(&zip64locrec_header[0],sizeof(char),zip64locrec_header.size(),fp);
    }
    //build footer
    std::vector<char> footer;
    footer += "PK"; //first part of sig
    footer += (uint16_t) 0x0605; //second part of sig
    footer += (uint16_t) 0; //number of this disk
    footer += (uint16_t) 0; //disk where footer starts
    footer += (uint16_t) (nrecs+1); //number of records on this disk
    footer += (uint16_t) (nrecs+1); //total number of records
    footer += (uint32_t) global_header.size(); //nbytes of global headers
    //offset of start of global headers
    //since global header now starts after newly written array
    footer += (global_header_offset >= ZIP64_LIMIT) ?
               (uint32_t) 0xFFFFFFFF : (uint32_t) (global_header_offset + nbytes + local_header.size());
    footer += (uint16_t) 0; //zip file comment length

    fwrite(&footer[0],sizeof(char),footer.size(),fp);
    fclose(fp);
}

template void npz_save<std::complex<double> >(std::string, std::string,
        const std::complex<double>*, const std::vector<size_t>&,
        std::string);
template void npz_save<double>(std::string, std::string,
        const double*, const std::vector<size_t>&, std::string);
template void npz_save<char>(std::string, std::string,
        const char*, const std::vector<size_t>&, std::string);

template<typename T>
void npz_save(std::string zipname, std::string fname,
        const std::vector<T> &data, std::string mode) {
    std::vector<size_t> shape;
    shape.push_back(data.size());
    npz_save(zipname, fname, &data[0], shape, mode);
}

template<typename T>
void npz_save(std::string zipname, std::string fname,
        NpyArray &array, std::string mode) {
    npz_save<T>(zipname, fname, array.data<T>(), array.shape, mode);
}

template<typename T>
void npz_add_array(npz_t &map, std::string fname,
        const T* data, const std::vector<size_t> shape) {
    size_t word_size = sizeof(T);
    char type = mapType<T>::value;
    bool fortran_order = false;
    NpyArray array(shape, word_size, type, fortran_order);
    memcpy(array.data<unsigned char>(), data, array.num_bytes());
    map[fname] = array;
}

void npz_clone_array(npz_t &map, std::string fname, std::string new_name) {
    auto array = map[fname];
    map[new_name] = array;
}

template void npz_add_array<std::complex<double> >(npz_t &, std::string,
        const std::complex<double>*, const std::vector<size_t>);
template void npz_add_array<float>(npz_t &, std::string,
        const float*, const std::vector<size_t>);
template void npz_add_array<int8_t>(npz_t &, std::string,
        const int8_t*, const std::vector<size_t>);
template void npz_add_array<uint8_t>(npz_t &, std::string,
        const uint8_t*, const std::vector<size_t>);
template void npz_add_array<int16_t>(npz_t &, std::string,
        const int16_t*, const std::vector<size_t>);
template void npz_add_array<uint16_t>(npz_t &, std::string,
        const uint16_t*, const std::vector<size_t>);
template void npz_add_array<uint32_t>(npz_t &, std::string,
        const uint32_t*, const std::vector<size_t>);
template void npz_add_array<int32_t>(npz_t &, std::string,
        const int32_t*, const std::vector<size_t>);

template<typename T>
void npz_add_array(npz_t &map, std::string fname,
        const std::vector<T> &data) {
    std::vector<size_t> shape;
    shape.push_back(data.size());
    npz_add_array(map, fname, &data[0], shape);
}

template void npz_add_array<std::complex<double> >(npz_t &, std::string,
        const std::vector<std::complex<double> > &);
template void npz_add_array<float>(npz_t &, std::string,
        const std::vector<float> &);
template void npz_add_array<int8_t>(npz_t &, std::string,
        const std::vector<int8_t> &);
template void npz_add_array<uint8_t>(npz_t &, std::string,
        const std::vector<uint8_t> &);
template void npz_add_array<int16_t>(npz_t &, std::string,
        const std::vector<int16_t> &);
template void npz_add_array<uint16_t>(npz_t &, std::string,
        const std::vector<uint16_t> &);
template void npz_add_array<int32_t>(npz_t &, std::string,
        const std::vector<int32_t> &);

void npz_save_all(std::string zipname, npz_t &map) {
    for (auto it = map.begin(); it != map.end(); it++) {
        std::string mode = (it == map.begin()) ? "w" : "a";
        NpyArray &arr = it->second;
        if (arr.type == 'f') {
            // support float only for now
            assert(arr.word_size = sizeof(float));
            npz_save<float>(zipname, it->first, it->second, mode);
        } else if (arr.type == 'i') {
            // support int8/int16/int32 only
            if (arr.word_size == sizeof(int8_t)) {
                npz_save<int8_t>(zipname, it->first, it->second, mode);
            } else if (arr.word_size == sizeof(int16_t)) {
                npz_save<int16_t>(zipname, it->first, it->second, mode);
            } else if (arr.word_size == sizeof(int32_t)) {
                npz_save<int32_t>(zipname, it->first, it->second, mode);
            } else {
                assert(0);
            }
        } else if (arr.type == 'u') {
            // support uint8/uint16/uint32
            if (arr.word_size == sizeof(uint8_t)) {
                npz_save<uint8_t>(zipname, it->first, it->second, mode);
            } else if (arr.word_size == sizeof(uint16_t)) {
                npz_save<uint16_t>(zipname, it->first, it->second, mode);
            } else if (arr.word_size == sizeof(uint32_t)) {
                npz_save<uint32_t>(zipname, it->first, it->second, mode);
            } else {
                assert(0);
            }
        } else if (arr.type == 'b') {
            // not support yet
            assert(0);
        } else if (arr.type == 'c') {
            // not support yet
            assert(0);
        } else {
            // invalid type
            std::cout << "libcnpy error: invalid array type "
                      << arr.type << ", for " << it->first << "\n";
            assert(0);
        }
    }
}

static NpyArray load_the_npy_file(FILE* fp) {
    std::vector<size_t> shape;
    size_t word_size;
    char type;
    bool fortran_order;
    parse_npy_header(fp,word_size,type,shape,fortran_order);

    NpyArray arr(shape, word_size, type, fortran_order);
    size_t nread = fread(arr.data<char>(),1,arr.num_bytes(),fp);
    if(nread != arr.num_bytes())
        throw std::runtime_error("load_the_npy_file: failed fread");
    return arr;
}

static NpyArray load_the_npz_array(FILE* fp, uint32_t compr_bytes,
        uint32_t uncompr_bytes) {
    std::vector<unsigned char> buffer_compr(compr_bytes);
    std::vector<unsigned char> buffer_uncompr(uncompr_bytes);
    size_t nread = fread(&buffer_compr[0],1,compr_bytes,fp);
    if(nread != compr_bytes)
        throw std::runtime_error("load_the_npy_file: failed fread");

    int err;
    z_stream d_stream;

    d_stream.zalloc = Z_NULL;
    d_stream.zfree = Z_NULL;
    d_stream.opaque = Z_NULL;
    d_stream.avail_in = 0;
    d_stream.next_in = Z_NULL;
    err = inflateInit2(&d_stream, -MAX_WBITS);
    assert(err = 0);

    d_stream.avail_in = compr_bytes;
    d_stream.next_in = &buffer_compr[0];
    d_stream.avail_out = uncompr_bytes;
    d_stream.next_out = &buffer_uncompr[0];

    err = inflate(&d_stream, Z_FINISH);
    assert(err = 0);
    err = inflateEnd(&d_stream);
    assert(err = 0);

    std::vector<size_t> shape;
    size_t word_size;
    char type;
    bool fortran_order;
    parse_npy_header(&buffer_uncompr[0],word_size,type,shape,fortran_order);

    NpyArray array(shape, word_size, type, fortran_order);

    size_t offset = uncompr_bytes - array.num_bytes();
    memcpy(array.data<unsigned char>(),&buffer_uncompr[0]+offset,array.num_bytes());

    return array;
}

npz_t npz_load(std::string fname) {
    npz_t arrays;
    arrays.clear();

    FILE* fp = fopen(fname.c_str(),"rb");
    if(!fp) {
        //throw std::runtime_error("npz_load: Error! Unable to open file "+fname+"!");
        return arrays;
    }

    while(1) {
        std::vector<char> local_header(30);
        size_t headerres = fread(&local_header[0],sizeof(char),30,fp);
        if(headerres != 30)
            break;

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        std::string varname(name_len,' ');
        size_t vname_res = fread(&varname[0],sizeof(char),name_len,fp);
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");

        //erase the lagging .npy
        varname.erase(varname.end()-4,varname.end());

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        if(extra_field_len > 0) {
            std::vector<char> buff(extra_field_len);
            size_t efield_res = fread(&buff[0],sizeof(char),extra_field_len,fp);
            if(efield_res != extra_field_len)
                throw std::runtime_error("npz_load: failed fread");
        }

        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[0]+8);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+22);

        if(compr_method == 0) {arrays[varname] = load_the_npy_file(fp);}
        else {arrays[varname] = load_the_npz_array(fp,compr_bytes,uncompr_bytes);}
    }

    fclose(fp);
    return arrays;
}

npz_t npz_load(FILE* fp) {
    npz_t arrays;
    arrays.clear();

    if(!fp) {
        //throw std::runtime_error("npz_load: Error! Unable to open file "+fname+"!");
        return arrays;
    }

    while(1) {
        std::vector<char> local_header(30);
        size_t headerres = fread(&local_header[0],sizeof(char),30,fp);
        if(headerres != 30)
            break;

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        std::string varname(name_len,' ');
        size_t vname_res = fread(&varname[0],sizeof(char),name_len,fp);
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");

        //erase the lagging .npy
        varname.erase(varname.end()-4,varname.end());

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        if(extra_field_len > 0) {
            std::vector<char> buff(extra_field_len);
            size_t efield_res = fread(&buff[0],sizeof(char),extra_field_len,fp);
            if(efield_res != extra_field_len)
                throw std::runtime_error("npz_load: failed fread");
        }

        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[0]+8);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+22);

        if(compr_method == 0) {arrays[varname] = load_the_npy_file(fp);}
        else {arrays[varname] = load_the_npz_array(fp,compr_bytes,uncompr_bytes);}
    }

    fclose(fp);
    return arrays;
}

NpyArray npz_load(std::string fname, std::string varname) {
    FILE* fp = fopen(fname.c_str(),"rb");

    if(!fp) throw std::runtime_error("npz_load: Unable to open file "+fname);

    while(1) {
        std::vector<char> local_header(30);
        size_t header_res = fread(&local_header[0],sizeof(char),30,fp);
        if(header_res != 30)
            throw std::runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        std::string vname(name_len,' ');
        size_t vname_res = fread(&vname[0],sizeof(char),name_len,fp);
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");
        vname.erase(vname.end()-4,vname.end()); //erase the lagging .npy

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        fseek(fp,extra_field_len,SEEK_CUR); //skip past the extra field

        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[0]+8);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+22);

        if(vname == varname) {
            NpyArray array  = (compr_method == 0) ? load_the_npy_file(fp)
                              : load_the_npz_array(fp,compr_bytes,uncompr_bytes);
            fclose(fp);
            return array;
        }
        else {
            //skip past the data
            uint32_t size = *(uint32_t*) &local_header[22];
            fseek(fp,size,SEEK_CUR);
        }
    }

    fclose(fp);

    //if we get here, we haven't found the variable in the file
    throw std::runtime_error("npz_load: Variable name "+varname+" not found in "+fname);
}

NpyArray npy_load(std::string fname) {

    FILE* fp = fopen(fname.c_str(), "rb");

    if(!fp) throw std::runtime_error("npy_load: Unable to open file "+fname);

    NpyArray arr = load_the_npy_file(fp);

    fclose(fp);
    return arr;
}

} // namespace cnpy
