//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include<string>
#include<cstring>
#include<stdexcept>
#include<sstream>
#include<vector>
#include<cstdio>
#include<typeinfo>
#include<iostream>
#include<cassert>
#include</usr/include/zlib.h>
#include<map>
#include<memory>
#include<stdint.h>
#include<numeric>

namespace cnpy {

struct NpyArray {
    NpyArray(const std::vector<size_t>& _shape, size_t _word_size,
        char _type, bool _fortran_order)
        : shape(_shape), word_size(_word_size),
          type(_type), fortran_order(_fortran_order) {
        num_vals = 1;
        for(size_t i = 0;i < shape.size();i++) num_vals *= shape[i];
        data_holder = std::shared_ptr<std::vector<char>>(
            new std::vector<char>(num_vals * word_size));
    }

    NpyArray() : shape(0), word_size(0), type(0), fortran_order(0), num_vals(0) {}

    template<typename T>
    T* data() {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template<typename T>
    const T* data() const {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template<typename T>
    std::vector<T> as_vec() const {
        const T* p = data<T>();
        return std::vector<T>(p, p+num_vals);
    }

    size_t num_bytes() const {
        return data_holder->size();
    }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t> shape;
    size_t word_size;
    char type;
    bool fortran_order;
    size_t num_vals;
};

using npz_t = std::map<std::string, NpyArray>;

std::vector<char> create_npy_header(const std::vector<size_t>& shape,
    size_t word_size, char type);
void parse_npy_header(FILE* fp,size_t& word_size, char& type,
        std::vector<size_t>& shape, bool& fortran_order);
void parse_npy_header(unsigned char* buffer, size_t& word_size, char& type,
        std::vector<size_t>& shape, bool& fortran_order);
void parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size,
        size_t& global_header_offset);
npz_t npz_load(FILE* fp);
npz_t npz_load(std::string fname);
NpyArray npz_load(std::string fname, std::string varname);
NpyArray npy_load(std::string fname);

template<typename T>
std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs);
template<>
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
template<>
std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);

template<typename T>
void npy_save(std::string fname, const T* data,
        const std::vector<size_t> shape, std::string mode = "w");
template<typename T>
void npy_save(std::string fname, const std::vector<T> data,
        std::string mode = "w");

template<typename T>
void npz_save(std::string zipname, std::string fname,
        const T* data, const std::vector<size_t>& shape,
        std::string mode = "w");
template<typename T>
void npz_save(std::string zipname, std::string fname,
        const std::vector<T> &data, std::string mode = "w");
template<typename T>
void npz_save(std::string zipname, std::string fname,
        NpyArray &array, std::string mode = "w");


template<typename T>
void npz_add_array(npz_t &map, std::string fname,
        const T* data, const std::vector<size_t> shape);
template<typename T>
void npz_add_array(npz_t &map, std::string fname,
        const std::vector<T> &data);

void npz_clone_array(npz_t &map, std::string fname, std::string new_name);

void npz_save_all(std::string zipname, npz_t &map);

} // namespace cnpy

#endif
