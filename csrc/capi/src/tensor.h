#pragma once

#include "flash_attn.h"

#include <vector>
#include <iostream>
#include <initializer_list>
#include <string>


struct InternalTensor{

    enum class DataType: unsigned {
        FP16 = 0,
        BF16,
        INT8,
        INT32,
        INT64,
        FP32,
        FP64,
        NONE,
    };

    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    DataType dtype;
    void *data;

    InternalTensor(void *data_,mcflashattnDataType_t dtype_,std::initializer_list<int64_t> list):
        data(data_){

        switch(dtype_){
            case MCFLASHATTN_DATATYPE_FP16:  dtype = DataType::FP16; break;
            case MCFLASHATTN_DATATYPE_BF16:  dtype = DataType::BF16; break;
            case MCFLASHATTN_DATATYPE_INT8:  dtype = DataType::INT8; break;
            case MCFLASHATTN_DATATYPE_INT32: dtype = DataType::INT32; break;
            case MCFLASHATTN_DATATYPE_INT64: dtype = DataType::INT64; break;
            case MCFLASHATTN_DATATYPE_FP32:  dtype = DataType::FP32; break;
            case MCFLASHATTN_DATATYPE_FP64:  dtype = DataType::FP64; break;

            default: dtype = DataType::NONE; break;
        }

        for(auto sz : list){
            sizes.push_back(sz);
        }

        auto count = sizes.size();
        strides.resize(count);
        if (count == 0) { return; }
        strides[count - 1] = 1;

        for(int i = count - 2; i >= 0; --i){
            int64_t stride = sizes[i + 1] * strides[i + 1];
            strides[i] = stride;
        }
    }

    void print(){

        std::string dtype_str;

        switch(dtype){
            case DataType::FP16:  dtype_str = "FP16"; break;
            case DataType::BF16:  dtype_str = "BF16"; break;
            case DataType::INT8:  dtype_str = "INT8"; break;
            case DataType::INT32: dtype_str = "INT32"; break;
            case DataType::INT64: dtype_str = "INT64"; break;
            case DataType::FP32:  dtype_str = "FP32"; break;
            case DataType::FP64:  dtype_str = "FP64"; break;

            default: dtype_str = "NONE"; break;
        }

        std::cout << "------------------------------------" << std::endl;
        std::cout << "type:" << dtype_str << std::endl;

        std::cout << "size:[";
        for(size_t i = 0; i < sizes.size(); ++ i) {
            if (i > 0) { std::cout << ","; }
            std::cout << sizes[i];
        }
        std::cout << "]" << std::endl;

        std::cout << "stride:[";
        for(size_t i = 0; i < strides.size(); ++ i) {
            if (i > 0) { std::cout << ","; }
            std::cout << strides[i];
        }
        std::cout << "]" << std::endl;

    }
};
