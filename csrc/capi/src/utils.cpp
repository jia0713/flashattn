#include "utils.h"
#include <cstring>

bool check_continues(const Tensor_t &tensor){

    if(tensor == nullptr) return false;
    if(tensor->data == nullptr) return false;

    auto t = (InternalTensor*)tensor->data;
    if(t->strides[t->strides.size() - 1] != 1){

        return false;
    }

    return true;
}

bool check_dtype(const Tensor_t &tensor, InternalTensor::DataType dtype){
    if(tensor == nullptr) return false;
    if(tensor->data == nullptr) return false;

    auto t = (InternalTensor*)tensor->data;

    if(t->dtype != dtype){

        return false;
    }

    return true;
}

bool check_shape(const Tensor_t &tensor, std::vector<int64_t> shape){

    if(tensor == nullptr) return false;
    if(tensor->data == nullptr) return false;

    auto t = (InternalTensor*)tensor->data;
    if(t->sizes.size() != shape.size() || t->sizes != shape) {

        return false;
    }

    return true;
}
