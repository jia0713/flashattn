
#include "tensor.h"


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

Tensor_t make_contiguous_tensor1d(void *data,mcflashattnDataType_t dtype,int width){

    if(width < 0) return nullptr;

    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{width});

    return tensor;
}
Tensor_t make_contiguous_tensor2d(void *data,mcflashattnDataType_t dtype,int height,int width){

    if(height < 0 || width < 0) return nullptr;

    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{height,width});

    return tensor;
}
Tensor_t make_contiguous_tensor3d(void *data,mcflashattnDataType_t dtype,int channel,int height,int width){

    if(channel < 0 || height < 0 || width < 0) return nullptr;

    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{channel,height,width});

    return tensor;
}
Tensor_t make_contiguous_tensor4d(void *data,mcflashattnDataType_t dtype,int batch,int channel,int height,int width){

    if(batch < 0 || channel < 0 || height < 0 || width < 0) return nullptr;

    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{batch,channel,height,width});

    return tensor;
}

Tensor_t make_contiguous_tensor5d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int size1,
    int size2,
    int size3,
    int size4
) {
    if(size0 < 0 || size1 < 0 || size2 < 0 || size3 < 0 || size4 < 0) return nullptr;

    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{size0,size1,size2,size3,size4});

    return tensor;
}



Tensor_t make_tensor1d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int stride0
){
    if(size0 < 0 || stride0 < 0) return nullptr;
    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{size0});

    // set stride
    ((InternalTensor*)(tensor->data))->strides = {stride0};

    return tensor;
}

Tensor_t make_tensor2d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int size1,
    int stride0,
    int stride1
){
    if(size0 < 0 || size1 < 0 || stride0 < 0 || stride1 < 0 || stride1 != 1) return nullptr;
    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{size0,size1});

    // set stride
    ((InternalTensor*)(tensor->data))->strides = {stride0,stride1};

    return tensor;

}

Tensor_t make_tensor3d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int size1,
    int size2,
    int stride0,
    int stride1,
    int stride2
) {
    if(size0 < 0 || size1 < 0 || size2 < 0) return nullptr;
    if(stride0 < 0 || stride1 < 0 || stride2 < 0 || stride2 != 1) return nullptr;

    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{size0,size1,size2});

    // set stride
    ((InternalTensor*)(tensor->data))->strides = {stride0,stride1,stride2};

    return tensor;

}

Tensor_t make_tensor4d(
    void *data,
    mcflashattnDataType_t dtype,
    int batch,
    int channel,
    int height,
    int width,
    int stride0,
    int stride1,
    int stride2,
    int stride3
){
    if(batch < 0 || channel < 0 || height < 0 || width < 0) return nullptr;
    if(stride0 < 0 || stride1 < 0 || stride2 < 0 || stride3 < 0 || stride3 != 1) return nullptr;
    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{batch,channel,height,width});

    // set stride
    ((InternalTensor*)(tensor->data))->strides = {stride0,stride1,stride2,stride3};

    return tensor;
}


Tensor_t make_tensor5d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int size1,
    int size2,
    int size3,
    int size4,
    int stride0,
    int stride1,
    int stride2,
    int stride3,
    int stride4
){
    if(size0 < 0 || size1 < 0 || size2 < 0 || size3 < 0 || size4 < 0) return nullptr;
    if(stride0 < 0 || stride1 < 0 || stride2 < 0 || stride3 < 0 || stride4 < 0 || stride4 != 1) return nullptr;
    Tensor_t tensor = make_tensor();
    tensor->data = new InternalTensor(data,dtype,{size0,size1,size2,size3,size4});

    // set stride
    ((InternalTensor*)(tensor->data))->strides = {stride0,stride1,stride2,stride3,stride4};

    return tensor;
}

void release_tensor(Tensor_t tensor){
    if(tensor){
        if(tensor->data){
            delete (InternalTensor*)tensor->data;
        }
        free(tensor);
    }
}

void* get_tensor_data(Tensor_t tensor){
    if(tensor && tensor->data){
        return ((InternalTensor*)tensor->data)->data;
    }
    return nullptr;
}
int get_tensor_size(Tensor_t tensor,int index){
    if(tensor && tensor->data){
        auto internal_tensor = ((InternalTensor*)tensor->data);
        const int ndim = static_cast<int>(internal_tensor->sizes.size());
        if(index < 0){
            int idx = ndim + index;
            if(idx < 0) return -1;
            return internal_tensor->sizes[idx];
        }else {
            if(index >= ndim) return -1;
            return internal_tensor->sizes[index];
        }
    }

    return -1;
}
int get_tensor_stride(Tensor_t tensor,int index){
    if(tensor && tensor->data){
        auto internal_tensor = ((InternalTensor*)tensor->data);
        const int ndim = static_cast<int>(internal_tensor->strides.size());
        if(index < 0){
            int idx = ndim + index;
            if(idx < 0) return -1;
            return internal_tensor->strides[idx];
        }else {
            if(index >= ndim) return -1;
            return internal_tensor->strides[index];
        }
    }

    return -1;
}
int get_tensor_dims(Tensor_t tensor){
    if(tensor && tensor->data){
        return ((InternalTensor*)tensor->data)->sizes.size();
    }
    return -1;
}

mcflashattnDataType_t get_tensor_dtype(Tensor_t tensor){
    if(tensor && tensor->data){
        auto dtype = ((InternalTensor*)tensor->data)->dtype;

        switch(dtype){
            case InternalTensor::DataType::FP16: return MCFLASHATTN_DATATYPE_FP16;
            case InternalTensor::DataType::BF16: return MCFLASHATTN_DATATYPE_BF16;
            case InternalTensor::DataType::INT8: return MCFLASHATTN_DATATYPE_INT8;
            case InternalTensor::DataType::INT32: return MCFLASHATTN_DATATYPE_INT32;
            case InternalTensor::DataType::FP32: return MCFLASHATTN_DATATYPE_FP32;
            case InternalTensor::DataType::FP64: return MCFLASHATTN_DATATYPE_FP64;
            default: return MCFLASHATTN_DATATYPE_NONE;
        }
    }

    return MCFLASHATTN_DATATYPE_NONE;
}


void print_tensor_info(Tensor_t tensor){
    if(tensor && tensor->data){
        auto t = (InternalTensor*)tensor->data;
        t->print();
    }
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
