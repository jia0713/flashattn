#include "flash_performance_mode.h"

int get_grid_type(std::string tag, int default_gtype) {
    char* gtype_str = std::getenv(tag.c_str());
    if (gtype_str != nullptr) {
        int debug_gtype = atoi(gtype_str);

        if (std::getenv("MHA_DEBUG_PARA")){
            printf("%s set value (%d), force cover default value (%d)\n",
                    tag.c_str(), debug_gtype, default_gtype);
        }
        return debug_gtype;
    }

    return default_gtype;
}

// grid switch type
// gtype   dim.x   dim.y   dim.z
// -----------------------------
//   0       x       h       b
//   1       x       b       h
//   2       h       x       b
//   3       b       x       h
//   4       b       h       x
//   5       h       b       x
dim3 flash_bwd_compute_grid_dim(int x, int h, int b, int type) {
    if(type == 0)       return dim3(x, h, b);
    else if(type == 1)  return dim3(x, b, h);
    else if(type == 2)  return dim3(h, x, b);
    else if(type == 3)  return dim3(b, x, h);
    else if(type == 4)  return dim3(b, h, x);
    else if(type == 5)  return dim3(h, b, x);
    else                return dim3(x, h, b);
}

dim3 flash_fwd_compute_grid_dim(int num_m_block, int h, int b, int  rowblock_parallel, int block_type) {

    int x = 0, y = 0, z = 0;
    switch (block_type) {
    case 0: {
            x = num_m_block, y = h, z = b;
            if (rowblock_parallel == 1)
                x = (num_m_block+1) / 2;
            else if (rowblock_parallel == 2)
                y = (h + 1) / 2;
        }
        break;
    case 1: {
            x = num_m_block, y = b, z = h;
            if (rowblock_parallel == 1)
                x = (num_m_block+1) / 2;
            else if (rowblock_parallel == 2)
                z = (h + 1) / 2;
        }
        break;
    case 2: {
            x = h, y = num_m_block, z = b;
            if (rowblock_parallel == 1)
                y = (num_m_block+1) / 2;
            else if (rowblock_parallel == 2)
                x = (h + 1) / 2;
        }
        break;
    case 3: {
            x = b, y = num_m_block, z = h;
            if (rowblock_parallel == 1)
                y = (num_m_block + 1) / 2;
            else if (rowblock_parallel == 2)
                z = (h + 1) / 2;
        }
        break;
     case 4: {
            x = b, y = h, z = num_m_block;
            if (rowblock_parallel == 1)
                z = (num_m_block + 1) / 2;
            else if (rowblock_parallel == 2)
                y = (h + 1) / 2;
        }
        break;
    case 5: {
            x = h, y = b, z = num_m_block;
            if (rowblock_parallel == 1)
                z = (num_m_block + 1) / 2;
            else if (rowblock_parallel == 2)
                x = (h + 1) / 2;
        }
        break;
    default: {
            x = num_m_block, y = h, z = b;
            if (rowblock_parallel == 1)
                x = (num_m_block+1) / 2;
            else if (rowblock_parallel == 2)
                y = (h + 1) / 2;
        }
    }

    dim3 grid(x, y, z);
    return grid;

}

dim3 flash_fwd_splitkv_compute_grid_dim_numsplits_one(const int& num_m_block, const int& h, const int &b, const int& rowblock_parallel, const int& block_type) {

    int x = 0, y = 0, z = 0;
    switch (block_type) {
    case 0: {
            x = num_m_block, y = h, z = b;
            if (rowblock_parallel == 1)
                x = (num_m_block+1) / 2;
            else if (rowblock_parallel == 2)
                y = (h + 1) / 2;
        }
        break;
    case 1: {
            x = num_m_block, y = b, z = h;
            if (rowblock_parallel == 1)
                x = (num_m_block+1) / 2;
            else if (rowblock_parallel == 2)
                z = (h + 1) / 2;
        }
        break;
    case 2: {
            x = h, y = num_m_block, z = b;
            if (rowblock_parallel == 1)
                y = (num_m_block+1) / 2;
            else if (rowblock_parallel == 2)
                x = (h + 1) / 2;
        }
        break;
    case 3: {
            x = b, y = num_m_block, z = h;
            if (rowblock_parallel == 1)
                y = (num_m_block + 1) / 2;
            else if (rowblock_parallel == 2)
                z = (h + 1) / 2;
        }
        break;
     case 4: {
            x = b, y = h, z = num_m_block;
            if (rowblock_parallel == 1)
                z = (num_m_block + 1) / 2;
            else if (rowblock_parallel == 2)
                y = (h + 1) / 2;
        }
        break;
    case 5: {
            x = h, y = b, z = num_m_block;
            if (rowblock_parallel == 1)
                z = (num_m_block + 1) / 2;
            else if (rowblock_parallel == 2)
                x = (h + 1) / 2;
        }
        break;
    default: {
            x = num_m_block, y = h, z = b;
            if (rowblock_parallel == 1)
                x = (num_m_block+1) / 2;
            else if (rowblock_parallel == 2)
                y = (h + 1) / 2;
        }
    }

    dim3 grid(x, y, z);
    return grid;

}

dim3 flash_fwd_splitkv_compute_grid_dim(const int& num_m_block,const int& num_splits, const int& h, const int &b, const int& block_type) {
    if(num_splits == 1)
        return flash_fwd_splitkv_compute_grid_dim_numsplits_one(num_m_block, h, b, 0, block_type);

    int x = 0, y = 0, z = 0;
    switch (block_type) {
        case 0:{
            x = num_splits;
            y = h * b;
            z = num_m_block;
        }
        break;
        case 1:{
            x = num_splits;
            y = num_m_block;
            z = h * b;
        }
        break;
        case 2:{
            x = h * b;
            y = num_splits;
            z = num_m_block;
        }
        break;
        case 3:{
            x = num_m_block;
            z = h * b;
            y = num_splits;
        }
        break;
        case 4:{
            x = h * b;
            y = num_m_block;
            z = num_splits;
        }
        break;
        case 5:{
            x = num_m_block;
            y = h * b;
            z = num_splits;
        }
        break;
        default:{
            x = num_splits;
            y = h * b;
            z = num_m_block;
        }
        break;
    }
    dim3 grid(x, y, z);
    return grid;
}
