#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP
#include <atomic>
#include <CL/cl.h>
#include "simcl.h"

namespace simcl {

template<typename T>
class Allocator_
{
public:
    using Pointer = cl_mem;
public:
    inline static Pointer get(std::size_t totalsize)
    {
        cl_mem ptr = clCreateBuffer(Device::context.ptr, CL_MEM_READ_WRITE,
                                    sizeof(T)*totalsize, nullptr, nullptr);
        return ptr;
    }

    inline static Pointer get(T* ptr, std::size_t totalsize)
    {
        cl_mem p = clCreateBuffer(Device::context.ptr, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(T)*totalsize, ptr, nullptr);
        return p;
    }
};



}

#endif // ALLOCATOR_HPP
