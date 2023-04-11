#ifndef EVALUATE_HPP
#define EVALUATE_HPP
#include <string>
#include "simcl.h"
#include "kernel.hpp"

namespace simcl {

class Operator
{
public:
    cl_program program;
    cl_kernel kernel;
public:
    explicit Operator(const std::string &funcName, const std::string &code)
    {
        const char* codestr = code.c_str();
        program = clCreateProgramWithSource(Device::context.ptr, 1, (const char **)&codestr,
                                            nullptr, nullptr);
        cl_int ret = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
        if (ret != CL_SUCCESS) {
            std::cout<<"build program failed"<<std::endl;
            return;
        }
        kernel = clCreateKernel(program, funcName.c_str(), nullptr);
        if (kernel == nullptr) {
             std::cout<<"create kernel failed"<<std::endl;
            return;
        }
    }
    ~Operator()
    {
        if (program != nullptr) {
            clReleaseProgram(program);
        }
        if (kernel != nullptr) {
            clReleaseKernel(kernel);
        }
    }


};

class BinaryOp : public Operator
{
public:
    explicit BinaryOp(const std::string &funcName, const std::string &code)
        :Operator(funcName, code){}
    template<typename T>
    inline int eval1d(T *xptr, T *xptr1, T *xptr2, std::size_t totalsize)
    {
        if (kernel == nullptr || program == nullptr) {
            return -1;
        }
        /* create buffer */
        cl_mem x  = clCreateBuffer(Device::context.ptr, CL_MEM_READ_WRITE,
                                   sizeof(T)*totalsize, nullptr, nullptr);
        cl_mem x1 = clCreateBuffer(Device::context.ptr, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(T)*totalsize, xptr1, nullptr);
        cl_mem x2 = clCreateBuffer(Device::context.ptr, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(T)*totalsize, xptr2, nullptr);

        /* set kernel function argument: from left to right */
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &x);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &x1);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &x2);

        size_t globalWorkSize[1] = { totalsize };
        size_t localWorkSize[1] = { 1 };

        clEnqueueNDRangeKernel(Device::context.cmdQueue,
                               kernel,
                               1,              // dimension of data
                               nullptr,
                               globalWorkSize, // shape of data
                               localWorkSize,  // excution unit of gpu
                               0, nullptr, nullptr);
        clFinish(Device::context.cmdQueue);
        /* read result from gpu */
        clEnqueueReadBuffer(Device::context.cmdQueue, x, CL_TRUE,
                            0, sizeof(T)*totalsize, xptr, 0, nullptr, nullptr);
        /* release */
        clReleaseMemObject(x);
        clReleaseMemObject(x1);
        clReleaseMemObject(x2);

        return 0;
    }
};


class UnaryOp : public Operator
{
public:
    explicit UnaryOp(const std::string &funcName, const std::string &code)
        :Operator(funcName, code){}
    template<typename T>
    inline int eval1d(T *yptr, T *xptr, std::size_t totalsize)
    {
        if (kernel == nullptr || program == nullptr) {
            return -1;
        }
        /* create buffer */
        cl_mem y = clCreateBuffer(Device::context.ptr, CL_MEM_READ_WRITE,
                                   sizeof(T)*totalsize, nullptr, nullptr);
        cl_mem x = clCreateBuffer(Device::context.ptr, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(T)*totalsize, x, nullptr);
        /* set kernel function argument: from left to right */
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &y);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &x);

        size_t globalWorkSize[1] = { totalsize };
        size_t localWorkSize[1] = { 1 };

        clEnqueueNDRangeKernel(Device::context.cmdQueue,
                               kernel,
                               1,              // dimension of data
                               nullptr,
                               globalWorkSize, // shape of data
                               localWorkSize,  // excution unit of gpu
                               0, nullptr, nullptr);
        clFinish(Device::context.cmdQueue);
        /* read result from gpu */
        clEnqueueReadBuffer(Device::context.cmdQueue, y, CL_TRUE,
                            0, sizeof(T)*totalsize, yptr, 0, nullptr, nullptr);

        /* release */
        clReleaseMemObject(y);
        clReleaseMemObject(x);
        return 0;
    }
};



class Add : public BinaryOp
{
public:
    Add():BinaryOp("add", simcl::kernel::Add){}
    template<typename T>
    inline int operator()(T *xptr, T *xptr1, T *xptr2, std::size_t totalsize)
    {
        return BinaryOp::eval1d(xptr, xptr1, xptr2, totalsize);
    }
};


class Sub : public BinaryOp
{
public:
    Sub():BinaryOp("sub", simcl::kernel::Sub){}
    template<typename T>
    inline int operator()(T *xptr, T *xptr1, T *xptr2, std::size_t totalsize)
    {
        return BinaryOp::eval1d(xptr, xptr1, xptr2, totalsize);
    }
};


class Mul : public BinaryOp
{
public:
    Mul():BinaryOp("mul", simcl::kernel::Mul){}
    template<typename T>
    inline int operator()(T *xptr, T *xptr1, T *xptr2, std::size_t totalsize)
    {
        return BinaryOp::eval1d(xptr, xptr1, xptr2, totalsize);
    }
};

class Div : public BinaryOp
{
public:
    Div():BinaryOp("div", simcl::kernel::Div){}
    template<typename T>
    inline int operator()(T *xptr, T *xptr1, T *xptr2, std::size_t totalsize)
    {
        return BinaryOp::eval1d(xptr, xptr1, xptr2, totalsize);
    }
};


class Exp : public UnaryOp
{
public:
    Exp():UnaryOp("expf", simcl::kernel::Exp){}
    template<typename T>
    inline int operator()(T *yptr, T *xptr, std::size_t totalsize)
    {
        return UnaryOp::eval1d(yptr, xptr, totalsize);
    }
};


}
#endif // EVALUATE_HPP
