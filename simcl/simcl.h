#ifndef SIMCL_H
#define SIMCL_H
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <CL/cl.h>
/*
    simple opencl
*/
namespace simcl {

enum Vendor {
    NVIDIA = 0,
    AMD,
    INTEL,
    UNKNOWN
};

namespace detail {

class Device
{
public:
    cl_device_id id;
    cl_device_type type;
    cl_uint vendorID;
    cl_uint maxWorkItemDim;
    size_t maxWorkGroupSize;
    cl_ulong maxMemorySize;
public:
    Device():id(0),type(0),vendorID(0),
          maxWorkItemDim(0),maxWorkGroupSize(0),maxMemorySize(0){}
    void print() const
    {
        std::cout<<"------------------------------------------"<<std::endl;
        std::cout<<"device id:               "<<id<<std::endl;
        std::cout<<"device type:             "<<type<<std::endl;
        std::cout<<"device vendorID:         "<<vendorID<<std::endl;
        std::cout<<"device maxWorkItemDim:   "<<maxWorkItemDim<<std::endl;
        std::cout<<"device maxWorkGroupSize: "<<maxWorkGroupSize<<std::endl;
        std::cout<<"device maxMemorySize:    "<<maxMemorySize<<std::endl;
        return;
    }
};

class Platform
{
public:
    cl_platform_id id;
    std::size_t vendorID;
    std::string profile;
    std::string version;
    std::string name;
    std::string vendor;
    std::string extentsion;
    std::vector<Device> devices;
    static std::map<int, Platform> platforms;
public:
    static int enumerate();
    static int enumerateDevices(cl_platform_id platformID, std::vector<Device> &devices);
    void print() const
    {
        std::cout<<"------------------------------------------"<<std::endl;
        std::cout<<"platform id:     "<<id<<std::endl;
        std::cout<<"platform profile:"<<profile<<std::endl;
        std::cout<<"platform version:"<<version<<std::endl;
        std::cout<<"platform name:   "<<name<<std::endl;
        std::cout<<"platform vendor: "<<vendor<<std::endl;
        return;
    }
};

class Context
{
public:
    cl_context ptr;
    /* for one device */
    cl_command_queue cmdQueue;
public:
    Context():ptr(nullptr),cmdQueue(nullptr){}
    ~Context()
    {
        if (ptr != nullptr) {
            clReleaseContext(ptr);
        }
        if (cmdQueue != nullptr) {
            clReleaseCommandQueue(cmdQueue);
        }
    }
};

} // detail

class Device
{
public:
    static detail::Context context;
    static int open(std::size_t vendor)
    {
        if (context.ptr != nullptr &&
                context.cmdQueue != nullptr) {
            return 0;
        }
        /* enumerate */
        detail::Platform::enumerate();
        auto it = detail::Platform::platforms.find(vendor);
        if (it == detail::Platform::platforms.end()) {
            return -1;
        }
        detail::Platform &platform = detail::Platform::platforms[vendor];
        platform.print();
        /* context */
        cl_int ret = 0;
        cl_platform_id pid = platform.id;
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)pid, 0};
        context.ptr = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU,
                                              nullptr, nullptr, &ret);
        /* cmd queue */
        size_t size = 0;
        ret = clGetContextInfo(context.ptr, CL_CONTEXT_DEVICES, 0, nullptr, &size);
        if (ret != CL_SUCCESS) {
            return -2;
        }
        cl_device_id id = platform.devices[0].id;
        platform.devices[0].print();
        /* OpenCL 2.0 APIs */
        context.cmdQueue = clCreateCommandQueue(context.ptr, id, 0, nullptr);
        if (context.cmdQueue == nullptr) {
            return -3;
        }
        return 0;
    }
};

}
#endif // SIMCL_H
