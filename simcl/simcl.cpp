#include "simcl.h"

simcl::detail::Context simcl::Device::context;
std::map<int, simcl::detail::Platform> simcl::detail::Platform::platforms;

int simcl::detail::Platform::enumerate()
{
    cl_uint platformNum = 0;
    cl_int ret = clGetPlatformIDs(0, nullptr, &platformNum);
    if (ret != CL_SUCCESS) {
        return -1;
    }
    if (platformNum == 0) {
        return -2;
    }

    std::unique_ptr<cl_platform_id[]> platformID(new cl_platform_id[platformNum]);
    ret = clGetPlatformIDs(platformNum, platformID.get(), nullptr);
    if (ret != CL_SUCCESS) {
        return -3;
    }
    for (std::size_t i = 0; i < platformNum; i++) {
        Platform platform;
        platform.id = platformID[i];
        /* profile */
        char buffer[1024];
        memset(buffer, 0, 1024);
        std::size_t size;
        cl_int ret = clGetPlatformInfo(platform.id, CL_PLATFORM_PROFILE, 1024, buffer, &size);
        if (ret != CL_SUCCESS) {
            continue;
        }
        platform.profile = std::string(buffer);
        /* version */
        memset(buffer, 0, 1024);
        size = 0;
        ret = clGetPlatformInfo(platform.id, CL_PLATFORM_VERSION, 1024, buffer, &size);
        if (ret != CL_SUCCESS) {
            continue;
        }
        platform.version = std::string(buffer);
        /* vendor */
        memset(buffer, 0, 1024);
        size = 0;
        ret = clGetPlatformInfo(platform.id, CL_PLATFORM_VENDOR, 1024, buffer, &size);
        if (ret != CL_SUCCESS) {
            continue;
        }
        platform.vendor = std::string(buffer);
        std::string vendor(platform.vendor);
        std::transform(vendor.begin(), vendor.end(), vendor.begin(), std::toupper);
        platform.vendorID = UNKNOWN;
        int r = vendor.find("NVIDIA");
        if (r >= 0) {
            platform.vendorID = NVIDIA;
        }
        r = vendor.find("INTEL");
        if (r >= 0) {
            platform.vendorID = INTEL;
        }
        r = vendor.find("AMD");
        if (r >= 0) {
            platform.vendorID = AMD;
        }
        /* name */
        memset(buffer, 0, 1024);
        size = 0;
        ret = clGetPlatformInfo(platform.id, CL_PLATFORM_NAME, 1024, buffer, &size);
        if (ret != CL_SUCCESS) {
            continue;
        }
        platform.name = std::string(buffer);
        /* extension */
        memset(buffer, 0, 1024);
        size = 0;
        ret = clGetPlatformInfo(platform.id, CL_PLATFORM_EXTENSIONS, 1024, buffer, &size);
        if (ret != CL_SUCCESS) {
            continue;
        }
        platform.extentsion = std::string(buffer);
        /* devices */
        ret = enumerateDevices(platform.id, platform.devices);
        if (ret != 0) {
            continue;
        }
        Platform::platforms.insert(std::pair<int, Platform>(platform.vendorID, platform));
    }
    return 0;
}

int simcl::detail::Platform::enumerateDevices(cl_platform_id platformID, std::vector<simcl::detail::Device> &devices)
{
    cl_uint deviceNum = 0;
    cl_int ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceNum);
    if (ret != CL_SUCCESS) {
        return -1;
    }
    if (deviceNum == 0) {
        return -2;
    }
    std::unique_ptr<cl_device_id[]> deviceID(new cl_device_id[deviceNum]);
    ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_ALL, deviceNum, deviceID.get(), nullptr);
    if (ret != CL_SUCCESS) {
        return -3;
    }
    for (std::size_t i = 0; i < deviceNum; i++) {
        Device dev;
        dev.id = deviceID[i];
        /* device type */
        cl_int ret = clGetDeviceInfo(dev.id, CL_DEVICE_TYPE,
                                     sizeof (cl_device_type), &dev.type, nullptr);
        if (ret != CL_SUCCESS) {
            continue;
        }
        /* vendor */
        ret = clGetDeviceInfo(dev.id, CL_DEVICE_VENDOR_ID,
                              sizeof(cl_uint), &dev.vendorID, nullptr);
        if (ret != CL_SUCCESS) {
            continue;
        }
        /* max dimension */
        ret = clGetDeviceInfo(dev.id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                              sizeof(cl_uint), &dev.maxWorkItemDim, nullptr);
        if (ret != CL_SUCCESS) {
            continue;
        }
        /* max work group size */
        ret = clGetDeviceInfo(dev.id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                              sizeof(size_t), &dev.maxWorkGroupSize, nullptr);
        if (ret != CL_SUCCESS) {
            continue;
        }
        /* max memory size */
        ret = clGetDeviceInfo(dev.id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                              sizeof(cl_ulong), &dev.maxMemorySize, nullptr);
        if (ret != CL_SUCCESS) {
            continue;
        }
        devices.push_back(dev);
    }
    return 0;
}
