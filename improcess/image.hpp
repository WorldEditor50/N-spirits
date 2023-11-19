#ifndef IMAGE_DEF_H
#define IMAGE_DEF_H
#include "../basic/tensor.hpp"
#include "../basic/point.hpp"

namespace imp {
using Size = Point2i;

enum HWC {
    HWC_H = 0,
    HWC_W = 1,
    HWC_C = 2
};

enum CHW {
    CHW_C = 0,
    CHW_H = 1,
    CHW_W = 2
};

enum Color {
    COLOR_RED = 0,
    COLOR_GREEN,
    COLOR_BLUE
};

constexpr static float pi = 3.1415926;

inline float clamp(float x, float max_, float min_)
{
    if (x > max_) {
        x = max_;
    } else if (x < min_) {
        x = min_;
    }
    return x;
}
inline void clamp(Tensor &img, float max_, float min_)
{
    for (std::size_t i = 0; i < img.totalSize; i++) {
        img[i] = clamp(img[i], max_, min_);
    }
    return;
}

inline bool isGray(const Tensor &x)
{
    return x.shape[HWC_C] == 1;
}

inline bool isRGB(const Tensor &x)
{
    return x.shape[HWC_C] == 3;
}

inline float area(const Tensor &x)
{
    return x.shape[HWC_H] * x.shape[HWC_W];
}

class Image
{
public:
    using Pointer = uint8_t*;
public:
    uint32_t height;
    uint32_t width;
    uint8_t channel;
    uint32_t widthstep;
    uint64_t totalsize;
    Pointer data;
public:
    Image():height(0),width(0),channel(0),widthstep(0),totalsize(0),data(nullptr){}
    explicit Image(uint32_t h, uint32_t w, uint8_t c, Pointer ptr)
        :height(h),width(w),channel(c),widthstep(w*c),totalsize(h*w*c),data(ptr){}

    explicit Image(uint32_t h, uint32_t w, uint8_t c)
        :height(h),width(w),channel(c),widthstep(w*c),totalsize(h*w*c)
    {
        data = new uint8_t[totalsize];
    }
    explicit Image(const Image &r)
        :height(r.height),width(r.width),channel(r.channel),
          widthstep(r.widthstep),totalsize(r.totalsize)
    {
        data = new uint8_t[totalsize];
        memcpy(data, r.data, totalsize);
    }

    Image(Image &&r) noexcept
        :height(r.height),width(r.width),channel(r.channel),
          widthstep(r.widthstep),totalsize(r.totalsize),data(r.data)
    {
        r.height = 0;
        r.width = 0;
        r.channel = 0;
        r.widthstep = 0;
        r.totalsize = 0;
        r.data = nullptr;
    }

    Image& operator=(const Image &r)
    {
        if (this == &r) {
            return *this;
        }
        height = r.height;
        width = r.width;
        channel = r.channel;
        widthstep = r.widthstep;
        if (totalsize < r.totalsize) {
            totalsize = r.totalsize;
            delete [] data;
            data = nullptr;
        }
        if (data == nullptr) {
            data = new uint8_t[totalsize];
        }
        memcpy(data, r.data, totalsize);
        return *this;
    }

    Image& operator = (Image &&r) noexcept
    {
        height = r.height;
        width = r.width;
        channel = r.channel;
        widthstep = r.widthstep;
        totalsize = r.totalsize;
        data = r.data;
        r.height = 0;
        r.width = 0;
        r.channel = 0;
        r.widthstep = 0;
        r.totalsize = 0;
        r.data = nullptr;
    }

    ~Image()
    {
        if (data != nullptr) {
            delete [] data;
        }
    }
    inline uint8_t operator()(std::size_t i, std::size_t j, std::size_t k)
    {
        /* hwc */
        return data[i*widthstep + j*channel + k];
    }
    inline uint8_t* operator()(std::size_t i, std::size_t j)
    {
        /* hwc */
        return data + i*widthstep + j*channel;
    }
    inline static uint32_t color(uint8_t r, uint8_t g, uint8_t b)
    {
        return uint32_t(r<<16 + g<<8 + b);
    }

    inline static uint8_t red(uint32_t c)
    {
        return (c >> 16)&0xff;
    }

    inline static uint8_t green(uint32_t c)
    {
        return (c >> 8)&0xff;
    }

    inline static uint8_t blue(uint32_t c)
    {
        return c&0xff;
    }

};



}
#endif // IMAGE_DEF_H
