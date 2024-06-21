#ifndef IMAGE_DEF_H
#define IMAGE_DEF_H
#include "improcess_def.h"
#include <string.h>

namespace imp {

class Image
{
public:
    uint32_t height;
    uint32_t width;
    uint32_t channel;
    uint32_t widthstep;
    uint64_t totalsize;
    uint8_t* data;
public:
    Image():height(0),width(0),channel(0),widthstep(0),totalsize(0),data(nullptr){}
    explicit Image(uint32_t h, uint32_t w, uint32_t c, uint8_t* d)
        :height(h),width(w),channel(c),widthstep(w*c),totalsize(h*w*c),data(d){}

    explicit Image(uint32_t h, uint32_t w, uint32_t c)
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
        if (this == &r) {
            return *this;
        }
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
        return *this;
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
