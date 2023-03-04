#ifndef IMAGE_HPP
#define IMAGE_HPP
#include <cstdlib>
#include <vector>
#include <memory>

#define RGB_MASK   0x00ffffff
#define RED_MASK   0x00ff0000
#define GREEN_MASK 0x0000ff00
#define BLUE_MASK  0x000000ff
class RGB
{
public:
    uint32_t data;
public:
    RGB():data(0){}
    RGB(uint8_t r, uint8_t g, uint8_t b)
    {
        int ir = (r & 0xff) << 16;
        int ig = (g & 0xff) << 8;
        int ib = b & 0xff;
        data = (ir + ig + ib)&RGB_MASK;
    }
    RGB(uint32_t rgb):data(rgb){}
    inline uint8_t R(){return ((data&RED_MASK) >> 16)&0xff;}
    inline uint8_t G(){return ((data&GREEN_MASK) >> 8)&0xff;}
    inline uint8_t B(){return (data&BLUE_MASK)&0xff;}
};


class Image
{
public:
    int width;
    int height;
    int channel;
    std::shared_ptr<uint8_t[]> data;
    int totalSize;
public:
    Image():width(0),height(0),channel(3),data(nullptr),totalSize(0){}
    Image(int h, int w, int c):width(w),height(h),channel(c)
    {
        int widthstep = (width * channel + 3)/4*4;
        totalSize = widthstep * height;
        data = std::shared_ptr<uint8_t[]>(new uint8_t[totalSize]);
    }
    Image(const Image &img)
        :width(img.width),height(img.height),channel(img.channel),totalSize(img.totalSize)
    {
        data = std::shared_ptr<uint8_t[]>(new uint8_t[totalSize]);
        memcpy(data.get(), img.data.get(), totalSize);
    }
    Image &operator=(const Image &img)
    {
        if (this == &img) {
            return *this;
        }
        width = img.width;
        height = img.height;
        channel = img.channel;
        totalSize = img.totalSize;
        data = std::shared_ptr<uint8_t[]>(new uint8_t[totalSize]);
        memcpy(data.get(), img.data.get(), totalSize);
        return *this;
    }
    Image(Image &&img)
        :width(img.width),height(img.height),channel(img.channel),
          data(img.data),totalSize(img.totalSize)
    {
        img.width = 0;
        img.height = 0;
        img.channel = 0;
        img.data = nullptr;
        img.totalSize = 0;
    }
    Image &operator=(Image &&img)
    {
        if (this == &img) {
            return *this;
        }
        width = img.width;
        height = img.height;
        channel = img.channel;
        totalSize =img.totalSize;
        data = img.data;
        img.width = 0;
        img.height = 0;
        img.channel = 0;
        img.data = nullptr;
        img.totalSize = 0;
        return *this;
    }

    inline uint8_t* scanline(int i)
    {
        return data.get() + i*(width*channel + 3)/4*4;
    }
    inline uint8_t* at(int i, int j)
    {
        return data.get() + i*(width*channel + 3)/4*4 + j*channel;
    }
};

#endif // IMAGE_HPP
