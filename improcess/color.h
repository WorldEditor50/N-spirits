#ifndef COLOR_H
#define COLOR_H
#include <cstdint>
#include "../basic/tensor.hpp"
#include "image.hpp"

namespace imp {

class Color3
{
public:
    uint8_t r;
    uint8_t g;
    uint8_t b;
public:
    Color3():r(0),g(0),b(0){}
    explicit Color3(uint8_t r_, uint8_t g_, uint8_t b_)
        :r(r_),g(g_),b(b_){}
};

class Color4
{
public:
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
public:
    Color4():r(0),g(0),b(0),a(0){}
    explicit Color4(uint8_t r_, uint8_t g_, uint8_t b_, uint8_t a_)
        :r(r_),g(g_),b(b_),a(a_){}
};

int RGB2CMY(OutTensor xo, InTensor xi);
int RGB2HSI(OutTensor xo, InTensor xi);
int HSI2RGB(OutTensor xo, InTensor xi);
int RGB2HSV(OutTensor xo, InTensor xi);
int HSV2RGB(OutTensor xo, InTensor xi);
int RGB2YUV(OutTensor xo, InTensor xi);
int YUV2RGB(OutTensor xo, InTensor xi);
int RGB2YIQ(OutTensor xo, InTensor xi);
int YIQ2RGB(OutTensor xo, InTensor xi);


}

#endif // COLOR_H
