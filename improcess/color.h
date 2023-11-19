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

int RGB2CMY(Tensor &xo, const Tensor &xi);
int RGB2HSI(Tensor &xo, const Tensor &xi);
int HSI2RGB(Tensor &xo, const Tensor &xi);
int RGB2HSV(Tensor &xo, const Tensor &xi);
int HSV2RGB(Tensor &xo, const Tensor &xi);
int RGB2YUV(Tensor &xo, const Tensor &xi);
int YUV2RGB(Tensor &xo, const Tensor &xi);
int RGB2YIQ(Tensor &xo, const Tensor &xi);
int YIQ2RGB(Tensor &xo, const Tensor &xi);


}

#endif // COLOR_H
