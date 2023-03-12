#ifndef IMAGE_DEF_H
#define IMAGE_DEF_H
#include "../basic/tensor.hpp"
#include "point.hpp"

namespace improcess {

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

}
#endif // IMAGE_DEF_H
