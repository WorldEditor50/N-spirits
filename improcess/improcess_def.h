#ifndef IMPROCESS_DEF_H
#define IMPROCESS_DEF_H
#include "../basic/tensor.hpp"
#include "../basic/point.hpp"
#include <memory>

namespace imp {
constexpr static float pi = 3.1415926;
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

enum InterplateType {
    INTERPOLATE_NEAREST = 0,
    INTERPOLATE_BILINEAR,
    INTERPOLATE_CUBIC
};

inline static double bound(double x, double min_, double max_)
{
    double value = x < min_ ? min_ : x;
    value = value > max_ ? max_ : x;
    return value;
}

inline static void bound(Tensor &img, float max_, float min_)
{
    for (std::size_t i = 0; i < img.totalSize; i++) {
        img[i] = bound(img[i], min_, max_);
    }
    return;
}
}

#endif // IMPROCESS_DEF_H
