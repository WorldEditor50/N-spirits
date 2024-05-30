#ifndef IMPROCESS_DEF_H
#define IMPROCESS_DEF_H
#include "../basic/tensor.hpp"
#include "../basic/point.hpp"
#include <memory>

namespace imp {
constexpr static float pi = 3.1415926;
using Size = Point2i;
using uint8ptr = std::shared_ptr<uint8_t[]>;

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

template <typename T>
class Rect_
{
public:
    T x;
    T y;
    T width;
    T height;
public:
    Rect_():x(0),y(0),width(0),height(0){}
    Rect_(T x_, T y_, T width_, T height_)
        :x(x_),y(y_),width(width_),height(height_){}
    Rect_& operator*=(T r)
    {
        x *= r;
        y *= r;
        width *= r;
        height *= r;
        return *this;
    }
    Rect_& operator/=(T r)
    {
        x /= r;
        y /= r;
        width /= r;
        height /= r;
        return *this;
    }
    Rect_ operator*(T r)
    {
        Rect_ rect;
        rect.x = x*r;
        rect.y = y*r;
        rect.width = width*r;
        rect.height = height*r;
        return rect;
    }
    Rect_ operator/(T r)
    {
        Rect_ rect;
        rect.x = x/r;
        rect.y = y/r;
        rect.width = width/r;
        rect.height = height/r;
        return rect;
    }

    Point2<T> topLeft() const {return Point2<T>(x, y);}
    Point2<T> size() const {return Point2<T>(height, width);}
};

using Rect = Rect_<int>;
using Rectf = Rect_<float>;

inline static Size imageSize(const Tensor &x)
{
    return Size(x.shape[HWC_H], x.shape[HWC_W]);
}

inline static double bound(double x, double min_, double max_)
{
    double value = x < min_ ? min_ : x;
    value = value > max_ ? max_ : x;
    return value;
}

inline static void bound(Tensor &img, float min_, float max_)
{
    for (std::size_t i = 0; i < img.totalSize; i++) {
        img[i] = bound(img[i], min_, max_);
    }
    return;
}
}

#endif // IMPROCESS_DEF_H
