#ifndef GRAPHIC_H
#define GRAPHIC_H
#include <vector>
#include "../basic/tensor.hpp"
#include "../basic/point.hpp"
#include "color.h"
namespace imp {

namespace graphic2D {
    inline int constraint(int x, int xL, int xU)
    {
        if (x < xL) {
            return xL;
        } else if (x > xU) {
            return xU;
        }
        return x;
    }
    int line(Tensor &img, const Point2i &p1, const Point2i &p2, Color3 color=Color3(0, 255, 0), float thick=1.0);
    int polygon(Tensor &img, const std::vector<Point2i> &p, imp::Color3 color, float thick=1.0);
    int circle(Tensor &img, const Point2i &center, float radius, Color3 color=Color3(0, 255, 0), float thick=1.0);
    int rectangle(Tensor &img, const Point2i &pos, const Point2i &size, Color3 color=Color3(0, 255, 0), float thick=1.0);
} // graphic

}// improcess

#endif // GRAPHIC_H
