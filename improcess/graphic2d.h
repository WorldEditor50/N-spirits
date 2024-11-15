#ifndef GRAPHIC2D_H
#define GRAPHIC2D_H
#include <vector>
#include "../basic/tensor.hpp"
#include "../basic/point.hpp"
#include "color.h"
namespace ns {

int line(Tensor &img, const Point2i &p1, const Point2i &p2, Color3 color=Color3(0, 255, 0), float thick=1.0);
int polygon(Tensor &img, const std::vector<Point2i> &p, ns::Color3 color, float thick=1.0);
int circle(Tensor &img, const Point2i &center, float radius, Color3 color=Color3(0, 255, 0), float thick=1.0);
int rectangle(Tensor &img, const Point2i &pos, const Point2i &size, Color3 color=Color3(0, 255, 0), float thick=1.0);

}// improcess

#endif // GRAPHIC_H
