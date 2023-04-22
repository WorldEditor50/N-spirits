#include "graphic2d.h"


/*

        o---------> x
        |
        |
        |
        V y

*/
int imp::graphic2D::line(Tensor &img, const Point2i &p1, const Point2i &p2, imp::Color3 color, float thick)
{
    /*
        (y - y2)/(x - x2) = (y2 - y1)/(x2 - x1) = k
        (y - y2)*(x2 - x1) = (y2 - y1)*(x - x2)
        (x2 - x1)*y - (x2 - x1)*y2 = (y2 - y1)*x - (y2 - y1)*x2

        (y1 - y2)*x + (x2 - x1)*y + (y2 - y1)*x2 - (x2 - x1)*y2 = 0
        (y1 - y2)*x + (x2 - x1)*y + x1*y2 - x2*y1 = 0
        a = y1 - y2
        b = x2 - x1
        c = x1*y2 - x2*y1
    */
    int a = p1.y - p2.y;
    int b = p2.x - p1.x;
    int c = p1.x*p2.y - p2.x*p1.y;
    float r = std::sqrt(a*a + b*b);
    int xL = std::min(p1.x, p2.x);
    int xU = std::max(p1.x, p2.x);
    int yL = std::min(p1.y, p2.y);
    int yU = std::max(p1.y, p2.y);
    for (int x = 0; x < img.shape[1]; x++) {
        for (int y = 0; y < img.shape[0]; y++) {
            if (x < xL || x > xU || y > yU || y < yL) {
                continue;
            }
            float d = 0;
            if (a == 0) {
                d = std::abs(y - p1.y);
            } else if (b == 0) {
                d = std::abs(x - p1.x);
            } else {
                d = std::abs(a*x + b*y + c)/r;
            }
            if (d < thick) {
                img(y, x, 0) = color.r;
                img(y, x, 1) = color.g;
                img(y, x, 2) = color.b;
            }
        }
    }
    return 0;
}

int imp::graphic2D::polygon(Tensor &img, const std::vector<Point2i> &p, imp::Color3 color, float thick)
{
    std::size_t i = 0;
    for (; i < p.size() - 1; i++) {
        line(img, p[i], p[i + 1], color, thick);
    }
    line(img, p[0], p[i], color, thick);
    return 0;
}

int imp::graphic2D::circle(Tensor &img, const Point2i &center, float radius, imp::Color3 color, float thick)
{
    int x0 = center.x;
    int y0 = center.y;
    float up = (radius + thick)*(radius + thick);
    float low = (radius - thick)*(radius - thick);
    for (int x = 0; x < img.shape[1]; x++) {
        for (int y = 0; y < img.shape[0]; y++) {
            float d = (x - x0)*(x - x0) + (y - y0)*(y - y0);
            if (d > low && d < up) {
                img(y, x, 0) = color.r;
                img(y, x, 1) = color.g;
                img(y, x, 2) = color.b;
            }
        }
    }
    return 0;
}

int imp::graphic2D::rectangle(Tensor &img, const Point2i &pos, const Point2i &size, imp::Color3 color, float thick)
{
    /* top left */
    int xL = pos.x;
    int yL = pos.y;
    /* right buttom */
    int xR = pos.x + size.x;
    int yR = pos.y + size.y;
    /* thick */
    int xLL = constraint(xL - thick, 0, img.shape[1]);
    int yLL = constraint(yL - thick, 0, img.shape[0]);
    int xLU = constraint(xL + thick, 0, img.shape[1]);
    int yLU = constraint(yL + thick, 0, img.shape[0]);
    int xRL = constraint(xR - thick, 0, img.shape[1]);
    int yRL = constraint(yR - thick, 0, img.shape[0]);
    int xRU = constraint(xR + thick, 0, img.shape[1]);
    int yRU = constraint(yR + thick, 0, img.shape[0]);
    for (int x = 0; x < img.shape[1]; x++) {
        /* vertical border */
        for (int y = 0; y < img.shape[0]; y++) {
            if (x < xLL || x > xRU || y > yRU || y < yLL) {
                continue;
            }
            if (x > xLU && x < xRL) {
                continue;
            }
            img(y, x, 0) = color.r;
            img(y, x, 1) = color.g;
            img(y, x, 2) = color.b;
        }
        /* horizontal border */
        for (int y = 0; y < img.shape[0]; y++) {
            if (x < xLL || x > xRU || y > yRU || y < yLL) {
                continue;
            }
            if (y > yLU && y < yRL) {
                continue;
            }
            img(y, x, 0) = color.r;
            img(y, x, 1) = color.g;
            img(y, x, 2) = color.b;
        }
    }
    return 0;
}
