#include "geometrytransform.h"

int imp::move(const Tensor &x, const Point2i &offset, Tensor &y)
{
    if (offset.x < 0 || offset.x > x.shape[HWC_W] ||
        offset.y < 0 || offset.y > x.shape[HWC_H]) {
        return -1;
    }
    for (int i = 0; i < x.shape[HWC_H]; i++) {
        for (int j = 0; j < x.shape[HWC_W]; j++) {
            if (i < offset.y || j < offset.x) {
                continue;
            }
            for (int k = 0; k < x.shape[HWC_C]; k++) {
                y(i, j, k) = x(i - offset.y, j - offset.x, k);
            }
        }
    }
    return 0;
}

int imp::transpose(const Tensor &x, Tensor &y)
{
    /* HWC -> WHC */
    y = x.permute(1, 0, 2);
    return 0;
}

int imp::horizontalFlip(const Tensor &x, Tensor &y)
{
    int w = x.shape[HWC_W];
    y = Tensor(x.shape);
    for (int i = 0; i < x.shape[HWC_H]; i++) {
        for (int j = 0; j < x.shape[HWC_W]; j++) {
            for (int k = 0; k < x.shape[HWC_C]; k++) {
                y(i, j, k) = x(i, w - j - 1, k);
            }
        }
    }
    return 0;
}

int imp::verticalFlip(const Tensor &x, Tensor &y)
{
    int h = x.shape[HWC_H];
    y = Tensor(x.shape);
    for (int i = 0; i < x.shape[HWC_H]; i++) {
        for (int j = 0; j < x.shape[HWC_W]; j++) {
            for (int k = 0; k < x.shape[HWC_C]; k++) {
                y(i, j, k) = x(h - i - 1, j, k);
            }
        }
    }
    return 0;
}

int imp::scale(const Tensor &x, float alpha, Tensor &y)
{
    int h = x.shape[HWC_H];
    int w = x.shape[HWC_W];
    y = Tensor(int(h*alpha), int(w*alpha), x.shape[HWC_C]);
    for (int i = 0; i < y.shape[HWC_H]; i++) {

        int u = int(float(i)/alpha + 0.5);

        for (int j = 0; j < y.shape[HWC_W]; j++) {

            int v = int(float(j)/alpha + 0.5);

            if (v < w && u < h) {
                y(i, j, 0) = x(u, v, 0);
                y(i, j, 1) = x(u, v, 1);
                y(i, j, 2) = x(u, v, 2);
            } else {
                y(i, j, 0) = 255;
                y(i, j, 1) = 255;
                y(i, j, 2) = 255;
            }
        }
    }
    return 0;
}

int imp::rotate(const Tensor &x, float angle, Tensor &y)
{
    /*
        rotate center = (h/2, w/2)

    */
    float theta = angle*pi/180;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    int h = x.shape[HWC_H];
    int w = x.shape[HWC_W];
    y = Tensor(x.shape);
    for (int i = 0; i < y.shape[HWC_H]; i++) {
        for (int j = 0; j < y.shape[HWC_W]; j++) {
            int u = i * cosTheta - j * sinTheta + 0.5;
            int v = j * cosTheta + i * sinTheta + 0.5;
            if (u < h && v < w && u >= 0 && v >= 0) {
                y(i, j, 0) = x(u, v, 0);
                y(i, j, 1) = x(u, v, 1);
                y(i, j, 2) = x(u, v, 2);
            } else {
                y(i, j, 0) = 0;
                y(i, j, 1) = 0;
                y(i, j, 2) = 0;
            }
        }
    }
    return 0;
}
