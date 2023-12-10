#include "geometrytransform.h"

int imp::move(OutTensor xo, InTensor xi, const Point2i &offset)
{
    if (offset.x < 0 || offset.x > xi.shape[HWC_W] ||
        offset.y < 0 || offset.y > xi.shape[HWC_H]) {
        return -1;
    }
    for (int i = 0; i < xi.shape[HWC_H]; i++) {
        for (int j = 0; j < xi.shape[HWC_W]; j++) {
            if (i < offset.y || j < offset.x) {
                continue;
            }
            for (int k = 0; k < xi.shape[HWC_C]; k++) {
                xo(i, j, k) = xi(i - offset.y, j - offset.x, k);
            }
        }
    }
    return 0;
}

int imp::transpose(OutTensor xo, InTensor xi)
{
    /* HWC -> WHC */
    xo = xi.permute(1, 0, 2);
    return 0;
}

int imp::horizontalFlip(OutTensor xo, InTensor xi)
{
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    for (int i = 0; i < xi.shape[HWC_H]; i++) {
        for (int j = 0; j < xi.shape[HWC_W]; j++) {
            for (int k = 0; k < xi.shape[HWC_C]; k++) {
                xo(i, j, k) = xi(i, w - j - 1, k);
            }
        }
    }
    return 0;
}

int imp::verticalFlip(OutTensor xo, InTensor xi)
{
    int h = xi.shape[HWC_H];
    xo = Tensor(xi.shape);
    for (int i = 0; i < xi.shape[HWC_H]; i++) {
        for (int j = 0; j < xi.shape[HWC_W]; j++) {
            for (int k = 0; k < xi.shape[HWC_C]; k++) {
                xo(i, j, k) = xi(h - i - 1, j, k);
            }
        }
    }
    return 0;
}

int imp::rotate(OutTensor xo, InTensor xi, float angle)
{
    /*
        rotate center = (h/2, w/2)

    */
    float theta = angle*pi/180;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    int ho = float(h*cosTheta + w*sinTheta + 0.5);
    int wo = float(w*cosTheta + h*sinTheta + 0.5);
    xo = Tensor(ho, wo, c);
    for (int i = 0; i < ho; i++) {
        for (int j = 0; j < wo; j++) {
            int u = imp::bound(i*cosTheta - j*sinTheta + 0.5, 0, h);
            int v = imp::bound(j*cosTheta + i*sinTheta + 0.5, 0, w);
            for (int k = 0; k < c; k++) {
                xo(i, j, k) = xi(u, v, k);
            }
        }
    }
    return 0;
}

int imp::nearestInterpolate(OutTensor xo, InTensor xi, const imp::Size &size)
{
    int height = xi.shape[HWC_H];
    int width = xi.shape[HWC_W];
    int channels = xi.shape[HWC_C];
    xo = Tensor(size.x, size.y, channels);
    int ho = size.x;
    int wo = size.y;
    double hr = double(height)/double(ho);
    double wr = double(width)/double(wo);
    for (int i = 1; i < ho + 1; i++) {
        for (int j = 1; j < wo + 1; j++) {
            int u = imp::bound(i*hr + 0.5, 1, height + 1);
            int v = imp::bound(j*wr + 0.5, 1, width + 1);
            for (int k = 0; k < channels; k++) {
                xo(i - 1, j - 1, k) = xi(u - 1, v - 1, k);
            }

        }
    }
    return 0;
}

int imp::bilinearInterpolate(OutTensor xo, InTensor xi, const Size &size)
{
    int height = xi.shape[HWC_H];
    int width = xi.shape[HWC_W];
    int channels = xi.shape[HWC_C];
    xo = Tensor(size.x, size.y, channels);
    double hr = double(height)/double(size.x);
    double wr = double(width)/double(size.y);

    for (int i = 0; i < size.x; i++) {
        for (int j = 0; j < size.y; j++) {
            for (int k = 0; k < channels; k++) {
                double rx = i*hr;
                double ry = j*wr;
                int xLeft = int(rx);
                int xRight = xLeft + 1;
                int yLeft = int(ry);
                int yRight = yLeft + 1;
                if (xRight >= height) {
                    xRight = height - 1;
                }
                if (yRight >= width) {
                    yRight = width - 1;
                }
                float y1 = xi(xRight, yLeft, k)*(rx - xLeft) - xi(xLeft, yLeft, k)*(rx - xLeft) + xi(xLeft, yLeft, k);
                float y2 = xi(xRight, yRight, k)*(rx - xLeft) - xi(xLeft, yRight, k)*(rx -xLeft) + xi(xLeft, yRight, k);
                float y3 = y2*(ry - yLeft) - y1*(ry - yLeft) + y1;

                xo(i, j, k) = clamp(y3, 255, 0);
            }
        }
    }
    return 0;
}

int imp::cubicInterpolate(OutTensor xo, InTensor xi, const imp::Size &size, float a)
{
    /*
        F(i + u, j + v) = ΣΣf(i + row, j + col)*S(row - u, col - v)

        S(x) = 1 - (a + 3)x^2 + (a + 2)|x|^3, for 0<|x|<1
        S(x) = -4a + 8a|x| - 5ax^2 + a|x|^3, for 1<|x|<2
    */

    return 0;
}
