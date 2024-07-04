#include "geometrytransform.h"

int imp::move(OutTensor xo, InTensor xi, const Size &offset)
{
    if (offset.x < 0 || offset.x > xi.shape[HWC_W] ||
        offset.y < 0 || offset.y > xi.shape[HWC_H]) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    xo = Tensor(h + offset.x, w + offset.y, c);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (i < offset.y || j < offset.x) {
                continue;
            }
            for (int k = 0; k < c; k++) {
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
    /* rotate center = (h/2, w/2) */
    float theta = angle*pi/180.0;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];

    /* original image's corners */
    float wi1 = -(w - 1)*0.5;
    float hi1 =  (h - 1)*0.5;
    float wi2 =  (w - 1)*0.5;
    float hi2 =  (h - 1)*0.5;
    float wi3 = -(w - 1)*0.5;
    float hi3 = -(h - 1)*0.5;
    float wi4 =  (w - 1)*0.5;
    float hi4 = -(h - 1)*0.5;
    /* output image's corners */
    float wo1 =  cosTheta*wi1 + sinTheta*hi1;
    float ho1 = -sinTheta*wi1 + cosTheta*hi1;
    float wo2 =  cosTheta*wi2 + sinTheta*hi2;
    float ho2 = -sinTheta*wi2 + cosTheta*hi2;
    float wo3 =  cosTheta*wi3 + sinTheta*hi3;
    float ho3 = -sinTheta*wi3 + cosTheta*hi3;
    float wo4 =  cosTheta*wi4 + sinTheta*hi4;
    float ho4 = -sinTheta*wi4 + cosTheta*hi4;
    /* output size */
    int wo = float(std::max(std::abs(wo4 - wo1), std::abs(wo3 - wo2)) + 0.5);
    int ho = float(std::max(std::abs(ho4 - ho1), std::abs(ho3 - ho2)) + 0.5);
    /* offset */
    float f1 = -0.5*(ho - 1)*cosTheta - 0.5*(wo - 1)*sinTheta + 0.5*(w - 1);
    float f2 =  0.5*(wo - 1)*sinTheta - 0.5*(ho - 1)*cosTheta + 0.5*(h - 1);
    xo = Tensor(ho, wo, c);
    for (int i = 0; i < ho; i++) {
        for(int j = 0; j < wo; j++) {
            int u = float(-j*sinTheta + i*cosTheta + f2 + 0.5);
            int v = float( j*cosTheta + i*sinTheta + f1 + 0.5);
            if (v >= 0 && v < w && u >= 0 && u < h) {
                for (int k = 0; k < c; k++) {
                    xo(ho - 1 - i, j, k) = xi(h - 1 - u, v, k);
                }
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

                xo(i, j, k) = imp::bound(y3, 0, 255);
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
