#ifndef FILTER_H
#define FILTER_H
#include <functional>
#include "../basic/linalg.h"
#include "../basic/ctensor.hpp"
#include "../basic/complex.hpp"
#include "improcess_def.h"

namespace imp {


/* naive conv2d */
inline void conv2d(OutTensor y, InTensor kernel, InTensor x, int stride=1, int padding=0)
{
    /*
        x: (h, w, c)
        y: (h, w, c)
        kernel: (h, w)
        (h, w, c) -> (h, w, c)
    */
    int hi = x.shape[HWC_H];
    int wi = x.shape[HWC_W];
    int ci = x.shape[HWC_C];
    int hk = kernel.shape[0];
    int wk = kernel.shape[1];
    int ho = std::floor((hi - hk + 2*padding)/stride) + 1;
    int wo = std::floor((wi - wk + 2*padding)/stride) + 1;
    y = Tensor(ho, wo, ci);
    for (int i = 0; i < ho; i++) {
        for (int j = 0; j < wo; j++) {
            for (int c = 0; c < ci; c++) {
                float yi = 0;
                /* kernels */
                for (int u = 0; u < hk; u++) {
                    for (int v = 0; v < wk; v++) {
                        /* map to input  */
                        int row = u + i*stride - padding;
                        int col = v + j*stride - padding;
                        if (row < 0 || row >= hi || col < 0 || col >= wi) {
                            continue;
                        }
                        //y(i, wo - j - 1, c) += kernel(u, v)*x(row, wo - col - 1, c);
                        float kt = kernel(u, v);
                        float xt = x(row, col, c);
                        yi += kt*xt;
                    }
                }
                y(i, j, c) = yi;
            }
        }
    }
    return;
}

int averageBlur(OutTensor xo, InTensor xi, const Size &size);
int gaussian3x3(OutTensor xo, InTensor xi);
int gaussianBlur5x5(OutTensor xo, InTensor xi);
int medianBlur(OutTensor xo, InTensor xi, const Size &size);

int sobel3x3(OutTensor xo, InTensor xi);
int sobel5x5(OutTensor xo, InTensor xi);
int laplacian3x3(OutTensor xo, InTensor xi);
int laplacian5x5(OutTensor xo, InTensor xi);
int prewitt3x3(OutTensor xo, InTensor xi);

int canny(OutTensor xo, InTensor xi, float minThres, float maxThres);

int FFT(Complex *xf, const Complex *xt, int t);
int iFFT(Complex *xt, const Complex *xf, int t);
int FFT2D(Tensor &spectrum, CTensor &xf, const Tensor &img);
int iFFT2D(Tensor &img, const CTensor &xf);
Tensor LPF(int h, int w, int freq);
Tensor gaussHPF(int h, int w, float sigma);
Tensor laplaceFilter(int h, int w);
Tensor invDegenerate(int h, int w);
Tensor invFilter(int h, int w, int rad);
Tensor wienerFilter(int h, int w, float K);

}

#endif // FILTER_H
