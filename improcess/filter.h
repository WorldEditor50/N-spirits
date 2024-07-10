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
    int h = x.shape[HWC_H];
    int w = x.shape[HWC_W];
    int c = x.shape[HWC_C];
    int kernelSize0 = kernel.shape[0];
    int kernelSize1 = kernel.shape[1];
    int ho = (h - kernelSize0 + 2*padding)/stride + 1;
    int wo = (w - kernelSize1 + 2*padding)/stride + 1;
    y = Tensor(ho, wo, c);
    for (int i = 0; i < ho; i++) {
        for (int j = 0; j < wo; j++) {
            for (int k = 0; k < c; k++) {
                float yijk = 0;
                /* kernels */
                for (int u = 0; u < kernelSize0; u++) {
                    for (int v = 0; v < kernelSize1; v++) {
                        /* map to input  */
                        int ui = u + i*stride - padding;
                        int vj = v + j*stride - padding;
                        if (ui < 0 || ui >= h || vj < 0 || vj >= w) {
                            continue;
                        }
                        yijk += kernel(u, v)*x(ui, w - 1 - vj, k);

                    }
                }
                y(i, j, k) = yijk;
            }
        }
    }
    return;
}

int averageBlur(OutTensor xo, InTensor xi, const Size &size);
int gaussianBlur3x3(OutTensor xo, InTensor xi);
int gaussianBlur5x5(OutTensor xo, InTensor xi);
int medianBlur(OutTensor xo, InTensor xi, const Size &size);

int sobel3x3(OutTensor xo, InTensor xi);
int sobel5x5(OutTensor xo, InTensor xi);
int laplacian3x3(OutTensor xo, InTensor xi);
int laplacian5x5(OutTensor xo, InTensor xi);
int prewitt3x3(OutTensor xo, InTensor xi);
int scharr3x3(OutTensor xo, InTensor xi);
int LOG5x5(OutTensor xo, InTensor xi);
int canny(OutTensor xo, InTensor xi, float minThres, float maxThres);

int FFT(Complex *xf, const Complex *xt, int t);
int iFFT(Complex *xt, const Complex *xf, int t);
int FFT2D(Tensor &spectrum, CTensor &xf, const Tensor &img);
int iFFT2D(Tensor &img, const CTensor &xf);
int wavelet2D(OutTensor xo, InTensor img, int level);
int iWavelet2D(OutTensor xo, InTensor img, int level);
Tensor LPF(int h, int w, int freq);
Tensor gaussHPF(int h, int w, float sigma);
Tensor laplaceFilter(int h, int w);
Tensor invDegenerate(int h, int w);
Tensor invFilter(int h, int w, int rad);
Tensor wienerFilter(int h, int w, float K);

}

#endif // FILTER_H
