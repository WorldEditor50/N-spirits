#ifndef FILTER_H
#define FILTER_H
#include <functional>
#include "../basic/util.hpp"
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
            /* kernels */
            for (int u = 0; u < hk; u++) {
                for (int v = 0; v < wk; v++) {
                    /* map to input  */
                    int row = u + i*stride - padding;
                    int col = v + j*stride - padding;
                    if (row < 0 || row >= hi || col < 0 || col >= wi) {
                        continue;
                    }
                    for (int c = 0; c < ci; c++) {
                        y(i, wo - j - 1, c) += kernel(u, v)*x(row, wo - col - 1, c);
                    }
                }
            }
        }
    }
    return;
}

int averageFilter(OutTensor xo, InTensor xi, const Size &size);
int gaussianFilter3x3(OutTensor xo, InTensor xi);
int gaussianFilter5x5(OutTensor xo, InTensor xi);
int medianFilter(OutTensor xo, InTensor xi, const Size &size);

int sobel3x3(OutTensor xo, InTensor xi);
int sobel5x5(OutTensor xo, InTensor xi);
int laplacian3x3(OutTensor xo, InTensor xi);
int prewitt3x3(OutTensor xo, InTensor xi);

int adaptiveMedianFilter(OutTensor xo, InTensor xi, std::size_t kernelSize);

int extendSize(int size);
int FFT(CTensor &xf, const CTensor &xt, int t);
int IFFT(CTensor &xt, const CTensor &xf, int t);
int FFT2D(Tensor &dst, CTensor &f, const Tensor &img, bool expand, unsigned char color);
int IFFT2D(Tensor &dst, const CTensor &xf);
Tensor freqLPF(const Tensor &img, int freq);
Tensor freqGaussHPF(const Tensor &img, float sigma);
Tensor freqLaplaceFilter(const Tensor &img);
Tensor freqInvDegenerate(const Tensor &img);
Tensor freqInvFilter(const Tensor &img, int rad);
Tensor freqWienerFilter(const Tensor &img, float K);
}

#endif // FILTER_H
