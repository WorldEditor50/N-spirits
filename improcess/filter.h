#ifndef FILTER_H
#define FILTER_H
#include <functional>
#include "../basic/util.hpp"
#include "../basic/ctensor.hpp"
#include "../basic/complexnumber.h"
#include "image.hpp"

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
    int ho = std::floor((x.shape[0] - kernel.shape[0] + 2*padding)/stride) + 1;
    int wo = std::floor((x.shape[1] - kernel.shape[1] + 2*padding)/stride) + 1;
    y = Tensor(ho, wo, x.shape[2]);
    for (int i = 0; i < y.shape[0]; i++) {
        for (int j = 0; j < y.shape[1]; j++) {
            /* kernels */
            for (int h = 0; h < kernel.shape[0]; h++) {
                for (int k = 0; k < kernel.shape[1]; k++) {
                    /* map to input  */
                    int row = h + i*stride - padding;
                    int col = k + j*stride - padding;
                    if (row < 0 || row >= x.shape[0] ||
                            col < 0 || col >= x.shape[1]) {
                        continue;
                    }
                    for (int c = 0; c < y.shape[2]; c++) {
                        y(i, wo - j - 1, c) += kernel(h, k)*x(row, col, c);
                        //y(i, j, c) += kernel(h, k)*x(row, wo - col - 1, c);
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
