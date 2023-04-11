#ifndef FILTER_H
#define FILTER_H
#include <functional>
#include "../basic/statistics.h"
#include "image.hpp"

namespace imp {


/* naive conv2d */
inline void conv2d(Tensor &y, const Tensor &kernel, const Tensor &x, int stride=1, int padding=0)
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

int averageFilter(Tensor &dst, const Tensor &src, const Size &size);
int gaussianFilter3x3(Tensor &dst, const Tensor &src);
int gaussianFilter5x5(Tensor &dst, const Tensor &src);
int medianFilter(Tensor &dst, const Tensor &src, const Size &size);

int sobel3x3(Tensor &dst, const Tensor &src);
int sobel5x5(Tensor &dst, const Tensor &src);
int laplacian3x3(Tensor &dst, const Tensor &src);
int prewitt3x3(Tensor &dst, const Tensor &src);
}

#endif // FILTER_H
