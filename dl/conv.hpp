#ifndef CONV_HPP
#define CONV_HPP
#include <cmath>
#include "../basic/tensor.hpp"
#include "../basic/util.hpp"

namespace conv {

inline int out(int i, int kernelSize, int stride, int padding)
{
    return std::floor((i - kernelSize + 2*padding)/stride) + 1;
}

inline void conv1d(Tensor &y, const Tensor &kernels, const Tensor &x, int stride=1, int padding=0)
{
    /*
        x: [n]
        kernels: [c, m]
        y: [c, p]
    */
    for (int i = 0; i < y.shape[0]; i++) {
        for (int j = 0; j < y.shape[1]; j++) {
            for (int h = 0; h < kernels.shape[1]; h++) {
                int k = h + j*stride - padding;
                if (k < 0 || k >= x.totalSize) {
                    continue;
                }
                y(i, j) += kernels(i, h)*x[k];
            }
        }
    }
    return;
}

/* naive conv2d */
inline void eval1(Tensor &y, const Tensor &kernels, const Tensor &x, int stride=1, int padding=0)
{
    /*
        on = bn + Kncij*Xcij
        example:
                in_chanels = 1
                out_channels = 1
                hi = 3
                wi = 3
                kernel_size = 3
                stride = 1
                padding = 1
                ho = (hi - kernel_size + 2*padding)/stride + 1 = 3
                wo = (wi - kernel_size + 2*padding)/stride + 1 = 3

                                kernel_11:

        0   0   0   0   0

        0   1   2   3   0        0    -1     0          -4    -2    -5

        0   4   5   6   0   *   -1     1    -1      =   -9    -15   -11

        0   7   8   9   0        0    -1     0          -5    -13   -5

        0   0   0   0   0

                                kernel_12:                     +

        0   0   0   0   0

        0   1   2   3   0        0    -1     0           2     3     8

        0   4   5   6   0   *    1     0    -1      =    1     4     11

        0   7   8   9   0        0     1     0          -12   -7     2

        0   0   0   0   0

                                kernel_13:                     +
        0   0   0   0   0

        0   1   2   3   0        1     0     1           4     8     2

        0   4   5   6   0   *    0    -1     0      =    6     15    4

        0   7   8   9   0        1     0     1          -2     2    -4

        0   0   0   0   0

                                                               +

                                                         bias_1:

                                                         0     0     0

                                                         0     0     0

                                                         0     0     0

                                                               ||


                                                         2     9     5

                                                        -2     4     4

                                                        -19   -18   -7
    */
    /* output */
    for (int n = 0; n < y.shape[0]; n++) {
        for (int i = 0; i < y.shape[1]; i++) {
            for (int j = 0; j < y.shape[2]; j++) {
                /* kernels */
                for (int c = 0; c < kernels.shape[1]; c++) {
                    for (int h = 0; h < kernels.shape[2]; h++) {
                        for (int k = 0; k < kernels.shape[3]; k++) {
                            /* map to input  */
                            int row = h + i*stride - padding;
                            int col = k + j*stride - padding;
                            if (row < 0 || row >= x.shape[1] ||
                                    col < 0 || col >= x.shape[2]) {
                                continue;
                            }
                            /* sum up all convolution result */
                            y(n, i, j) += kernels(n, c, h, k)*x(c, row, col);
                        }
                    }
                }
            }
        }
    }
    return;
}

/* conv2d */

inline void im2col(Tensor &slice, const Tensor &img,
                  int kernelSize, int stride, int padding,
                  int c, int i, int j)
{
    for (int h = 0; h < kernelSize; h++) {
        for (int k = 0; k < kernelSize; k++) {
            /* map to input  */
            float& val = slice.val[h*kernelSize + k];
            val = 0;
            int row = h + i*stride - padding;
            int col = k + j*stride - padding;
            if (row < 0 || row >= img.shape[1] ||
                    col < 0 || col >= img.shape[2]) {
                continue;
            }
            val = img(c, row, col);
        }
    }
    return;
}

inline void eval2(Tensor &y, const Tensor &kernels, const Tensor &x, int stride=1, int padding=0)
{
    /* avoid visiting tensor's element with operator() */
    int kernelSize = kernels.shape[2];
    Tensor yn(y.shape[1], y.shape[2]);
    Tensor kernel(kernelSize, kernelSize);
    Tensor img(kernelSize, kernelSize);
    for (int n = 0; n < y.shape[0]; n++) {
        for (int c = 0; c < kernels.shape[1]; c++) {
            kernels.slice(kernel, n, c);
            for (int i = 0; i < y.shape[1]; i++) {
                for (int j = 0; j < y.shape[2]; j++) {
                    /* image subset to vector */
                    im2col(img, x, kernelSize, stride, padding, c, i, j);
                    /* convolution */
                    float s = util::dot(kernel, img);
                    y(n, i, j) += s;
                }
            }
        }
    }
    return;
}


}
#endif // CONV_HPP
