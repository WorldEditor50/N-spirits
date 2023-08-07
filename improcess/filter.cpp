#include "filter.h"

int imp::averageFilter(Tensor &dst, const Tensor &src, const imp::Size &size)
{
    if (size.x != size.y) {
        return -1;
    }
    Tensor kernel(size.x, size.y);
    kernel = 1.0/(size.x*size.y);
    conv2d(dst, kernel, src, 1);
    return 0;
}

int imp::gaussianFilter3x3(Tensor &dst, const Tensor &src)
{
    Tensor kernel({3, 3}, {1.0/16, 2.0/16, 1.0/16,
                           2.0/16, 4.0/16, 2.0/16,
                           1.0/16, 2.0/16, 1.0/16});
    conv2d(dst, kernel, src, 1);
    return 0;
}

int imp::gaussianFilter5x5(Tensor &dst, const Tensor &src)
{
    Tensor kernel({5, 5}, {1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273,
                           4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
                           7.0/273, 26.0/273, 41.0/273, 4.0/273, 7.0/273,
                           4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
                           1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273});
    conv2d(dst, kernel, src, 1);
    return 0;
}

int imp::medianFilter(Tensor &dst, const Tensor &src, const imp::Size &size)
{
    if (size.x != size.y) {
        return -1;
    }
    int stride = 1;
    int padding = 0;
    const Tensor &x = src;
    Tensor &y = dst;
    int ho = std::floor((x.shape[0] - size.x + 2*padding)/stride) + 1;
    int wo = std::floor((x.shape[1] - size.y + 2*padding)/stride) + 1;
    y = Tensor(ho, wo, x.shape[2]);
    Tensor box(size.x, size.y);
    for (int c = 0; c < y.shape[2]; c++) {
        for (int i = 0; i < y.shape[0]; i++) {
            for (int j = 0; j < y.shape[1]; j++) {
                /* kernels */
                for (int h = 0; h < size.x; h++) {
                    for (int k = 0; k < size.y; k++) {
                        /* map to input  */
                        int row = h + i*stride - padding;
                        int col = k + j*stride - padding;
                        if (row < 0 || row >= x.shape[0] ||
                                col < 0 || col >= x.shape[1]) {
                            continue;
                        }
                        box(h, k) = x(row, col, c);
                    }
                }
                /* sort and find middle value */
                std::sort(box.begin(), box.end());
                y(i, wo - j - 1, c) = box(size.x/2, size.y/2);

            }
        }
    }

    return 0;
}

int imp::sobel3x3(Tensor &dst, const Tensor &src)
{
    Tensor kernelx({3, 3}, {-1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1});
    Tensor gradx;
    imp::conv2d(gradx, kernelx, src, 1);
    Tensor kernely({3, 3}, { 1,  2,  1,
                             0,  0,  0,
                            -1, -2, -1});
    Tensor grady;
    imp::conv2d(grady, kernely, src, 1);
    dst = util::sqrt(gradx*gradx + grady*grady);
    return 0;
}

int imp::sobel5x5(Tensor &dst, const Tensor &src)
{
    Tensor kernelx({5, 5}, {-5,  -4,  0, 4,  5,
                            -8,  -10, 0, 10, 8,
                            -10, -20, 0, 20, 10,
                            -8,  -10, 0, 10, 8,
                            -5,  -4,  0, 4,  5});
    Tensor gradx;
    imp::conv2d(gradx, kernelx, src, 1);
    Tensor kernely({5, 5}, {5,   8,  10,   8,  5,
                            4,  10,  20,  10,  4,
                            0,   0,   0,   0,  0,
                           -4, -10, -20, -10, -4,
                           -5,  -8, -10,  -8, -5});
    Tensor grady;
    imp::conv2d(grady, kernely, src, 1);
    //dst = sqrt(gradx*gradx + grady*grady);
    dst = util::sqrt(gradx*gradx + grady*grady);
    return 0;
}

int imp::laplacian3x3(Tensor &dst, const Tensor &src)
{
    Tensor kernel({3, 3}, {0, -1,  0,
                          -1,  4, -1,
                           0, -1,  0});
    conv2d(dst, kernel, src, 1);
    return 0;
}


int imp::prewitt3x3(Tensor &dst, const Tensor &src)
{
    Tensor kernelx({3, 3}, {-1, 0,  1,
                            -1, 0,  1,
                            -1, 0,  1});
    Tensor gradx;
    imp::conv2d(gradx, kernelx, src, 1);
    Tensor kernely({3, 3}, { 1,  1,  1,
                             0,  0,  0,
                            -1, -1, -1});
    Tensor grady;
    imp::conv2d(grady, kernely, src, 1);
    dst = util::sqrt(gradx*gradx + grady*grady);
    return 0;
}
