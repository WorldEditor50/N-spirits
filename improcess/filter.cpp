#include "filter.h"
#include <algorithm>

int imp::averageFilter(OutTensor xo, InTensor xi, const imp::Size &size)
{
    if (size.x != size.y) {
        return -1;
    }
    Tensor kernel(size.x, size.y);
    kernel = 1.0/(size.x*size.y);
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int imp::gaussianFilter3x3(OutTensor xo, InTensor xi)
{
    Tensor kernel({3, 3}, {1.0/16, 2.0/16, 1.0/16,
                           2.0/16, 4.0/16, 2.0/16,
                           1.0/16, 2.0/16, 1.0/16});
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int imp::gaussianFilter5x5(OutTensor xo, InTensor xi)
{
    Tensor kernel({5, 5}, {1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273,
                           4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
                           7.0/273, 26.0/273, 41.0/273, 4.0/273, 7.0/273,
                           4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
                           1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273});
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int imp::medianFilter(OutTensor xo, InTensor xi, const imp::Size &size)
{
    if (size.x != size.y) {
        return -1;
    }
    int stride = 1;
    int padding = 0;
    int ho = std::floor((xi.shape[0] - size.x + 2*padding)/stride) + 1;
    int wo = std::floor((xi.shape[1] - size.y + 2*padding)/stride) + 1;
    xo = Tensor(ho, wo, xi.shape[2]);
    Tensor box(size.x, size.y);
    for (int c = 0; c < xo.shape[2]; c++) {
        for (int i = 0; i < xo.shape[0]; i++) {
            for (int j = 0; j < xo.shape[1]; j++) {
                /* kernels */
                for (int h = 0; h < size.x; h++) {
                    for (int k = 0; k < size.y; k++) {
                        /* map to input  */
                        int row = h + i*stride - padding;
                        int col = k + j*stride - padding;
                        if (row < 0 || row >= xi.shape[0] ||
                                col < 0 || col >= xi.shape[1]) {
                            continue;
                        }
                        box(h, k) = xi(row, wo - col - 1, c);
                    }
                }
                /* sort and find middle value */
                std::sort(box.begin(), box.end());
                xo(i, wo - j - 1, c) = box(size.x/2, size.y/2);

            }
        }
    }

    return 0;
}

int imp::sobel3x3(OutTensor xo, InTensor xi)
{
    Tensor kernelx({3, 3}, {-1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1});
    Tensor gradx;
    imp::conv2d(gradx, kernelx, xi, 1);
    Tensor kernely({3, 3}, { 1,  2,  1,
                             0,  0,  0,
                            -1, -2, -1});
    Tensor grady;
    imp::conv2d(grady, kernely, xi, 1);
    xo = LinAlg::sqrt(gradx*gradx + grady*grady)/2;
    //xo = gradx + grady;
    return 0;
}

int imp::sobel5x5(OutTensor xo, InTensor xi)
{
    Tensor kernelx({5, 5}, {-5,  -4,  0, 4,  5,
                            -8,  -10, 0, 10, 8,
                            -10, -20, 0, 20, 10,
                            -8,  -10, 0, 10, 8,
                            -5,  -4,  0, 4,  5});
    Tensor gradx;
    imp::conv2d(gradx, kernelx, xi, 1);
    Tensor kernely({5, 5}, {5,   8,  10,   8,  5,
                            4,  10,  20,  10,  4,
                            0,   0,   0,   0,  0,
                           -4, -10, -20, -10, -4,
                           -5,  -8, -10,  -8, -5});
    Tensor grady;
    imp::conv2d(grady, kernely, xi, 1);
    xo = LinAlg::sqrt(gradx*gradx + grady*grady);
    return 0;
}

int imp::laplacian3x3(OutTensor xo, InTensor xi)
{
    Tensor kernel({3, 3}, {0, -1,  0,
                          -1,  4, -1,
                           0, -1,  0});
    conv2d(xo, kernel, xi, 1);
    return 0;
}


int imp::prewitt3x3(OutTensor xo, InTensor xi)
{
    Tensor kernel0({3, 3}, { 1,  1,  1,
                             0,  0,  0,
                            -1, -1, -1});

    Tensor grad0;
    imp::conv2d(grad0, kernel0, xi, 1);
    Tensor kernel45({3, 3}, {-1, -1,  0,
                             -1,  0,  1,
                              0,  1,  1});

    Tensor grad45;
    imp::conv2d(grad45, kernel45, xi, 1);

    Tensor kernel90({3, 3}, {-1, 0,  1,
                             -1, 0,  1,
                             -1, 0,  1});

    Tensor grad90;
    imp::conv2d(grad90, kernel90, xi, 1);

    Tensor kernel135({3, 3}, { 0,  1,  1,
                              -1,  0,  1,
                              -1, -1,  0});

    Tensor grad135;
    imp::conv2d(grad135, kernel135, xi, 1);
    xo = LinAlg::sqrt(grad0*grad0 + grad45*grad45 + grad90*grad90 + grad135*grad135);
    return 0;
}


int imp::adaptiveMedianFilter(OutTensor xo, InTensor xi, size_t kernelSize)
{
    return 0;
}

int imp::extendSize(int size)
{
    int extSize = 1;
    while (extSize*2 <= size) {
        extSize *= 2;
    }
    if (extSize != size) {
        extSize *= 2;
    }
    return 0;
}

int imp::FFT(CTensor &xf, const CTensor &xt, int t)
{
    int length = 1 << t;
    CTensor w(length/2);
    CTensor x1(length);
    CTensor x2(length);
    CTensor x(length);
    /* init weight */
    for (std::size_t i = 0; i < w.totalSize; i++) {
        float theta = -2*pi*i/float(length);
        w[i] = Complex(std::cos(theta), std::sin(theta));
    }
    /* align */
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        x1[i] = xt[i];
    }
    /* FFT */
    for (int k = 0; k < t; k++) {
        for (int j = 0; j < (1 << k); j++) {
            int bfsize = 1 << (t - k);
            for (int i = 0; i < bfsize/2; i++) {
                int p = j*bfsize;
                x2[i + p] = x1[i + p] + x1[i + p + bfsize/2];
                x2[i + p + bfsize/2] = w[i*(1<<k)]*(x1[i + p] - x1[i + p + bfsize/2]);
            }
        }
        x = x1;
        x1 = x2;
        x2 = x;
    }
    /* sort */
    for (int j = 0; j < length; j++) {
        int p = 0;
        for (int i = 0; i < t; i++) {
            if (j&(1<<i)) {
                p += 1<<(t - i - 1);
            }
        }
        xf[j] = x1[p];
    }
    return 0;
}

int imp::IFFT(CTensor &xt, const CTensor &xf, int t)
{
    int length = 1 << t;
    CTensor x(length);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x[i] = xf[i];
    }
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x[i] = x[i].conjugate();
    }
    FFT(x, xt, t);
    for (std::size_t i = 0; i < xt.totalSize; i++) {
        xt[i] = xt[i].conjugate()/float(length);
    }
    return 0;
}

int imp::FFT2D(Tensor &dst, CTensor &f, const Tensor &img, bool expand, unsigned char color)
{

    int w = 1;
    int h = 1;
    int wp = 0;
    int hp = 0;
    while (w*2 <= img.shape[HWC_W]) {
        w *= 2;
        wp++;
    }
    while (h*2 <= img.shape[HWC_H]) {
        h *= 2;
        hp++;
    }
    if (expand && w != img.shape[HWC_W] && h != img.shape[HWC_H]) {
        w *= 2;
        wp++;
        h *= 2;
        hp++;
    }
    
    CTensor xt(h, w);
    CTensor xf(h, w);
    
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (expand) {
                if (i > img.shape[HWC_H] && j > img.shape[HWC_W]) {
                    xt(i, j) = Complex(img(i, j), 0);
                } else {
                    xt(i, j) = Complex(color, 0);
                }
            } else {
                xt(i, j) = Complex(img(i, j), 0);
            }
        }
    }
    for (int i = 0; i < w; i++) {

    }
    for (int i = 0; i < h; i++) {

    }
    return 0;
}

int imp::IFFT2D(Tensor &dst, const CTensor &xf)
{

    return 0;
}

Tensor imp::freqLPF(const Tensor &img, int freq)
{
    int h = extendSize(img.shape[HWC_H]);
    int w = extendSize(img.shape[HWC_W]);
    Tensor H(h, w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float f = (i - h/2)*(i - h/2) + (j - w/2)*(j - w/2);
            int u = i < h/2 ? i + h/2 : i - h/2;
            int v = j < w/2 ? j + w/2 : j - w/2;
            if (std::sqrt(f) > freq) {
                H(u, v) = 1;
            } else {
                H(u, v) = 0;
            }

        }
    }
    return H;
}

Tensor imp::freqGaussHPF(const Tensor &img, float sigma)
{
    /*
        H(u, v) = 1 - e^(-[(u - M/2)^2 + (v - N/2)^2]/2)/sigma^2)

    */
    int h = extendSize(img.shape[HWC_H]);
    int w = extendSize(img.shape[HWC_W]);
    Tensor H(h, w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float f = (i - h/2)*(i - h/2) + (j - w/2)*(j - w/2);
            int u = i < h/2 ? i + h/2 : i - h/2;
            int v = j < w/2 ? j + w/2 : j - w/2;
            H(u, v) = 1 - std::exp(-f/2)/(sigma*sigma);

        }
    }
    return H;
}

Tensor imp::freqLaplaceFilter(const Tensor &img)
{
    int h = extendSize(img.shape[HWC_H]);
    int w = extendSize(img.shape[HWC_W]);
    Tensor H(h, w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float f = (i - h/2)*(i - h/2) + (j - w/2)*(j - w/2);
            int u = i < h/2 ? i + h/2 : i - h/2;
            int v = j < w/2 ? j + w/2 : j - w/2;
            H(u, v) = -f;

        }
    }
    return H;
}

Tensor imp::freqInvDegenerate(const Tensor &img)
{
    /*
        H(u, v) = exp(k*((u - M/2)^2 + (v - N/2)^2)^(5/6))
    */
    int h = extendSize(img.shape[HWC_H]);
    int w = extendSize(img.shape[HWC_W]);
    Tensor H(h, w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float f = (i - h/2)*(i - h/2) + (j - w/2)*(j - w/2);
            f = std::pow(f, 5.0/6);
            f *= -0.0025;
            f = std::exp(f);
            int u = i < h/2 ? i + h/2 : i - h/2;
            int v = j < w/2 ? j + w/2 : j - w/2;
            H(u, v) = f;

        }
    }
    return H;
}

Tensor imp::freqInvFilter(const Tensor &img, int rad)
{
    int h = extendSize(img.shape[HWC_H]);
    int w = extendSize(img.shape[HWC_W]);
    Tensor H(h, w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float f = (i - h/2)*(i - h/2) + (j - w/2)*(j - w/2);
            float d = std::sqrt(f);
            if (d <= rad) {
                f = std::pow(f, 5.0/6);
                f *= -0.0025;
                f = std::exp(f);
                int u = i < h/2 ? i + h/2 : i - h/2;
                int v = j < w/2 ? j + w/2 : j - w/2;
                H(u, v) = 1.0/(f + 1e-5);
            }

        }
    }
    return H;
}

Tensor imp::freqWienerFilter(const Tensor &img, float K)
{
    int h = extendSize(img.shape[HWC_H]);
    int w = extendSize(img.shape[HWC_W]);
    Tensor H(h, w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float f = (i - h/2)*(i - h/2) + (j - w/2)*(j - w/2);
            f = std::pow(f, 5.0/6);
            f *= -0.0025;
            f = std::exp(f);
            int u = i < h/2 ? i + h/2 : i - h/2;
            int v = j < w/2 ? j + w/2 : j - w/2;
            H(u, v) = f*f/(f*f*K*(f + 1e-5));

        }
    }
    return H;
}

