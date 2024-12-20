#include "filter.h"
#include <algorithm>

int ns::averageBlur(OutTensor xo, InTensor xi, const ns::Size &size)
{
    if (size.x != size.y) {
        return -1;
    }
    Tensor kernel(size.x, size.y);
    kernel = 1.0/(size.x*size.y);
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int ns::gaussianBlur3x3(OutTensor xo, InTensor xi)
{
    Tensor kernel({3, 3}, {1.0/16, 2.0/16, 1.0/16,
                           2.0/16, 4.0/16, 2.0/16,
                           1.0/16, 2.0/16, 1.0/16});
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int ns::gaussianBlur5x5(OutTensor xo, InTensor xi)
{
    Tensor kernel({5, 5}, {1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273,
                           4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
                           7.0/273, 26.0/273, 41.0/273, 4.0/273, 7.0/273,
                           4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
                           1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273});
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int ns::medianBlur(OutTensor xo, InTensor xi, const ns::Size &size)
{
    if (size.x != size.y) {
        return -1;
    }
    int stride = 1;
    int padding = 0;
    int h = xi.shape[0];
    int w = xi.shape[1];
    int ho = std::floor((xi.shape[0] - size.x + 2*padding)/stride) + 1;
    int wo = std::floor((xi.shape[1] - size.y + 2*padding)/stride) + 1;
    xo = Tensor(ho, wo, xi.shape[2]);
    Tensor box(size.x, size.y);
    for (int c = 0; c < xo.shape[2]; c++) {
        for (int i = 0; i < xo.shape[0]; i++) {
            for (int j = 0; j < xo.shape[1]; j++) {
                /* kernels */
                for (int u = 0; u < size.x; u++) {
                    for (int v = 0; v < size.y; v++) {
                        /* map to input  */
                        int ui = u + i*stride - padding;
                        int vj = v + j*stride - padding;
                        if (ui < 0 || ui >= h || vj < 0 || vj >= w) {
                            continue;
                        }
                        box(u, v) = xi(ui, vj, c);
                    }
                }
                /* sort and find middle value */
                std::sort(box.begin(), box.end());
                xo(i, j, c) = box(size.x/2, size.y/2);

            }
        }
    }
    return 0;
}


int ns::bilateralBlur(OutTensor xo, InTensor xi, const ns::Size &size, float sigma1, float sigma2)
{
    int h = xi.shape[0];
    int w = xi.shape[1];
    int c = xi.shape[2];
    int kernelSize1 = size.x;
    int kernelSize2 = size.y;
    int padding1 = kernelSize1/2;
    int padding2 = kernelSize2/2;
    int stride1 = 1;
    int stride2 = 1;
    int km1 = kernelSize1/2;
    int km2 = kernelSize2/2;
    int ho = std::floor(float(h - kernelSize1 + 2*padding1)/stride1) + 1;
    int wo = std::floor(float(w - kernelSize2 + 2*padding2)/stride2) + 1;
    xo = Tensor(ho, wo, c);
    for (int i = 0; i < ho; i++) {
        for (int j = 0; j < wo; j++) {
            for (int k = 0; k < c; k++) {
                int um = km1 + i*stride1 - padding1;
                int vm = km2 + j*stride2 - padding2;
                if (um < 0 || vm < 0 || um >= h || vm >= w) {
                    continue;
                }
                float xw = 0;
                float sw = 0;
                for (int u = 0; u < kernelSize1; u++) {
                    for (int v = 0; v < kernelSize2; v++) {
                        int ui = u + i*stride1 - padding1;
                        int vj = v + j*stride2 - padding2;
                        if (ui < 0 || vj < 0 || ui >= h || vj >= w) {
                            continue;
                        }
                        /* spatial kernel */
                        float du = u - km1;
                        float dv = v - km2;
                        float w1 = std::exp(-(du*du + dv*dv)/(2*sigma1*sigma1));
                        /* pixel kernel */
                        float delta = xi(ui, vj, k) - xi(um, vm, k);
                        float w2 = std::exp(-(delta*delta)/(2*sigma2*sigma2));
                        xw += w1*w2*xi(ui, vj, k);
                        sw += w1*w2;
                    }
                }
                xo(i, j, k) = xw/sw;
            }
        }
    }
    return 0;
}

int ns::curvatureBlur3x3(OutTensor xo, InTensor xi)
{
    int h = xi.shape[0];
    int w = xi.shape[1];
    int c = xi.shape[2];
    xo = Tensor(h, w, c);
    Tensor d(3, 3);
    for (int k = 0; k < c; k++) {
        for (int i = 1; i < h - 1; i++) {
            for (int j = 1; j < w - 1; j++) {
                d[0] = (xi(i - 1, j, k) + xi(i + 1, j, k))/2 - xi(i, j, k);
                d[1] = (xi(i, j - 1, k) + xi(i, j + 1, k))/2 - xi(i, j, k);
                d[2] = (xi(i - 1, j - 1, k) + xi(i + 1, j + 1, k))/2 - xi(i, j, k);
                d[3] = (xi(i - 1, j + 1, k) + xi(i + 1, j - 1, k))/2 - xi(i, j, k);

                d[4] = xi(i - 1, j, k) + xi(i, j - 1, k) - xi(i - 1, j - 1, k) - xi(i, j, k);
                d[5] = xi(i - 1, j, k) + xi(i, j + 1, k) - xi(i - 1, j + 1, k) - xi(i, j, k);
                d[6] = xi(i, j - 1, k) + xi(i + 1, j, k) - xi(i + 1, j - 1, k) - xi(i, j, k);
                d[7] = xi(i, j + 1, k) + xi(i + 1, j, k) - xi(i + 1, j + 1, k) - xi(i, j, k);
                /* sort and find middle value */
                std::sort(d.begin(), d.end());
                xo(i, j, k) = xi(i, j , k) + d(1, 1);
            }
        }
    }
    return 0;
}

int ns::sobel3x3(OutTensor xo, InTensor xi)
{
    Tensor kernelx({3, 3}, {-1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1});
    Tensor gradx;
    ns::conv2d(gradx, kernelx, xi, 1);
    Tensor kernely({3, 3}, { 1,  2,  1,
                             0,  0,  0,
                            -1, -2, -1});
    Tensor grady;
    ns::conv2d(grady, kernely, xi, 1);
    xo = LinAlg::sqrt(gradx*gradx + grady*grady);
    return 0;
}

int ns::sobel5x5(OutTensor xo, InTensor xi)
{
    Tensor kernelx({5, 5}, {-5,  -4,  0, 4,  5,
                            -8,  -10, 0, 10, 8,
                            -10, -20, 0, 20, 10,
                            -8,  -10, 0, 10, 8,
                            -5,  -4,  0, 4,  5});
    Tensor gradx;
    ns::conv2d(gradx, kernelx, xi, 1);
    Tensor kernely({5, 5}, {5,   8,  10,   8,  5,
                            4,  10,  20,  10,  4,
                            0,   0,   0,   0,  0,
                           -4, -10, -20, -10, -4,
                           -5,  -8, -10,  -8, -5});
    Tensor grady;
    ns::conv2d(grady, kernely, xi, 1);
    xo = LinAlg::sqrt(gradx*gradx + grady*grady);
    return 0;
}

int ns::laplacian3x3(OutTensor xo, InTensor xi)
{
    Tensor kernel({3, 3}, {0, -1,  0,
                          -1,  4, -1,
                           0, -1,  0});
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int ns::laplacian5x5(OutTensor xo, InTensor xi)
{
    Tensor kernel({5, 5}, {0,   0, -1,  0,  0,
                           0,  -1, -2, -1,  0,
                          -1,  -2, 16, -2, -1,
                           0,  -1, -2, -1,  0,
                           0,   0, -1,  0,  0});
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int ns::prewitt3x3(OutTensor xo, InTensor xi)
{
    Tensor kernel0({3, 3}, { 1,  1,  1,
                             0,  0,  0,
                            -1, -1, -1});

    Tensor grad0;
    ns::conv2d(grad0, kernel0, xi, 1);
    Tensor kernel45({3, 3}, {-1, -1,  0,
                             -1,  0,  1,
                              0,  1,  1});

    Tensor grad45;
    ns::conv2d(grad45, kernel45, xi, 1);

    Tensor kernel90({3, 3}, {-1, 0,  1,
                             -1, 0,  1,
                             -1, 0,  1});

    Tensor grad90;
    ns::conv2d(grad90, kernel90, xi, 1);

    Tensor kernel135({3, 3}, { 0,  1,  1,
                              -1,  0,  1,
                              -1, -1,  0});

    Tensor grad135;
    ns::conv2d(grad135, kernel135, xi, 1);
    xo = LinAlg::sqrt(grad0*grad0 + grad45*grad45 + grad90*grad90 + grad135*grad135);
    return 0;
}

int ns::scharr3x3(OutTensor xo, InTensor xi)
{
    Tensor kernelx({3, 3}, {-3,  0, 3,
                            -10, 0, 10,
                            -3,  0, 3});
    Tensor gradx;
    ns::conv2d(gradx, kernelx, xi, 1);
    Tensor kernely({3, 3}, {-3, -10, -3,
                             0,   0,  0,
                             3,  10,  3});
    Tensor grady;
    ns::conv2d(grady, kernely, xi, 1);
    xo = LinAlg::sqrt(gradx*gradx + grady*grady);
    return 0;
}

int ns::LOG5x5(OutTensor xo, InTensor xi)
{
    Tensor kernel({5, 5}, {-2,  -4, -4, -4, -2,
                           -4,   0,  8,  0, -4,
                           -4,   8, 24,  8, -4,
                           -4,   0,  8,  0, -4,
                           -2,  -4, -4, -4, -2});
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int ns::canny(OutTensor xo, InTensor xi, float minThres, float maxThres)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    /* step1: gaussian blur */
    Tensor img;
    gaussianBlur5x5(img, xi);
    /* step2: grad and theta */
    Tensor kx({3, 3}, {-1, 0, 1,
                       -2, 0, 2,
                       -1, 0, 1});
    Tensor ky({3, 3}, { -1,  -2,  -1,
                         0,   0,   0,
                         1,   2,   1});
    Tensor imgx;
    conv2d(imgx, kx, img, 1, 0);
    Tensor imgy;
    conv2d(imgy, ky, img, 1, 0);
    Tensor g = LinAlg::sqrt(imgx*imgx + imgy*imgy);
    Tensor theta(g.shape);
    for (std::size_t i = 0; i < theta.totalSize; i++) {
        theta[i] = std::atan2(imgy[i], imgx[i] + 1e-9)*180/pi;
    }

    /* step3: Non-Maximum Suppression */
    int h = g.shape[HWC_H];
    int w = g.shape[HWC_W];
    Tensor mask(g.shape);
    for (int i = 1; i < h - 1; i++) {
        for (int j = 1; j < w - 1; j++) {
            if (g(i, j) == 0) {
                continue;
            }
            float thetaij = theta(i, j);
            float p1 = 0;
            float p2 = 0;
            if (thetaij >= 0 && thetaij < 45) {
                float g1 = g(i + 1, j - 1);
                float g2 = g(i + 1, j);
                float g3 = g(i - 1, j + 1);
                float g4 = g(i - 1, j);
                float w = std::abs(std::tan(thetaij*pi/180));
                p1 = w*g1 + (1 - w)*g2;
                p2 = w*g3 + (1 - w)*g4;
            } else if (thetaij >= 45 && thetaij < 90) {
                float g1 = g(i + 1, j - 1);
                float g2 = g(i, j - 1);
                float g3 = g(i - 1, j + 1);
                float g4 = g(i, j + 1);
                float w = std::abs(std::tan((thetaij - 90)*pi/180));
                p1 = w*g1 + (1 - w)*g2;
                p2 = w*g3 + (1 - w)*g4;
            } else if (thetaij >= -90 && thetaij < -45) {
                float g1 = g(i - 1, j - 1);
                float g2 = g(i, j - 1);
                float g3 = g(i + 1, j + 1);
                float g4 = g(i, j + 1);
                float w = std::abs(std::tan((thetaij - 90)*pi/180));
                p1 = w*g1 + (1 - w)*g2;
                p2 = w*g3 + (1 - w)*g4;
            } else if (thetaij >= -45 && thetaij < 0) {
                float g1 = g(i + 1, j + 1);
                float g2 = g(i + 1, j);
                float g3 = g(i - 1, j - 1);
                float g4 = g(i - 1, j);
                float w = std::abs(std::tan(thetaij*pi/180));
                p1 = w*g1 + (1 - w)*g2;
                p2 = w*g3 + (1 - w)*g4;
            }
            float p = g(i, j);
            if (p > p1 && p > p2) {
                mask(i, j) = p;
            }
        }
    }

    /* step4: threshold */
    Tensor strongEdge(g.shape);
    Tensor weakEdge(g.shape);

    int hi[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int hj[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int i = 1; i < h - 1; i++) {
        for (int j = 1; j < w - 1; j++) {
            float p = mask(i, j);
            if (p == 0) {
                continue;
            }
            if (p > maxThres) {
                strongEdge(i, j) = 255;
            } else if (p < maxThres && p > minThres) {
                for (int k = 0; k < 8; k++) {
                    float u = i + hi[k];
                    float v = j + hj[k];
                    if (mask(u, v) > maxThres) {
                        weakEdge(i, j) = 255;
                        break;
                    }
                }
            }
        }
    }
    xo = strongEdge + weakEdge;
    return 0;
}

int ns::FFT(Complex *xf, const Complex *xt, int t)
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

int ns::iFFT(Complex *xt, const Complex *xf, int t)
{
    int length = 1 << t;
    CTensor xfi(length);
    for (int i = 0; i < length; i++) {
        xfi[i] = xf[i].conjugate();
    }
    FFT(xt, xfi.ptr(), t);
    for (int i = 0; i < length; i++) {
        xt[i] = xt[i].conjugate()/float(length);
    }
    return 0;
}

int ns::FFT2D(Tensor &spectrum, CTensor &xf, const Tensor &img)
{
    if (img.shape[HWC_C] != 1) {
        return -1;
    }
    int th = std::floor(std::log2(img.shape[0])) + 1;
    int tw = std::floor(std::log2(img.shape[1])) + 1;
    int w = 1 << tw;//std::pow(2, tw);
    int h = 1 << th;//std::pow(2, th);
    int ph = (h - img.shape[0])/2;
    int pw = (w - img.shape[1])/2;
    CTensor xt(h, w);
    xf = CTensor(h, w);
    //std::cout<<"h:"<<h<<",w:"<<w<<std::endl;
    /* copy image */
    for (int i = 0; i < img.shape[0]; i++) {
        for (int j = 0; j < img.shape[1]; j++) {
            float p = img(i, j, 0);
            xt(i + ph, j + pw).re = p;
            xt(i + ph, j + pw).im = 0;
        }
    }
    /* FFT on row */
    for (int i = 0; i < h; i++) {
        Complex* xti = xt.ptr() + w*i;
        Complex* xfi = xf.ptr() + w*i;
        FFT(xfi, xti, tw);
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            xt[j*h + i] = xf[i*w + j];
        }
    }
    /* FFT on column */
    for (int i = 0; i < w; i++) {
        Complex* xti = xt.ptr() + h*i;
        Complex* xfi = xf.ptr() + h*i;
        FFT(xfi, xti, th);
    }
    CTensor xf_= xf.tr();
    xf = xf_;
    //xf.printValue();
    /* spectrum */
    spectrum = Tensor(h, w, 1);
    float maxVal = 0;
    float minVal = 1e9;
    for (std::size_t i = 0; i < xf.totalSize; i++) {
        float d = xf[i].modulus()/100;
        d = std::log(1 + d);
        maxVal = std::max(maxVal, d);
        minVal = std::min(minVal, d);
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float d = xf[j*h + i].modulus()/100;
            d = std::log(1 + d);
            float p = (d - minVal)/(maxVal - minVal)*255.0;
            int u = i < h/2 ? i + h/2 : i - h/2;
            int v = j < w/2 ? j + w/2 : j - w/2;
            spectrum(u, v, 0) = p;
        }
    }
    return 0;
}

int ns::iFFT2D(Tensor &img, const CTensor &xf_)
{
    CTensor xf = xf_;
    CTensor xt(xf.shape);
    int h = xf.shape[0];
    int w = xf.shape[1];
    int th = std::log2(h);
    int tw = std::log2(w);
    /* iFFT on row */
    for (int i = 0; i < h; i++) {
        Complex* xti = xt.ptr() + w*i;
        Complex* xfi = xf.ptr() + w*i;
        iFFT(xti, xfi, tw);
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            xf[j*h + i] = xt[i*w + j];
        }
    }
    /* iFFT on column */
    for (int i = 0; i < w; i++) {
        Complex* xti = xt.ptr() + h*i;
        Complex* xfi = xf.ptr() + h*i;
        iFFT(xti, xfi, th);
    }
    /* image */
    img = Tensor(h, w, 1);
    float maxVal = 0;
    float minVal = 1e9;
    for (std::size_t i = 0; i < xt.totalSize; i++) {
        float d = xt[i].modulus();
        maxVal = std::max(maxVal, d);
        minVal = std::min(minVal, d);
    }
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float d = xt[j*h + i].modulus();
            float p = (d - minVal)/(maxVal - minVal)*255.0;
            img(i, j, 0) = p;
        }
    }
    return 0;
}

int ns::HarrWavelet2D(OutTensor xo, InTensor xi, int depth)
{
    Tensor row(xi.shape);
    Tensor col(xi.shape);
    Tensor wavelet = xi;
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    int depth_ = 1;
    while (depth_ <= depth) {
        h = xi.shape[HWC_H]/depth_;
        w = xi.shape[HWC_W]/depth_;
        /* wavelet on rows */
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w/2; j++) {
                for (int k = 0; k < c; k++) {
                    /* mean */
                    row(i, j, k) = (wavelet(i, 2*j, k) + wavelet(i, 2*j + 1, k))*0.5;
                    /* delta */
                    row(i, j + w/2, k) = (wavelet(i, 2*j, k) - wavelet(i, 2*j + 1, k))*0.5;
                }
            }
        }
        /* wavelet on columns */
        for (int i = 0; i < h / 2; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < c; k++) {
                    col(i, j, k) = (row(2*i, j, k) + row(2*i + 1, j, k))*0.5;
                    col(i + h/2, j, k) = (row(2*i, j, k) - row(2*i + 1, j, k))*0.5;
                }
            }
        }
        wavelet = col;
        depth_++;
    }
    xo = wavelet;
    return 0;
}

int ns::iHarrWavelet2D(OutTensor xo, InTensor xi, int depth)
{
    Tensor row(xi.shape);
    Tensor col = xi;
    Tensor iWavelet = xi;
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    int depth_ = depth;
    while (depth_ > 0) {
        h = xi.shape[HWC_H]/depth_;
        w = xi.shape[HWC_W]/depth_;
        for (int i = 0; i < h - 1; i+=2) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < c; k++) {
                    row(i, j, k) = iWavelet(i/2, j, k) + iWavelet(i/2 + h/2, j, k);
                    row(i + 1, j, k) = iWavelet(i/2, j, k) - iWavelet(i/2 + h/2, j, k);
                }
            }
        }

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w - 1; j+=2) {
                for (int k = 0; k < c; k++) {
                    col(i, j, k) = row(i, j/2, k) + row(i, j/2 + w/2, k);
                    col(i, j + 1, k) = row(i, j/2, k) - row(i, j/2 + w/2, k);
                }
            }
        }
        iWavelet = col;
        depth_--;
    }
    xo = iWavelet;
    return 0;
}

Tensor ns::LPF(int h, int w, int freq)
{
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

Tensor ns::gaussHPF(int h, int w, float sigma)
{
    /*
        H(u, v) = 1 - e^(-[(u - M/2)^2 + (v - N/2)^2]/2)/sigma^2)
    */
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

Tensor ns::laplaceFilter(int h, int w)
{
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

Tensor ns::invDegenerate(int h, int w)
{
    /*
        H(u, v) = exp(k*((u - M/2)^2 + (v - N/2)^2)^(5/6))
    */
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

Tensor ns::invFilter(int h, int w, int rad)
{
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

Tensor ns::wienerFilter(int h, int w, float K)
{
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
