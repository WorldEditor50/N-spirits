#include "filter.h"
#include <algorithm>

int imp::averageBlur(OutTensor xo, InTensor xi, const imp::Size &size)
{
    if (size.x != size.y) {
        return -1;
    }
    Tensor kernel(size.x, size.y);
    kernel = 1.0/(size.x*size.y);
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int imp::gaussian3x3(OutTensor xo, InTensor xi)
{
    Tensor kernel({3, 3}, {1.0/16, 2.0/16, 1.0/16,
                           2.0/16, 4.0/16, 2.0/16,
                           1.0/16, 2.0/16, 1.0/16});
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int imp::gaussianBlur5x5(OutTensor xo, InTensor xi)
{
    Tensor kernel({5, 5}, {1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273,
                           4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
                           7.0/273, 26.0/273, 41.0/273, 4.0/273, 7.0/273,
                           4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
                           1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273});
    conv2d(xo, kernel, xi, 1);
    return 0;
}

int imp::medianBlur(OutTensor xo, InTensor xi, const imp::Size &size)
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

int imp::laplacian5x5(OutTensor xo, InTensor xi)
{
    Tensor kernel({5, 5}, {0,   0, -1,  0,  0,
                           0,  -1, -2, -1,  0,
                          -1,  -2, 16, -2, -1,
                           0,  -1, -2, -1,  0,
                           0,   0, -1,  0,  0});
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


int imp::canny(OutTensor xo, InTensor xi, float minThres, float maxThres)
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
    Tensor ky({3, 3}, { 1,  2,  1,
                        0,  0,  0,
                       -1, -2, -1});
    Tensor imgx;
    conv2d(imgx, kx, img, 1, 0);
    Tensor imgy;
    conv2d(imgy, ky, img, 1, 0);
    Tensor grad = LinAlg::sqrt(imgx*imgx + imgy*imgy);
    Tensor theta(grad.shape);
    for (std::size_t i = 0; i < theta.totalSize; i++) {
        theta[i] = std::atan2(imgy[i], imgx[i] + 1e-9)*180/pi;
    }

    /* step3: Non-Maximum Suppression */
    int h = grad.shape[HWC_H];
    int w = grad.shape[HWC_W];
    Tensor mask(grad.shape);
    for (int i = 1; i < h - 1; i++) {
        for (int j = 1; j < w - 1; j++) {
            if (grad(i, j) == 0) {
                continue;
            }
            float thetaij = theta(i, j);
            float p1 = 0;
            float p2 = 0;
            if (thetaij >= 0 && thetaij < 45) {
                float g1 = grad(i + 1, j - 1);
                float g2 = grad(i + 1, j);
                float g3 = grad(i - 1, j + 1);
                float g4 = grad(i - 1, j);
                float w = std::abs(std::tan(thetaij*pi/180));
                p1 = w*g1 + (1 - w)*g2;
                p2 = w*g3 + (1 - w)*g4;
            } else if (thetaij >= 45 && thetaij < 90) {
                float g1 = grad(i + 1, j - 1);
                float g2 = grad(i, j - 1);
                float g3 = grad(i - 1, j + 1);
                float g4 = grad(i, j + 1);
                float w = std::abs(std::tan((thetaij - 90)*pi/180));
                p1 = w*g1 + (1 - w)*g2;
                p2 = w*g3 + (1 - w)*g4;
            } else if (thetaij >= -90 && thetaij < -45) {
                float g1 = grad(i - 1, j - 1);
                float g2 = grad(i, j - 1);
                float g3 = grad(i + 1, j + 1);
                float g4 = grad(i, j + 1);
                float w = std::abs(std::tan((thetaij - 90)*pi/180));
                p1 = w*g1 + (1 - w)*g2;
                p2 = w*g3 + (1 - w)*g4;
            } else if (thetaij >= -45 && thetaij < 0) {
                float g1 = grad(i + 1, j + 1);
                float g2 = grad(i + 1, j);
                float g3 = grad(i - 1, j - 1);
                float g4 = grad(i - 1, j);
                float w = std::abs(std::tan(thetaij*pi/180));
                p1 = w*g1 + (1 - w)*g2;
                p2 = w*g3 + (1 - w)*g4;
            }
            float p = grad(i, j);
            if (p > p1 && p > p2) {
                mask(i, j) = p;
            }
        }
    }

    /* step4: threshold */
    Tensor strongEdge(grad.shape);
    Tensor weakEdge(grad.shape);

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

int imp::FFT(Complex *xf, const Complex *xt, int t)
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

int imp::iFFT(Complex *xt, const Complex *xf, int t)
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

int imp::FFT2D(Tensor &spectrum, CTensor &xf, const Tensor &img)
{
    if (img.shape[HWC_C] != 1) {
        return -1;
    }
    int th = std::floor(std::log2(img.shape[0])) + 1;
    int tw = std::floor(std::log2(img.shape[1])) + 1;
    int w = std::pow(2, tw);
    int h = std::pow(2, th);
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

int imp::iFFT2D(Tensor &img, const CTensor &xf_)
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

Tensor imp::LPF(int h, int w, int freq)
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

Tensor imp::gaussHPF(int h, int w, float sigma)
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

Tensor imp::laplaceFilter(int h, int w)
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

Tensor imp::invDegenerate(int h, int w)
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

Tensor imp::invFilter(int h, int w, int rad)
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

Tensor imp::wienerFilter(int h, int w, float K)
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
