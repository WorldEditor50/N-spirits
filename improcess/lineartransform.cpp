#include "lineartransform.h"


int imp::histogram1(OutTensor hist, InTensor gray)
{
    if (gray.shape[HWC_C] != 1) {
        return -1;
    }
    hist = Tensor(256);
    for (std::size_t i = 0; i < gray.totalSize; i++) {
        int pixel = gray.val[i];
        hist[pixel]++;
    }
    float h = gray.shape[HWC_H];
    float w = gray.shape[HWC_W];
    float s = h*w;
    hist /= s;
    return 0;
}

int imp::histogram3(OutTensor hist, InTensor rgb)
{
    hist = Tensor(3, 256);
    for (int i = 0; i < rgb.shape[HWC_H]; i++) {
        for (int j = 0; j < rgb.shape[HWC_W]; j++) {
            int r = int(rgb(i, j, 0));
            hist(0, r)++;
            int g = int(rgb(i, j, 1));
            hist(1, g)++;
            int b = int(rgb(i, j, 2));
            hist(2, b)++;
        }
    }
    float h = rgb.shape[HWC_H];
    float w = rgb.shape[HWC_W];
    float s = h*w;
    hist /= s;
    return 0;
}

int imp::linearTransform(OutTensor xo, InTensor xi, float alpha, float beta)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    xo = Tensor(xi.shape);
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        xo.val[i] = bound(xi.val[i]*alpha + beta, 0, 255);
    }
    return 0;
}

int imp::logTransform(OutTensor xo, InTensor xi, float c)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    xo = Tensor(xi.shape);
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        xo.val[i] = bound(c*std::log(xi.val[i] + 1), 0, 255);
    }
    return 0;
}

int imp::gammaTransform(OutTensor xo, InTensor xi, float esp, float gamma)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    xo = Tensor(xi.shape);
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        float p = std::pow((xi.val[i] + esp)/255, gamma)*255;
        xo.val[i] = bound(p, 0, 255);
    }
    return 0;
}

int imp::threshold(OutTensor xo, InTensor xi, float thres, float max_, float min_)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    xo = Tensor(xi.shape);
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        if (xi.val[i] < thres) {
            xo.val[i] = min_;
        } else {
            xo.val[i] = max_;
        }
    }
    return 0;
}

int imp::histogramEqualize(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    /*
        PDF : ∫p(x)dx=1

        origin image PDF: pr
        equalize image PDF: ps, ∫ps(x)dx=1
        ps * ds = ps * ds

        CDF: s = ∫pr(x)dx

    */
    xo = Tensor(xi.shape);
    /* 1. histogram */
    Tensor hist;
    histogram1(hist, xi);
    /* 2. equalize */
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        float cdf = 0;
        for (std::size_t j = 0; j < xi.val[i]; j++) {
            cdf += hist.val[j];
        }
        xo.val[i] = bound(cdf*255.0, 0, 255);
    }
    return 0;
}
