#include "lineartransform.h"


int imp::histogram1(OutTensor hist, InTensor gray)
{
    if (isGray(gray) == false) {
        return -1;
    }
    hist = Tensor(256);
    for (std::size_t i = 0; i < gray.totalSize; i++) {
        int pixel = gray.val[i];
        hist[pixel]++;
    }
    float s = area(gray);
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
    float s = area(rgb);
    hist /= s;
    return 0;
}

int imp::linearTransform(OutTensor xo, InTensor xi, float alpha, float beta)
{
    if (imp::isGray(xi) == false) {
        return -1;
    }
    xo = Tensor(xi.shape);
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        xo.val[i] = clamp(xi.val[i]*alpha + beta, 0, 255);
    }
    return 0;
}

int imp::logTransform(OutTensor xo, InTensor xi, float c)
{
    if (isGray(xi) == false) {
        return -1;
    }
    xo = Tensor(xi.shape);
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        xo.val[i] = clamp(c*std::log(xi.val[i] + 1), 0, 255);
    }
    return 0;
}

int imp::gammaTransform(OutTensor xo, InTensor xi, float esp, float gamma)
{
    if (isGray(xi) == false) {
        return -1;
    }
    xo = Tensor(xi.shape);
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        float p = std::pow((xi.val[i] + esp)/255, gamma)*255;
        xo.val[i] = clamp(p, 0, 255);
    }
    return 0;
}

int imp::threshold(OutTensor xo, InTensor xi, float thres, float max_, float min_)
{
    if (isGray(xi) == false) {
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
    if (isGray(xi) == false) {
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
        xo.val[i] = clamp(cdf*255.0, 0, 255);
    }
    return 0;
}
