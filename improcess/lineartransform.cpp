#include "lineartransform.h"


int improcess::histogram1(const Tensor &gray, Tensor &hist)
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

int improcess::histogram3(const Tensor &rgb, Tensor &hist)
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

int improcess::linearTransform(const Tensor &x, float alpha, float beta, Tensor &y)
{
    if (isGray(x) == false) {
        return -1;
    }
    y = Tensor(x.shape);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = clamp(x.val[i]*alpha + beta, 0, 255);
    }
    return 0;
}

int improcess::logTransform(const Tensor &x, float c, Tensor &y)
{
    if (isGray(x) == false) {
        return -1;
    }
    y = Tensor(x.shape);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = clamp(c*std::log(x.val[i] + 1), 0, 255);
    }
    return 0;
}

int improcess::gammaTransform(const Tensor &x, float esp, float gamma, Tensor &y)
{
    if (isGray(x) == false) {
        return -1;
    }
    y = Tensor(x.shape);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        float p = std::pow((x.val[i] + esp)/255, gamma)*255;
        y.val[i] = clamp(p, 0, 255);
    }
    return 0;
}

int improcess::threshold(const Tensor &x, float thres, float max_, float min_, Tensor &y)
{
    if (isGray(x) == false) {
        return -1;
    }
    y = Tensor(x.shape);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        if (x.val[i] < thres) {
            y.val[i] = min_;
        } else {
            y.val[i] = max_;
        }
    }
    return 0;
}

int improcess::histogramEqualize(const Tensor &x, Tensor &y)
{
    if (isGray(x) == false) {
        return -1;
    }
    /*
        PDF : ∫p(x)dx=1

        origin image PDF: pr
        equalize image PDF: ps, ∫ps(x)dx=1
        ps * ds = ps * ds

        CDF: s = ∫pr(x)dx

    */
    y = Tensor(x.shape);
    /* 1. histogram */
    Tensor hist;
    histogram1(x, hist);
    /* 2. equalize */
    for (std::size_t i = 0; i < x.totalSize; i++) {
        float cdf = 0;
        for (std::size_t j = 0; j < x.val[i]; j++) {
            cdf += hist.val[j];
        }
        y.val[i] = clamp(cdf*255.0, 0, 255);
    }
    return 0;
}
