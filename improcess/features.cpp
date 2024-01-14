#include "features.h"


int imp::histogram(OutTensor hist, InTensor gray)
{
    if (gray.shape[HWC_C] != 1) {
        return -1;
    }
    hist = Tensor(256);
    for (std::size_t i = 0; i < gray.totalSize; i++) {
        int pixel = gray.val[i];
        hist[pixel]++;
    }
    hist /= gray.totalSize;
    return 0;
}

int imp::mean(InTensor hist, float &m)
{
    m = 0;
    for (std::size_t i = 0; i < 256; i++) {
        m += i*hist[i];
    }
    return 0;
}

int imp::moment(InTensor hist, float m, int n, float &mu)
{
    mu = 0;
    for (std::size_t i = 0; i < 256; i++) {
        mu += std::pow(i - m, n)*hist[i];
    }
    return 0;
}


int imp::entropy(InTensor hist, float &e)
{
    e = 0;
    for (std::size_t i = 0; i < 256; i++) {
        e += -hist[i]*std::log(hist[i]);
    }
    return 0;
}

int imp::grayConjugateMatrix(OutTensor xo, InTensor xi, const Point2i &p1, const Point2i &p2)
{
    int maxGray = xi.max();
    xo = Tensor(maxGray + 1, maxGray + 1);
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int g = xi(i, j);
            int u = i + p1.x;
            int v = j + p1.y;
            if (u >= 0 && u < h && v >= 0 && v < w) {
                int g1 = xi(u, v);
                xo(g, g1) += 1;
            }
            u = i + p2.x;
            v = j + p2.y;
            if (u >= 0 && u < h && v >= 0 && v < w) {
                int g2 = xi(u, v);
                xo(g, g2) += 1;
            }
        }
    }
    return 0;
}


