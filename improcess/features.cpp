#include "features.h"


int imp::histogram(OutTensor hist, InTensor gray)
{
    if (gray.shape[HWC_C] != 1) {
        return -1;
    }
    hist = Tensor(256, 1);
    for (std::size_t i = 0; i < gray.totalSize; i++) {
        int pixel = gray.val[i];
        hist[pixel]++;
    }
    return 0;
}

int imp::uniformHistogram(OutTensor hist, InTensor gray)
{
    int ret = histogram(hist, gray);
    int h = gray.shape[HWC_H];
    int w = gray.shape[HWC_W];
    hist /= (h*w);
    return ret;
}

int imp::moment0(OutTensor m0, InTensor hist)
{
    m0 = Tensor(256, 1);
    float s = 0;
    for (std::size_t i = 0; i < 256; i++) {
        s += hist[i];
        m0[i] = s;
    }
    return 0;
}

int imp::moment1(OutTensor m1, InTensor hist)
{
    m1 = Tensor(256, 1);
    float s = 0;
    for (std::size_t i = 0; i < 256; i++) {
        s += i*hist[i];
        m1[i] = s;
    }
    return 0;
}

int imp::entropy(InTensor img, int &thres)
{
    if (img.shape[HWC_C] != 1) {
        return -1;
    }
    /* histogram */
    Tensor hist;
    uniformHistogram(hist, img);
    /* moment0 */
    Tensor m0;
    moment0(m0, hist);
    /* f = f1 + f2 */
    Tensor f(256, 1);
    for (std::size_t t = 0; t < 256; t++) {
        float f1 = 0;
        for (std::size_t i = 0; i < t + 1; i++) {
            if (std::abs(m0[i]) < 1e-6) {
                f1 = 0;
            } else {
                f1 += -(hist[i]/m0[i])*std::log(hist[i]/m0[i] + 1e-6);
            }
        }
        float f2 = 0;
        for (std::size_t i = t + 1; i < 256; i++) {
            if (std::abs(1 - m0[i]) < 1e-6) {
                f2 = 0;
            } else {
                f1 += -(hist[i]/(1 - m0[i]))*std::log(hist[i]/(1 - m0[i]) + 1e-6);
            }
        }

        f[t] = f1 + f2;
        //std::cout<<f[t]<<std::endl;
    }
    thres = f.argmax();
    return 0;
}
int imp::otsu(InTensor img, int &thres)
{
    if (img.shape[HWC_C] != 1) {
        return -1;
    }
    /* histogram */
    Tensor hist;
    uniformHistogram(hist, img);
    /* moment0 */
    Tensor m0;
    moment0(m0, hist);
    /* moment1 */
    Tensor m1;
    moment1(m1, hist);
    float u = m1[255];
    /* variance */
    Tensor sigma(256, 1);
    for (int i = 0; i < 256; i++) {
        if (m0[i] == 0 || m0[i] == 1) {
            sigma[i] = 0;
            continue;
        }
        sigma[i] = (u*m0[i] - m1[i])*(u*m0[i] - m1[i])/(m0[i]*(1 - m0[i]));
    }
    thres = sigma.argmax();
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

int imp::barycenter(InTensor img, Point2i &center)
{
    if (img.shape[HWC_C] != 1) {
        return -1;
    }
    float s = img.sum();
    int w = img.shape[HWC_W];
    int h = img.shape[HWC_H];
    float x = 0;
    float y = 0;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int pixel = img(i, j);
            x += pixel*i;
            y += pixel*j;
        }
    }
    center = Point2i(x/s, y/s);
    return 0;
}

