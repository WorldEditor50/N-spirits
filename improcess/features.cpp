#include "features.h"
#include "filter.h"

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

int imp::entropy(InTensor img, uint8_t &thres)
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
int imp::otsu(InTensor img, uint8_t &thres)
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


int imp::LBP(OutTensor feature, InTensor gray)
{
    if (gray.shape[HWC_C] != 1) {
        return -1;
    }
    feature = Tensor(gray.shape);
    int h = gray.shape[HWC_H];
    int w = gray.shape[HWC_W];
    for (int i = 1; i < h - 1; i++) {
        for (int j = 1; j < w - 1; j++) {
            uint8_t p = gray(i, j, 0);
            uint8_t code = 0;
            code |= uint8_t(gray(i-1, j-1)>=p)<<7;
            code |= uint8_t(gray(i-1, j  )>=p)<<6;
            code |= uint8_t(gray(i-1, j+1)>=p)<<5;
            code |= uint8_t(gray(i  , j+1)>=p)<<4;
            code |= uint8_t(gray(i+1, j+1)>=p)<<3;
            code |= uint8_t(gray(i+1, j  )>=p)<<2;
            code |= uint8_t(gray(i+1, j-1)>=p)<<1;
            code |= uint8_t(gray(i  , j-1)>=p)<<0;
            feature(i - 1, j - 1, 0) = code;
        }
    }
    return 0;
}

int imp::circleLBP(OutTensor feature, InTensor gray, int radius, int neighbors, bool rotationInvariance)
{
    if (gray.shape[HWC_C] != 1) {
        return -1;
    }
    int h = gray.shape[HWC_H];
    int w = gray.shape[HWC_W];
    feature = Tensor(h - 2*radius, w - 2*radius, 1);
    for (int i = radius; i < h - radius; i++) {
        for (int j = radius; j < w - radius; j++) {
            uint8_t p = gray(i, j, 0);
            uint8_t code = 0;
            for (int k = 0; k < neighbors; k++) {
                /* offset to sampling center */
                float theta = 2.0*pi*k/neighbors;
                int rx = radius*std::cos(theta);
                int ry = -radius*std::sin(theta);
                /* bilinear interpolation */
                int x1 = std::floor(rx);
                int x2 = std::ceil(rx);
                int y1 = std::floor(ry);
                int y2 = std::ceil(ry);
                /* map offset to (0, 1) */
                int tx = rx - x1;
                int ty = ry - y1;
                /* weight */
                float w1 = (1 - tx)*(1 - ty);
                float w2 = tx*(1 - ty);
                float w3 = (1 - tx)*ty;
                float w4 = tx*ty;
                /* sample */
                uint8_t neighbor = gray(i + y1, j + x1)*w1 +
                        gray(i + y2, j + x1)*w2 +
                        gray(i + y1, j + x2)*w3 +
                        gray(i + y2, j + x2)*w4;
                /* encode to binary */
                code |= (neighbor > p)<<(uint8_t)(neighbors - k - 1);
            }
            feature(i - radius, j - radius, 0) = code;
        }
    }
    /* rotation invariance */
    if (rotationInvariance) {
        for (int i = 0; i < feature.shape[0]; i++) {
            for (int j = 0; j < feature.shape[1]; j++) {
                uint8_t p = feature(i, j, 0);
                uint8_t minValue = p;
                for (int k = 1;  k < neighbors; k++) {
                    uint8_t code = (uint8_t)(p>>(neighbors-k))|(uint8_t)(p<<k);
                    if (code < minValue) {
                        minValue = code;
                    }
                }
                feature(i, j, 0) = minValue;
            }
        }
    }
    return 0;
}

int imp::multiScaleBlockLBP(OutTensor feature, InTensor gray, float scale)
{
    if (gray.shape[HWC_C] != 1) {
        return -1;
    }
    int h = gray.shape[HWC_H];
    int w = gray.shape[HWC_W];
    int cellSize = scale/3;
    int offset = float(cellSize)/2;
    /* average blur */
    Tensor cellImage(h - 2*offset, w - 2*offset, 1);
    for (int i = offset; i < h - offset; i++) {
        for (int j = offset; j < w - offset; j++) {
            float s = 0;
            for (int h = -offset; h < offset + 1; h++) {
                for (int k = -offset; k < offset + 1; k++) {
                    s += gray(i + h, j + k, 0);
                }
            }
            s /= cellSize*cellSize;
            cellImage(i - offset, j - offset, 0) = s;
        }
    }
    return LBP(feature, cellImage);
}
