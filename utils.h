#ifndef UTILS_H
#define UTILS_H
#include "mat.h"

namespace Utils {

inline void sqrt(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::sqrt(x.val[i]);
    }
    return;
}

inline void exp(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::exp(x.val[i]);
    }
    return;
}

inline void sin(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::sin(x.val[i]);
    }
    return;
}

inline void cos(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::cos(x.val[i]);
    }
    return;
}

inline void tanh(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::tanh(x.val[i]);
    }
    return;
}

inline void uniform(Mat &x)
{
    std::uniform_real_distribution<float> distibution(-1, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.val[i] = distibution(Mat::engine);
    }
    return;
}

inline void uniform(Mat &x, float x1, float x2)
{
    std::uniform_real_distribution<float> distibution(x1, x2);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.val[i] = distibution(Mat::engine);
    }
    return;
}

inline float dot(const Mat &x1, const Mat &x2)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += x1.val[i] * x2.val[i];
    }
    return s;
}

inline void normalize(Mat &x)
{
    x /= std::sqrt(Utils::dot(x, x));
    return;
}

inline void EMA(Mat &xh, const Mat &x, float rho)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        xh.val[i] = (1 - rho)*x.val[i] + rho*x.val[i];
    }
    return;
}
namespace Norm {

inline float l1(const Mat &x1, const Mat &x2)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += x1.val[i] - x2.val[i];
    }
    return s;
}

inline float l2(const Mat &x1, const Mat &x2)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += (x1.val[i] - x2.val[i])*(x1.val[i] - x2.val[i]);
    }
    return std::sqrt(s);
}

inline float lp(const Mat &x1, const Mat &x2, float p)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        float delta = x1.val[i] - x2.val[i];
        s += std::pow(delta, p);
    }
    return std::pow(s, 1/p);
}

}

inline void cov(Mat &y, const Mat &x)
{
    Mat a(x);
    for (std::size_t i = 0; i < x.cols; i++) {
        /* mean of column */
        float u = 0;
        for (std::size_t j = 0; j < x.rows; j++) {
            u += a(j, i);
        }
        u /= float(x.rows);
        /* delta */
        for (std::size_t j = 0; j < x.rows; j++) {
            a(j, i) -= u;
        }
    }
    /* y = A^T*A */
    Mat::Multiply::kikj(y, a, a);
    y /= float(a.rows);
    return;
}

}
#endif // UTILS_H
