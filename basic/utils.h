#ifndef UTILS_H
#define UTILS_H
#include <random>
#include "mat.h"

namespace Utils {
static std::default_random_engine engine;

template<typename T>
inline void sqrt(const T &x, T &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::sqrt(x.val[i]);
    }
    return;
}

template<typename T>
inline void exp(const T &x, T &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::exp(x.val[i]);
    }
    return;
}

template<typename T>
inline void sin(const T &x, T &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::sin(x.val[i]);
    }
    return;
}

template<typename T>
inline void cos(const T &x, T &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::cos(x.val[i]);
    }
    return;
}

template<typename T>
inline void tanh(const T &x, T &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::tanh(x.val[i]);
    }
    return;
}
template<typename T>
inline void uniform(T &x)
{
    std::uniform_real_distribution<float> distibution(-1, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.val[i] = distibution(engine);
    }
    return;
}
template<typename T>
inline void uniform(T &x, float x1, float x2)
{
    std::uniform_real_distribution<float> distibution(x1, x2);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.val[i] = distibution(engine);
    }
    return;
}
template<typename T>
void bernoulli(T &x, float p)
{
    std::bernoulli_distribution distribution(p);
    for (std::size_t i = 0; i < x.totalSize; i++) {
      x.val[i] = distribution(engine);
    }
    return;
}
template<typename T>
inline void add(T &y, const T &x1, const T &x2)
{
    for (std::size_t i = 0; i < y.totalSize; i++) {
        y.val[i] = x1.val[i] + x2.val[i];
    }
    return;
}
template<typename T>
inline void minus(T &y, const T &x1, const T &x2)
{
    for (std::size_t i = 0; i < y.totalSize; i++) {
        y.val[i] = x1.val[i] + x2.val[i];
    }
    return;
}
template<typename T>
inline void multi(T &y, const T &x1, const T &x2)
{
    for (std::size_t i = 0; i < y.totalSize; i++) {
        y.val[i] = x1.val[i] * x2.val[i];
    }
    return;
}
template<typename T>
inline void divide(T &y, const T &x1, const T &x2)
{
    for (std::size_t i = 0; i < y.totalsize; i++) {
        y.val[i] = x1.val[i] / x2.val[i];
    }
    return;
}

template<typename T>
inline typename T::ValueType dot(const T &x1, const T &x2)
{
    typename T::ValueType s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += x1.val[i] * x2.val[i];
    }
    return s;
}

template<typename T>
inline void normalize(T &x)
{
    x /= std::sqrt(Utils::dot(x, x));
    return;
}
template<typename T>
inline void EMA(T &xh, const T &x, float rho)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        xh.val[i] = (1 - rho)*x.val[i] + rho*x.val[i];
    }
    return;
}
namespace Norm {

template <typename T>
inline float l1(const T &x1, const T &x2)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += x1.val[i] - x2.val[i];
    }
    return s;
}

template <typename T>
inline float l2(const T &x1, const T &x2)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += (x1.val[i] - x2.val[i])*(x1.val[i] - x2.val[i]);
    }
    return std::sqrt(s);
}

template <typename T>
inline float lp(const T &x1, const T &x2, float p)
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
