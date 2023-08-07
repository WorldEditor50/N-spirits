#ifndef UTIL_H
#define UTIL_H
#include <random>
#include "mat.h"

namespace util {
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
inline T sqrt(const T &x)
{
    T y(x);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::sqrt(x.val[i]);
    }
    return y;
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
    std::uniform_real_distribution<typename T::ValueType> distibution(-1, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.val[i] = distibution(engine);
    }
    return;
}
template<typename T>
inline void uniform(T &x, typename T::ValueType x1, typename T::ValueType x2)
{
    std::uniform_real_distribution<typename T::ValueType> distibution(x1, x2);
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
void gaussian(T &x, typename T::ValueType mu, typename T::ValueType sigma)
{
    std::random_device device;
    std::default_random_engine engine(device());
    std::normal_distribution<typename T::ValueType> distribution(mu, sigma);
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
inline void sub(T &y, const T &x1, const T &x2)
{
    for (std::size_t i = 0; i < y.totalSize; i++) {
        y.val[i] = x1.val[i] + x2.val[i];
    }
    return;
}
template<typename T>
inline void mul(T &y, const T &x1, const T &x2)
{
    for (std::size_t i = 0; i < y.totalSize; i++) {
        y.val[i] = x1.val[i] * x2.val[i];
    }
    return;
}
template<typename T>
inline void div(T &y, const T &x1, const T &x2)
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
inline T sum(const std::vector<T> &x)
{
    T s = x[0];
    for (std::size_t i = 1; i < x.size(); i++) {
        s += x[i];
    }
    return s;
}

template<typename T>
inline T variance(const std::vector<T> &x, T u)
{
    std::vector<T> delta(x.size());
    for (std::size_t i = 0; i < x.size(); i++) {
        delta[i] = x[i] - u;
    }
    T s = delta[0]*delta[0];
    for (std::size_t i = 1; i < x.size(); i++) {
        s += delta[i]*delta[i];
    }
    s /= x.size();
    return s;
}


template<typename T>
inline void normalize(T &x)
{
    x /= std::sqrt(util::dot(x, x));
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
inline typename T::ValueType l1(const T &x1, const T &x2)
{
    typename T::ValueType s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += x1.val[i] - x2.val[i];
    }
    return s;
}

template <typename T>
inline typename T::ValueType l2(const T &x1, const T &x2)
{
    typename T::ValueType s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += (x1.val[i] - x2.val[i])*(x1.val[i] - x2.val[i]);
    }
    return std::sqrt(s);
}

template <typename T>
inline typename T::ValueType lp(const T &x1, const T &x2, float p)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        float delta = x1.val[i] - x2.val[i];
        s += std::pow(delta, p);
    }
    return std::pow(s, 1/p);
}

template <typename T>
inline typename T::ValueType l8(const T &x1, const T &x2)
{
    float s = x1.val[0] - x2.val[0];
    for (std::size_t i = 1; i < x1.totalSize; i++) {
        float delta = x1.val[i] - x2.val[i];
        if (delta > s) {
            s = delta;
        }
    }
    return s;
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
        /* delta: x - u */
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
#endif // UTIL_H
