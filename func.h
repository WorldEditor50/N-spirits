#ifndef FUNC_H
#define FUNC_H
#include "mat.h"


inline void sqrt(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::sqrt(x.data[i]);
    }
    return;
}

inline void exp(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::exp(x.data[i]);
    }
    return;
}

inline void sin(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::sin(x.data[i]);
    }
    return;
}

inline void cos(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::cos(x.data[i]);
    }
    return;
}

inline void tanh(const Mat &x, Mat &y)
{
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::tanh(x.data[i]);
    }
    return;
}

inline void uniform(Mat &x)
{
    std::uniform_real_distribution<float> distibution(-1, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.data[i] = distibution(Mat::engine);
    }
    return;
}

inline void uniform(Mat &x, float x1, float x2)
{
    std::uniform_real_distribution<float> distibution(x1, x2);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.data[i] = distibution(Mat::engine);
    }
    return;
}
#endif // FUNC_H