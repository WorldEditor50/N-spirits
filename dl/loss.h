#ifndef LOSS_H
#define LOSS_H
#include <cmath>
#include "../basic/tensor.hpp"

struct Loss
{
    static Tensor MSE(const Tensor& yp, const Tensor& yt)
    {
        Tensor loss(yp.shape);
        loss = yp - yt;
        loss *= 2;
        return loss;
    }

    static Tensor CROSS_EMTROPY(const Tensor& yp, const Tensor& yt)
    {
        Tensor loss(yp.shape);
        for (std::size_t i = 0; i < yp.totalSize; i++) {
            loss[i] = -yt[i] * std::log(yp[i]);
        }
        return loss;
    }
    static Tensor BCE(const Tensor& yp, const Tensor& yt)
    {
        Tensor loss(yt.shape);
        for (std::size_t i = 0; i < yp.totalSize; i++) {
            loss[i] = -(yt[i] * std::log(yp[i]) + (1 - yt[i]) * std::log(1 - yp[i]));
        }
        return loss;
    }
};
#endif // LOSS_H
