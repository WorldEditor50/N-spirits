#ifndef OPTIMIZE_H
#define OPTIMIZE_H
#include "../basic/tensor.hpp"

namespace Optimize {

class SGD
{
public:
    SGD(){}
    explicit SGD(const std::vector<int> &){}
    inline void operator()(Tensor& w, Tensor& dw, float learningRate, float decay, bool clipGrad)
    {
        if (clipGrad) {
            dw /= dw.norm2() + 1e-8;
        }
        for (std::size_t i = 0; i < w.totalSize; i++) {
            w.val[i] = (1 - decay)*w.val[i] - learningRate*dw.val[i];
        }
        dw.zero();
        return;
    }
};

class RMSProp
{
public:
    float rho;
    Tensor v;
public:
    RMSProp():rho(0.9f){}
    explicit RMSProp(const std::vector<int> &shape)
        :rho(0.9f)
    {
        v = Tensor(shape);
    }
    inline void operator()(Tensor& w, Tensor& dw, float learningRate, float decay, bool clipGrad)
    {
        if (clipGrad) {
            dw /= dw.norm2() + 1e-8;
        }
        for (std::size_t i = 0; i < w.totalSize; i++) {
            v.val[i] = rho*v.val[i] + (1 - rho) * dw.val[i]*dw.val[i];
            w.val[i] = (1 - decay)*w.val[i] - learningRate*dw.val[i]/(std::sqrt(v.val[i]) + 1e-9);
        }
        dw.zero();
        return;
    }
};

class Adam
{
public:
    float alpha;
    float beta;
    float alpha_;
    float beta_;
    Tensor v;
    Tensor m;
public:
    Adam():alpha(0.9f),beta(0.99f){}
    explicit Adam(const std::vector<int> &shape)
        :alpha(0.9f),beta(0.99f),alpha_(1),beta_(1)
    {
        v = Tensor(shape);
        m = Tensor(shape);
    }
    inline void operator()(Tensor& w, Tensor& dw, float learningRate, float decay, bool clipGrad)
    {
        alpha_ *= alpha;
        beta_ *= beta;
        if (clipGrad) {
            dw /= dw.norm2() + 1e-8;
        }
        for (std::size_t i = 0; i < w.totalSize; i++) {
            m[i] = alpha*m[i] + (1 - alpha)*dw[i];
            v[i] = beta*v[i] + (1 - beta)*dw[i]*dw[i];
            float m_ = m[i]/(1 - alpha_);
            float v_ = v[i]/(1 - beta_);
            w[i] = (1 - decay)*w[i] - learningRate*m_/(std::sqrt(v_) + 1e-9);
        }
        dw.zero();
        return;
    }
};

}
#endif // OPTIMIZE_H
