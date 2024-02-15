#ifndef OPTIMIZE_H
#define OPTIMIZE_H
#include "../basic/tensor.hpp"

namespace Optimize {

class SGD
{
public:
    float decay;
public:
    SGD():decay(0){}
    explicit SGD(const std::vector<int> &):decay(0){}
    inline void operator()(Tensor& w, Tensor& dw, float learningRate)
    {
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
    float decay;
    float rho;
    Tensor s;
public:
    RMSProp():decay(0.0f),rho(0.9f){}
    explicit RMSProp(const std::vector<int> &shape)
        :decay(0.0f),rho(0.9f)
    {
        s = Tensor(shape);
    }
    inline void operator()(Tensor& w, Tensor& dw, float learningRate)
    {
        for (std::size_t i = 0; i < w.totalSize; i++) {
            s.val[i] = rho*s.val[i] + (1 - rho) * dw.val[i]*dw.val[i];
            w.val[i] = (1 - decay)*w.val[i] - learningRate*dw.val[i]/(std::sqrt(s.val[i]) + 1e-9);
        }
        dw.zero();
        return;
    }
};

class Adam
{
public:
    float decay;
    float alpha;
    float beta;
    float alpha_;
    float beta_;
    Tensor s;
    Tensor v;
public:
    Adam():decay(0),alpha(0.9f),beta(0.99f){}
    explicit Adam(const std::vector<int> &shape)
        :decay(0),alpha(0.9f),beta(0.99f),alpha_(1),beta_(1)
    {
        s = Tensor(shape);
        v = Tensor(shape);
    }
    inline void operator()(Tensor& w, Tensor& dw, float learningRate)
    {
        alpha_ *= alpha;
        beta_ *= beta;
        for (std::size_t i = 0; i < w.totalSize; i++) {
            v[i] = alpha*v[i] + (1 - alpha)*dw[i];
            s[i] = beta*s[i] + (1 - beta)*dw[i]*dw[i];
            float v_ = v[i]/(1 - alpha_);
            float s_ = s[i]/(1 - beta_);
            w[i] = (1 - decay)*w[i] - learningRate*v_/(std::sqrt(s_) + 1e-9);
        }
        dw.zero();
        return;
    }
};

}
#endif // OPTIMIZE_H
