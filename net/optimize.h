#ifndef OPTIMIZE_H
#define OPTIMIZE_H
#include "../basic/tensor.h"

namespace Optimize {
class SGD
{
public:
    static float decay;
public:
    SGD(){}
    void operator()(Tensor& w, Tensor& dw, float learningRate)
    {
        for (std::size_t i = 0; i < w.totalsize; i++) {
            w.val[i] = (1 - decay)*w.val[i] - learningRate*dw.val[i];
        }
        dw.zero();
        return;
    }
};
float SGD::decay = 0;

class RMSProp
{
public:
    static float decay;
    static float rho;
    Tensor s;
public:
    RMSProp(){}
    explicit RMSProp(const std::vector<int> &shape)
    {
        s = Tensor(shape);
    }
    void operator()(Tensor& w, Tensor& dw, float learningRate)
    {
        for (std::size_t i = 0; i < w.totalsize; i++) {
            s.val[i] = rho*s.val[i] + (1 - rho) * dw.val[i]*dw.val[i];
            w.val[i] = (1 - decay)*w.val[i] - learningRate*dw.val[i]/(sqrt(s.val[i]) + 1e-9);
        }
        return;
    }
};
float RMSProp::decay = 0;
float RMSProp::rho = 0.9;

class Adam
{
public:
    static float decay;
    static float alpha;
    static float beta;
    float alpha_;
    float beta_;
    Tensor s;
    Tensor v;
public:
    Adam(){}
    explicit Adam(const std::vector<int> &shape)
        :alpha_(1),beta_(1)
    {
        s = Tensor(shape);
        v = Tensor(shape);
    }
    void operator()(Tensor& w, Tensor& dw, float learningRate)
    {
        alpha_ *= alpha;
        beta_ *= beta;
        for (std::size_t i = 0; i < w.totalsize; i++) {
            v[i] = alpha*v[i] + (1 - alpha)*dw[i];
            s[i] = beta*s[i] + (1 - beta)*dw[i]*dw[i];
            float v_ = v[i]/(1 - alpha_);
            float s_ = s[i]/(1 - beta_);
            w[i] = (1 - decay)*w[i] - learningRate*v_/(sqrt(s_) + 1e-9);
        }
        return;
    }
};
float Adam::decay = 0;
float Adam::alpha = 0.9;
float Adam::beta = 0.99;
}
#endif // OPTIMIZE_H
