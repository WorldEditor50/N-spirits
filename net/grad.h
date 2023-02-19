#ifndef GRAD_H
#define GRAD_H
#include "../basic/tensor.h"
#include "layerdef.h"
#include "layerparam.h"

class FcGrad : public FcParam
{
public:
    Tensor dw;
    Tensor db;
    Tensor delta;
public:
    FcGrad(){}
    explicit FcGrad(const FcParam &param)
        :FcParam(param)
    {
        dw = Tensor(outputDim, inputDim);
        if (bias == true) {
            db = Tensor(outputDim, 1);
        }
        delta = Tensor(outputDim, 1);
    }

    void eval(const Tensor &x, const Tensor &o)
    {
        Tensor dy(outputDim, 1);
        Active::func[activeType].df(dy, o);
        dy *= delta;
        Tensor::MatOp::kikj(dw, dy, x);
        db += dy;
        delta.zero();
        return;
    }

    void backward(const Tensor &w, Tensor &delta_)
    {
        Tensor::MatOp::kikj(delta_, w, delta);
        return;
    }

};

class SoftmaxGrad : public FcGrad
{
public:
    void eval(const Tensor &x, const Tensor &o, const Tensor &yt)
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalsize; i++) {
            dy.val[i] = o.val[i] - yt.val[i];
        }
        Tensor::MatOp::kikj(dw, dy, x);
        db += dy;
        delta.zero();
        return;
    }
};

class DropoutGrad : public FcGrad
{
public:
    void backward(const Tensor &w, const Tensor &mask, Tensor &delta_)
    {
        delta_ *= mask;
        Tensor::MatOp::kikj(delta_, w, delta);
        return;
    }
};

class LayerNormGrad : public FcGrad
{
public:
    void backward(const Tensor &w, float gamma, Tensor &delta_)
    {
        delta_ *= gamma;
        Tensor::MatOp::kikj(delta_, w, delta);
        return;
    }
};

#endif // GRAD_H
