#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <fstream>
#include "tensor.h"
#include "activate.h"
#include "grad.h"

class FcLayer: public FcParam
{
public:
    using ParamType = FcParam;
    using GradType = FcGrad;
public:
    Tensor w;
    Tensor b;
    Tensor o;
public:
    FcLayer(){}
    FcLayer(int inDim_, int outDim_, bool bias_, int activeType_)
        : FcParam(inDim_, outDim_, bias_, activeType_)
    {
        w = Tensor(outputDim, inputDim);
        if (bias == true) {
            b = Tensor(outputDim);
        }
        o = Tensor(outputDim);
        /* init */
        Utils::uniform(w, -1, 1);
        Utils::uniform(b, -1, 1);
    }

    virtual void forward(const Tensor &x)
    {
        Tensor::MM::kikj(o, w, x);
        if (bias == true) {
            o += b;
        }
        Active::func[activeType].f(o, o);
        return;
    }
    void save(std::ofstream &file)
    {
        for (std::size_t j = 0; j < w.totalsize; j++) {
            file << w.val[j];
        }
        for (std::size_t j = 0; j < b.totalsize; j++) {
            file << b.val[j];
        }
        file << std::endl;
        return;
    }

    void load(std::ifstream &file)
    {
        for (std::size_t j = 0; j < w.totalsize; j++) {
            file >> w[j];
        }
        for (std::size_t j = 0; j < b.totalsize; j++) {
            file >> b[j];
        }
        return;
    }
};

class Concat
{
public:
    struct Offset {
        std::size_t from;
        std::size_t to;
    };
public:
    Tensor o;
    std::vector<Tensor> inputs;
    std::vector<Offset> offset;
public:
    template<typename ...Input>
    explicit Concat(Input&& ...input):
        inputs(input...)
    {
        int size = 0;
        offset = std::vector<Offset>(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); i++) {
            size += inputs[i].totalsize;
            if (i == 0) {
                offset[i].from = 0;
            } else {
                offset[i].from = offset[i - 1].to;
            }
            offset[i].to = offset[i].from + size;
        }
        o = Tensor(size, 1);
    }

    void forward()
    {
        std::size_t k = 0;
        for (std::size_t i = 0; i < inputs.size(); i++) {
            for (std::size_t j = 0; j < inputs[i].totalsize; j++) {
                o[k] = inputs[i][j];
                k++;
            }
        }
        return;
    }

    void backward(const Tensor &delta)
    {
        std::size_t k = 0;
        for (std::size_t i = 0; i < inputs.size(); i++) {
            for (std::size_t j = offset[i].from; j < offset[i].to; j++) {
                inputs[i][j] = delta[k];
                k++;
            }
        }
        return;
    }
};

class Softmax : public FcLayer
{
public:
    using GradType = SoftmaxGrad;
public:
    Softmax(){}
    ~Softmax(){}
    explicit Softmax(int inDim_, int outDim_)
        :FcLayer(inDim_, outDim_, false, ACTIVE_LINEAR){}

    void forward(const Tensor &x) override
    {
        FcLayer::forward(x);
        Utils::exp(o, o);
        float s = o.sum();
        o /= s;
        return;
    }
};

class Dropout : public FcLayer
{
public:
    using GradType = DropoutGrad;
public:
    float p;
    bool withGrad;
    Tensor mask;
public:
    Dropout(){}
    ~Dropout(){}
    explicit Dropout(int inDim_, int outDim_, bool bias_, int activeType_, float p_, bool withGrad_)
        :FcLayer(inDim_, outDim_, bias_, activeType_),p(p_), withGrad(withGrad_),mask(outDim_, 1){}

    void forward(const Tensor &x) override
    {
        FcLayer::forward(x);
        if (withGrad == true) {
            Tensor::bernoulli(mask, p);
            mask /= (1 - p);
            FcLayer::o *= mask;
        }
        return;
    }
};

class LayerNorm : public FcLayer
{
public:
    using GradType = LayerNormGrad;
public:
    float gamma;
public:
    LayerNorm(){}
    explicit LayerNorm(int inDim_, int outDim_, bool bias_, int activeType_)
        :FcLayer(inDim_, outDim_, bias_, activeType_), gamma(1){}

    void forward(const Tensor &x) override
    {
        Tensor::MM::kikj(o, w, x);

        float u = o.mean();
        float sigma = o.variance(u);
        gamma = 1/sqrt(sigma + 1e-9);

        for (std::size_t i = 0; i < o.totalsize; i++) {
            o.val[i] = gamma*(o.val[i] - u);
        }
        return;
    }
};

class BatchNorm
{
public:
    /* (in_channels) */
    Tensor u;
    /* (in_channels) */
    Tensor sigma;
    std::vector<Tensor> xh;
public:
    BatchNorm(){}
    explicit BatchNorm(int inChannels)
    {
        u = Tensor(inChannels);
        sigma = Tensor(inChannels);
    }
    void forward(const std::vector<Tensor> &x)
    {
        /* batchsize */
        float batchsize = x.size();
        /* u */
        for (std::size_t i = 0; i < x.size(); i++) {
            u += x[i];
        }
        u /= batchsize;
        /* xh */
        for (std::size_t i = 0; i < x.size(); i++) {
            Utils::minus(xh[i], x[i], u);
        }
        /* sigma */
        std::vector<Tensor> sigmas(x.size(), Tensor(u.totalsize));
        for (std::size_t i = 0; i < x.size(); i++) {
            Utils::multi(sigmas[i], xh[i], xh[i]);
        }
        for (std::size_t i = 0; i < x.size(); i++) {
            sigma += x[i];
        }
        sigma /= batchsize;
        sigma += 1e-9;
        Utils::sqrt(sigma, sigma);
        /* xh */
        for (std::size_t i = 0; i < x.size(); i++) {
            xh[i] /= sigma;
        }

        return;
    }
};

#endif // LAYER_H
