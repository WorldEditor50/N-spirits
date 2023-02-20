#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <fstream>
#include "activate.h"
#include "utils.h"
#include "layerparam.h"

class FcLayer: public FcParam
{
public:
    using ParamType = FcParam;
    class Grad : public FcParam
    {
    public:
        Tensor dw;
        Tensor db;
        Tensor delta;
    public:
        Grad(){}
        explicit Grad(const FcParam &param)
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
            /*
                dw: (outputDim, inputDim)
                dy: (outputDim, 1)
                x:  (inputDim, 1)

                dy = dActive(o)*delta
                dw = dy * x^T
            */
            Tensor::MatOp::ikjk(dw, dy, x);
            db += dy;
            delta.zero();
            return;
        }

        void backward(FcLayer &layer, Tensor &delta_)
        {
            /*
                delta_: (inputDim, 1)
                w:      (outputDim, inputDim)
                delta:  (outputDim, 1)
                delta_ = w^T * delta
            */
            Tensor::MatOp::kikj(delta_, layer.w, delta);
            return;
        }

    };

    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        Optimizer optW;
        Optimizer optB;
    public:
        OptimizeBlock(){}
        OptimizeBlock(const FcLayer &layer)
        {
            optW = Optimizer(layer.w.shape);
            optB = Optimizer(layer.b.shape);
        }
        void operator()(FcLayer& layer, Grad& grad, float learningRate)
        {
            optW(layer.w, grad.dw, learningRate);
            optB(layer.b, grad.db, learningRate);
            return;
        }
    };

public:
    Tensor w;
    Tensor b;
    Tensor o;
public:
    FcLayer(){}
    explicit FcLayer(int inDim_, int outDim_, bool bias_, int activeType_)
        : FcParam(inDim_, outDim_, bias_, activeType_)
    {
        w = Tensor(outputDim, inputDim);
        if (bias == true) {
            b = Tensor(outputDim);
        }
        o = Tensor(outputDim, 1);
        /* init */
        Utils::uniform(w, -1, 1);
        Utils::uniform(b, -1, 1);
    }

    virtual Tensor& forward(const Tensor &x)
    {
        /*
           w: (outputDim, inputDim)
           x: (inputDim, 1)
           o = Active(w * x + b)
        */
        Tensor::MatOp::ikkj(o, w, x);
        if (bias == true) {
            o += b;
        }
        Active::func[activeType].f(o, o);
        return o;
    }

    void save(std::ofstream &file)
    {
        for (std::size_t j = 0; j < w.totalSize; j++) {
            file << w.val[j];
        }
        for (std::size_t j = 0; j < b.totalSize; j++) {
            file << b.val[j];
        }
        file << std::endl;
        return;
    }

    void load(std::ifstream &file)
    {
        for (std::size_t j = 0; j < w.totalSize; j++) {
            file >> w[j];
        }
        for (std::size_t j = 0; j < b.totalSize; j++) {
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
            size += inputs[i].totalSize;
            if (i == 0) {
                offset[i].from = 0;
            } else {
                offset[i].from = offset[i - 1].to;
            }
            offset[i].to = offset[i].from + size;
        }
        o = Tensor(size, 1);
    }

    Tensor& forward()
    {
        std::size_t k = 0;
        for (std::size_t i = 0; i < inputs.size(); i++) {
            for (std::size_t j = 0; j < inputs[i].totalSize; j++) {
                o[k] = inputs[i][j];
                k++;
            }
        }
        return o;
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
    class Grad : public FcLayer::Grad
    {
    public:
        Grad(){}
        explicit Grad(const FcParam &param)
            :FcLayer::Grad(param){}
        void eval(const Tensor &x, const Tensor &o, const Tensor &yt)
        {
            Tensor dy(outputDim, 1);
            for (std::size_t i = 0; i < dy.totalSize; i++) {
                dy.val[i] = o.val[i] - yt.val[i];
            }
            Tensor::MatOp::ikjk(dw, dy, x);
            db += dy;
            delta.zero();
            return;
        }
    };
public:
    Softmax(){}
    ~Softmax(){}
    explicit Softmax(int inDim_, int outDim_)
        :FcLayer(inDim_, outDim_, false, ACTIVE_LINEAR){}

    Tensor& forward(const Tensor &x) override
    {
        FcLayer::forward(x);
        Utils::exp(o, o);
        float s = o.sum();
        o /= s;
        return o;
    }
};

class Dropout : public FcLayer
{
public:
    class Grad : public FcLayer::Grad
    {
    public:
        Grad(){}
        explicit Grad(const FcParam &param)
            :FcLayer::Grad(param){}
        void backward(Dropout &layer, Tensor &delta_)
        {
            delta_ *= layer.mask;
            Tensor::MatOp::kikj(delta_, layer.w, delta);
            return;
        }
    };

public:
    float p;
    bool withGrad;
    Tensor mask;
public:
    Dropout(){}
    ~Dropout(){}
    explicit Dropout(int inDim_, int outDim_, bool bias_, int activeType_, float p_, bool withGrad_)
        :FcLayer(inDim_, outDim_, bias_, activeType_),p(p_), withGrad(withGrad_),mask(outDim_, 1){}

    Tensor& forward(const Tensor &x) override
    {
        FcLayer::forward(x);
        if (withGrad == true) {
            Utils::bernoulli(mask, p);
            mask /= (1 - p);
            FcLayer::o *= mask;
        }
        return o;
    }
};

class LayerNorm : public FcLayer
{
public:
    class Grad : public FcLayer::Grad
    {
    public:
        Grad(){}
        explicit Grad(const FcParam &param)
            :FcLayer::Grad(param){}
        void backward(LayerNorm &layer, Tensor &delta_)
        {
            delta_ *= layer.gamma;
            Tensor::MatOp::kikj(delta_, layer.w, delta);
            return;
        }
    };

public:
    float gamma;
public:
    LayerNorm(){}
    explicit LayerNorm(int inDim_, int outDim_, bool bias_, int activeType_)
        :FcLayer(inDim_, outDim_, bias_, activeType_), gamma(1){}

    Tensor& forward(const Tensor &x) override
    {
        Tensor::MatOp::kikj(o, w, x);

        float u = o.mean();
        float sigma = o.variance(u);
        gamma = 1/std::sqrt(sigma + 1e-9);

        for (std::size_t i = 0; i < o.totalSize; i++) {
            o.val[i] = gamma*(o.val[i] - u);
        }
        Active::func[activeType].f(o, o);
        return o;
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
        std::vector<Tensor> sigmas(x.size(), Tensor(u.totalSize));
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

class RBM
{
public:
    Tensor w;
    Tensor b;
    Tensor a;
public:
    explicit RBM(int visibleDim, int hiddenDim)
    {
        w = Tensor(visibleDim, hiddenDim);
        a = Tensor(visibleDim, 1);
        b = Tensor(hiddenDim, 1);
    }

    Tensor sample(const Tensor &p)
    {
        Tensor values(p.shape);
        for (std::size_t i = 0; i < p.totalSize; i++) {
            std::bernoulli_distribution distribution(p.val[i]);
            values.val[i] = distribution(Utils::engine);
        }
        return values;
    }

    Tensor sampleHidden(const Tensor &v)
    {
        /* p = sigmoid(w^T*v + b) */
        Tensor p = Sigmoid::f(w.tr()*v + b);
        return sample(p);
    }

    Tensor sampleVisible(const Tensor &h)
    {
        Tensor p = Sigmoid::f(w*h + a);
        return sample(p);
    }

    void train(const std::vector<Tensor> &x, std::size_t maxEpoch, float learningRate=1e-3)
    {
        for (std::size_t i = 0; i < maxEpoch; i++) {
            for (std::size_t j = 0; j < x.size(); j++) {
                Tensor h = sampleHidden(x[j]);
                Tensor v1 = sampleVisible(h);
                Tensor h1 = sampleHidden(v1);
                w += (x[j]*h - v1*h1)*learningRate;
                a += (x[j] - v1)*learningRate;
                b += (h - h1)*learningRate;
            }
        }
        return;
    }

    Tensor generate()
    {
        Tensor h(b.shape);
        return sampleVisible(h);
    }
};

#endif // LAYER_H
