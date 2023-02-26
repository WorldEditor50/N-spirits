#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <fstream>
#include "activate.h"
#include "../basic/utils.h"
#include "layerdef.h"

class FcParam
{
public:
    int id;
    int inputDim;
    int outputDim;
    bool bias;
    /* type */
    int opType;
    int activeType;
    int layerType;
public:
    FcParam():inputDim(0),outputDim(0),bias(false),
    opType(OP_FORWARD),activeType(ACTIVE_LINEAR),layerType(LAYER_FC){}
    FcParam(int inDim_, int outDim_, bool bias_, int activeType_):
        inputDim(inDim_),outputDim(outDim_),bias(bias_),
        opType(OP_FORWARD),activeType(activeType_),layerType(LAYER_FC){}
    FcParam(const FcParam &param):
        inputDim(param.inputDim),outputDim(param.outputDim),bias(param.bias),
        opType(param.opType),activeType(param.activeType),layerType(param.layerType){}
};

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
        inline Tensor& deltaRef()
        {
            return delta;
        }
        void backward(const FcLayer &layer, Tensor &delta_)
        {
            /*
                delta_: (inputDim, 1)
                w:      (outputDim, inputDim)
                delta:  (outputDim, 1)
                delta_ = w^T * delta
            */
            Tensor::Mul::kikj(delta_, layer.w, delta);
            return;
        }

        void eval(const Tensor &x, Tensor &o)
        {
            Tensor &dy = o;
            Active::func[activeType].df(dy);
            dy *= delta;
            /*
                dw: (outputDim, inputDim)
                dy: (outputDim, 1)
                x:  (inputDim, 1)

                dy = dActive(o)*delta
                dw = dy * x^T
            */
            Tensor::Mul::ikjk(dw, dy, x);
            db += dy;
            delta.zero();
            return;
        }

    };
    /* optimizer */
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
        inline void operator()(FcLayer& layer, Grad& grad, float learningRate)
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
        o.zero();
        Tensor::Mul::ikkj(o, w, x);
        if (bias == true) {
            o += b;
        }
        Active::func[activeType].f(o);
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

class SoftmaxLayer : public FcLayer
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
            Utils::minus(dy, o, yt);
            Tensor::Mul::ikjk(dw, dy, x);
            delta.zero();
            return;
        }
    };
public:
    SoftmaxLayer(){}
    ~SoftmaxLayer(){}
    explicit SoftmaxLayer(int inDim_, int outDim_)
        :FcLayer(inDim_, outDim_, false, ACTIVE_LINEAR)
    {
        layerType = LAYER_SOFTMAX;
    }

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
        static bool enable;
    public:
        Grad(){}
        explicit Grad(const FcParam &param)
            :FcLayer::Grad(param)
        {
            enable = true;
        }
        void backward(Dropout &layer, Tensor &delta_)
        {
            delta_ *= layer.mask;
            Tensor::Mul::kikj(delta_, layer.w, delta);
            return;
        }
    };

public:
    float p;
    Tensor mask;
public:
    Dropout(){}
    ~Dropout(){}
    explicit Dropout(int inDim_, int outDim_, bool bias_, int activeType_, float p_)
        :FcLayer(inDim_, outDim_, bias_, activeType_),p(p_), mask(outDim_, 1)
    {
        layerType = LAYER_DROPOUT;
    }

    Tensor& forward(const Tensor &x) override
    {
        FcLayer::forward(x);
        if (Grad::enable == true) {
            Utils::bernoulli(mask, p);
            mask /= (1 - p);
            FcLayer::o *= mask;
        }
        return o;
    }
};
bool Dropout::Grad::enable = false;

class LayerNorm : public FcLayer
{
public:
    class Grad : public FcLayer::Grad
    {
    public:
        Grad(){}
        explicit Grad(const FcParam &param)
            :FcLayer::Grad(param){}
        void backward(const LayerNorm &layer, Tensor &delta_)
        {
            delta_ *= layer.gamma;
            Tensor::Mul::kikj(delta_, layer.w, delta);
            return;
        }
    };

public:
    float gamma;
public:
    LayerNorm(){}
    explicit LayerNorm(int inDim_, int outDim_, bool bias_, int activeType_)
        :FcLayer(inDim_, outDim_, bias_, activeType_), gamma(1)
    {
        layerType = LAYER_NORM;
    }

    Tensor& forward(const Tensor &x) override
    {
        o.zero();
        Tensor::Mul::ikkj(o, w, x);
        float u = o.mean();
        float sigma = o.variance(u);
        gamma = 1/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o.val[i] = gamma*(o.val[i] - u);
        }
        if (bias == true) {
            o += b;
        }
        Active::func[activeType].f(o);
        return o;
    }
};

class BatchNorm1dParam
{
public:
    int inputDim;
    int outputDim;
    int batchSize;
public:
    BatchNorm1dParam(){}
    BatchNorm1dParam(const BatchNorm1dParam &param)
        :inputDim(param.inputDim),
         outputDim(param.outputDim),batchSize(param.batchSize){}
};
class BatchNorm1d : public BatchNorm1dParam
{
public:
    using ParamType = BatchNorm1dParam;
    /* grad */
    class Grad : public BatchNorm1dParam
    {
    public:
        Tensor dGamma;
        Tensor dBeta;
        std::vector<Tensor> deltas;
    public:
        Grad(){}
        Grad(const BatchNorm1dParam &param)
            :BatchNorm1dParam((param))
        {
            dGamma = Tensor(outputDim, 1);
            dBeta  = Tensor(outputDim, 1);
            deltas = std::vector<Tensor>(batchSize, Tensor(outputDim, 1));
        }
        inline Tensor& deltaRef()
        {
            return deltas[0];
        }
        void backward(const BatchNorm1d &layer, Tensor &delta/* output */)
        {
            return;
        }
        void eval(const Tensor &x, const Tensor &o)
        {

            return;
        }
    };
    /* optimizer */
    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        Optimizer optGamma;
        Optimizer optBeta;
    public:
        OptimizeBlock(){}
        OptimizeBlock(const BatchNorm1d &layer)
        {
            optGamma = Optimizer(layer.gamma.shape);
            optBeta = Optimizer(layer.beta.shape);
        }
        inline void operator()(BatchNorm1d& layer, Grad& grad, float learningRate)
        {
            optGamma(layer.gamma, grad.dGamma, learningRate);
            optBeta(layer.beta, grad.dBeta, learningRate);
            return;
        }
    };
public:
    Tensor u;
    Tensor sigma;
    std::vector<Tensor> xh;
    std::vector<Tensor> o;
    Tensor beta;
    Tensor gamma;
public:
    BatchNorm1d(){}
    explicit BatchNorm1d(int inputDim, int outputDim, int batchSize)
    {
        u     = Tensor(inputDim, 1);
        sigma = Tensor(inputDim, 1);

        beta     = Tensor(outputDim, 1);
        gamma = Tensor::ones(outputDim, 1);

        o     = std::vector<Tensor>(batchSize, Tensor(outputDim, 1));
        xh    = std::vector<Tensor>(batchSize, Tensor(outputDim, 1));
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
        xh = std::vector<Tensor>(batchsize, Tensor(u.shape));
        for (std::size_t i = 0; i < x.size(); i++) {
            Utils::minus(xh[i], x[i], u);
        }
        /* sigma */
        Tensor r(u.shape);
        for (std::size_t i = 0; i < xh.size(); i++) {
            Utils::multi(r, xh[i], xh[i]);
            sigma += r;
        }
        sigma /= batchsize;
        sigma += 1e-9;
        Utils::sqrt(sigma, sigma);
        /* xh = (xi - u)/sqrt(sigma + 1e-9) */
        for (std::size_t i = 0; i < x.size(); i++) {
            xh[i] /= sigma;
        }
        /* o = gamma*xh + b */
        for (std::size_t i = 0; i < xh.size(); i++) {
            Utils::multi(o[i], gamma, xh[i]);
            o[i] += beta;
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


class Gate
{
public:
    Tensor w;
    Tensor u;
    Tensor b;
    Tensor o;
public:
    Gate(){}
    Gate(int inputDim, int hiddenDim, int outputDim)
    {
        w = Tensor(outputDim, inputDim);
        u = Tensor(hiddenDim, hiddenDim);
        b = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
    }
    Tensor& forward(const Tensor &x, const Tensor &h)
    {
        Tensor::Mul::ikkj(o, w, x);
        Tensor::Mul::ikkj(o, u, h);
        o += b;
        return o;
    }
};

#endif // LAYER_H
