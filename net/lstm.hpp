#ifndef LSTM_H
#define LSTM_H
#include <vector>
#include "tensor.hpp"
#include "activate.h"

class LSTMParam
{
public:
    int inputDim;
    int hiddenDim;
    int outputDim;
public:
    LSTMParam():inputDim(0),hiddenDim(0),outputDim(0){}
    explicit LSTMParam(const LSTMParam &param)
        :inputDim(param.inputDim),hiddenDim(param.hiddenDim),outputDim(param.outputDim){}
    explicit LSTMParam(int inputDim_, int hiddenDim_, int outputDim_)
        :inputDim(inputDim_),hiddenDim(hiddenDim_),outputDim(outputDim_){}
};

/* SOA style */
class LSTMCells : public LSTMParam
{
public:
    /* input gate */
    Tensor Wi;
    Tensor Ui;
    Tensor Bi;
    /* generate */
    Tensor Wg;
    Tensor Ug;
    Tensor Bg;
    /* forget gate */
    Tensor Wf;
    Tensor Uf;
    Tensor Bf;
    /* output gate */
    Tensor Wo;
    Tensor Uo;
    Tensor Bo;
    /* predict */
    Tensor W;
    Tensor B;
public:
    LSTMCells(){}
    explicit LSTMCells(const LSTMParam &param)
        :LSTMParam(param)
    {
        Wi = Tensor(hiddenDim, inputDim);
        Wg = Tensor(hiddenDim, inputDim);
        Wf = Tensor(hiddenDim, inputDim);
        Wo = Tensor(hiddenDim, inputDim);

        Ui = Tensor(hiddenDim, hiddenDim);
        Ug = Tensor(hiddenDim, hiddenDim);
        Uf = Tensor(hiddenDim, hiddenDim);
        Uo = Tensor(hiddenDim, hiddenDim);

        Bi = Tensor(hiddenDim, 1);
        Bg = Tensor(hiddenDim, 1);
        Bf = Tensor(hiddenDim, 1);
        Bo = Tensor(hiddenDim, 1);

        W = Tensor(outputDim, hiddenDim);
        B = Tensor(outputDim, 1);
    }
    explicit LSTMCells(int inputDim_, int hiddenDim_, int outputDim_)
        :LSTMParam(inputDim_, hiddenDim_, outputDim_)
    {
        Wi = Tensor(hiddenDim, inputDim);
        Wg = Tensor(hiddenDim, inputDim);
        Wf = Tensor(hiddenDim, inputDim);
        Wo = Tensor(hiddenDim, inputDim);

        Ui = Tensor(hiddenDim, hiddenDim);
        Ug = Tensor(hiddenDim, hiddenDim);
        Uf = Tensor(hiddenDim, hiddenDim);
        Uo = Tensor(hiddenDim, hiddenDim);

        Bi = Tensor(hiddenDim, 1);
        Bg = Tensor(hiddenDim, 1);
        Bf = Tensor(hiddenDim, 1);
        Bo = Tensor(hiddenDim, 1);

        W = Tensor(outputDim, hiddenDim);
        B = Tensor(outputDim, 1);
    }

    void zero()
    {
        Wi.zero(); Wg.zero(); Wf.zero(); Wo.zero();
        Ui.zero(); Ug.zero(); Uf.zero(); Uo.zero();
        Bi.zero(); Bg.zero(); Bf.zero(); Bo.zero();
        W.zero(); B.zero();
        return;
    }
    void random()
    {
        Utils::uniform(Wi, -1, 1);
        Utils::uniform(Wg, -1, 1);
        Utils::uniform(Wf, -1, 1);
        Utils::uniform(Wo, -1, 1);

        Utils::uniform(Ui, -1, 1);
        Utils::uniform(Ug, -1, 1);
        Utils::uniform(Uf, -1, 1);
        Utils::uniform(Uo, -1, 1);

        Utils::uniform(Bi, -1, 1);
        Utils::uniform(Bg, -1, 1);
        Utils::uniform(Bf, -1, 1);
        Utils::uniform(Bo, -1, 1);

        Utils::uniform(W, -1, 1);
        Utils::uniform(B, -1, 1);
        return;
    }
};

class State
{
public:
    Tensor i;
    Tensor f;
    Tensor g;
    Tensor o;
    Tensor c;
    Tensor h;
    Tensor y;
public:
    State(){}
    State(std::size_t hiddenDim, std::size_t outputDim):
        i(hiddenDim, 1),f(hiddenDim, 1),g(hiddenDim, 1),
        o(hiddenDim, 1),c(hiddenDim, 1),h(hiddenDim, 1),
        y(outputDim, 1){}
    void zero()
    {
        i.zero(); f.zero(); g.zero(); o.zero();
        c.zero(); h.zero(); y.zero();
        return;
    }
};

class LSTM : public LSTMCells
{
public:
    using ParamType = LSTMParam;
    class Grad : public LSTMParam
    {
    public:
        int t;
        LSTMCells d;
        State sdelta;
        //Tensor delta;
    public:
        Grad(){}
        explicit Grad(const LSTMParam& param)
            :LSTMParam(param), t(0), d(param)
        {
            sdelta = State(hiddenDim, outputDim);
        }
        void backwardAtTime(LSTM &lstm, const Tensor &loss, const Tensor &x)
        {
            State delta(hiddenDim, outputDim);
            /* delta = W^T * E */
            Tensor::MatOp::kikj(delta.h, lstm.W, loss);
            /* delta = U * delta_ */
            Tensor::MatOp::ikkj(delta.h, lstm.Uf, sdelta.i);
            Tensor::MatOp::ikkj(delta.h, lstm.Ui, sdelta.f);
            Tensor::MatOp::ikkj(delta.h, lstm.Ug, sdelta.g);
            Tensor::MatOp::ikkj(delta.h, lstm.Uo, sdelta.o);
            /*
                δht = E + δht+1
                δct = δht ⊙ ot ⊙ dtanh(ct) + δct+1 ⊙ ft+1
                δot = δht ⊙ tanh(ct) ⊙ dsigmoid(ot)
                δgt = δct ⊙ it ⊙ dtanh(gt)
                δit = δct ⊙ gt ⊙ dsigmoid(it)
                δft = δct ⊙ ct-1 ⊙ dsigmoid(ft)
            */
            auto& states = lstm.states;
            Tensor f_ = t < states.size() - 1 ? states[t + 1].f : Tensor(hiddenDim, 1);
            Tensor _c = t > 0 ? states[t - 1].c : Tensor(hiddenDim, 1);
            for (std::size_t i = 0; i < delta.o.totalSize; i++) {
                delta.c[i] = delta.h[i] * states[t].o[i] * Tanh::df(states[t].c[i]) + sdelta.c[i] * f_[i];
                delta.o[i] = delta.h[i] * Tanh::f(states[t].c[i]) * Sigmoid::df(states[t].o[i]);
                delta.g[i] = delta.c[i] * states[t].i[i] * Tanh::df(states[t].g[i]);
                delta.i[i] = delta.c[i] * states[t].g[i] * Sigmoid::df(states[t].i[i]);
                delta.f[i] = delta.c[i] * _c[i] * Sigmoid::df(states[t].f[i]);
            }
            /* gradient */
            /*
                dw:(outputDim, hiddenDim)
                E: (outputDim, 1)
                h: (hiddenDim, 1)
                dw = E * h^T
            */
            Tensor::MatOp::ikjk(d.W, loss, states[t].h);
            Tensor::MatOp::ikjk(d.B, loss, states[t].y);
            /*
                dw:    (hiddenDim, inputDim)
                delta: (hiddenDim, 1)
                x:     (inputDim, 1)
                dw = delta * x^T
            */
            Tensor::MatOp::ikjk(d.Wi, delta.i, x);
            Tensor::MatOp::ikjk(d.Wf, delta.f, x);
            Tensor::MatOp::ikjk(d.Wg, delta.g, x);
            Tensor::MatOp::ikjk(d.Wo, delta.o, x);

            /*
                du:    (hiddenDim, hiddenDim)
                delta: (hiddenDim, 1)
                _h:    (hiddenDim, 1)
                du = delta * _h^T
            */
            Tensor _h = t > 0 ? states[t - 1].h : Tensor(hiddenDim, 1);
            Tensor::MatOp::ikjk(d.Ui, delta.i, _h);
            Tensor::MatOp::ikjk(d.Uf, delta.f, _h);
            Tensor::MatOp::ikjk(d.Ug, delta.g, _h);
            Tensor::MatOp::ikjk(d.Uo, delta.o, _h);

            d.Bi += delta.i;
            d.Bf += delta.f;
            d.Bg += delta.g;
            d.Bo += delta.o;
            /* next */
            sdelta = delta;
            t--;
            return;
        }

        void backward(LSTM &lstm, const std::vector<Tensor> &loss, const std::vector<Tensor> &x)
        {
            sdelta.zero();
            t = lstm.states.size() - 1;
            /* backward through time */
            for (std::size_t i = 0; i < x.size(); i++) {
                backwardAtTime(lstm, loss[t], x[t]);
            }
            lstm.states.clear();
            return;
        }

    };

    /* optimizer */
    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        Optimizer Wi;
        Optimizer Ui;
        Optimizer Bi;
        Optimizer Wg;
        Optimizer Ug;
        Optimizer Bg;
        Optimizer Wf;
        Optimizer Uf;
        Optimizer Bf;
        Optimizer Wo;
        Optimizer Uo;
        Optimizer Bo;
        Optimizer W;
        Optimizer B;
    public:
        OptimizeBlock(){}
        OptimizeBlock(const LSTM &layer)
        {
            Wi = Optimizer(layer.Wi.shape);
            Ui = Optimizer(layer.Ui.shape);
            Bi = Optimizer(layer.Bi.shape);
            Wg = Optimizer(layer.Wg.shape);
            Ug = Optimizer(layer.Ug.shape);
            Bg = Optimizer(layer.Bi.shape);
            Wf = Optimizer(layer.Wf.shape);
            Uf = Optimizer(layer.Uf.shape);
            Bf = Optimizer(layer.Bf.shape);
            Wo = Optimizer(layer.Wo.shape);
            Uo = Optimizer(layer.Uo.shape);
            Bo = Optimizer(layer.Bo.shape);
            W  = Optimizer(layer.W.shape);
            B  = Optimizer(layer.B.shape);
        }
        void operator()(LSTM& layer, Grad& grad, float learningRate)
        {
            Wi(layer.Wi, grad.d.Wi, learningRate);
            Ui(layer.Ui, grad.d.Ui, learningRate);
            Bi(layer.Bi, grad.d.Bi, learningRate);
            Wg(layer.Wg, grad.d.Wg, learningRate);
            Ug(layer.Ug, grad.d.Ug, learningRate);
            Bg(layer.Bg, grad.d.Bg, learningRate);
            Wf(layer.Wf, grad.d.Wf, learningRate);
            Uf(layer.Uf, grad.d.Uf, learningRate);
            Bf(layer.Bf, grad.d.Bf, learningRate);
            Wo(layer.Wo, grad.d.Wo, learningRate);
            Uo(layer.Uo, grad.d.Uo, learningRate);
            Bo(layer.Bo, grad.d.Bo, learningRate);
            W(layer.W, grad.d.W, learningRate);
            B(layer.B, grad.d.B, learningRate);
            return;
        }
    };

public:
    Tensor h;
    Tensor c;
    Tensor y;
    /* state */
    std::vector<State> states;
public:
    LSTM(){}
    LSTM(int inputDim_, int hiddenDim_, int outputDim_)
        :LSTMCells(inputDim_, hiddenDim_, outputDim_)
    {
        h = Tensor(hiddenDim, 1);
        c = Tensor(hiddenDim, 1);
        y = Tensor(outputDim, 1);
        LSTMCells::random();
    }

    void reset()
    {
        h.zero();
        c.zero();
        return;
    }

    State feedForward(const Tensor &x, const Tensor &_h, const Tensor &_c)
    {
        /*
                                                             y
                                                             |
                                                            h(t)
                                          c(t)               |
            c(t-1) -->--x-----------------+----------------------->-- c(t)
                        |                 |             |    |
                        |                 |            tanh  |
                        |                 |             |    |
                        |          -------x      -------x-----
                     f  |        i |      | g    | o    |
                        |          |      |      |      |
                     sigmoid    sigmoid  tanh  sigmoid  |
                        |          |      |      |      |
            h(t-1) -->----------------------------      --------->--- h(t)
                        |
                        x(t)

            ft = sigmoid(Wf*xt + Uf*ht-1 + bf);
            it = sigmoid(Wi*xt + Ui*ht-1 + bi);
            gt = tanh(Wg*xt + Ug*ht-1 + bg);
            ot = sigmoid(Wo*xt + Uo*ht-1 + bo);
            ct = ft ⊙ ct-1 + it ⊙ gt
            ht = ot ⊙ tanh(ct)
            yt = linear(W*ht + b)
        */
        State state(hiddenDim, outputDim);

        Tensor::MatOp::ikkj(state.f, Wf, x);
        Tensor::MatOp::ikkj(state.i, Wi, x);
        Tensor::MatOp::ikkj(state.g, Wg, x);
        Tensor::MatOp::ikkj(state.o, Wo, x);

        Tensor::MatOp::ikkj(state.f, Uf, _h);
        Tensor::MatOp::ikkj(state.i, Ui, _h);
        Tensor::MatOp::ikkj(state.g, Ug, _h);
        Tensor::MatOp::ikkj(state.o, Uo, _h);

        for (std::size_t i = 0; i < state.f.totalSize; i++) {
            state.f[i] = Sigmoid::f(state.f[i] + Bf[i]);
            state.i[i] = Sigmoid::f(state.i[i] + Bi[i]);
            state.g[i] =    Tanh::f(state.g[i] + Bg[i]);
            state.o[i] = Sigmoid::f(state.o[i] + Bo[i]);
            state.c[i] = state.f[i] * _c[i] + state.i[i]*state.g[i];
            state.h[i] = state.o[i] * Tanh::f(state.c[i]);
        }

        Tensor::MatOp::ikkj(state.y, W, state.h);
        state.y += B;
        return state;
    }

    void forward(const std::vector<Tensor> &sequence)
    {
        h.zero();
        c.zero();
        for (auto &x : sequence) {
            State state = feedForward(x, h, c);
            h = state.h;
            c = state.c;
            states.push_back(state);
        }
        return;
    }

    Tensor &forward(const Tensor &x)
    {
        State state = feedForward(x, h, c);
        h = state.h;
        c = state.c;
        y = state.y;
        states.push_back(state);
        return y;
    }

};

#endif // LSTM_H
