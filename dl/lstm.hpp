#ifndef LSTM_H
#define LSTM_H
#include <vector>
#include "activate.hpp"
#include "../basic/tensor.hpp"


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
        LinAlg::uniform(Wi, -1, 1);
        LinAlg::uniform(Wg, -1, 1);
        LinAlg::uniform(Wf, -1, 1);
        LinAlg::uniform(Wo, -1, 1);
        LinAlg::uniform(Ui, -1, 1);
        LinAlg::uniform(Ug, -1, 1);
        LinAlg::uniform(Uf, -1, 1);
        LinAlg::uniform(Uo, -1, 1);
        LinAlg::uniform(Bi, -1, 1);
        LinAlg::uniform(Bg, -1, 1);
        LinAlg::uniform(Bf, -1, 1);
        LinAlg::uniform(Bo, -1, 1);
        LinAlg::uniform(W, -1, 1);
        LinAlg::uniform(B, -1, 1);
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
        i(Tensor(hiddenDim, 1)),f(Tensor(hiddenDim, 1)),g(Tensor(hiddenDim, 1)),
        o(Tensor(hiddenDim, 1)),c(Tensor(hiddenDim, 1)),h(Tensor(hiddenDim, 1)),
        y(Tensor(outputDim, 1)){}
    State(const State &r):
        i(r.i),f(r.f),g(r.g),
        o(r.o),c(r.c),h(r.h),y(r.y){}
    State& operator = (const State &r)
    {
        if (this == &r) {
            return *this;
        }
        i = r.i;
        f = r.f;
        g = r.g;
        o = r.o;
        c = r.c;
        h = r.h;
        y = r.y;
        return *this;
    }
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
        LSTMCells d;
        /* buffer loss */
        std::vector<Tensor> loss;
        /* buffer input */
        std::vector<Tensor> x;
    public:
        Grad(){}
        explicit Grad(const LSTMParam& param)
            :LSTMParam(param), d(param){}

        void cache(const Tensor &loss_, const Tensor &x_)
        {
            /* cache loss and input */
            loss.push_back(loss_);
            x.push_back(x_);
            return;
        }

        void backwardThroughTime(LSTM &lstm)
        {
            State delta_(hiddenDim, outputDim);
            std::vector<State>& states = lstm.states;
            int te = lstm.states.size() - 1;
            for (int t = te; t >= 0; t--) {
                State delta(hiddenDim, outputDim);
                /* δh = W^T * E */
                Tensor::MM::kikj(delta.h, lstm.W, loss[t]);
                if (t < te) {
                    /* δh += U^T * delta_ */
                    Tensor::MM::kikj(delta.h, lstm.Uf, delta_.i);
                    Tensor::MM::kikj(delta.h, lstm.Ui, delta_.f);
                    Tensor::MM::kikj(delta.h, lstm.Ug, delta_.g);
                    Tensor::MM::kikj(delta.h, lstm.Uo, delta_.o);
                }
                /*
                    δht = E + δht+1
                    δct = δht ⊙ ot ⊙ dtanh(ct) + δct+1 ⊙ ft+1
                    δot = δht ⊙ tanh(ct) ⊙ dsigmoid(ot)
                    δgt = δct ⊙ it ⊙ dtanh(gt)
                    δit = δct ⊙ gt ⊙ dsigmoid(it)
                    δft = δct ⊙ ct-1 ⊙ dsigmoid(ft)

                    notaion:
                                A_ -> At+1
                                _A -> At-1
                */
                Tensor f_ = t < te ? states[t + 1].f : Tensor(hiddenDim, 1);
                Tensor _c = t > 0 ? states[t - 1].c : Tensor(hiddenDim, 1);
                for (std::size_t i = 0; i < delta.o.totalSize; i++) {
                    delta.c[i] = delta.h[i] * states[t].o[i] * Tanh::df(states[t].c[i]) + delta_.c[i] * f_[i];
                    delta.o[i] = delta.h[i] * Tanh::f(states[t].c[i]) * Sigmoid::df(states[t].o[i]);
                    delta.g[i] = delta.c[i] * states[t].i[i] * Tanh::df(states[t].g[i]);
                    delta.i[i] = delta.c[i] * states[t].g[i] * Sigmoid::df(states[t].i[i]);
                    delta.f[i] = delta.c[i] * _c[i] * Sigmoid::df(states[t].f[i]);
                }
                /* gradient */

                /*
                    dw:    (hiddenDim, inputDim)
                    delta: (hiddenDim, 1)
                    x:     (inputDim, 1)
                    dw = delta * x^T
                */
                Tensor::MM::ikjk(d.Wi, delta.i, x[t]);
                Tensor::MM::ikjk(d.Wf, delta.f, x[t]);
                Tensor::MM::ikjk(d.Wg, delta.g, x[t]);
                Tensor::MM::ikjk(d.Wo, delta.o, x[t]);

                /*
                    du:    (hiddenDim, hiddenDim)
                    delta: (hiddenDim, 1)
                    _h:    (hiddenDim, 1)
                    du = delta * _h^T
                */
                Tensor _h = t > 0 ? states[t - 1].h : Tensor(hiddenDim, 1);
                Tensor::MM::ikjk(d.Ui, delta.i, _h);
                Tensor::MM::ikjk(d.Uf, delta.f, _h);
                Tensor::MM::ikjk(d.Ug, delta.g, _h);
                Tensor::MM::ikjk(d.Uo, delta.o, _h);

                d.Bi += delta.i;
                d.Bf += delta.f;
                d.Bg += delta.g;
                d.Bo += delta.o;
                /*
                    dw:(outputDim, hiddenDim)
                    E: (outputDim, 1)
                    h: (hiddenDim, 1)
                    dw = E * h^T
                */
                Tensor::MM::ikjk(d.W, loss[t], states[t].h);
                d.B += loss[t];
                /* next */
                delta_ = delta;
            }
            /* clear */
            loss.clear();
            x.clear();
            lstm.states.clear();
            return;
        }
    };

    /* optimizer */
    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        Optimizer optWi;
        Optimizer optUi;
        Optimizer optBi;
        Optimizer optWg;
        Optimizer optUg;
        Optimizer optBg;
        Optimizer optWf;
        Optimizer optUf;
        Optimizer optBf;
        Optimizer optWo;
        Optimizer optUo;
        Optimizer optBo;
        Optimizer optW;
        Optimizer optB;
    public:
        OptimizeBlock(){}
        explicit OptimizeBlock(const LSTM &layer)
        {
            optWi = Optimizer(layer.Wi.shape);
            optUi = Optimizer(layer.Ui.shape);
            optBi = Optimizer(layer.Bi.shape);
            optWg = Optimizer(layer.Wg.shape);
            optUg = Optimizer(layer.Ug.shape);
            optBg = Optimizer(layer.Bi.shape);
            optWf = Optimizer(layer.Wf.shape);
            optUf = Optimizer(layer.Uf.shape);
            optBf = Optimizer(layer.Bf.shape);
            optWo = Optimizer(layer.Wo.shape);
            optUo = Optimizer(layer.Uo.shape);
            optBo = Optimizer(layer.Bo.shape);
            optW  = Optimizer(layer.W.shape);
            optB  = Optimizer(layer.B.shape);
        }
        void operator()(LSTM& lstm, Grad& grad, float learningRate)
        {
            /* backward and eval */
            grad.backwardThroughTime(lstm);
            /* update */
            optWi(lstm.Wi, grad.d.Wi, learningRate);
            optUi(lstm.Ui, grad.d.Ui, learningRate);
            optBi(lstm.Bi, grad.d.Bi, learningRate);
            optWg(lstm.Wg, grad.d.Wg, learningRate);
            optUg(lstm.Ug, grad.d.Ug, learningRate);
            optBg(lstm.Bg, grad.d.Bg, learningRate);
            optWf(lstm.Wf, grad.d.Wf, learningRate);
            optUf(lstm.Uf, grad.d.Uf, learningRate);
            optBf(lstm.Bf, grad.d.Bf, learningRate);
            optWo(lstm.Wo, grad.d.Wo, learningRate);
            optUo(lstm.Uo, grad.d.Uo, learningRate);
            optBo(lstm.Bo, grad.d.Bo, learningRate);
            optW(lstm.W, grad.d.W, learningRate);
            optB(lstm.B, grad.d.B, learningRate);
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
                        |                 |             V    ^
                        ^                 ^            tanh  |
                        |                 |             |    |
                        |          -------x      -------x-----
                     f  |        i |      | g    | o    |
                        |          |      |      |      |
                     sigmoid    sigmoid  tanh  sigmoid  |
                        |          |      |      |      |
            h(t-1) -->----------------------------      --------->--- h(t)
                        |
                        ^
                        |
                        x(t)

            ft = sigmoid(Wf*xt + Uf*ht-1 + bf);
            it = sigmoid(Wi*xt + Ui*ht-1 + bi);
            gt =    tanh(Wg*xt + Ug*ht-1 + bg);
            ot = sigmoid(Wo*xt + Uo*ht-1 + bo);
            ct = ft ⊙ ct-1 + it ⊙ gt
            ht = ot ⊙ tanh(ct)
            yt = linear(W*ht + b)
        */
        State state(hiddenDim, outputDim);

        Tensor::MM::ikkj(state.f, Wf, x);
        Tensor::MM::ikkj(state.i, Wi, x);
        Tensor::MM::ikkj(state.g, Wg, x);
        Tensor::MM::ikkj(state.o, Wo, x);

        Tensor::MM::ikkj(state.f, Uf, _h);
        Tensor::MM::ikkj(state.i, Ui, _h);
        Tensor::MM::ikkj(state.g, Ug, _h);
        Tensor::MM::ikkj(state.o, Uo, _h);

        for (std::size_t i = 0; i < state.f.totalSize; i++) {
            state.f[i] = Sigmoid::f(state.f[i] + Bf[i]);
            state.i[i] = Sigmoid::f(state.i[i] + Bi[i]);
            state.g[i] =    Tanh::f(state.g[i] + Bg[i]);
            state.o[i] = Sigmoid::f(state.o[i] + Bo[i]);
            state.c[i] = state.f[i] * _c[i] + state.i[i]*state.g[i];
            state.h[i] = state.o[i] * Tanh::f(state.c[i]);
        }

        Tensor::MM::ikkj(state.y, W, state.h);
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
