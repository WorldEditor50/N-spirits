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
        i(Tensor(hiddenDim, 1)),f(Tensor(hiddenDim, 1)),g(Tensor(hiddenDim, 1)),
        o(Tensor(hiddenDim, 1)),c(Tensor(hiddenDim, 1)),h(Tensor(hiddenDim, 1)),
        y(Tensor(outputDim, 1)){}
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
            for (int t = lstm.states.size() - 1; t >= 0; t--) {
                State delta(hiddenDim, outputDim);
                /* delta = W^T * E */
                Tensor::Mul::kikj(delta.h, lstm.W, loss[t]);
                /* delta = U * delta_ */
                Tensor::Mul::ikkj(delta.h, lstm.Uf, delta_.i);
                Tensor::Mul::ikkj(delta.h, lstm.Ui, delta_.f);
                Tensor::Mul::ikkj(delta.h, lstm.Ug, delta_.g);
                Tensor::Mul::ikkj(delta.h, lstm.Uo, delta_.o);
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
                    delta.c[i] = delta.h[i] * states[t].o[i] * Tanh::df(states[t].c[i]) + delta_.c[i] * f_[i];
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
                Tensor::Mul::ikjk(d.W, loss[t], states[t].h);
                Tensor::Mul::ikjk(d.B, loss[t], states[t].y);
                /*
                    dw:    (hiddenDim, inputDim)
                    delta: (hiddenDim, 1)
                    x:     (inputDim, 1)
                    dw = delta * x^T
                */
                Tensor::Mul::ikjk(d.Wi, delta.i, x[t]);
                Tensor::Mul::ikjk(d.Wf, delta.f, x[t]);
                Tensor::Mul::ikjk(d.Wg, delta.g, x[t]);
                Tensor::Mul::ikjk(d.Wo, delta.o, x[t]);

                /*
                    du:    (hiddenDim, hiddenDim)
                    delta: (hiddenDim, 1)
                    _h:    (hiddenDim, 1)
                    du = delta * _h^T
                */
                Tensor _h = t > 0 ? states[t - 1].h : Tensor(hiddenDim, 1);
                Tensor::Mul::ikjk(d.Ui, delta.i, _h);
                Tensor::Mul::ikjk(d.Uf, delta.f, _h);
                Tensor::Mul::ikjk(d.Ug, delta.g, _h);
                Tensor::Mul::ikjk(d.Uo, delta.o, _h);

                d.Bi += delta.i;
                d.Bf += delta.f;
                d.Bg += delta.g;
                d.Bo += delta.o;
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
        explicit OptimizeBlock(const LSTM &layer)
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
        void operator()(LSTM& lstm, Grad& grad, float learningRate)
        {
            /* backward and eval */
            grad.backwardThroughTime(lstm);
            /* update */
            Wi(lstm.Wi, grad.d.Wi, learningRate);
            Ui(lstm.Ui, grad.d.Ui, learningRate);
            Bi(lstm.Bi, grad.d.Bi, learningRate);
            Wg(lstm.Wg, grad.d.Wg, learningRate);
            Ug(lstm.Ug, grad.d.Ug, learningRate);
            Bg(lstm.Bg, grad.d.Bg, learningRate);
            Wf(lstm.Wf, grad.d.Wf, learningRate);
            Uf(lstm.Uf, grad.d.Uf, learningRate);
            Bf(lstm.Bf, grad.d.Bf, learningRate);
            Wo(lstm.Wo, grad.d.Wo, learningRate);
            Uo(lstm.Uo, grad.d.Uo, learningRate);
            Bo(lstm.Bo, grad.d.Bo, learningRate);
            W(lstm.W, grad.d.W, learningRate);
            B(lstm.B, grad.d.B, learningRate);
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

        Tensor::Mul::ikkj(state.f, Wf, x);
        Tensor::Mul::ikkj(state.i, Wi, x);
        Tensor::Mul::ikkj(state.g, Wg, x);
        Tensor::Mul::ikkj(state.o, Wo, x);

        Tensor::Mul::ikkj(state.f, Uf, _h);
        Tensor::Mul::ikkj(state.i, Ui, _h);
        Tensor::Mul::ikkj(state.g, Ug, _h);
        Tensor::Mul::ikkj(state.o, Uo, _h);

        for (std::size_t i = 0; i < state.f.totalSize; i++) {
            state.f[i] = Sigmoid::f(state.f[i] + Bf[i]);
            state.i[i] = Sigmoid::f(state.i[i] + Bi[i]);
            state.g[i] =    Tanh::f(state.g[i] + Bg[i]);
            state.o[i] = Sigmoid::f(state.o[i] + Bo[i]);
            state.c[i] = state.f[i] * _c[i] + state.i[i]*state.g[i];
            state.h[i] = state.o[i] * Tanh::f(state.c[i]);
        }

        Tensor::Mul::ikkj(state.y, W, state.h);
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
