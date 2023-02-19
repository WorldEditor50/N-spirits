#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>
#include "../basic/tensor.h"

template<typename Net, typename OptimizeMethod>
class Optimizer
{
public:
    using Layers = typename Net::Layers;
    using Grads = typename Net::Grads;
    float learningRate;
    Net &net;
    Grads grads;
    std::vector<OptimizeMethod> optimizers;
public:
    /* generate grad */
    template<int N>
    struct GenerateGrad {
        static void impl(Layers& layers, Grads &grads)
        {
            GenerateGrad<N - 1>::impl(layers, grads);
            auto &layer = std::get<N - 1>(layers);
            auto &grad = std::get<N - 1>(grads);
            using ParamType = typename decltype (layer)::ParamType;
            grad = std::tuple_element<N - 1, Grads>::type(static_cast<ParamType>(layer));
            return;
        }
    };

    template<>
    struct GenerateGrad<1> {
        static void impl(Layers& layers, Grads &grads)
        {
            auto &layer = std::get<0>(layers);
            auto &grad = std::get<0>(grads);
            using ParamType = typename decltype (layer)::ParamType;
            grad = std::tuple_element<0, Grads>::type(static_cast<ParamType>(layer));
            return;
        }
    };
    /* backward */
    template<int N>
    struct Backward {
        static void impl(Layers& layers, Grads &grads, const Tensor& loss)
        {
            std::get<N - 1>(layers).backward(loss);
            Tensor& delta = std::get<N>(layers).delta;
            Backward<N - 1>::impl(delta);
            return;
        }
    };

    template<>
    struct Backward<1> {
        static void impl(Layers& layers, Grads &grads, const Tensor& loss)
        {
            auto &layer = std::get<0>(layers);
            return;
        }
    };
    /* grad */
    template<int N>
    struct Gradient {
        static void eval(Layers& layers, Grads &grads, const Tensor &x)
        {
            Gradient<N - 1>::eval(layers, grads, x);
            auto &layer = std::get<N - 1>(layers);
            auto &grad = std::get<N - 1>(grads);
            Tensor &x_ = std::get<N - 2>(layers).o;
            grad.eval(x_, layer.o);
            return;
        }
    };

    template<>
    struct Gradient<1> {
        static void eval(Layers& layers, Grads &grads, const Tensor &x)
        {
            auto &layer = std::get<0>(layers);
            auto &grad = std::get<0>(grads);
            grad.eval(x, layer.o);
            return;
        }
    };

    /* update */
    template<int N>
    struct Update {
        static void impl(Layers& layers, Grads &grads, std::vector<Optimizer> &opt, float learningRate)
        {
            Update<N - 1>::impl(layers, opt, learningRate);
            auto &layer = std::get<N - 1>(layers);
            auto &grad = std::get<N - 1>(grads);
            opt[N - 1](layer.w, grad.dw, learningRate);
            opt[N - 1](layer.b, grad.db, learningRate);
            return;
        }
    };

    template<>
    struct Update<1> {
        static void impl(Layers& layers, Grads &grads, std::vector<Optimizer> &opt, float learningRate)
        {
            auto &layer = std::get<0>(layers);
            auto &grad = std::get<0>(grads);
            opt[0](layer.w, grad.dw, learningRate);
            opt[0](layer.b, grad.db, learningRate);
            return;
        }
    };
public:
    explicit Optimizer(Net &net_, float learningRate_)
        :learningRate(learningRate_),net(net_)
    {
        GenerateGrad<Net::N>::impl(net_.layers, grads);
    }

    void backward(const Tensor &loss)
    {
        Backward<Net::N>::impl(net.layers, optimizers, loss);
        return;
    }
    void grad()
    {
        /* gradient */
        Gradient<Net::N>::eval(net.layers, grads);

        return;
    }
    void update()
    {
        /* update */
        Update<Net::N>::impl(net.layers, optimizers, learningRate);
        return;
    }
};


#endif // OPTIMIZER_H
