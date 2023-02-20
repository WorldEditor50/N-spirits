#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>
#include "../basic/tensor.hpp"
#include "optimize.h"
#include <typeinfo>

template<typename Net, typename OptimizeMethod>
class Optimizer
{
public:
    using Layers = typename Net::Layers;
    using Grads = typename Net::Grads;
    using Optimizers = typename Net::template OptimzeBlocks<OptimizeMethod>;
    float learningRate;
    Net &net;
    Grads grads;
    Optimizers optimizers;
public:
    /* generate grad */
    template<int N>
    struct Generate {
        static void impl(Layers& layers, Grads &grads, Optimizers &optimizers)
        {
            Generate<N - 1>::impl(layers, grads, optimizers);
            auto &layer = std::get<N - 1>(layers);
            /* grad */
            auto &grad = std::get<N - 1>(grads);
            using Layer = typename std::remove_reference<decltype(layer)>::type;
            using ParamType = typename Layer::ParamType;
            using GradType = typename Layer::Grad;
            grad = GradType(static_cast<ParamType>(layer));
            /* optimizer */
            using OptimizeBlock = typename Layer::template OptimizeBlock<OptimizeMethod>;
            auto& opt = std::get<N - 1>(optimizers);
            opt = OptimizeBlock(layer);
            return;
        }
    };

    template<>
    struct Generate<1> {
        static void impl(Layers& layers, Grads &grads, Optimizers &optimizers)
        {
            auto &layer = std::get<0>(layers);
            auto &grad = std::get<0>(grads);
            /* grad */
            using Layer = typename std::remove_reference<decltype(layer)>::type;
            using ParamType = typename Layer::ParamType;
            using GradType = typename Layer::Grad;
            grad = GradType(static_cast<ParamType>(layer));
            /* optimizer */
            using OptimizeBlock = typename Layer::template OptimizeBlock<OptimizeMethod>;
            auto& opt = std::get<0>(optimizers);
            opt = OptimizeBlock(layer);
            return;
        }
    };
    /* backward */
    template<int N>
    struct Backward {
        static void impl(Grads &grads, Layers& layers)
        {
            auto& layer = std::get<N - 1>(layers);
            auto& grad = std::get<N - 1>(grads);
            auto& preGrad = std::get<N - 2>(grads);
            grad.backward(layer, preGrad.delta);
            Backward<N - 1>::impl(grads, layers);
            return;
        }
    };

    template<>
    struct Backward<1> {
        static void impl(Grads &grads, Layers& layers)
        {
            auto& layer = std::get<1>(layers);
            auto& grad = std::get<1>(grads);
            auto& preGrad = std::get<0>(grads);
            grad.backward(layer, preGrad.delta);
            return;
        }
    };
    /* grad */
    template<int N>
    struct Gradient {
        static void eval(Grads &grads, Layers& layers, const Tensor &x)
        {
            Gradient<N - 1>::eval(grads, layers, x);
            auto &layer = std::get<N - 1>(layers);
            auto &grad = std::get<N - 1>(grads);
            Tensor &x_ = std::get<N - 2>(layers).o;
            grad.eval(x_, layer.o);
            return;
        }
    };

    template<>
    struct Gradient<1> {
        static void eval(Grads &grads, Layers& layers, const Tensor &x)
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
        static void impl(Optimizers &optimizers, Layers& layers, Grads &grads, float learningRate)
        {
            Update<N - 1>::impl(optimizers, layers, grads, learningRate);
            auto &layer = std::get<N - 1>(layers);
            auto &grad = std::get<N - 1>(grads);
            auto &opt = std::get<N - 1>(optimizers);
            opt(layer, grad, learningRate);
            return;
        }
    };

    template<>
    struct Update<1> {
        static void impl(Optimizers &optimizers, Layers& layers, Grads &grads, float learningRate)
        {
            auto &layer = std::get<0>(layers);
            auto &grad = std::get<0>(grads);
            auto &opt = std::get<0>(optimizers);
            opt(layer, grad, learningRate);
            return;
        }
    };
public:
    explicit Optimizer(Net &net_, float learningRate_)
        :learningRate(learningRate_),net(net_)
    {
        Generate<Net::N>::impl(net_.layers, grads, optimizers);
    }

    void backward(const Tensor &loss)
    {
        auto &grad = std::get<Net::N - 1>(grads);
        grad.delta = loss;
        Backward<Net::N>::impl(grads, net.layers);
        return;
    }
    void grad(const Tensor &x)
    {
        /* gradient */
        Gradient<Net::N>::eval(grads, net.layers,  x);
        return;
    }
    void update()
    {
        /* update */
        Update<Net::N>::impl(optimizers, net.layers, grads, learningRate);
        return;
    }
};


#endif // OPTIMIZER_H
