#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>
#include "../basic/tensor.hpp"
#include "optimize.h"
#include "layerdef.h"

class FcLayer;
class SoftmatLayer;
class LSTM;
class Conv2d;
class MaxPooling2d;
class AvgPooling2d;

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
    template<int N, typename TLayer>
    struct Backward {
        static void impl(Grads &grads, Layers& layers, const Tensor &x)
        {
            auto& layer = std::get<N - 1>(layers);
            auto& grad = std::get<N - 1>(grads);
            auto& preLayer = std::get<N - 2>(layers);
            auto& preGrad = std::get<N - 2>(grads);
            using PreLayer = typename std::remove_reference<decltype(preLayer)>::type;
            /* backward */
            grad.backward(layer, preGrad.delta);
            /* evaluate */
            Tensor &x_ = std::get<N - 2>(layers).o;
            grad.eval(x_, layer.o);
            Backward<N - 1, PreLayer>::impl(grads, layers, x);
            return;
        }
    };

    template<typename TLayer>
    struct Backward<1, TLayer> {
        static void impl(Grads &grads, Layers& layers, const Tensor &x)
        {
            auto& layer = std::get<1>(layers);
            auto& grad = std::get<1>(grads);
            auto& preGrad = std::get<0>(grads);
            /* backward */
            grad.backward(layer, preGrad.delta);
            /* evaluate */
            auto& preLayer = std::get<0>(layers);
            preGrad.eval(x, preLayer.o);
            return;
        }
    };

    template<>
    struct Backward<1, LSTM> {
        static void impl(Grads &grads, Layers& layers, const Tensor &x)
        {
            auto& layer = std::get<1>(layers);
            auto& grad = std::get<1>(grads);
            auto& preGrad = std::get<0>(grads);
            /* backward through time */
            grad.backwardAtTime(layer, grad.delta, x);
            return;
        }
    };
    template<int N>
    struct Backward<N, MaxPooling2d> {
        static void impl(Grads &grads, Layers& layers, const Tensor &x)
        {
            auto& layer = std::get<N - 1>(layers);
            auto& grad = std::get<N - 1>(grads);
            auto& preLayer = std::get<N - 2>(layers);
            auto& preGrad = std::get<N - 2>(grads);
            using PreLayer = typename std::remove_reference<decltype(preLayer)>::type;
            /* backward */
            grad.backward(layer, preGrad.delta);
            /* no gradient */
            Backward<N - 1, PreLayer>::impl(grads, layers, x);
            return;
        }
    };

    template<int N>
    struct Backward<N, AvgPooling2d> {
        static void impl(Grads &grads, Layers& layers, const Tensor &x)
        {
            auto& layer = std::get<N - 1>(layers);
            auto& grad = std::get<N - 1>(grads);
            auto& preLayer = std::get<N - 2>(layers);
            auto& preGrad = std::get<N - 2>(grads);
            using PreLayer = typename std::remove_reference<decltype(preLayer)>::type;
            /* backward */
            grad.backward(layer, preGrad.delta);
            /* no gradient */
            Backward<N - 1, PreLayer>::impl(grads, layers, x);
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

    void backward(const Tensor &loss, const Tensor &x)
    {
        auto &grad = std::get<Net::N - 1>(grads);
        grad.delta = loss;
        auto &layer = std::get<Net::N - 1>(net.layers);
        using Layer = typename std::remove_reference<decltype(layer)>::type;
        Backward<Net::N, Layer>::impl(grads, net.layers, x);
        return;
    }

    void backward(const Tensor &loss, const Tensor &x, const Tensor &yt)
    {
        auto &grad = std::get<Net::N - 1>(grads);
        grad.delta = loss;
        auto &layer = std::get<Net::N - 1>(net.layers);
        Tensor &x_ = std::get<Net::N - 2>(net.layers).o;
        grad.eval(x_, layer.o, yt);
        using Layer = typename std::remove_reference<decltype(layer)>::type;
        Backward<Net::N - 1, Layer>::impl(grads, net.layers,  x);
        return;
    }

    void backward(const std::vector<Tensor> &loss, const std::vector<Tensor> &x)
    {
        for (int t = x.size() - 1; t >= 0; t--) {
            backward(loss[t], x[t]);
        }
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
