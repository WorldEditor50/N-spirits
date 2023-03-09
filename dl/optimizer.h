#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>
#include "optimize.h"
#include "layerdef.h"
#include "../basic/tensor.hpp"


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
    template<typename Layers, typename Grads, typename Optimizers, std::size_t N>
    struct Generate {
        inline static void impl(Layers& layers, Grads &grads, Optimizers &optimizers)
        {
            //std::cout<<"N:"<<N<<std::endl;
            Generate<Layers, Grads, Optimizers, N - 1>::impl(layers, grads, optimizers);

            using Layer = std::tuple_element_t<N - 1, Layers>;
            using Grad = typename Layer::Grad;
            Layer &layer = std::get<N - 1>(layers);
            /* grad */
            Grad &grad = std::get<N - 1>(grads);
            using Param = typename Layer::ParamType;
            grad = Grad(static_cast<Param>(layer));
            /* optimizer */
            using OptimizeBlock = typename Layer::template OptimizeBlock<OptimizeMethod>;
            auto& opt = std::get<N - 1>(optimizers);
            opt = OptimizeBlock(layer);
            return;
        }
    };

    template<typename Layers, typename Grads, typename Optimizers>
    struct Generate<Layers, Grads, Optimizers, 1> {
        inline static void impl(Layers& layers, Grads &grads, Optimizers &optimizers)
        {
            auto &layer = std::get<0>(layers);
            auto &grad = std::get<0>(grads);
            /* grad */
            using Layer = std::tuple_element_t<0, Layers>;
            using Param = typename Layer::ParamType;
            using Grad = typename Layer::Grad;
            grad = Grad(static_cast<Param>(layer));
            /* optimizer */
            using OptimizeBlock = typename Layer::template OptimizeBlock<OptimizeMethod>;
            auto& opt = std::get<0>(optimizers);
            opt = OptimizeBlock(layer);
            return;
        }
    };
    /* backward: LayerN1 -> LayerN2 */
    template<typename Layers, typename Grads, typename LayerN1, typename LayerN2, std::size_t N>
    struct Backward {
        inline static void impl(Grads &grads, Layers& layers, const Tensor &x)
        {
            auto& layerN1 = std::get<N - 1>(layers);
            auto& gradN1 = std::get<N - 1>(grads);
            auto& gradN2 = std::get<N - 2>(grads);
            /* backward */
            gradN1.backward(layerN1, gradN2.delta);
            /* evaluate */
            Tensor &x1 = std::get<N - 2>(layers).o;
            gradN1.eval(x1, layerN1.o);
            /* next */
            using LayerN3 = std::tuple_element_t<N - 3, Layers>;
            Backward<Layers, Grads, LayerN2, LayerN3, N - 1>::impl(grads, layers, x);
            return;
        }
    };

    template<typename Layers, typename Grads, typename LayerN1, typename LayerN2>
    struct Backward<Layers, Grads, LayerN1, LayerN2, 2> {
        inline static void impl(Grads &grads, Layers& layers, const Tensor &x)
        {
            auto& layer0 = std::get<0>(layers);
            auto& grad0 = std::get<0>(grads);
            auto& layer1 = std::get<1>(layers);
            auto& grad1 = std::get<1>(grads);
            /* backward */
            grad1.backward(layer1, grad0.delta);
            /* evaluate */
            grad1.eval(layer0.o, layer1.o);
            grad0.eval(x, layer0.o);
            return;
        }
    };
    /* backward: FcLayer -> LSTM */
    template<typename Layers, typename Grads>
    struct Backward<Layers, Grads, FcLayer, LSTM, 2> {
        inline static void impl(Grads &grads, Layers& layers, const Tensor &x)
        {
            auto& lstm = std::get<0>(layers);
            auto& lstmGrad = std::get<0>(grads);
            auto& layer1 = std::get<1>(layers);
            auto& grad1 = std::get<1>(grads);
            /* backward */
            Tensor loss(lstm.y.shape);
            grad1.backward(layer1, loss);
            lstmGrad.cache(loss, x);
            /* evaluate */
            grad1.eval(lstm.y, layer1.o);
            return;
        }
    };

    /* backward: fc -> MaxPooling2d */
    template<typename Layers, typename Grads, std::size_t N>
    struct Backward<Layers, Grads, FcLayer, MaxPooling2d, N> {
        inline static void impl(Grads &grads, Layers& layers, const Tensor &x)
        {
            auto& fcLayer = std::get<N - 1>(layers);
            auto& fcGrad = std::get<N - 1>(grads);
            auto& maxPooling2d = std::get<N - 2>(layers);
            auto& maxPooling2dGrad = std::get<N - 2>(grads);
            Tensor &o = maxPooling2d.o;
            /* backward */
            Tensor delta(int(o.totalSize), 1);
            fcGrad.backward(fcLayer, delta);
            maxPooling2dGrad.delta.val = delta.val;
            /* evaluate fcLayer */
            fcGrad.eval(Tensor(std::vector<int>{int(o.totalSize), 1}, o.val), fcLayer.o);
            /* next */
            using LayerN3 = std::tuple_element_t<N - 3, Layers>;
            Backward<Layers, Grads, MaxPooling2d, LayerN3, N - 1>::impl(grads, layers, x);
            return;
        }
    };

    /* update */
    template< typename Optimizers, typename Layers, typename Grads, std::size_t N>
    struct Update {
        inline static void impl(Optimizers &optimizers, Layers& layers, Grads &grads, float learningRate)
        {
            Update<Optimizers, Layers, Grads, N - 1>::impl(optimizers, layers, grads, learningRate);
            auto &layer = std::get<N - 1>(layers);
            auto &grad = std::get<N - 1>(grads);
            auto &opt = std::get<N - 1>(optimizers);
            opt(layer, grad, learningRate);
            return;
        }
    };

    template< typename Optimizers, typename Layers, typename Grads>
    struct Update<Optimizers, Layers, Grads, 1> {
        inline static void impl(Optimizers &optimizers, Layers& layers, Grads &grads, float learningRate)
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
        Generate<Layers, Grads, Optimizers, Net::N>::impl(net_.layers, grads, optimizers);
    }

    void backward(const Tensor &loss, const Tensor &x)
    {
        auto &grad = std::get<Net::N - 1>(grads);
        grad.delta = loss;
        using LayerN1 = std::tuple_element_t<Net::N - 1, Layers>;
        using LayerN2 = std::tuple_element_t<Net::N - 2, Layers>;
        Backward<Layers, Grads, LayerN1, LayerN2, Net::N>::impl(grads, net.layers, x);
        return;
    }

    void backward(const Tensor &loss, const Tensor &x, const Tensor &yt)
    {
        auto &grad1 = std::get<Net::N - 1>(grads);
        auto &grad2 = std::get<Net::N - 2>(grads);
        auto &layer1 = std::get<Net::N - 1>(net.layers);
        auto &layer2 = std::get<Net::N - 2>(net.layers);
        /* backward */
        grad1.delta = loss;
        grad1.backward(layer1, grad2.delta);
        /* evaluate gradient */
        grad1.eval(layer2.o, layer1.o, yt);
        /* next layer */
        using LayerN2 = std::tuple_element_t<Net::N - 2, Layers>;
        using LayerN3 = std::tuple_element_t<Net::N - 3, Layers>;
        Backward<Layers, Grads, LayerN2, LayerN3, Net::N - 1>::impl(grads, net.layers, x);
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
        Update<Optimizers, Layers, Grads, Net::N>::impl(optimizers, net.layers, grads, learningRate);
        return;
    }
};


#endif // OPTIMIZER_H
