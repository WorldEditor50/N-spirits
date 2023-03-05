#ifndef NET_H
#define NET_H
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <tuple>
#include <map>
#include <type_traits>
#include "../basic/tensor.hpp"
#include "layerdef.h"


template<typename ...TLayer>
class Net
{
public:
    using Layers = std::tuple<TLayer...>;
    using Grads = std::tuple<typename TLayer::Grad...>;
    template<typename Optimizer>
    using OptimzeBlocks = std::tuple<typename TLayer::template OptimizeBlock<Optimizer>...>;
    constexpr static std::size_t N = sizeof... (TLayer);
    Layers layers;
public:
    /* use template specialization to control forwarding between the layers */

    /* forward: LayerN1 <- LayerN2 */
    template<typename Layers, typename LayerN1, typename LayerN2, std::size_t Ni>
    struct Forward {
        inline static Tensor& impl(Layers& layers, const Tensor& x)
        {
            using LayerN3 = std::tuple_element_t<Ni - 3, Layers>;
            /* LayerN2 <- LayerN3 */
            Tensor& o = Forward<Layers, LayerN2, LayerN3, Ni - 1>::impl(layers, x);
            /* LayerN1 <- LayerN2 */
            return std::get<Ni - 1>(layers).forward(o);
        }
    };

    template<typename Layers, typename LayerN1, typename LayerN2>
    struct Forward<Layers, LayerN1, LayerN2, 2> {
        inline static Tensor& impl(Layers& layers, const Tensor& x)
        {
            Tensor& o = std::get<0>(layers).forward(x);
            return std::get<1>(layers).forward(o);
        }
    };

    /* forward: FcLayer <- MaxPooling2d */
    template<typename Layers, std::size_t Ni>
    struct Forward<Layers, FcLayer, MaxPooling2d, Ni> {
        inline static Tensor& impl(Layers& layers, const Tensor& x)
        {
            auto& fcLayer = std::get<Ni - 1>(layers);
            /* MaxPooling2d */
            Tensor& o = Forward<Layers, MaxPooling2d, Conv2d, Ni - 1>::impl(layers, x);
            /* FcLayer forward */
            return fcLayer.forward(Tensor(std::vector<int>{int(o.totalSize), 1}, o.val));
        }
    };

    /* forward: FcLayer <- AvgPooling2d */
    template<typename Layers, std::size_t Ni>
    struct Forward<Layers, FcLayer, AvgPooling2d, Ni> {
        inline static Tensor& impl(Layers& layers, const Tensor& x)
        {
            auto& fcLayer = std::get<Ni - 1>(layers);
            /* AvgPooling2d */
            Tensor& o = Forward<Layers, AvgPooling2d, Conv2d, Ni - 1>::impl(layers, x);
            /* FcLayer forward */
            return fcLayer.forward(Tensor(std::vector<int>{int(o.totalSize), 1}, o.val));
        }
    };

    /* forward: FcLayer <- Conv2d */
    template<typename Layers, std::size_t Ni>
    struct Forward<Layers, FcLayer, Conv2d, Ni> {
        inline static Tensor& impl(Layers& layers, const Tensor& x)
        {
            auto& fcLayer = std::get<Ni - 1>(layers);
            /* conv2d */
            using LayerN3 = std::tuple_element_t<Ni - 3, Layers>;
            Tensor& o = Forward<Layers, Conv2d, LayerN3, Ni - 1>::impl(layers, x);
            /* FcLayer forward */
            return fcLayer.forward(Tensor(std::vector<int>{int(o.totalSize), 1}, o.val));
        }
    };

    /* forward: LSTM <- MaxPooling2d */
    template<typename Layers, std::size_t Ni>
    struct Forward<Layers, LSTM, MaxPooling2d, Ni> {
        inline static Tensor& impl(Layers& layers, const Tensor& x)
        {
            auto& lstm = std::get<Ni - 1>(layers);
            /* MaxPooling2d */
            Tensor& o = Forward<Layers, MaxPooling2d, Conv2d, Ni - 1>::impl(layers, x);
            /* lstm forward */
            return lstm.forward(Tensor(std::vector<int>{int(o.totalSize), 1}, o.val));
        }
    };


    /* save */
    template<typename Layers, std::size_t Ni>
    struct Save {
        inline static void impl(Layers& layers, std::ofstream &file)
        {
            Save<Layers, Ni - 1>::impl(layers, file);
            std::get<Ni - 1>(layers).save(file);
            return;
        }
    };

    template<typename Layers>
    struct Save<Layers, 1> {
        inline static void impl(Layers& layers, std::ofstream &file)
        {
            std::get<0>(layers).save(file);
            return;
        }
    };
    /* load */
    template<typename Layers, std::size_t Ni>
    struct Load {
        inline static void impl(Layers& layers, std::ifstream &file)
        {
            Load<Layers, Ni - 1>::impl(layers, file);
            std::get<Ni - 1>(layers).load(file);
            return;
        }
    };

    template<typename Layers>
    struct Load<Layers, 1> {
        inline static void impl(Layers& layers, std::ifstream &file)
        {
            std::get<0>(layers).load(file);
            return;
        }
    };
public:
    explicit Net(TLayer&& ...layer)
        :layers(layer...){}

    inline Tensor& operator()(const Tensor &x)
    {  
        using LayerN2 = std::tuple_element_t<Net::N - 2, Layers>;
        using LayerN1 = std::tuple_element_t<Net::N - 1, Layers>;
        return Forward<Layers, LayerN1, LayerN2, N>::impl(layers, x);
    }

    void load(const std::string& fileName)
    {
        std::ifstream file;
        file.open(fileName);
        Load<Layers, N>::impl(layers);
        file.close();
        return;
    }

    void save(const std::string& fileName)
    {
        std::ofstream file;
        file.open(fileName);
        Save<Layers, N>::impl(layers);
        file.close();
        return;
    }
};

#endif // NET_H
