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
    /* forward */
    template<typename Layers, std::size_t Ni, typename LayerN1>
    struct Forward {
        static Tensor& impl(Layers& layers, const Tensor& x)
        {
            using LayerN2 = std::tuple_element_t<Ni - 2, Layers>;
            Tensor& o = Forward<Layers, Ni - 1, LayerN2>::impl(layers, x);
            return std::get<Ni - 1>(layers).forward(o);
        }
    };

    template<typename Layers, typename LayerN1>
    struct Forward<Layers, 1, LayerN1> {
        static Tensor& impl(Layers& layers, const Tensor& x)
        {
            return std::get<0>(layers).forward(x);
        }
    };

    template<typename Layers>
    struct Forward<Layers, 1, BatchNorm1d> {
        static Tensor& impl(Layers& layers, const Tensor& x)
        {
            return std::get<0>(layers).forward(x);
        }
    };

    /* save */
    template<typename Layers, std::size_t Ni>
    struct Save {
        static void impl(Layers& layers, std::ofstream &file)
        {
            Save<Layers, Ni - 1>::impl(layers, file);
            std::get<Ni - 1>(layers).save(file);
            return;
        }
    };

    template<typename Layers>
    struct Save<Layers, 1> {
        static void impl(Layers& layers, std::ofstream &file)
        {
            std::get<0>(layers).save(file);
            return;
        }
    };
    /* load */
    template<typename Layers, int Ni>
    struct Load {
        static void impl(Layers& layers, std::ifstream &file)
        {
            Load<Layers, Ni - 1>::impl(layers, file);
            std::get<Ni - 1>(layers).load(file);
            return;
        }
    };

    template<typename Layers>
    struct Load<Layers, 1> {
        static void impl(Layers& layers, std::ifstream &file)
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
        using LayerN1 = std::tuple_element_t<Net::N - 1, Layers>;
        return Forward<Layers, N, LayerN1>::impl(layers, x);
    }

    inline void operator()(const std::vector<Tensor> &x, std::vector<Tensor> &y)
    {
        using LayerN1 = std::tuple_element_t<Net::N - 1, Layers>;
        y = std::vector<Tensor>(x.size());
        for (std::size_t i = 0; i < x.size(); i++) {
            y[i] = Forward<Layers, N, LayerN1>::impl(layers, x[i]);
        }
        return;
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
