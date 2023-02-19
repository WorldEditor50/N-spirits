#ifndef NET_H
#define NET_H
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <tuple>
#include <map>
#include "layer.h"


template<typename ...TLayer>
class Net
{
public:
    using Layers = std::tuple<TLayer...>;
    using Grads = std::tuple<typename TLayer::GradType...>;
    constexpr static int N = sizeof... (TLayer);
    Layers layers;
public:
    /* forward */
    template<int Ni>
    struct Forward {
        static void impl(Layers& layers, const Tensor& x)
        {
            Forward<Ni - 1>::impl(x);
            Tensor& o = std::get<Ni - 2>(layers).o;
            std::get<Ni - 1>(layers).forward(o);
            return;
        }
    };

    template<>
    struct Forward<1> {
        static void impl(Layers& layers, const Tensor& x)
        {
            std::get<0>(layers).forward(x);
            return;
        }
    };

    /* save */
    template<int Ni>
    struct Save {
        static void impl(Layers& layers, std::ofstream &file)
        {
            Save<Ni - 1>::impl(layers);
            std::get<Ni - 1>(layers).save(file);
            return;
        }
    };

    template<>
    struct Save<1> {
        static void impl(Layers& layers, std::ofstream &file)
        {
            std::get<0>(layers).save(file);
            return;
        }
    };
    /* load */
    template<int Ni>
    struct Load {
        static void impl(Layers& layers, std::ifstream &file)
        {
            Save<Ni - 1>::impl(layers);
            std::get<Ni - 1>(layers).load(file);
            return;
        }
    };

    template<>
    struct Load<1> {
        static void impl(Layers& layers, std::ifstream &file)
        {
            std::get<0>(layers).load(file);
            return;
        }
    };
public:
    explicit Net(TLayer&& ...layer)
        :layers(layer...){}

    void operator()(const Tensor &x)
    {
        Forward<N>::impl(layers, x);
        return;
    }

    void load(const std::string& fileName)
    {
        std::ifstream file;
        file.open(fileName);
        Load<N>::impl(layers);
        file.close();
        return;
    }

    void save(const std::string& fileName)
    {
        std::ofstream file;
        file.open(fileName);
        Save<N>::impl(layers);
        file.close();
        return;
    }
};

#endif // NET_H