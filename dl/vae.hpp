#ifndef VAE_HPP
#define VAE_HPP
#include <tuple>
#include "activate.hpp"
#include "layer.hpp"
#include "../basic/tensor.hpp"
#include "../basic/linalg.h"

class VAE
{
private:
    /* encoder */
    FcLayer encode1;
    FcLayer encode2;
    FcLayer encodeMu;
    FcLayer encodeSigma;
    /* decoder */
    FcLayer decode1;
    FcLayer decode2;
public:
    explicit VAE(int inputDim, int hiddenDim, int zDim)
    {
        /* encoder */
        encode1 = FcLayer(inputDim, hiddenDim, false, Fn_Tanh);
        encode2 = FcLayer(hiddenDim, zDim, false, Fn_Linear);
        encodeMu = FcLayer(hiddenDim, zDim, false, Fn_Linear);
        encodeSigma = FcLayer(hiddenDim, zDim, false, Fn_Linear);
        /* decoder */
        decode1 = FcLayer(zDim, hiddenDim, false, Fn_LeakyRelu);
        decode2 = FcLayer(hiddenDim, inputDim, false, Fn_Sigmoid);
    }
    void encode(const Tensor &img, Tensor &mu, Tensor &sigma)
    {
        Tensor& o1 = encode1.forward(img);
        Tensor& o2 = encode2.forward(o1);
        mu = encodeMu.forward(o2);
        sigma = encodeSigma.forward(o2);
        return;
    }

    Tensor& decode(const Tensor &z)
    {
        Tensor& h = decode1.forward(z);
        return decode2.forward(h);
    }

    Tensor& forward(const Tensor &xi)
    {
        Tensor mu;
        Tensor sigma;
        encode(xi, mu, sigma);
        Tensor epsilon = Tensor(sigma.shape);
       //LinAlg::normal(epsilon, -1, 1);
        Tensor z = mu + epsilon*sigma;
        return decode(z);
    }

    void backward(const Tensor &x)
    {

    }

    Tensor& operator()(const Tensor &img)
    {
        Tensor mu;
        Tensor sigma;
        encode(img, mu, sigma);
        Tensor epsilon = Tensor(sigma.shape);
        LinAlg::uniform(epsilon, -1, 1);
        Tensor z = mu + epsilon*sigma;
        return decode(z);
    }


};
#endif // VAE_HPP
