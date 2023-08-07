#ifndef VAE_HPP
#define VAE_HPP
#include <tuple>
#include "activate.hpp"
#include "layer.hpp"
#include "../basic/tensor.hpp"
#include "../basic/util.hpp"

class VAE
{  
public:
    /* encoder */
    FcLayer fromImg;
    FcLayer map2Mu;
    FcLayer map2Sigma;
    /* decoder */
    FcLayer fromZ;
    FcLayer map2Img;
public:
    explicit VAE(int imgSize, int hiddenDim, int zDim)
    {
        /* encoder */
        fromImg = FcLayer(imgSize, hiddenDim, false, ACTIVE_LEAKRELU);
        map2Mu = FcLayer(hiddenDim, zDim, false, ACTIVE_LINEAR);
        map2Sigma = FcLayer(hiddenDim, zDim, false, ACTIVE_LINEAR);
        /* decoder */
        fromZ = FcLayer(zDim, hiddenDim, false, ACTIVE_LEAKRELU);
        map2Img = FcLayer(hiddenDim, imgSize, false, ACTIVE_SIGMOID);
    }
    void encoder(const Tensor &img, Tensor &mu, Tensor &sigma)
    {
        Tensor& h = fromImg.forward(img);
        mu = map2Mu.forward(h);
        sigma = map2Sigma.forward(h);
        return;
    }

    Tensor& decoder(const Tensor &z)
    {
        Tensor& h = fromZ.forward(z);
        return map2Img.forward(h);
    }

    std::tuple<Tensor, Tensor, Tensor> forward(const Tensor &img)
    {
        Tensor mu;
        Tensor sigma;
        encoder(img, mu, sigma);
        Tensor epsilon = Tensor(sigma.shape);
        util::uniform(epsilon, -1, 1);
        Tensor z = mu + epsilon;
        Tensor img_ = decoder(z);
        return std::tuple<Tensor, Tensor, Tensor>(img_, mu, sigma);
    }

    Tensor& operator()(const Tensor &img)
    {
        Tensor mu;
        Tensor sigma;
        encoder(img, mu, sigma);
        Tensor epsilon = Tensor(sigma.shape);
        util::uniform(epsilon, -1, 1);
        Tensor z = mu + epsilon;
        return decoder(z);
    }

};
#endif // VAE_HPP
