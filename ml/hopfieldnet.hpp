#ifndef HOPFILED_NET_HPP
#define HOPFILED_NET_HPP
#include <vector>
#include "../basic/tensor.hpp"
#include "../basic/linalg.h"


class HopfieldNet
{
private:
    int featureDim;
    Tensor theta;
    Tensor w;
public:
    HopfieldNet(){}
    HopfieldNet(int featureDim_, float theta_)
        :featureDim(featureDim_)
    {
        theta = Tensor(featureDim, 1);
        theta.fill(theta_);
        w = Tensor(featureDim, featureDim);
    }

    HopfieldNet(int featureDim_, const Tensor &theta_)
        :featureDim(featureDim_),theta(theta_)
    {
        w = Tensor(featureDim, featureDim);
    }

    static Tensor sign(const Tensor &x)
    {
        Tensor y(x.shape);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            y[i] = x[i] >= 0 ? 1 : -1;
        }
        return y;
    }

    float energy(const Tensor &x)
    {
        /* e = 0.5*∑w*x^2 + ∑θ*x  */
        return 0.5*LinAlg::dot(w%x, x) + LinAlg::dot(x, theta);
    }

    void fit(const std::vector<Tensor> &x)
    {
        int N = x.size();
        float u = 0;
        for (int i = 0; i < N; i++) {
            u += x[i].sum();
        }
        u /= N*featureDim;
        for (int i = 0; i < N; i++) {
            Tensor xi = x[i] - u;
            w += xi%xi.tr();
        }
        w /= N;
        /* no self connection */
        for (int i = 0; i < featureDim; i++) {
            w(i, i) = 0;
        }
        return;
    }
    /*
        x:[1, -1, 1, -1, ...] --> noised data
        xn --> restore data
    */
    Tensor operator()(const Tensor &x, int maxIterate)
    {
        Tensor xn(x);
        float e = energy(xn);
        for (int i = 0; i < maxIterate; i++) {
            xn = sign(w%xn - theta);
            float ei = energy(xn);
            if (e == ei) {
                break;
            }
            e = ei;
        }
        return xn;
    }
    
};


#endif // HOPFILED_NET_HPP
