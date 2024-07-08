#ifndef MEANSHIFT_HPP
#define MEANSHIFT_HPP
#include <cmath>
#include <random>
#include <functional>
#include "../basic/tensor.hpp"
#include "../basic/linalg.h"

class MeanShift
{
public:
    using FnKernel = std::function<float(const Tensor &x1, const Tensor &x2)>;
public:
    float h;
    FnKernel kernel;
    std::size_t topicDim;
    std::size_t featureDim;
    std::vector<Tensor> x;
public:
    MeanShift(){}
    explicit MeanShift(std::size_t k, std::size_t featureDim_)
        :h(1),topicDim(k),featureDim(featureDim_)
    {
        kernel = [=](const Tensor &x1, const Tensor &x2)->float {
            return LinAlg::Kernel::rbf(x1, x2, 1.0/h);
        };
    }
    void cluster(const std::vector<Tensor> &xi, std::size_t maxEpoch, float eps=1e-6)
    {
        x = xi;
        std::vector<Tensor> mean(x.size(), Tensor(featureDim, 1));
        for (std::size_t epoch = 0; epoch < maxEpoch; epoch++) {
            /* mean */
            for (std::size_t i = 0; i < x.size(); i++) {
                float s = 0;
                for (std::size_t j = 0; j < x.size(); j++) {
                    if (i == j) {
                        continue;
                    }
                    s += kernel(x[i], x[j]);
                }
                Tensor xs(featureDim, 1);
                for (std::size_t j = 0; j < x.size(); j++) {
                    if (i == j) {
                        continue;
                    }
                    xs += x[i]*kernel(x[i], x[j]);
                }
                mean[i] = xs/s - x[i];
            }

            /* shift */
            for (std::size_t i = 0; i < x.size(); i++) {
                x[i] += mean[i];
            }
            /* error */
            float delta = 0;
            for (std::size_t i = 0; i < topicDim; i++) {
                delta += mean[i].norm2();
            }
            delta /= float(topicDim);
            std::cout<<"epoch:"<<epoch<<", error:"<<delta<<std::endl;
            if (delta < eps && epoch > maxEpoch/2) {
                /* finished */
                std::cout<<"finished"<<std::endl;
                break;
            }
        }
        return;
    }

    std::size_t operator()(const Tensor &xi)
    {
        float maxD = -1;
        std::size_t topic = 0;
        for (std::size_t i = 0; i < x.size(); i++) {
            float d = kernel(x[i], xi);
            if (d > maxD) {
                maxD = d;
                topic = i;
            }
        }
        return topic;
    }
};

#endif // MEANSHIFT_HPP
