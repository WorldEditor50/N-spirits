#ifndef LOGISTIC_H
#define LOGISTIC_H
#include "../basic/tensor.hpp"
#include "../basic/linalg.h"

class Logistic
{
public:
    std::size_t maxEpoch;
    std::size_t batchSize;
    float learningRate;
    Tensor w;
    float b;
public:
    Logistic(){}
    explicit Logistic(int topicDim,
                      int featureDim,
                      std::size_t maxEpoch_,
                      std::size_t batchSize_,
                      float learningRate_)
        :maxEpoch(maxEpoch_),batchSize(batchSize_),learningRate(learningRate_)
    {
        w = Tensor(featureDim, 1);
    }
    static float sigmoid(float x)
    {
        return 1/(1 + std::exp(-x));
    }
    static float dSigmoid(float y)
    {
        return y*(1 - y);
    }
    float operator()(const Tensor &x)
    {
        return sigmoid(LinAlg::dot(w, x) + b);
    }
    void update(const Tensor &x, float y, float yt, float learningRate)
    {
        /* sgd */
        for (std::size_t i = 0; i < w.totalSize; i++) {
            w[i] -= learningRate*(y - yt)*dSigmoid(y)*x[i];
        }
        b -= learningRate*(y - yt)*dSigmoid(y);
        return;
    }
    void fit(const std::vector<Tensor> &x, const Tensor &yt, const Tensor &sampleWeight)
    {
        for (std::size_t i = 0; i < maxEpoch; i++) {
            for (std::size_t j = 0; j < batchSize; j++) {
                int k = rand() % x.size();
                float y = sigmoid(LinAlg::dot(w, x[k]) + b);
                update(x[k], y, yt[k], learningRate);
            }
        }
        return;
    }
};

#endif // LOGISTIC_H
