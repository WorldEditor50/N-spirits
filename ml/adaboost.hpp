#ifndef ADABOOST_HPP
#define ADABOOST_HPP
#include <vector>
#include <set>
#include "../basic/tensor.hpp"
#include "../basic/linalg.h"


template<typename Estimator, int N>
class AdaBoost
{
protected:
    std::size_t maxEpoch;
    std::size_t batchSize;
    float learningRate;
    float alpha[N];
    Estimator estimators[N];
public:
    AdaBoost(){}
    explicit AdaBoost(int topicDim,
                      int featureDim,
                      std::size_t maxEpoch_,
                      std::size_t batchSize_,
                      float learningRate_)
        :maxEpoch(maxEpoch_),batchSize(batchSize_),learningRate(learningRate_)
    {
        for (int i = 0; i < N; i++) {
            estimators[i] = Estimator(topicDim,
                                      featureDim,
                                      maxEpoch,
                                      batchSize,
                                      learningRate);
        }
    }
    void fit(const std::vector<Tensor> &x, const Tensor &y)
    {
        /*
            x:(n, featureDim)
        */
        int n = x.size();
        Tensor sampleWeight = Tensor::ones(n, 1);
        sampleWeight /= n;
        for (int i = 0; i < N; i++) {
            estimators[i].fit(x, y, sampleWeight);
            float error = estimators[i].optError;
            alpha[i] = 0.5*std::log((1 - error)/error);
            for (std::size_t j = 0; j < n; j++) {
                float yp = estimators[i](x[j]);
                sampleWeight[j] *= std::exp(-alpha[i]*yp*y[j]);
            }
            sampleWeight /= sampleWeight.sum();
        }
        return;
    }

    float operator()(const Tensor &x)
    {
        /*
            x:(featureDim, 1)
        */
        float s = 0;
        for (int i = 0; i < N; i++) {
            s += estimators[i](x)*alpha[i];
        }
        return 2*(s > 0)- 1;
    }
};
#endif // ADABOOST_HPP
