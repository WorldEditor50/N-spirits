#ifndef DECISIONTREE_HPP
#define DECISIONTREE_HPP
#include <vector>
#include <set>
#include "../basic/tensor.hpp"
#include "../basic/linalg.h"

class DecisionTree
{
public:
    float optError;
protected:
    int topicDim;
    int featureDim;
    int optId;
    int optOp;
    float optThres;
public:
    DecisionTree()
        :topicDim(0),featureDim(0){}
    explicit DecisionTree(int topicDim_,
                          int featureDim_,
                          std::size_t maxEpoch_,
                          std::size_t batchSize_,
                          float learningRate_)
        :topicDim(topicDim_),featureDim(featureDim_),
          optError(1),optId(0),optThres(0),optOp(1){}

    void fit(const std::vector<Tensor> &xi, const Tensor &y, const Tensor &sampleWeight)
    {
        /*
            xi:(n, featureDim)
            y:(n, 1)
            weight:(n, 1)
            x:(featureDim, n)
        */
        Tensor xt = Tensor::fromVector(xi).tr();
        std::vector<Tensor> x;
        xt.toVector(x);
        for (std::size_t i = 0; i < featureDim; i++) {
            Tensor feature = x[i];
            std::set<float> featureFilter(feature.begin(), feature.end());
            Tensor uniqueFeature(featureFilter.begin(), featureFilter.end());
            for (std::size_t j = 0; j < uniqueFeature.size() - 1; j++) {
                float thres = (uniqueFeature[j] + uniqueFeature[j + 1])/2;
                for (int op = 0; op < 2; op++) {
                    float error = 0;
                    for (std::size_t k = 0; k < y.totalSize; k++) {
                        float yk = 0;
                        if (op == 1) {
                            yk = 2*(feature[k] >= thres) - 1;
                        } else {
                            yk = 2*(feature[k] < thres) - 1;
                        }
                        error += (yk != y[k])*sampleWeight[k];
                    }
                    if (error < optError) {
                        optError = error;
                        optOp = op;
                        optId = i;
                        optThres = thres;
                    }
                }
            }
        }
        return;
    }

    float operator()(const Tensor &x)
    {
        /*
            x: (featureDim, 1)
        */
        float p = 0;
        if (optOp == 1) {
            p = 2*(x[optId] >= optThres) - 1;
        } else {
            p = 2*(x[optId] < optThres) - 1;
        }
        return p;
    }

};

#endif // DECISIONTREE_HPP
