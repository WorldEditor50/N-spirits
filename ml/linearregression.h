#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include "../basic/mat.h"
#include "../basic/statistics.h"

class LinearModel
{
public:
    Mat w;
    float b;
public:
    LinearModel(){}
    static float sigmoid(float x)
    {
        return 1/(1 + exp(-x));
    }
    static float dSigmoid(float y)
    {
        return y*(1 - y);
    }
    float operator()(const Mat &x)
    {
        return sigmoid(Statistics::dot(w, x) + b);
    }
    void update(const Mat &x, float y, float yt, float learningRate)
    {
        /* sgd */
        for (std::size_t i = 0; i < w.totalSize; i++) {
            w[i] -= learningRate*(y - yt)*dSigmoid(y)*x[i];
        }
        b -= learningRate*(y - yt)*dSigmoid(y);
        return;
    }
    void train(const std::vector<Mat> &x, const Mat &yt, std::size_t maxEpoch, std::size_t batchSize=30, float learningRate=1e-3)
    {
        for (std::size_t i = 0; i < maxEpoch; i++) {
            for (std::size_t j = 0; j < batchSize; j++) {
                int k = rand() % x.size();
                float y = sigmoid(Statistics::dot(w, x[k]) + b);
                update(x[k], y, yt[k], learningRate);
            }
        }
        return;
    }
};

#endif // LINEARREGRESSION_H
