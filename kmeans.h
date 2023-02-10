#ifndef KMEANS_H
#define KMEANS_H
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <random>
#include "mat.h"

class KMeans
{
public:
    std::size_t topicDim;
    std::size_t featureDim;
    std::vector<Mat> centers;
public:
    KMeans(){}
    explicit KMeans(std::size_t k):topicDim(k),featureDim(0){}

    void cluster(const std::vector<Mat> &x, std::size_t maxEpoch)
    {
        /* init center */
        std::uniform_int_distribution<int> distribution(0, x.size() - 1);
        std::default_random_engine engine;
        centers = std::vector<Mat>(topicDim);
        for (std::size_t i = 0; i < centers.size(); i++) {
            int j = distribution(engine);
            centers[i] = x[j];
        }
        /* cluster */
        std::vector<std::vector<std::size_t> > groups(topicDim);
        for (std::size_t epoch = 0; epoch < maxEpoch; epoch++) {
            /* pick the nearest topic */
            for (std::size_t i = 0; i < x.size(); i++) {
                float maxD = -1;
                std::size_t topic = 0;
                for (std::size_t j = 0; j < centers.size(); j++) {
                    float d = distance(x[i], centers[j]);
                    if (d > maxD) {
                        maxD = d;
                        topic = j;
                    }
                }
                groups[topic].push_back(i);
            }
            /* calculate center */
            for (std::size_t i = 0; i < groups.size(); i++) {
                centers[i].zero();
                for (std::size_t j = 0; j < groups[i].size(); j++) {
                    std::size_t h = groups[i][j];
                    centers[i] += x[h];
                }
                centers[i] /= float(groups[i].size());
            }
            /* clear */
            for (std::size_t i = 0; i < centers.size(); i++) {
                groups[i].clear();
            }
        }
        return;
    }

    void predict(const std::vector<Mat> &x, std::vector<std::size_t> &y)
    {
        y = std::vector<std::size_t>(topicDim);
        for (std::size_t i = 0; i < x.size(); i++) {
            float maxD = -1;
            std::size_t topic = 0;
            for (std::size_t j = 0; j < centers.size(); j++) {
                float d = distance(x[i], centers[j]);
                if (d > maxD) {
                    maxD = d;
                    topic = j;
                }
            }
            y[i] = topic;
        }
        return;
    }

    std::size_t operator()(const Mat &x)
    {
        float maxD = -1;
        std::size_t topic = 0;
        for (std::size_t i = 0; i < centers.size(); i++) {
            float d = distance(x, centers[i]);
            if (d > maxD) {
                maxD = d;
                topic = i;
            }
        }
        return topic;
    }

    static float distance(const Mat &p1, const Mat &p2)
    {
        float d = 0;
        for (std::size_t i = 0; i < p1.totalSize; i++) {
            d += (p1[i] - p2[i])*(p1[i] - p2[i]);
        }
        return std::sqrt(d);
    }

};



#endif // KMEANS_H
