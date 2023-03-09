#ifndef KMEANS_H
#define KMEANS_H
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <random>
#include "../basic/mat.h"
#include "../basic/statistics.h"

class KMeans
{
public:
    std::size_t topicDim;
    std::size_t featureDim;
    std::vector<Mat> centers;
public:
    KMeans(){}
    explicit KMeans(std::size_t k):topicDim(k),featureDim(0){}

    void cluster(const std::vector<Mat> &x, std::size_t maxEpoch, float eps=1e-6)
    {
        /* init center */
        featureDim = x[0].cols;
        std::uniform_int_distribution<int> distribution(0, x.size() - 1);
        std::default_random_engine engine;
        centers = std::vector<Mat>(topicDim);
        for (std::size_t i = 0; i < centers.size(); i++) {
            int j = distribution(engine);
            centers[i] = x[j];
        }
        /* cluster */
        std::vector<std::vector<std::size_t> > groups(topicDim);
        std::vector<Mat> centers_(topicDim, Mat(1, featureDim));
        for (std::size_t epoch = 0; epoch < maxEpoch; epoch++) {
            /* pick the nearest topic */
            for (std::size_t i = 0; i < x.size(); i++) {
                float maxD = -1;
                std::size_t topic = 0;
                for (std::size_t j = 0; j < centers.size(); j++) {
                    float d = Statistics::Norm::l2(x[i], centers[j]);
                    if (d > maxD) {
                        maxD = d;
                        topic = j;
                    }
                }
                groups[topic].push_back(i);
            }
            /* calculate center */
            for (std::size_t i = 0; i < groups.size(); i++) {
                centers_[i].zero();
                for (std::size_t j = 0; j < groups[i].size(); j++) {
                    std::size_t h = groups[i][j];
                    centers_[i] += x[h];
                }
                centers_[i] /= float(groups[i].size());
            }
            /* error */
            float delta = 0;
            for (std::size_t i = 0; i < centers.size(); i++) {
                delta += Statistics::Norm::l2(centers[i], centers_[i]);
            }
            delta /= float(topicDim);
            if (delta < eps && epoch > maxEpoch/4) {
                /* finished */
                break;
            }
            if (epoch % (maxEpoch/10) == 0) {
                std::cout<<"error:"<<delta<<std::endl;
            }
            /* update center */
            centers = centers_;

            /* clear */
            for (std::size_t i = 0; i < centers.size(); i++) {
                groups[i].clear();
            }
        }
        return;
    }

    void operator()(const std::vector<Mat> &x, std::vector<std::size_t> &y)
    {
        y = std::vector<std::size_t>(topicDim);
        for (std::size_t i = 0; i < x.size(); i++) {
            float maxD = -1;
            std::size_t topic = 0;
            for (std::size_t j = 0; j < centers.size(); j++) {
                float d = Statistics::Norm::l2(x[i], centers[j]);
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
            float d = Statistics::Norm::l2(x, centers[i]);
            if (d > maxD) {
                maxD = d;
                topic = i;
            }
        }
        return topic;
    }

};



#endif // KMEANS_H
