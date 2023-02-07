#ifndef GMM_H
#define GMM_H
#include "kmeans.h"

class GMM
{
public:
    class Gaussian
    {
    public:
        float prior;
        float minSigma;
        Vec u;
        Vec sigma;
    public:
        Gaussian():prior(0){}
        Gaussian(std::size_t featureDim)
        {
            u = Vec(featureDim, 0);
            sigma = Vec(featureDim, 0);
        }
        float operator()(const Vec &x)
        {
            /* N(x;u,sigma) = exp((x-u)^2/sigma)/sqrt(2*pi*sigma) */
            float p = prior;
            for (std::size_t i = 0; i < u.size(); i++) {
                p *= std::exp(-0.5*(x[i] - u[i])*(x[i] - u[i])/sigma[i])/std::sqrt(2*pi*sigma[i]);
            }
            return p;
        }
        void zero()
        {
            prior = 0;
            u.zero();
            sigma.zero();
            return;
        }
    };
public:
    std::size_t topicDim;
    std::size_t featureDim;
    std::vector<Gaussian> gaussians;
public:
    GMM():topicDim(0), featureDim(0){}
    explicit GMM(std::size_t k, std::size_t featureDim_)
        :topicDim(k), featureDim(featureDim_)
    {
        gaussians = std::vector<Gaussian>(topicDim, Gaussian(featureDim));
    }
    void init(const std::vector<Vec> &x, std::size_t maxEpoch)
    {
        /* run kmeans to select centers */
        KMeans kmeans(topicDim);
        kmeans.cluster(x, maxEpoch);
        std::vector<std::size_t> y;
        kmeans.predict(x, y);
        /* mean of all data */
        Vec u(topicDim, 0);
        for (std::size_t i = 0; i < x.size(); i++) {
            u += x[i];
        }
        u /= float(x.size());
        /* variance of all data */
        for (std::size_t i = 0; i < x.size(); i++) {
            std::size_t topic = y[i];
            gaussians[topic].minSigma += Vec::dot(x[i], x[i]);
        }
        for (std::size_t i = 0; i < u.size(); i++) {
            gaussians[i].minSigma = std::max(1e-10, 0.01*(gaussians[i].minSigma/float(x.size()) - u[i]*u[i]));
        }
        /* prior of each topic */
        Vec N(topicDim, 0);
        for (std::size_t i = 0; i < x.size(); i++) {
            std::size_t topic = y[i];
            N[topic]++;
        }
        for (std::size_t i = 0; i < gaussians.size(); i++) {
            gaussians[i].prior = float(N[i])/float(x.size());
        }
        /* mean of each topic */
        for (std::size_t i = 0; i < kmeans.centers.size(); i++) {
            gaussians[i].u = kmeans.centers[i];
        }
        /* variance of each topic */
        for (std::size_t i = 0; i < x.size(); i++) {
            std::size_t topic = y[i];
            Vec& uc = kmeans.centers[topic];
            for (std::size_t j = 0; j < gaussians[topic].sigma.size(); j++) {
                gaussians[topic].sigma[j] += (x[i][j] - uc[j])*(x[i][j] - uc[j]);
            }
        }
        /* constraint */
        for (std::size_t i = 0; i < gaussians.size(); i++) {
            if (gaussians[i].prior > 0) {
                for (std::size_t j = 0; j < gaussians[i].sigma.size(); j++) {
                    gaussians[i].sigma[j] /= N[i];
                    if (gaussians[i].sigma[j] < gaussians[i].minSigma) {
                        gaussians[i].sigma[j] = gaussians[i].minSigma;
                    }
                }
            } else {
                for (std::size_t j = 0; j < gaussians[i].sigma.size(); j++) {
                    gaussians[i].sigma[j] = gaussians[i].minSigma;
                }
            }
        }
        return;
    }

    void cluster(const std::vector<Vec> &x, std::size_t maxEpoch)
    {
        /* init */
        init(x, maxEpoch);
        /* estimate */
        std::vector<Gaussian> s(x.size(), Gaussian(x[0].size()));
        for (std::size_t epoch = 0; epoch < maxEpoch; epoch++) {
            for (std::size_t i = 0; i < s.size(); i++) {
                s[i].zero();
            }
            for (std::size_t i = 0; i < x.size(); i++) {
                /*
                   E-Step:
                       γ(i, k) = pi_k*N(x_i | u_k, sigma_k)/sum_(j=1)^K pi_j*N(x_i | u_j, sigma_j)
                   M-Step:
                       N_k = Σ γ(i,k)
                       pi_k = N_k/N
                       u_k = Σ γ(i,k)*x_i/N_k
                       sigma_k = Σ γ(i,k)*(x_i - u_k)*(x_i - u_k)/N_k
                */
                /* E-Step */
                float sp = 0;
                for (std::size_t j = 0; j < gaussians.size(); j++) {
                    sp += gaussians[j](x[i]);
                }
                for (std::size_t j = 0; j < s.size(); j++) {
                    /* E-Step */
                    float p = gaussians[j].prior*gaussians[j](x[i])/sp;
                    s[j].prior += p;
                    /* M-Step */
                    for (std::size_t k = 0; k < gaussians[j].u.size(); k++) {
                        s[j].u[k] += p*x[i][k];
                        s[j].sigma[k] += p*x[i][k]*x[i][k];
                    }
                }
            }
            /* update */
            for (std::size_t i = 0; i < gaussians.size(); i++) {
                float& prior = gaussians[i].prior;
                float& minSigma = gaussians[i].minSigma;
                Vec& u = gaussians[i].u;
                Vec& sigma = gaussians[i].sigma;
                /* M-Step */
                prior = s[i].prior/float(x.size());
                if (prior <= 0) {
                    return;
                }
                for (std::size_t j = 0; j < u.size(); j++) {
                    u[j] = s[i].u[j]/s[i].prior;
                    sigma[j] = s[i].sigma[j]/s[i].prior - u[j]*u[j];
                    if (sigma[j] < minSigma) {
                        sigma[j] = minSigma;
                    }
                }
            }
        }
        return;
    }

    std::size_t operator()(const Vec &x)
    {
        float maxP = -1;
        std::size_t topic = 0;
        for (std::size_t i = 0; i < gaussians.size(); i++) {
            float p = gaussians[i](x);
            if (p > maxP) {
                maxP = p;
                topic = i;
            }
        }
        return topic;
    }
};

#endif // GMM_H
