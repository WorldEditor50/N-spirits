#ifndef GMM_H
#define GMM_H
#include "kmeans.h"
#include "../basic/linalg.h"

class GMM
{
public:
    class Gaussian
    {
    public:
        Tensor u;
        Tensor sigma;
    public:
        Gaussian(){}
        explicit Gaussian(int featureDim)
        {
            u = Tensor(featureDim, 1);
            sigma = Tensor(featureDim, 1);
        }
        Gaussian(const Gaussian &r)
            :u(r.u),sigma(r.sigma){}
        float operator()(const Tensor &x)
        {
            /* N(x;u,sigma) = exp((x-u)^2/sigma)/sqrt(2*pi*sigma) */
            float p = 1;
            for (std::size_t i = 0; i < u.totalSize; i++) {
                p *= std::exp(-0.5*(x[i] - u[i])*(x[i] - u[i])/sigma[i])/std::sqrt(2*3.14159*sigma[i]);
            }
            return p;
        }

        void zero()
        {
            u.zero();
            sigma.zero();
            return;
        }
    };
public:
    std::size_t componentDim;
    std::size_t featureDim;
    Tensor priors;
    Tensor minSigma;
    std::vector<Gaussian> gaussians;
public:
    GMM():componentDim(0), featureDim(0){}
    explicit GMM(std::size_t k, std::size_t featureDim_)
        :componentDim(k), featureDim(featureDim_)
    {
        priors = Tensor(componentDim, 1);
        minSigma = Tensor(componentDim, 1);
        gaussians = std::vector<Gaussian>(componentDim, Gaussian(featureDim));
    }
    void init(const std::vector<Tensor> &x, std::size_t maxEpoch)
    {
        /* run kmeans to select centers */
        Kmeans model(componentDim, featureDim);
        model.cluster(x, maxEpoch);
        std::vector<std::size_t> y;
        model(x, y);
        /* mean of all data */
        Tensor u(componentDim, 1);
        for (std::size_t i = 0; i < x.size(); i++) {
            u += x[i];
        }
        u /= float(x.size());
        /* variance of all data */
        for (std::size_t i = 0; i < x.size(); i++) {
            std::size_t topic = y[i];
            minSigma[topic] += LinAlg::dot(x[i], x[i]);
        }
        for (std::size_t i = 0; i < u.totalSize; i++) {
            minSigma[i] = std::max(1e-10, 0.01*(minSigma[i]/float(x.size()) - u[i]*u[i]));
        }
        /* prior of each topic */
        Tensor N(componentDim, 1);
        for (std::size_t i = 0; i < x.size(); i++) {
            std::size_t topic = y[i];
            N[topic]++;
        }
        for (std::size_t i = 0; i < priors.totalSize; i++) {
            priors[i] = float(N[i])/float(x.size());
        }
        /* mean of each topic */
        for (std::size_t i = 0; i < model.centers.size(); i++) {
            gaussians[i].u = model.centers[i];
        }
        /* variance of each topic */
        for (std::size_t i = 0; i < x.size(); i++) {
            std::size_t topic = y[i];
            Tensor& uc = model.centers[topic];
            for (std::size_t j = 0; j < gaussians[topic].sigma.totalSize; j++) {
                gaussians[topic].sigma[j] += (x[i][j] - uc[j])*(x[i][j] - uc[j]);
            }
        }
        /* constraint */
        for (std::size_t i = 0; i < gaussians.size(); i++) {
            if (priors[i] > 0) {
                for (std::size_t j = 0; j < gaussians[i].sigma.totalSize; j++) {
                    gaussians[i].sigma[j] /= N[i];
                    if (gaussians[i].sigma[j] < minSigma[i]) {
                        gaussians[i].sigma[j] = minSigma[i];
                    }
                }
            } else {
                gaussians[i].sigma = minSigma;
            }
        }
        return;
    }

    void cluster(const std::vector<Tensor> &x, std::size_t maxEpoch, float eps)
    {
        /* init */
        init(x, maxEpoch);
        /* estiTensore */
        Tensor priors_(componentDim, 1);
        std::vector<Gaussian> s(x.size(), Gaussian(featureDim));
        for (std::size_t epoch = 0; epoch < maxEpoch; epoch++) {
            for (std::size_t i = 0; i < s.size(); i++) {
                s[i].zero();
                priors_.zero();
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
                    float gamma = priors[i]*gaussians[j](x[i])/sp;
                    priors_[j] += gamma;
                    /* M-Step */
                    for (std::size_t k = 0; k < gaussians[j].u.totalSize; k++) {
                        s[j].u[k] += gamma*x[i][k];
                        s[j].sigma[k] += gamma*x[i][k]*x[i][k];
                    }
                }
            }
            /* update */
            for (std::size_t i = 0; i < gaussians.size(); i++) {
                /* M-Step */
                priors[i] = priors_[i]/float(x.size());
                if (priors[i] <= 0) {
                    continue;
                }
                Tensor& u = gaussians[i].u;
                for (std::size_t j = 0; j < u.totalSize; j++) {
                    u[j] = s[i].u[j]/priors_[i];
                    /* Cov(X，Y)=E(XY)-E(X)E(Y) */
                    gaussians[i].sigma[j] = s[i].sigma[j]/priors_[i] - u[j]*u[j];
                    if (gaussians[i].sigma[j] < minSigma[i]) {
                        gaussians[i].sigma[j] = minSigma[i];
                    }
                }
            }
        }
        return;
    }

    std::size_t operator()(const Tensor &x)
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
