#ifndef HMM_H
#define HMM_H
#include "vec.h"

class HMM
{
public:
    std::size_t hiddenDim;
    std::size_t observeDim;
    /* input parameter */
    std::vector<Vec> A;
    std::vector<Vec> B;
    Vec p;
public:
    HMM(){}
    explicit HMM(std::size_t hiddenDim_, std::size_t observeDim_)
        :hiddenDim(hiddenDim_),observeDim(observeDim_)
    {
        p = Vec(hiddenDim, 0);
        A = std::vector<Vec>(hiddenDim, Vec(hiddenDim, 0));
        B = std::vector<Vec>(hiddenDim, Vec(observeDim, 0));

    }
    float forward(const Vec &x, std::vector<Vec> &alpha)
    {
        /* calculate P(O|A,B,p) */
        //std::vector<Vec> alpha(x.size(), Vec(hiddenDim, 0));
        for (std::size_t i = 0; i < alpha[0].size(); i++) {
            std::size_t k = x[0];
            alpha[0][i] = p[i]*B[i][k];
        }
        for (std::size_t t = 0; t < alpha.size(); t++) {
            for (std::size_t i = 0; i < alpha[0].size(); i++) {
                float a = 0;
                for (std::size_t j = 0; j < A.size(); j++) {
                    a += alpha[t - 1][j]*A[j][i];
                }
                std::size_t k = x[t - 1];
                alpha[t][i] = a*B[t][k];
            }
        }
        float pr = 0;
        std::size_t T = x.size() - 1;
        for (std::size_t i = 0; i < alpha[0].size(); i++) {
            pr += alpha[T][i];
        }
        return pr;
    }

    float backward(const Vec &x, std::vector<Vec> &beta)
    {
        //std::vector<Vec> beta(x.size(), Vec(hiddenDim, 0));
        std::size_t T = x.size() - 1;
        for (std::size_t i = 0; i < beta[0].size(); i++) {
            beta[T][i] = 1;
        }
        for (std::size_t t = T - 1; t >= 0; t--) {
            for (std::size_t i = 0; i < beta[0].size(); i++) {
                float b = 0;
                std::size_t k = x[t - 1];
                for (std::size_t j = 0; j < A.size(); j++) {
                    b += beta[t + 1][j]*A[i][j]*B[j][k];
                }
                beta[t][i] = b;
            }
        }
        float pr = 0;
        std::size_t k = x[0];
        for (std::size_t i = 0; i < beta[0].size(); i++) {
            pr += p[i]*B[i][k]*beta[1][i];
        }
        return pr;
    }

    void baumWelch(const Vec &x, std::size_t maxEpoch)
    {
        /* estimate model parameters (A, B, p) */
        std::vector<Vec> alpha(x.size(), Vec(hiddenDim, 0));
        std::vector<Vec> beta(x.size(), Vec(hiddenDim, 0));
        std::vector<Vec> gamma(x.size(), Vec(hiddenDim, 0));
        for (std::size_t epoch = 0; epoch < maxEpoch; epoch++) {
            forward(x, alpha);
            backward(x, beta);
            for (std::size_t i = 0; i < gamma.size(); i++) {
                Vec::multi(gamma[i], alpha[i], beta[i]);
            }
            for (std::size_t i = 0; i < A.size(); i++) {
                for (std::size_t j = 0; j < A[0].size(); j++) {
                    //A[i][j];
                }
            }

        }
        return;
    }
    void viterbi(const Vec &x)
    {
        /* estimate state sequence from observed seuqence */

        return;
    }

    void decode()
    {

        return;
    }

};

#endif // HMM_H
