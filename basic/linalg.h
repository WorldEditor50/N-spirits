#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H
#include <cmath>
#include <random>
#include "mat.h"
#include "tensor.hpp"

namespace LinAlg {

struct Random {
    static std::default_random_engine engine;
    static std::random_device device;
    static std::mt19937 generator;
};
void add(Tensor &y, const Tensor &x1, const Tensor &x2);
void sub(Tensor &y, const Tensor &x1, const Tensor &x2);
void mul(Tensor &y, const Tensor &x1, const Tensor &x2);
void div(Tensor &y, const Tensor &x1, const Tensor &x2);

Tensor abs(const Tensor &x);
Tensor sqrt(const Tensor &x);
Tensor exp(const Tensor &x);
Tensor sin(const Tensor &x);
Tensor cos(const Tensor &x);
Tensor tanh(const Tensor &x);

Tensor lerp(const Tensor &x1, const Tensor &x2, float alpha);

namespace Interplate {
    float lagrange(const Tensor &x, const Tensor &y, float xi, int n);
    float newton(const Tensor &x, const Tensor &y, float xi, int n);
}

template<typename T>
inline void uniform(T &x, float x1, float x2)
{
    std::uniform_real_distribution<float> distibution(x1, x2);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.val[i] = distibution(Random::engine);
    }
    return;
}
void bernoulli(Tensor &x, float p);
void gaussian(Tensor &x, float mu, float sigma);
float normL1(const Tensor &x1, const Tensor &x2);
float normL2(const Tensor &x1, const Tensor &x2);
float normLp(const Tensor &x1, const Tensor &x2, float p);
float normL8(const Tensor &x1, const Tensor &x2);
float dot(const Tensor &x1, const Tensor &x2);
void mean(const Tensor &x, Tensor &u);
void variance(const Tensor &x, const Tensor &u, Tensor &sigma);
void mean(const std::vector<Tensor> &x, Tensor &u);
void variance(const std::vector<Tensor> &x, const Tensor &u, Tensor &sigma);
Tensor mean(const std::vector<Tensor> &x);
Tensor variance(const std::vector<Tensor> &x, const Tensor &u);
void cov(const Tensor &x, Tensor &y);
/* exchange */
void exchangeRow(Tensor &x, int i1, int i2);
void exchangeCol(Tensor &x, int j1, int j2);
/* embedding */
void embeddingRow(Tensor &x, int i, const Tensor &r);
void embeddingCol(Tensor &x, int j, const Tensor &c);
/* indentiy */
Tensor eye(int n);
/* transpose */
Tensor transpose(const Tensor &x);
/* trace */
int trace(const Tensor& x, float &value);
/* diag */
Tensor diag(const Tensor &x);
/* gaussian elimination */
namespace GaussianElimination {
    void solve(const Tensor &a, Tensor &u);
    void evaluate(const Tensor &u, Tensor &x);
}
int gaussSeidel(const Tensor &a, const Tensor &b, Tensor &x, int iteration, float eps=1e-4);
/* det */
int det(const Tensor &x, float &value);
/* rank */
int rank(const Tensor &x);
/* LU */
namespace LU {
    int solve(const Tensor &x, Tensor& l, Tensor &u);
    int inv(const Tensor &x, Tensor& xi);
};
/* QR */
namespace QR {
    int solve(const Tensor &x, Tensor &q, Tensor &r);
    int iterate(const Tensor &x, Tensor &q, Tensor &r);
    int eigen(const Tensor &x, Tensor &, float eps=1e-8);
};
/* svd */
namespace SVD {
    float normalize(Tensor &x, float eps);
    float qrIteration(Tensor &a, const Tensor &q, float eps);
    int solve(const Tensor &x, Tensor &u, Tensor &s, Tensor &v, float eps=1e-7, std::size_t maxEpoch=1e6);
};
/* cholesky */
namespace Cholesky {
    int solve(const Tensor &x, Tensor &l);
}
/* pca */
namespace PCA {
    void solve(const Tensor &x, Tensor &u);
    void project(const Tensor &x, const Tensor &u, int k, Tensor &y);
};


}
#endif // LINEARALGEBRA_H
