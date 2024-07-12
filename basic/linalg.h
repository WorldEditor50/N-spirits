#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H
#include <cmath>
#include <random>
#include "mat.h"
#include "tensor.hpp"

namespace LinAlg {
constexpr static float pi = 3.1415926535897932384626433832795;
constexpr static float pi2 = 2*pi;
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
void lerp(Tensor &x1, const Tensor &x2, float alpha);

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
void gaussian1(Tensor &x, float mu, float sigma);
float normL1(const Tensor &x1, const Tensor &x2);
float normL2(const Tensor &x1, const Tensor &x2);
float normLp(const Tensor &x1, const Tensor &x2, float p);
float normL8(const Tensor &x1, const Tensor &x2);
float dot(const Tensor &x1, const Tensor &x2);
float product(const Tensor &x);
float cosine(const Tensor &x1, const Tensor &x2);
namespace Kernel {
    float rbf(const Tensor &x1, const Tensor &x2, float gamma);
    float laplace(const Tensor &x1, const Tensor &x2, float gamma);
    float tanh(const Tensor &x1, const Tensor &x2, float c1, float c2);
    float polynomial(const Tensor& x1, const Tensor& x2, float d, float p);
}

void mean(const Tensor &x, Tensor &u);
void variance(const Tensor &x, const Tensor &u, Tensor &sigma);
void mean(const std::vector<Tensor> &x, Tensor &u);
void variance(const std::vector<Tensor> &x, const Tensor &u, Tensor &sigma);
Tensor mean(const std::vector<Tensor> &x);
Tensor variance(const std::vector<Tensor> &x, const Tensor &u);
Tensor cov(const Tensor &x);
float gaussian(const Tensor &xi, const Tensor &ui, const Tensor &sigmai);
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
Tensor diagInv(const Tensor &x);
/* inverse */
int invert(const Tensor &x, Tensor &ix);
Tensor inv(const Tensor &x);
/* eigen */
float eigen(const Tensor &x, Tensor &vec, int maxIterateCount=1000, float eps=1e-8);
int eigen(const Tensor &x, Tensor &vec, Tensor &value, int maxIterateCount=1000, float eps=1e-8);
/* xTAx */
void xTAx(Tensor &y, const Tensor &x, const Tensor &a);
void USVT(Tensor &y, const Tensor &u, const Tensor &s, const Tensor &v);
Tensor USVT(const Tensor &u, const Tensor &s, const Tensor &v);
/* gaussian elimination */
namespace GaussianElimination {
    void solve(const Tensor &a, Tensor &u);
    void evaluate(const Tensor &u, Tensor &x);
}
int gaussSeidel(const Tensor &a, const Tensor &b, Tensor &x, int iteration, float eps=1e-4);
/* det */
float det(const Tensor &x);
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
    int solve(const Tensor &x, Tensor &u, Tensor &s, Tensor &v, float eps=1e-7, std::size_t maxEpoch=100);
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
