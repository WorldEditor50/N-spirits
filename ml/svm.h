#ifndef SVM_H
#define SVM_H
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "../basic/util.hpp"
#include "../basic/tensor.hpp"


namespace kernel {

struct RBF {
    static float f(const Tensor& x1, const Tensor& x2)
    {
        float sigma = 1;
        float xL2 = util::dot(x1, x1) + util::dot(x2, x2) - 2*util::dot(x1, x2);
        xL2 = xL2/(-2*sigma*sigma);
        return std::exp(xL2);
    }
};
struct Laplace {
    static float f(const Tensor& x1, const Tensor& x2)
    {
        float sigma = 1;
        float xL2 = util::dot(x1, x1) + util::dot(x2, x2) - 2*util::dot(x1, x2);
        xL2 = -sqrt(xL2)/sigma;
        return exp(xL2);
    }
};

struct Sigmoid {
    static float f(const Tensor& x1, const Tensor& x2)
    {
        float beta1 = 1;
        float theta = -1;
        return std::tanh(beta1 * util::dot(x1, x2) + theta);
    }
};

struct Polynomial {
    static float f(const Tensor& x1, const Tensor& x2)
    {
        float d = 1.0;
        float p = 100;
        return pow(util::dot(x1, x2) + d, p);
    }
};

struct Linear {
    static float f(const Tensor& x1, const Tensor& x2)
    {
        return util::dot(x1, x2);
    }
};

}

template<typename Kernel>
class SVM
{
public:
    struct Vector {
        float alpha;
        float y;
        Tensor x;
    };
    float b;
    float c;
    float tolerance;
    std::vector<Vector> vectors;
public:
    SVM():c(1),b(0),tolerance(1e-3){}

    int operator()(const Tensor& xi)
    {
        int label = 0.0;
        float s = 0.0;
        for (std::size_t j = 0; j < vectors.size(); j++) {
            s += vectors[j].alpha * vectors[j].y * Kernel::f(vectors[j].x, xi);
        }
        s += b;
        /* f(x) = sign(sum(alpha_j * yj * K(xj, x))) */
        if (s >= 0) {
            label = 1.0;
        } else {
            label = -1.0;
        }
        return label;
    }

    void SMO(const std::vector<Tensor> &x, const Tensor &y, int maxEpoch)
    {
        int k = 0;
        Tensor alpha(x.size(), 0);
        while (k < maxEpoch) {
            bool alphaOptimized = false;
            for (std::size_t i = 0; i < x.size(); i++) {
                float Ei = g(alpha, x, x[i], y) - y[i];
                float alpha_i = alpha[i];
                /* KKT */
                if (KKT(y[i], Ei, alpha[i]) == false) {
                    continue;
                }
                int j = random(x.size(), i);
                float Ej = g(alpha, x, x[j], y) - y[j];
                float alpha_j = alpha[j];
                /* optimize alpha[j] */
                float L = 0;
                float H = 0;
                if (y[i] != y[j]) {
                    L = std::max(0.0f, alpha[j] - alpha[i]);
                    H = std::min(c, c + alpha[j] - alpha[i]);
                } else {
                    L = std::max(0.0f, alpha[j] + alpha[i] - c);
                    H = std::min(c, alpha[j] + alpha[i]);
                }
                if (L == H) {
                    continue;
                }
                float Kii = Kernel::f(x[i], x[i]);
                float Kjj = Kernel::f(x[j], x[j]);
                float Kij = Kernel::f(x[i], x[j]);
                float eta = Kii + Kjj - 2 * Kij;
                if (eta <= 0) {
                    continue;
                }
                alpha[j] += y[j]*(Ei - Ej)/eta;
                if (alpha[j] > H) {
                    alpha[j] = H;
                } else if (alpha[j] < L) {
                    alpha[j] = L;
                }
                if (std::abs(alpha[j] - alpha_j) < tolerance) {
                    continue;
                }
                /* optimize alpha[i] */
                alpha[i] += y[i]*y[j]*(alpha_j - alpha[j]);
                /* update b */
                float b1 = b - Ei - y[i]*Kii*(alpha[i] - alpha_i) - y[j]*Kij*(alpha[j] - alpha_j);
                float b2 = b - Ej - y[i]*Kij*(alpha[i] - alpha_i) - y[j]*Kjj*(alpha[j] - alpha_j);
                if (alpha[i] > 0 && alpha[i] < c) {
                    b = b1;
                } else if (alpha[j] > 0 && alpha[j] < c) {
                    b = b2;
                } else {
                    b = (b1 + b2)/2;
                }
                alphaOptimized = true;
            }

            if (alphaOptimized == false) {
                k++;
            } else {
                k = 0;
            }
        }
        /* save support vectors */
        for (std::size_t i = 0; i < alpha.size(); i++) {
            if (alpha[i] > 0) {
                Vector v;
                v.alpha = alpha[i];
                v.y = y[i];
                v.x = x[i];
                vectors.push_back(v);
            }
        }
        return;
    }

    bool KKT(float yi, float Ei, float alpha_i)
    {
        return ((yi*Ei < -tolerance) && (alpha_i < c)) || ((yi*Ei > tolerance) && (alpha_i > 0));
    }

    float g(const Tensor &alpha, const std::vector<Tensor> &x,
            const Tensor& xi, const Tensor &y)
    {
        float s = 0.0;
        for (std::size_t j = 0; j < x.size(); j++) {
            s += alpha[j]*y[j]*Kernel::f(x[j], xi);
        }
        s += b;
        return s;
    }

    static int random(std::size_t s, int i)
    {
        int j = 0;
        j = rand() % s;
        while (j == i) {
            j = rand() % s;
        }
        return j;
    }

};
#endif // SVM_H
