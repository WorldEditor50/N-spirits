#ifndef SVM_H
#define SVM_H
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include "../basic/linalg.h"
#include "../basic/tensor.hpp"

class SVM
{
public:
    using FnKernel = std::function<float(const Tensor &x1, const Tensor &x2)>;
    struct Vector {
        float alpha;
        float y;
        Tensor x;
    };
protected:
    float b;
    float c;
    float tolerance;
    FnKernel kernel;
    std::vector<Vector> vectors;
protected:
    bool KKT(float yi, float Ei, float alpha_i)
    {
        return ((yi*Ei < -tolerance) &&
                (alpha_i < c)) || ((yi*Ei > tolerance) &&
                                   (alpha_i > 0));
    }

    float g(const Tensor &alpha, const std::vector<Tensor> &x,
            const Tensor& xi, const Tensor &y)
    {
        float s = 0.0;
        for (std::size_t j = 0; j < x.size(); j++) {
            s += alpha[j]*y[j]*kernel(x[j], xi);
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

public:
    SVM():c(1),b(0),tolerance(1e-3){}
    explicit SVM(const FnKernel &func, float tolerance_, float c_=1)
        :c(c_),b(0),tolerance(tolerance_),kernel(func)
    {

    }

    void fit(const std::vector<Tensor> &x, const Tensor &y, int maxEpoch)
    {
        /* smo */
        int k = 0;
        Tensor alpha(x.size(), 1);
        while (k < maxEpoch) {
            bool alphaOptimized = false;
            for (std::size_t i = 0; i < x.size(); i++) {
                float Ei = g(alpha, x, x[i], y) - y[i];
                float alphai = alpha[i];
                /* KKT */
                if (KKT(y[i], Ei, alpha[i]) == false) {
                    continue;
                }
                int j = random(x.size(), i);
                float Ej = g(alpha, x, x[j], y) - y[j];
                float alphaj = alpha[j];
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
                float Kii = kernel(x[i], x[i]);
                float Kjj = kernel(x[j], x[j]);
                float Kij = kernel(x[i], x[j]);
                float eta = Kii + Kjj - 2*Kij;
                if (eta <= 0) {
                    continue;
                }
                alpha[j] += y[j]*(Ei - Ej)/eta;
                if (alpha[j] > H) {
                    alpha[j] = H;
                } else if (alpha[j] < L) {
                    alpha[j] = L;
                }
                if (std::abs(alpha[j] - alphaj) < tolerance) {
                    continue;
                }
                /* optimize alpha[i] */
                alpha[i] += y[i]*y[j]*(alphaj - alpha[j]);
                /* update b */
                float b1 = b - Ei - y[i]*Kii*(alpha[i] - alphai) - y[j]*Kij*(alpha[j] - alphaj);
                float b2 = b - Ej - y[i]*Kij*(alpha[i] - alphai) - y[j]*Kjj*(alpha[j] - alphaj);
                if (alpha[i] > 0 && alpha[i] < c) {
                    b = b1;
                } else if (alpha[j] > 0 && alpha[j] < c) {
                    b = b2;
                } else {
                    b = (b1 + b2)/2;
                }
                alphaOptimized = true;
            }

            if (!alphaOptimized) {
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

    float operator()(const Tensor& xi)
    {
        /* f(x) = sign(sum(alpha_i * yi * K(xi, x))) */
        float s = 0.0;
        for (std::size_t i = 0; i < vectors.size(); i++) {
            s += vectors[i].alpha * vectors[i].y * kernel(vectors[i].x, xi);
        }
        s += b;
        return s;
    }
};
#endif // SVM_H
