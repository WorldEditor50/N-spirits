#ifndef SVM_H
#define SVM_H
#include <iostream>
#include <cmath>
#include "../basic/vec.h"
#include "../basic/kernel.h"

template<typename KernelF>
class SVM
{
public:
    struct Vector {
        float alpha;
        float y;
        Vec x;
    };
    float b;
    float c;
    float tolerance;
    std::vector<Vector> vectors;
public:
    SVM():c(1),b(0),tolerance(1e-3){}

    int predict(const Vec& xi)
    {
        int label = 0.0;
        float sum = 0.0;
        for (int j = 0; j < vectors.size(); j++) {
            sum += vectors[j].alpha*vectors[j].y*KernelF::f(vectors[j].x, xi);
        }
        sum += b;
        /* f(x) = sign(sum(alpha_j * yj * K(xj, x))) */
        if (sum >= 0) {
            label = 1.0;
        } else {
            label = -1.0;
        }
        return label;
    }

    void SMO(const std::vector<Vec> &x, const Vec &y, int maxEpoch)
    {
        int k = 0;
        Vec alpha(x.size(), 0);
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
                float Kii = KernelF::f(x[i], x[i]);
                float Kjj = KernelF::f(x[j], x[j]);
                float Kij = KernelF::f(x[i], x[j]);
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

    float g(const Vec &alpha, const std::vector<Vec> &x, const Vec& xi, const Vec &y)
    {
        float sum = 0.0;
        for (std::size_t j = 0; j < x.size(); j++) {
            sum += alpha[j]*y[j]*KernelF::f(x[j], xi);
        }
        sum += b;
        return sum;
    }

    int random(std::size_t s, int i)
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
