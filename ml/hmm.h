#ifndef HMM_H
#define HMM_H
#include <vector>
#include "../basic/tensor.hpp"
#include "../basic/linalg.h"

class HMM
{
public:
    std::size_t N;
    std::size_t M;
    /* state transition probability */
    Tensor A;
    /* emission probability */
    Tensor B;
    /* initial state probability */
    Tensor Pi;
protected:
    float forward(const Tensor &O, Tensor &alpha)
    {
        int T = O.size();

        for (int i = 0; i < N; i++) {
            alpha(0, i) = Pi[i]*B(i, O[0]);
        }

        for (int t = 0; t < T - 1; t++) {
            for (int i = 0; i < N; i++) {
                float s = 0;
                for (int j = 0; j < N; j++) {
                    s += alpha(t, j)*A(j, i);
                }
                alpha(t + 1, i) = s*B(i, O[t + 1]);
            }
        }

        float p = 0.0;
        for (int i = 0; i < N; i++) {
            p += alpha(T-1, i);
        }
        return p;
    }

    float backward(const Tensor &O, Tensor &beta)
    {
        int T = O.size();
        beta = Tensor(T, N);
        for (int i = 0; i < N; i++) {
            beta(T-1, i) = 1.0;
        }

        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < N; i++) {
                float s = 0;
                for (int j = 0; j < N; j++) {
                    s += A(i, j)*B(j, O[t + 1])*beta(t + 1, j);
                }
                beta(t, i) = s;
            }
        }

        float p = 0.0;
        for (int i = 0; i < N; i++) {
            p += Pi[i] * B(i, O[0]) * beta(0, i);
        }
        return p;
    }

    Tensor computeGamma(const Tensor &alpha, const Tensor &beta, const Tensor &O)
    {
        int T = O.size();
        Tensor gamma(T, N);
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < N; i++) {
                float s = 0;
                for (int j = 0; j < N; j++) {
                    s += alpha(t,j) * beta(t,j);
                }
                gamma(t, i) = alpha(t, i)*beta(t, i)/s;
            }
        }
        return gamma;
    }

    Tensor computeXi(const Tensor &alpha, const Tensor &beta, const Tensor &O)
    {
        int T = O.size();
        Tensor xi(T-1, N, N);
        for (int t = 0; t < T - 1; t++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float numerator = alpha(t,i) * A[i,j] * B(j, O[t+1]) * beta(t+1,j);
                    float denominator = 0;
                    for (int h = 0; h < N; h++) {
                        for (int k = 0; k < N; k++) {
                            denominator += alpha(t, h)*A(h, k)*B(k, O[t+1])*beta(t+1, k);
                        }
                    }
                    xi(t,i,j) = numerator / denominator;
                }
            }
        }
        return xi;
    }

    void baumWelch(const Tensor &O, float eps)
    {
        // given O list finding lambda model(can derive T form O list)
        // also given N, M,
        int T = O.size();
        Tensor V(M, 1);
        for (int k = 0; k < M; k++) {
            V[k] = k;
        }

        float delta = eps + 1;
        Tensor alpha(T, N);
        Tensor beta(T, N);
        while (delta > eps) {
            float p1 = forward(O, alpha);
            float p2 = backward(O, beta);
            Tensor gamma = computeGamma(alpha, beta, O);
            Tensor xi = computeXi(alpha, beta, O);

            Tensor A_ = A;
            Tensor B_ = B;
            Tensor Pi_ = Pi;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float numerator = 0;
                    float denominator = 0;
                    for (int t = 0; t < T - 1; t++) {
                        numerator += xi(t, i, j);
                        denominator += gamma(t, i);
                    }
                    A(i, j) = numerator/denominator;
                }
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    float numerator = 0;
                    float denominator = 0;
                    for (int t = 0; t < T; t++) {
                        if (O[t] == V[j]) {
                            numerator += gamma(t, i);
                        }
                        denominator += gamma(t, i);
                    }
                    B(i, j) = numerator/denominator;
                }
            }
            for (int i = 0; i < N; i++) {
                Pi[i] = gamma(0, i);
            }

            delta = LinAlg::normL2(A_, A) +
                    LinAlg::normL2(B_, B) +
                    LinAlg::normL2(Pi_, Pi);
        }
        return;
    }

    void scaleForward(const Tensor &O, Tensor &alpha, Tensor &c)
    {
        int T = O.size();
        Tensor alpha_(T, N);
        for (int i = 0; i < T; i++) {
            c[i] = i;
        }
        for (int i = 0; i < T; i++) {
            alpha_(0, i) = Pi[i]*B(i, O[0]);
        }

        c[0] = 1.0 / alpha_.sum(0);

        for (int i = 0; i < T; i++) {
            alpha(0, i) = c[0]*alpha_(0, i);
        }

        for (int t = 0; t < T - 1; t++) {
            for (int i = 0; i < N; i++) {
                float s = 0.0;
                for (int j = 0; j < N; j++) {
                    s += alpha(t, j)*A(j, i);
                    alpha_(t+1, i) = s * B(i, O[t+1]);
                }
            }
            c[t + 1] = 1.0 / alpha_.sum(t + 1);

            for (int i = 0; i < N; i++) {
                alpha(t + 1, i) = c[t + 1] * alpha_(t + 1, i);
            }
        }
        return;
    }

    void scaleBackward(const Tensor &O, const Tensor &c, Tensor &beta)
    {
        int T = O.size();
        Tensor beta_raw(T, N);
        for (int i = 0; i < N; i++) {
            beta_raw(T - 1, i) = 1.0;
            beta(T - 1, i) = c[T - 1] * beta_raw(T - 1, i);
        }
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < N; i++) {
                float s = 0.0;
                for (int j = 0; j < N; j++) {
                    s += A(i, j) * B(j, O[t + 1]) * beta(t + 1, j);
                }
                beta(t,i) = c[t] * s;
            }
        }
        return;
    }

    void scaledBaumWelch(const Tensor &O, float eps)
    {
        int T = O.size();
        Tensor V(M, 1);
        for (int i = 0; i < M; i++) {
            V[i] = i;
        }
        float delta = eps + 1;
        Tensor c(T, 1);
        Tensor alpha(T, N);
        Tensor beta(T, N);
        while (delta > eps) {
            alpha.zero();
            beta.zero();
            scaleForward(O, alpha, c);
            scaleBackward(O, c, beta);

            Tensor A_ = A;
            Tensor B_ = B;
            Tensor Pi_ = Pi;

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float numerator = 0;
                    float denominator = 0;
                    for (int t = 0; t < T - 1; t++) {
                        numerator += alpha(t, i)*A(i, j)*B(j, O[t + 1])*beta(t + 1, j);
                        denominator += alpha(t, i)*beta(t, i)/c[t];
                    }
                    A(i, j) = numerator / denominator;
                }
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    float numerator = 0;
                    float denominator = 0;
                    for (int t = 0; t < T; t++) {
                        if (O[t] == V[j]) {
                            numerator += alpha(t, i)*beta(t, i)/c[t];
                        }
                        denominator += alpha(t, i)*beta(t, i)/c[t];
                    }
                    B(i, j) = numerator / denominator;
                }
            }

            // Pi have no business with c
            float denominator_Pi = 0;
            for (int i = 0; i < N; i++) {
                denominator_Pi += alpha(0, i) * beta(0, i);
            }
            for (int i = 0; i < N; i++) {
                Pi[i] = alpha(0,i) * beta(0,i) / denominator_Pi;
            }
            //# if sum directly, there will be positive and negative offset
            delta = LinAlg::normL2(A_, A) +
                    LinAlg::normL2(B_, B) +
                    LinAlg::normL2(Pi_, Pi);
        }
        return;
    }
public:
    HMM(){}
    explicit HMM(const Tensor &A_,
                 const Tensor &B_,
                 const Tensor &Pi_)
        :A(A_),B(B_),Pi(Pi_)
    {
        N = A.shape[0];
        M = B.shape[1];
    }

    Tensor operator()(const Tensor &O)
    {
        /* viterbi: estimate state sequence from observed seuqence */
        std::size_t T = O.size();
        Tensor I(T, 1);

        Tensor delta(T, N);
        Tensor psi(T, N);

        for (int i = 0; i < N; i++) {
            delta(0, i) = Pi[i] * B(i, O[0]);
            psi(0, i) = 0;
        }

        for (int t = 1; t < T; t++) {
            for (int i = 0; i < N; i++) {
                Tensor da(N, 1);
                for (int j = 0; j < N; j++) {
                    da(j, 0) = delta(t - 1, j)*A(j, i);
                }
                float max_ = da.max();
                delta(t, i) = B(i, O[t]) * max_;
                psi(t, i) = da.argmax();
            }
        }
        Tensor delta_ = delta.sub(T - 1);
        float PT = delta_.max();
        I[T - 1] = delta_.argmax();

        for (int t = T - 2; t >= 0; t--) {
            I[t] = psi(t+1, I[t+1]);
        }
        return I;
    }

    void fit(const std::vector<Tensor> &observeSeq, float eps)
    {
        for (std::size_t i = 0; i < observeSeq.size(); i++) {
            scaledBaumWelch(observeSeq[i], eps);
        }
        return;
    }
};

#endif // HMM_H
