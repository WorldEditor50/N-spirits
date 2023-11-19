#ifndef FFT_H
#define FFT_H
#include <vector>
#include "ctensor.hpp"

namespace DFT {

    inline CTensor transform1D(const CTensor &x)
    {
        /*
            X(k) = Σx(n)*exp(-2Πi*n*k/N)
            exp(i*x) = cos(x) + i*sin(x)
        */
        int N = x.totalSize;
        CTensor X(N);
        for (int k = 0; k < N; k++) {
            for (int n = 0; n < N; n++) {
                Complex w(0, -2*pi*n*k/N);
                X[k] += exp(w)*x[n];
            }
        }
        return X;
    }
    inline CTensor inverse1D(const CTensor &X)
    {
        /*
            x(k) =1/N * Σ X(n)*exp(2Πi*n*k/N)
        */
        int N = X.totalSize;
        CTensor x(N);
        for (int k = 0; k < N; k++) {
            for (int n = 0; n < N; n++) {
                Complex w(0, 2*pi*n*k/N);
                x[k] += exp(w)*X[n];
            }
        }
        for (int i = 0; i < N; i++) {
            x[i] /= N;
        }
        return x;
    }

    inline CTensor transform2D(const CTensor &x)
    {
        /*
            X(u, v) = ΣΣ x(i,j)e^(-2*pi*(u*i/M + v*j/N)
        */
        int M = x.shape[0];
        int N = x.shape[1];
        CTensor X(M, N);
        for (int u = 0; u < M; u++) {
            for (int v = 0; v < N; v++) {
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        Complex w(0, -2*pi*(float(u*i)/M + float(v*j)/N));
                        X(u, v) += exp(w)*x(i, j);
                    }
                }
            }
        }
        return X;
    }

    inline CTensor inverse2D(const CTensor &X)
    {
        /*
            x(i, j) = 1/(M*N) * ΣΣ X(u,v)e^(2*pi*(u*i/M + v*j/N)
        */
        int M = X.shape[0];
        int N = X.shape[1];
        CTensor x(M, N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                for (int u = 0; u < M; u++) {
                    for (int v = 0; v < N; v++) {
                        Complex w(0, 2*pi*(float(u*i)/M + float(v*j)/N));
                        x(i, j) += exp(w)*X(u, v);
                    }
                }
            }
        }

        for (int i = 0; i < x.totalSize; i++) {
            x[i] /= M*N;
        }
        return x;
    }

}

namespace FFT {


    inline void naiveTransform1D(CTensor &x)
    {
        int N = x.totalSize;
        if (N == 1) {
            return;
        }
        CTensor xR(N/2);
        CTensor xL(N/2);
        for (int i = 0; i < N; i+=2) {
            xL[i/2] = x[i];
            xR[i/2] = x[i + 1];
        }
        naiveTransform1D(xL);
        naiveTransform1D(xR);
        /* e^(2*pi*i/N) */
        Complex e = exp(Complex(0, 2*pi/N));
        /* 1 */
        Complex c(1, 0);
        for (int i = 0; i < N/2; i++) {
            /* X(k) = xL(k */
            x[i]       = xL[i] + c*xR[i];
            x[i + N/2] = xL[i] - c*xR[i];
            c *= e;
        }
        return;
    }

    inline int byteReverse(int x, int n)
    {
        int y = 0;
        for (int i = 0; i < n; i++) {
            y <<= 1;
            y |= x&0x01;
            x >>= 1;
        }
        return y;
    }

    inline void transform1D(CTensor &xf, const CTensor &xt)
    {
        int N = xt.totalSize;
        std::vector<Complex> w(N/2);
        std::vector<Complex> x1(N);
        std::vector<Complex> x2(N);

        for (int i = 0; i < N/2; i++) {
            float n = -2*pi*i/N;
            w[i] = Complex(std::cos(n), sin(n));
        }

        x1 = xt.val;
        int r = std::log2(N);
        for (int k = 0; k < r; k++) {
            for (int j = 0; j < 1 << k; j++) {
                int n = 1 << (r - k);
                for (int i = 0; i < n/2; i++) {
                    int h = j*n;
                    x2[i + h] = x1[i + h] + x1[i + h + n/2];
                    x2[i + h + n/2] = (x1[i + h] - x1[i + h + n/2])*w[i*(1<<k)];
                }
            }
            std::swap(x1, x2);
        }

        for (int j = 0; j < N; j++) {
            int k = 0;
            for (int i = 0; i < r; i++) {
                if (j&(1<<i)) {
                    k += 1<<(r - i - 1);
                }
            }
            xf[j] = x1[k];
        }
        return;
    }

    inline void transform1D(CTensor &c, int opt)
    {
        int N = c.totalSize;
        CTensor cr(N);
        int N2 = std::ceil(std::log2(N));
        for (int i = 0; i < N; i++) {
            int index = byteReverse(i, N2);
            cr[i] = c[index];
        }
        std::vector<Complex> w(N/2);
        for (int i = 0; i < N/2; i++) {
            float n = 2*pi*i/N;
            w[i] = Complex(std::cos(n), -opt*std::sin(n));
        }

        for (int h = 2; h < N + 1; h*=2) {
            for (int s = 0; s < N/h; s++) {
                for (int i = 0; i < h/2; i++) {
                    int n1 = i * s * h;
                    int n2 = n1 + h/2;
                    Complex c0 = cr[n1]*w[N/h*i];
                    cr[n2] = cr[n1] - c0;
                    cr[n1] = cr[n1] + c0;
                }
            }
        }
        c = cr;
        if (opt == -1) {
            for (std::size_t i = 0; i < c.totalSize; i++) {
                c[i] = cr[i]/N;
            }
        } else {
            c = cr;
        }
        return;
    }

    inline void transform2D(CTensor &src, CTensor &dst, int opt)
    {
        for (int i = 0; i < src.shape[0]; i++) {
            CTensor r = src.row(i);
            transform1D(r, opt);
            dst.row(i, r);
        }

        for (int i = 0; i < src.shape[1]; i++) {
            CTensor c = src.column(i);
            transform1D(c, opt);
            dst.column(i, c);
        }
        return;
    }
}


#endif // FFT_H
