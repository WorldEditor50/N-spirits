#ifndef FFT_H
#define FFT_H
#include <vector>
#include "tensor.h"
#include "complexnumber.h"

using Tensorc = Tensor_<Complex>;

namespace FFT {

    inline int reverse(int index, int n)
    {
        int r = 0;
        for (int i = 0; i < n; i++) {
            if (index&(1 << i)) {
                r |= (1 << (n - i - 1));
            }
        }
        return index;
    }
    inline void transform1D(Tensorc &c, int opt)
    {
        int N = c.totalsize;
        Tensorc cr(N);
        for (int i = 0; i < N; i++) {
            int index = reverse(i, std::log2(N));
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
            for (std::size_t i = 0; i < c.totalsize; i++) {
                c[i] = cr[i]/N;
            }
        } else {
            c = cr;
        }
        return;
    }

    inline void transform2D(Tensorc &src, Tensorc &dst, int opt)
    {
        for (int i = 0; i < src.shape[0]; i++) {
            Tensorc row(1, src.shape[1]);
            Tensorc::MatOp::row(src, i, row);
            transform1D(row, opt);
            Tensorc::MatOp::setRow(dst, i, row);
        }

        for (int i = 0; i < src.shape[1]; i++) {
            Tensorc column(src.shape[0], 1);
            Tensorc::MatOp::row(src, i, column);
            transform1D(column, opt);
            Tensorc::MatOp::setColumn(dst, i, column);
        }
        return;
    }
}


#endif // FFT_H
