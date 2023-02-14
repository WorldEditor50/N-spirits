#ifndef FFT_H
#define FFT_H
#include <vector>
#include "complexnumber.h"

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
    inline int fft1d(std::vector<Complex> &x, int N, int opt)
    {
        std::vector<Complex> xa(N);
        for (int i = 0; i < N; i++) {
            int index = reverse(i, std::log2(N));
            xa[i] = x[index];
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
                    Complex c = xa[n1]*w[N/h*i];
                    xa[n2] = xa[n1] - c;
                    xa[n1] = xa[n1] + c;
                }
            }
        }
        x = xa;
        if (opt == -1) {
            for (std::size_t i = 0; i < x.size(); i++) {
                x[i] = xa[i]/N;
            }
        } else {
            for (std::size_t i = 0; i < x.size(); i++) {
                x[i] = xa[i];
            }
        }
        return 0;
    }

}


#endif // FFT_H
