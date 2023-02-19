#ifndef CONV_H
#define CONV_H
#include "../basic/tensor.h"
#include "activate.h"
#include "layerdef.h"

class Conv2dParam
{
public:
    /* conv */
    int inChannels;
    int outChannels;
    int kernelSize;
    int stride;
    int padding;
    bool bias;
    bool withGrad;
    /* i/o */
    int hi;
    int wi;
    int ho;
    int wo;
    /* layer */
    int id;
    int opType;
    int activeType;
    int layerType;
public:
    Conv2dParam():inChannels(0),outChannels(0),kernelSize(0),stride(0),padding(0){}
    explicit Conv2dParam(int inChannels_,
                         int outChannels_ ,
                         int kernelSize_=3,
                         int stride_=1,
                         int padding_=0,
                         bool bias_=false,
                         bool withGrad_=false):
        inChannels(inChannels_),outChannels(outChannels_),kernelSize(kernelSize_),
    stride(stride_),padding(padding_),bias(bias_),withGrad(withGrad_),
    id(0),opType(OP_FORWARD),activeType(ACTIVE_LINEAR),layerType(LAYER_CONV2D){}
};

class Conv2dGrad
{
public:
    /* grad */
    Tensor dkernels;
    Tensor db;
    Tensor delta;
public:

};

class Conv2d : public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    using GradType = Conv2dGrad;
public:
    /* (N, c, kernelSize, kernelSize) */
    Tensor kernels;
    /* (N, ho, wo) */
    Tensor o;
    /* (N, ho, wo) */
    Tensor b;
    /* grad */
    Tensor dkernels;
    Tensor db;
    Tensor delta;
public:
    Conv2d(){}
    explicit Conv2d(int inChannels_,
                    int h,
                    int w,
                    int outChannels_,
                    int kernelSize_=3,
                    int stride_=1,
                    int padding_=0,
                    bool bias_=false,
                    bool withGrad_=false):
        Conv2dParam(inChannels_, outChannels_, kernelSize_, stride_, padding_, bias_, withGrad_)
    {
        kernels = Tensor(outChannels, inChannels, kernelSize, kernelSize);
        hi = h;
        wi = w;
        ho = std::floor((hi - kernelSize + 2*padding)/stride) + 1;
        wo = std::floor((wi - kernelSize + 2*padding)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        if (bias == true) {
            b = Tensor(outChannels, kernelSize, kernelSize);
        }
        /* grad */
        if (withGrad == true) {
            dkernels = Tensor(outChannels, inChannels, kernelSize, kernelSize);
            if (bias == true) {
                db = Tensor(outChannels, kernelSize, kernelSize);
            }
            delta = Tensor(outChannels, ho, wo);
        }
    }
    void forward(const Tensor &x)
    {
        /* conv */
        conv(o, kernels, x, stride, padding);
        /* bias */
        if (bias == true) {
            o += b;
        }
        /* activate */
        Sigmoid::f(o, o);
        return;
    }

    static void conv(Tensor &y, const Tensor &kernels, const Tensor &x, int stride=1, int padding=0)
    { 
        /*
            on = bn + Kncij*Xcij
            example:
                    in_chanels = 1
                    out_channels = 1
                    hi = 3
                    wi = 3
                    kernel_size = 3
                    stride = 1
                    padding = 1
                    ho = (hi - kernel_size + 2*padding)/stride + 1 = 3
                    wo = (wi - kernel_size + 2*padding)/stride + 1 = 3

                                    kernel_11:

            0   0   0   0   0

            0   1   2   3   0        0    -1     0          -4    -2    -5

            0   4   5   6   0   *   -1     1    -1      =   -9    -15   -11

            0   7   8   9   0        0    -1     0          -5    -13   -5

            0   0   0   0   0

                                    kernel_12:                     +

            0   0   0   0   0

            0   1   2   3   0        0    -1     0           2     3     8

            0   4   5   6   0   *    1     0    -1      =    1     4     11

            0   7   8   9   0        0     1     0          -12   -7     2

            0   0   0   0   0

                                    kernel_13:                     +
            0   0   0   0   0

            0   1   2   3   0        1     0     1           4     8     2

            0   4   5   6   0   *    0    -1     0      =    6     15    4

            0   7   8   9   0        1     0     1          -2     2    -4

            0   0   0   0   0

                                                                   +

                                                             bias_1:

                                                             0     0     0

                                                             0     0     0

                                                             0     0     0

                                                                   ||


                                                             2     9     5

                                                            -2     4     4

                                                            -19   -18   -7
        */
        /* output */
        for (int n = 0; n < y.shape[0]; n++) {
            for (int i = 0; i < y.shape[1]; i++) {
                for (int j = 0; j < y.shape[2]; j++) {
                    /* kernels */
                    for (int h = 0; h < kernels.shape[2]; h++) {
                        for (int k = 0; k < kernels.shape[3]; k++) {
                            for (int c = 0; c < kernels.shape[1]; c++) {
                                /* map to input  */
                                int row = h + i*stride - padding;
                                int col = k + j*stride - padding;
                                if (row < 0 || row >= x.shape[1] ||
                                        col < 0 || col >= x.shape[2]) {
                                    continue;
                                }
                                /* sum up all convolution result */
                                y(n, i, j) += kernels(n, c, h, k)*x(c, row, col);
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    void backward(Tensor &delta_)
    {
        /* delta_: previous delta, the shape is same as delta and output */
        for (int n = 0; n < delta_.shape[0]; n++) {
            for (int i = 0; i < delta_.shape[1]; i++) {
                for (int j = 0; j < delta_.shape[2]; j++) {

                    int h0 = (i - kernelSize + 1)/stride;
                    h0 = h0 > 0 ? std::ceil(h0):0;
                    int hn = i/stride;
                    hn = hn < ho ? std::floor(hn):ho;

                    int k0 = (j - kernelSize + 1)/stride;
                    k0 = k0 > 0 ? std::ceil(k0):0;
                    int kn = j/stride;
                    kn = kn < wo ? std::floor(kn):wo;

                    for (int c = 0; c < kernels.shape[0]; c++) {
                        for (int h = h0; h < hn; h++) {
                            for (int k = k0; k < kn; k++) {
                                delta_(n, i, j) += kernels(n, c, i - h*stride, j - k*stride)*delta(c, h, k);
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    void grad(const Tensor &x)
    {
        for (int n = 0; n < x.shape[0]; n++) {
            for (int i = 0; i < x.shape[1]; i++) {
                for (int j = 0; j < x.shape[2]; j++) {

                    int h0 = (i - kernelSize + 1)/stride;
                    h0 = h0 > 0 ? std::ceil(h0):0;
                    int hn = i/stride;
                    hn = hn < ho ? std::floor(hn):ho;

                    int k0 = (j - kernelSize + 1)/stride;
                    k0 = k0 > 0 ? std::ceil(k0):0;
                    int kn = j/stride;
                    kn = kn < wo ? std::floor(kn):wo;

                    for (int c = 0; c < dkernels.shape[1]; c++) {
                        for (int h = h0; h < hn; h++) {
                            for (int k = k0; k < kn; k++) {
                                dkernels(n, c, h, k) += x(n, i, j)*delta(n, h, k);
                            }
                        }
                    }
                }
            }
        }
        return;
    }
};

class MaxPooling2d: public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    using GradType = Conv2dGrad;
public:
    Tensor o;
    Tensor mask;
    Tensor delta;
public:
    MaxPooling2d(){}
    explicit MaxPooling2d(int inChannels_,
                          int h,
                          int w,
                          int kernelSize_=2,
                          int stride_=2):
        Conv2dParam(inChannels_, inChannels_, kernelSize_, stride_, 0, false)
    {
        hi = h;
        wi = w;
        ho = std::floor((hi - kernelSize)/stride) + 1;
        wo = std::floor((wi - kernelSize)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        if (withGrad == true) {
            mask = Tensor(outChannels, ho, wo);
            delta = Tensor(outChannels, ho, wo);
        }
    }
    void forward(const Tensor &x)
    {
        /* input shape is same as output shape */
        for (int n = 0; n < outChannels; n++) {
            for (int i = 0; i < ho; i++) {
                for (int j = 0; j < wo; j++) {
                    float maxValue = 0;
                    for (int h = 0; h < kernelSize; h++) {
                        for (int k = 0; k < kernelSize; k++) {
                            float value = x(n, h + i*stride, k + j*stride);
                            if (value > maxValue) {
                                maxValue = value;
                                mask(n, h + i*stride, k + j*stride) = 1;
                            }
                        }
                    }
                    o(n, i, j) = maxValue;
                }
            }
        }
        return;
    }

    void backward(Tensor &delta_)
    {
        /* delta_: previous delta, the shape is same as delta and output */
        for (int n = 0; n < delta_.shape[0]; n++) {
            for (int i = 0; i < delta_.shape[1]; i++) {
                for (int j = 0; j < delta_.shape[2]; j++) {

                    int h0 = (i - kernelSize + 1)/stride;
                    h0 = h0 > 0 ? std::ceil(h0):0;
                    int hn = i/stride;
                    hn = hn < ho ? std::floor(hn):ho;

                    int k0 = (j - kernelSize + 1)/stride;
                    k0 = k0 > 0 ? std::ceil(k0):0;
                    int kn = j/stride;
                    kn = kn < wo ? std::floor(kn):wo;

                    for (int h = h0; h < hn; h++) {
                        for (int k = k0; k < kn; k++) {
                            delta_(n, i, j) += mask(n, h, k)*delta(n, h, k);
                        }
                    }
                }
            }
        }
        mask.zero();
        return;
    }
};

class AveragePooling2d: public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    using GradType = Conv2dGrad;
public:
    Tensor o;
    Tensor delta;
public:
    AveragePooling2d(){}
    explicit AveragePooling2d(int inChannels_,
                              int h,
                              int w,
                              int kernelSize_=2,
                              int stride_=2):
        Conv2dParam(inChannels_, inChannels_, kernelSize_, stride_, 0, false)
    {
        hi = h;
        wi = w;
        ho = std::floor((hi - kernelSize)/stride) + 1;
        wo = std::floor((wi - kernelSize)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        if (withGrad == true) {
            delta = Tensor(outChannels, ho, wo);
        }
    }
    void forward(const Tensor &x)
    {
        /* conv */
        for (int n = 0; n < outChannels; n++) {
            for (int i = 0; i < ho; i++) {
                for (int j = 0; j < wo; j++) {
                    float u = 0;
                    for (int h = 0; h < kernelSize; h++) {
                        for (int k = 0; k < kernelSize; k++) {
                            u += x(n, h + i*stride, k + j*stride);
                        }
                    }
                    o(n, i, j) = u/(kernelSize*kernelSize);
                }
            }
        }
        return;
    }
    void backward(Tensor &delta_)
    {
        /* delta_: previous delta, the shape is same as delta and output */
        for (int n = 0; n < delta_.shape[0]; n++) {
            for (int i = 0; i < delta_.shape[1]; i++) {
                for (int j = 0; j < delta_.shape[2]; j++) {

                    int h0 = (i - kernelSize + 1)/stride;
                    h0 = h0 > 0 ? std::ceil(h0):0;
                    int hn = i/stride;
                    hn = hn < ho ? std::floor(hn):ho;

                    int k0 = (j - kernelSize + 1)/stride;
                    k0 = k0 > 0 ? std::ceil(k0):0;
                    int kn = j/stride;
                    kn = kn < wo ? std::floor(kn):wo;

                    for (int h = h0; h < hn; h++) {
                        for (int k = k0; k < kn; k++) {
                            delta_(n, i, j) += delta(n, h, k);
                        }
                    }
                }
            }
        }
        return;
    }
};

#endif // CONV_H