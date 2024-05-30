#ifndef CONV2D_HPP
#define CONV2D_HPP
#include "../basic/tensor.hpp"
#include "activate.hpp"
#include "layerdef.h"
#include "conv.hpp"

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
    Conv2dParam()
        :inChannels(0),outChannels(0),kernelSize(0),stride(0),padding(0),
         hi(0),wi(0),ho(0),wo(0){}
    explicit Conv2dParam(const Conv2dParam &param)
        : inChannels(param.inChannels),outChannels(param.outChannels),kernelSize(param.kernelSize),
          stride(param.stride),padding(param.padding),bias(param.bias),
          hi(param.hi),wi(param.wi),ho(param.ho),wo(param.wo),
          id(0),opType(OP_FORWARD),activeType(Active_Linear),layerType(Layer_Conv2d){}
    explicit Conv2dParam(int inChannels_,
                         int h,
                         int w,
                         int outChannels_ ,
                         int kernelSize_=3,
                         int stride_=1,
                         int padding_=0,
                         bool bias_=false,
                         int activeType_=Active_LeakyRelu):
        inChannels(inChannels_),outChannels(outChannels_),kernelSize(kernelSize_),
    stride(stride_),padding(padding_),bias(bias_),
    hi(h),wi(w),
    id(0),opType(OP_FORWARD),activeType(activeType_),layerType(Layer_Conv2d){}
};

class Conv2d : public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    class Grad : public Conv2dParam
    {
    public:
        /* grad */
        Tensor dkernels;
        Tensor db;
        Tensor e;
    public:
        Grad(){}
        explicit Grad(const Conv2dParam &param)
            :Conv2dParam(param)
        {
            dkernels = Tensor(outChannels, inChannels, kernelSize, kernelSize);
            if (bias == true) {
                db = Tensor(outChannels, kernelSize, kernelSize);
            }
            e = Tensor(outChannels, ho, wo);
            layerType = Layer_Conv2d;
        }

        inline Tensor& loss() {return e;}

        void backward(const Conv2d &layer, Tensor &delta_)
        {
            /* delta_: previous delta, the shape is same as input */

#if 1
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

                        for (int c = 0; c < layer.kernels.shape[0]; c++) {
                            for (int h = h0; h < hn; h++) {
                                for (int k = k0; k < kn; k++) {
                                    delta_(n, i, j) += layer.kernels(n, c, i - h*stride, j - k*stride)*e(c, h, k);
                                }
                            }
                        }
                    }
                }
            }
#else

            /* kernels shape */
            for (int c = 0; c < dkernels.shape[1]; c++) {
                for (int h = 0; h < dkernels.shape[2]; h++) {
                    for (int k = 0; k < dkernels.shape[3]; k++) {
                        /* output shape */
                        for (int n = 0; n < delta.shape[0]; n++) {
                            for (int i = 0; i < delta.shape[1]; i++) {
                                for (int j = 0; j < delta.shape[2]; j++) {
                                    /* input shape */
                                    int row = h + i*stride - padding;
                                    int col = k + j*stride - padding;
                                    if (row < 0 || row >= delta_.shape[1] ||
                                            col < 0 || col >= delta_.shape[2]) {
                                        continue;
                                    }
                                    delta_(c, row, col) += layer.kernels(n, c, h, k)*delta(n, i, j);
                                }
                            }
                        }
                    }
                }
            }
#endif
            return;
        }

        void eval(Conv2d &conv, const Tensor &x)
        {
            Tensor &o = conv.o;
            Tensor dy = Fn::df(activeType, o);
            dy *= e;
            /* db */
            if (bias == true) {
                db += dy;
            }
            /* dkernel */
#if 1
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
                                    dkernels(n, c, h, k) += x(n, i, j)*dy(n, h, k);
                                }
                            }
                        }
                    }
                }
            }

#else
            /* kernels shape */
            for (int n = 0; n < dkernels.shape[0]; n++) {
                for (int c = 0; c < dkernels.shape[1]; c++) {
                    for (int h = 0; h < dkernels.shape[2]; h++) {
                        for (int k = 0; k < dkernels.shape[3]; k++) {
                            /* output shape */
                            for (int i = 0; i < dy.shape[1]; i++) {
                                for (int j = 0; j < dy.shape[2]; j++) {
                                    /* input shape */
                                    int row = h + i*stride - padding;
                                    int col = k + j*stride - padding;
                                    if (row < 0 || row >= x.shape[1] ||
                                            col < 0 || col >= x.shape[2]) {
                                        continue;
                                    }
                                    dkernels(n, c, h, k) += x(c, row, col)*dy(n, i, j);
                                }
                            }
                        }
                    }
                }
            }
#endif
            return;
        }

    };

    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        Optimizer optKernels;
        Optimizer optB;
    public:
        OptimizeBlock(){}
        explicit OptimizeBlock(const Conv2d &layer)
        {
            optKernels = Optimizer(layer.kernels.shape);
            if (layer.bias == true) {
                optB = Optimizer(layer.b.shape);
            }
        }
        void operator()(Conv2d& layer, Grad& grad, float learningRate)
        {
            optKernels(layer.kernels, grad.dkernels, learningRate);
            if (layer.bias == true) {
                optB(layer.b, grad.db, learningRate);
            }
            return;
        }
    };
public:
    /* (N, c, kernelSize, kernelSize) */
    Tensor kernels;
    /* (N, ho, wo) */
    Tensor o;
    /* (N, ho, wo) */
    Tensor b;
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
                    int activeType_=Active_LeakyRelu):
        Conv2dParam(inChannels_, h, w, outChannels_, kernelSize_, stride_, padding_, bias_, activeType_)
    {
        kernels = Tensor(outChannels, inChannels, kernelSize, kernelSize);
        util::uniform(kernels, -1, 1);
        ho = std::floor((hi - kernelSize + 2*padding)/stride) + 1;
        wo = std::floor((wi - kernelSize + 2*padding)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        if (bias == true) {
            b = Tensor(outChannels, kernelSize, kernelSize);
            util::uniform(b, -1, 1);
        }
        layerType = Layer_Conv2d;
    }

    inline Tensor& output() {return o;}

    Tensor& forward(const Tensor &x)
    {           
        /* conv */
        o.zero();
        conv::eval2(o, kernels, x, stride, padding);
        /* bias */
        if (bias == true) {
            o += b;
        }
        /* activate */
        Fn::f(activeType, o);
#if 0
        float _max = std::abs(o.max());
        o /= _max;
#endif
        return o;
    }


};

class MaxPooling2d: public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    /* grad */
    class Grad : public Conv2dParam
    {
    public:
        Tensor e;
    public:
        Grad(){}
        explicit Grad(const Conv2dParam &param)
            :Conv2dParam(param)
        {
            e = Tensor(outChannels, ho, wo);
        }
        inline Tensor& loss() {return e;}
        void backward(MaxPooling2d &layer, Tensor &delta_)
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
                                delta_(n, i, j) += layer.mask(n, h, k)*e(n, h, k);
                            }
                        }
                    }
                }
            }
            layer.mask.zero();
            return;
        }
        /* no gradient */
        void eval(MaxPooling2d &layer, const Tensor &)
        {

        }
    };
    /* optimizer */
    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        OptimizeBlock(){}
        explicit OptimizeBlock(const MaxPooling2d &){}
        void operator()(MaxPooling2d&, Grad&, float){}
    };
public:
    Tensor o;
    Tensor mask;
public:
    MaxPooling2d(){}
    explicit MaxPooling2d(int inChannels_,
                          int h,
                          int w,
                          int kernelSize_=2,
                          int stride_=2):
        Conv2dParam(inChannels_, h, w, inChannels_, kernelSize_, stride_, 0, false)
    {
        ho = std::floor((hi - kernelSize)/stride) + 1;
        wo = std::floor((wi - kernelSize)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        mask = Tensor(outChannels, ho, wo);
        layerType = Layer_MaxPooling2d;
    }
    inline Tensor& output() {return o;}
    Tensor& forward(const Tensor &x)
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
                                mask(n, h, k) = 1;
                            }
                        }
                    }
                    o(n, i, j) = maxValue;
                }
            }
        }
        return o;
    }

};

class AvgPooling2d: public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    /* grad */
    class Grad : public Conv2dParam
    {
    public:
        Tensor e;
    public:
        Grad(){}
        explicit Grad(const Conv2dParam &param)
            :Conv2dParam(param)
        {
            e = Tensor(outChannels, ho, wo);
        }
        inline Tensor& loss() {return e;}
        void backward(AvgPooling2d &layer, Tensor &delta_)
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
                                delta_(n, i, j) += e(n, h, k);
                            }
                        }
                    }
                }
            }
            return;
        }
        /* no gradient */
        void eval(AvgPooling2d &layer, const Tensor &){}
    };

    /* optimizer */
    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        OptimizeBlock(){}
        explicit OptimizeBlock(const AvgPooling2d &){}
        void operator()(AvgPooling2d&, Grad&, float){}
    };
public:
    Tensor o;
public:
    AvgPooling2d(){}
    explicit AvgPooling2d(int inChannels_,
                          int h,
                          int w,
                          int kernelSize_=2,
                          int stride_=2):
        Conv2dParam(inChannels_, h, w, inChannels_, kernelSize_, stride_, 0, false)
    {
        ho = std::floor((hi - kernelSize)/stride) + 1;
        wo = std::floor((wi - kernelSize)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        layerType = Layer_AvgPooling2d;
    }
    inline Tensor& output() {return o;}
    Tensor& forward(const Tensor &x)
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
        return o;
    }

};


class ResidualConv2d : public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    class Grad: public Conv2dParam
    {
    public:
        Conv2d::Grad convGrad1;
        Conv2d::Grad convGrad2;
    public:
        explicit Grad(const Conv2dParam &param)
            :Conv2dParam(param)
        {
            convGrad1 = Conv2d::Grad(param);
            convGrad2 = Conv2d::Grad(param);
        }
        inline Tensor& loss() {return convGrad2.e;}
        void backward(const ResidualConv2d &layer, Tensor &delta_)
        {
            convGrad2.backward(layer.conv2, convGrad1.e);
            convGrad1.backward(layer.conv1, delta_);
            return;
        }
        void eval(ResidualConv2d& layer, const Tensor &x)
        {
             convGrad1.eval(layer.conv1, x);
             convGrad2.eval(layer.conv2, layer.conv1.o);
             /* residual part differentiate */
             convGrad2.dkernels += 1;
             if (convGrad2.bias == true) {
                 convGrad2.db += 1;
             }
             return;
        }
    };

    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        Conv2d::OptimizeBlock<Optimizer> opt1;
        Conv2d::OptimizeBlock<Optimizer> opt2;
    public:
        OptimizeBlock(){}
        explicit OptimizeBlock(const ResidualConv2d &layer)
        {
            opt1 = Conv2d::OptimizeBlock<Optimizer>(layer.conv1);
            opt2 = Conv2d::OptimizeBlock<Optimizer>(layer.conv2);
        }
        void operator()(ResidualConv2d& layer, Grad& grad, float learningRate)
        {
            opt1(layer.conv1, grad.convGrad1, learningRate);
            opt2(layer.conv2, grad.convGrad2, learningRate);
            return;
        }
    };

public:
    Conv2d conv1;
    Conv2d conv2;
public:
    ResidualConv2d(){}
    explicit ResidualConv2d(int inChannels_,
                    int h,
                    int w,
                    int kernelSize_=3,
                    int stride_=1,
                    int padding_=0,
                    bool bias_=false,
                    int activeType_=Active_LeakyRelu):
        Conv2dParam(inChannels_, h, w, inChannels_, kernelSize_, stride_, padding_, bias_, activeType_)
    {
        conv1 = Conv2d(inChannels, h, w, inChannels, kernelSize, stride, padding, bias, activeType);
        conv2 = Conv2d(inChannels, h, w, inChannels, kernelSize, stride, padding, bias, activeType);
    }
    inline Tensor& output() {return conv2.o;}
    Tensor& forward(const Tensor &x)
    {
        Tensor& c1 = conv1.forward(x);
        /* c2 = conv(c1) + x */
        Tensor& c2 = conv2.forward(c1);
        c2 += x;
        return c2;
    }
};


class NMS: public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    /* grad */
    class Grad : public Conv2dParam
    {
    public:
        Tensor e;
    public:
        Grad(){}
        explicit Grad(const Conv2dParam &param)
            :Conv2dParam(param)
        {
            e = Tensor(inChannels, hi, wi);
        }
        void backward(NMS &layer, Tensor &delta_)
        {
            delta_ = e;
            return;
        }
        /* no gradient */
        void eval(NMS&, const Tensor &){}
    };
    /* optimizer */
    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        OptimizeBlock(){}
        explicit OptimizeBlock(const NMS &){}
        void operator()(NMS&, Grad&, float){}
    };
public:
    Tensor o;
public:
    NMS(){}
    explicit NMS(int inChannels_, int h, int w)
        :Conv2dParam(inChannels_, h, w, inChannels_, 0, 0, 0, false)
    {
        o = Tensor(inChannels, hi, wi);
    }
    Tensor& forward(const Tensor &x)
    {
        float max_ = x.max();
        o = x / max_;
        return o;
    }

};


#endif // CONV2D_HPP
