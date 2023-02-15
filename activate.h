#ifndef ACTIVATE_H
#define ACTIVATE_H
#include <cmath>
#include <map>
#include "tensor.h"
#include "utils.h"

struct Sigmoid {
    inline static float f(float x) {return std::exp(x)/(1 + std::exp(x));}
    inline static float df(float y) {return y*(1 - y);}
    inline static void f(Tensor& y, const Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalsize; i++) {
            y.val[i] = 1/(1 + exp(-x.val[i]));
        }
        return;
    }

    inline static void df(Tensor &dy, const Tensor &y)
    {
        for (std::size_t i = 0; i < dy.totalsize; i++) {
            dy.val[i] = y.val[i]*(1 + y.val[i]);
        }
        return;
    }
};

struct Tanh {
    inline static float f(float x) {return tanh(x);}
    inline static float df(float y) {return 1 - y*y;}
    inline static void f(Tensor& y, const Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalsize; i++) {
            y.val[i] = tanh(-x.val[i]);
        }
        return;
    }

    inline static void df(Tensor &dy, const Tensor &y)
    {
        for (std::size_t i = 0; i < dy.totalsize; i++) {
            dy.val[i] = 1 - y.val[i]*y.val[i];
        }
        return;
    }
};

struct Relu {
    inline static float f(float x) {return x > 0 ? x : 0;}
    inline static float df(float y) {return y > 0 ? 1 : 0;}
    inline static void f(Tensor& y, const Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalsize; i++) {
            y.val[i] = x.val[i] > 0 ? x.val[i] : 0;
        }
        return;
    }

    inline static void df(Tensor &dy, const Tensor &y)
    {
        for (std::size_t i = 0; i < dy.totalsize; i++) {
            dy.val[i] = y.val[i] > 0 ? 1 : 0;
        }
        return;
    }
};

struct LeakyRelu {
    inline static float f(float x) {return x > 0 ? x : 0.01*x;}
    inline static float df(float y) {return y > 0 ? 1 : 0.01;}
    inline static void f(Tensor& y, const Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalsize; i++) {
            y.val[i] = x.val[i] > 0 ? x.val[i] : 0.01*x.val[i];
        }
        return;
    }

    inline static void df(Tensor &dy, const Tensor &y)
    {
        for (std::size_t i = 0; i < dy.totalsize; i++) {
            dy.val[i] = y.val[i] > 0 ? 1 : 0.01;
        }
        return;
    }
};

struct Linear {
    inline static float f(float x) {return x;}
    inline static float df(float) {return 1;}
    inline static void f(Tensor& y, const Tensor &x)
    {
        y = x;
        return;
    }

    inline static void df(Tensor &dy, const Tensor &y)
    {
        dy.fill(1);
        return;
    }
};

struct Swish {
    static constexpr float beta = 1.0;//1.702;
    inline static float f(float x) {return x*Sigmoid::f(beta*x);}
    inline static float d(float x)
    {
        float s = Sigmoid::f(beta*x);
        return s + x*s*(1 - s);
    }
};

struct Gelu {
    static constexpr float c1 = 0.79788456080287;/* sqrt(2/pi) */
    static constexpr float c2 = 0.044715;
    inline static float f(float x)
    {
        return 0.5*x*(1 + tanh(c1*(x + c2*x*x*x)));
    }
    inline static float df(float x)
    {
        float t = tanh(c1*(x + c2*x*x*x));
        return 0.5*(1 + t + x*(c1*(1 + 3*c2*x*x)*(1 - t*t)));
    }
};

struct Softmax_ {

    inline static void f(Tensor& y, const Tensor &x)
    {
        Utils::exp(x, y);
        float s = Tensor::sum(y);
        y /= s;
        return;
    }
    inline static void df(Tensor &dy, const Tensor &y, const Tensor &yt)
    {
        Tensor::minus(dy, y, yt);

        return;
    }
};

enum ActiveType {
    ACTIVE_LINEAR = 0,
    ACTIVE_SIGMOID,
    ACTIVE_RELU,
    ACTIVE_LEAKRELU,
    ACTIVE_TANH,
    ACTIVE_GELU
};

class Active
{
public:
    struct Functor {
        void(*f)(Tensor& y, const Tensor &x);
        void(*df)(Tensor &dy, const Tensor &y);
    };
public:
    static std::map<int, Functor> func;
};

std::map<int, Active::Functor> Active::func = {
    {ACTIVE_LINEAR, {&Linear::f, &Linear::df}},
    {ACTIVE_SIGMOID, {&Sigmoid::f, &Sigmoid::df}},
    {ACTIVE_RELU, {&Relu::f, &Relu::df}},
    {ACTIVE_LEAKRELU, {&LeakyRelu::f, &LeakyRelu::df}},
    {ACTIVE_TANH, {&Tanh::f, &Tanh::df}}
};

#endif // ACTIVATE_H
