#ifndef ACTIVATE_H
#define ACTIVATE_H
#include <cmath>
#include <map>
#include "../basic/utils.h"
#include "../basic/tensor.hpp"

struct Sigmoid {
    inline static float f(float x) {return 1/(1 + std::exp(-x));}
    inline static float df(float y) {return y*(1 - y);}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = 1/(1 + std::exp(-x.val[i]));
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i]*(1 - y.val[i]);
        }
        return;
    }
    inline static Tensor f(const Tensor &x)
    {
        Tensor y(x.shape);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            y.val[i] = 1/(1 + std::exp(-x.val[i]));
        }
        return y;
    }

    inline static Tensor df(const Tensor &y)
    {
        Tensor dy(y.shape);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy.val[i] = y.val[i]*(1 - y.val[i]);
        }
        return dy;
    }
};

struct Tanh {
    inline static float f(float x) {return std::tanh(x);}
    inline static float df(float y) {return 1 - y*y;}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = std::tanh(x.val[i]);
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = 1 - y.val[i]*y.val[i];
        }
        return;
    }

    inline static Tensor f(const Tensor &x)
    {
        Tensor y(x.shape);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            y.val[i] = std::tanh(x.val[i]);
        }
        return y;
    }

    inline static Tensor df(const Tensor &y)
    {
        Tensor dy(y.shape);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy.val[i] = 1 - y.val[i]*y.val[i];
        }
        return dy;
    }
};

struct Relu {
    inline static float f(float x) {return x > 0 ? x : 0;}
    inline static float df(float y) {return y > 0 ? 1 : 0;}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = x.val[i] > 0 ? x.val[i] : 0;
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i] > 0 ? 1 : 0;
        }
        return;
    }

    inline static Tensor f(const Tensor &x)
    {
        Tensor y(x.shape);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            y.val[i] = x.val[i] > 0 ? x.val[i] : 0;
        }
        return y;
    }

    inline static Tensor df(const Tensor &y)
    {
        Tensor dy(y.shape);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy.val[i] = y.val[i] > 0 ? 1 : 0;
        }
        return dy;
    }
};

struct LeakyRelu {
    inline static float f(float x) {return x > 0 ? x : 0.01*x;}
    inline static float df(float y) {return y > 0 ? 1 : 0.01;}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = x.val[i] > 0 ? x.val[i] : 0.01*x.val[i];
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i] > 0 ? 1 : 0.01;
        }
        return;
    }
    inline static Tensor f(const Tensor &x)
    {
        Tensor y(x.shape);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            y.val[i] = x.val[i] > 0 ? x.val[i] : 0.01*x.val[i];
        }
        return y;
    }

    inline static Tensor df(const Tensor &y)
    {
        Tensor dy(y.shape);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy.val[i] = y.val[i] > 0 ? 1 : 0.01;
        }
        return dy;
    }
};

struct Linear {
    inline static float f(float x) {return x;}
    inline static float df(float) {return 1;}
    inline static void f(Tensor &x){}

    inline static void df(Tensor &y)
    {
        y.fill(1);
        return;
    }
    inline static Tensor f(const Tensor &x)
    {
        return x;
    }

    inline static Tensor df(const Tensor &y)
    {
        Tensor dy = Tensor::ones();
        return dy;
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
        float s = y.sum();
        y /= s;
        return;
    }
    inline static void df(Tensor &dy, const Tensor &y, const Tensor &yt)
    {
        Utils::sub(dy, y, yt);
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
        void(*f)(Tensor &x);
        void(*df)(Tensor &y);
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
