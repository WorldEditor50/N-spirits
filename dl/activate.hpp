#ifndef ACTIVATE_H
#define ACTIVATE_H
#include <cmath>
#include <map>
#include "../basic/linalg.h"
#include "../basic/tensor.hpp"
enum FnType {
    Fn_Linear = 0,
    Fn_Sigmoid,
    Fn_Relu,
    Fn_LeakyRelu,
    Fn_Tanh,
    Fn_Gelu
};
struct Sigmoid {
    inline static float f(float x) {return 1/(1 + std::exp(-x));}
    inline static float df(float y) {return y*(1 - y);}
    inline static Tensor& f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = 1/(1 + std::exp(-x.val[i]));
        }
        return x;
    }

    inline static Tensor& df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i]*(1 - y.val[i]);
        }
        return y;
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
    inline static Tensor& f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = std::tanh(x.val[i]);
        }
        return x;
    }

    inline static Tensor& df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = 1 - y.val[i]*y.val[i];
        }
        return y;
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
    inline static Tensor& f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = x.val[i] > 0 ? x.val[i] : 0;
        }
        return x;
    }

    inline static Tensor& df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i] > 0 ? 1 : 0;
        }
        return y;
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
    inline static Tensor& f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = x.val[i] > 0 ? x.val[i] : 0.01*x.val[i];
        }
        return x;
    }

    inline static Tensor& df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i] > 0 ? 1 : 0.01;
        }
        return y;
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
    inline static Tensor& f(Tensor &x){return x;}

    inline static Tensor& df(Tensor &y)
    {
        y.fill(1);
        return y;
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


struct Fn {
    inline static float f(int type, float x)
    {
        switch (type) {
        case Fn_Linear:
            return Linear::f(x);
        case Fn_Sigmoid:
            return Sigmoid::f(x);
        case Fn_Relu:
            return Relu::f(x);
        case Fn_LeakyRelu:
            return LeakyRelu::f(x);
        case Fn_Tanh:
            return Tanh::f(x);
        default:
            break;
        }
        return Sigmoid::f(x);
    }
    inline static float df(int type, float y)
    {
        switch (type) {
        case Fn_Linear:
            return Linear::df(y);
        case Fn_Sigmoid:
            return Sigmoid::df(y);
        case Fn_Relu:
            return Relu::df(y);
        case Fn_LeakyRelu:
            return LeakyRelu::df(y);
        case Fn_Tanh:
            return Tanh::df(y);
        default:
            break;
        }
        return Sigmoid::df(y);
    }
    inline static Tensor& f(int type, Tensor &x)
    {
        switch (type) {
        case Fn_Linear:
            return Linear::f(x);
        case Fn_Sigmoid:
            return Sigmoid::f(x);
        case Fn_Relu:
            return Relu::f(x);
        case Fn_LeakyRelu:
            return LeakyRelu::f(x);
        case Fn_Tanh:
            return Tanh::f(x);
        default:
            break;
        }
        return Sigmoid::f(x);
    }

    inline static Tensor& df(int type, Tensor &y)
    {
        switch (type) {
        case Fn_Linear:
            return Linear::df(y);
        case Fn_Sigmoid:
            return Sigmoid::df(y);
        case Fn_Relu:
            return Relu::df(y);
        case Fn_LeakyRelu:
            return LeakyRelu::df(y);
        case Fn_Tanh:
            return Tanh::df(y);
        default:
            break;
        }
        return Sigmoid::df(y);
    }
    inline static Tensor f(int type, const Tensor &x)
    {
        switch (type) {
        case Fn_Linear:
            return Linear::f(x);
        case Fn_Sigmoid:
            return Sigmoid::f(x);
        case Fn_Relu:
            return Relu::f(x);
        case Fn_LeakyRelu:
            return LeakyRelu::f(x);
        case Fn_Tanh:
            return Tanh::f(x);
        default:
            break;
        }
        return Sigmoid::f(x);
    }

    inline static Tensor df(int type, const Tensor &y)
    {
        switch (type) {
        case Fn_Linear:
            return Linear::df(y);
        case Fn_Sigmoid:
            return Sigmoid::df(y);
        case Fn_Relu:
            return Relu::df(y);
        case Fn_LeakyRelu:
            return LeakyRelu::df(y);
        case Fn_Tanh:
            return Tanh::df(y);
        default:
            break;
        }
        return Sigmoid::df(y);
    }
};

#endif // ACTIVATE_H
