#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <cmath>
#include <vector>
#include <iostream>

/* forward mode */
class Varibable
{
public:
    float val;
    float dval;

public:
    Varibable():val(0),dval(0){}
    Varibable(float val_, float dval_=1.0):val(val_),dval(dval_){}

    Varibable operator + (const Varibable &x)
    {
        Varibable y;
        y.val = val + x.val;
        y.dval = dval + x.dval;
        return y;
    }

    Varibable operator - (const Varibable &x)
    {
        Varibable y;
        y.val = val - x.val;
        y.dval = dval - x.dval;
        return y;
    }

    Varibable operator * (const Varibable &x)
    {
        /*
            (uv)' = u'v + uv'
        */
        Varibable y;
        y.val = val * x.val;
        y.dval = dval*x.val + val*x.dval;
        return y;
    }

    Varibable operator / (const Varibable &x)
    {
        /*
            (u/v)' = u'/v - (u*v')/(v*v)
                   = (u'v - uv')/(v*v)
        */
        Varibable y;
        y.val = val / x.val;
        y.dval = (dval*x.val - val*x.dval)/(x.val*x.val);
        return y;
    }

    Varibable operator + (float c)
    {
        Varibable y;
        y.val = val + c;
        y.dval = dval;
        return y;
    }

    Varibable operator - (float c)
    {
        Varibable y;
        y.val = val - c;
        y.dval = dval;
        return y;
    }

    Varibable operator * (float c)
    {
        Varibable y;
        y.val = val * c;
        y.dval = dval*c;
        return y;
    }

    Varibable operator / (float c)
    {
        Varibable y;
        y.val = val / c;
        y.dval = dval / c;
        return y;
    }

    static Varibable sin(const Varibable &x)
    {
        Varibable y;
        y.val = std::sin(x.val);
        y.dval = std::cos(x.val)*x.dval;
        return y;
    }

    static Varibable cos(const Varibable &x)
    {
        Varibable y;
        y.val = std::cos(x.val);
        y.dval = -1*std::sin(x.val)*x.dval;
        return y;
    }

    static Varibable log(const Varibable &x)
    {
        Varibable y;
        y.val = std::log(x.val);
        y.dval = x.dval/x.val;
        return y;
    }

    static Varibable exp(const Varibable &x)
    {
        Varibable y;
        y.val = std::exp(x.val);
        y.dval = std::exp(x.val)*x.dval;
        return y;
    }

    static Varibable sqrt(const Varibable &x)
    {
        Varibable y;
        y.val = std::sqrt(x.val);
        y.dval = 0.5*x.dval/std::sqrt(x.val);
        return y;
    }

    void show() const
    {
        std::cout<<"val="<<val<<", dval="<<dval<<std::endl;
        return;
    }


    static void test()
    {
        /*
            f(x) = ln(x) + sin(x)
            f'(x) = 1/x + cos(x)
            f(1) = ln(1) + sin(1)
            f'(1) = 1 + cos(1)
        */
        Varibable x(1);
        Varibable y1 = log(x) + sin(x);
        y1.show();

        /*
            f(x) = exp(x) * sin(x)
            f'(x) = exp(x)*(sin(x) + cos(x))
            f(1) = exp(1) * sin(1)
            f'(1) = exp(1) * (sin(1) + cos(1))
        */
        Varibable y2 = exp(x)*sin(x);
        y2.show();
    }
};

#endif // AUTOGRAD_H
