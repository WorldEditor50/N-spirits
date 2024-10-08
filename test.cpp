#include <iostream>
#include <tuple>
#include "basic/mat.h"
#include "basic/mats.hpp"
#include "basic/linalg.h"
#include "basic/tensor.hpp"
#include "basic/complex.hpp"
#include "basic/quaternion.hpp"
#include "basic/optimization.hpp"
#include "basic/fft.h"
#include "utils/csv.h"
#include "utils/dataset.h"
#include "utils/clock.hpp"
/*
    http://www.yunsuan.info/matrixcomputations/solveqrfactorization.html
*/

void test_rank()
{
    Tensor x1({4, 4}, {1,0,1,1,
                      1,0,0,1,
                      1,0,1,0,
                      0,0,1,1});

    int r1 = LinAlg::rank(x1);
    std::cout<<"rank:"<<r1<<std::endl;
    Tensor x2({4, 4}, {1,0,1,1,
                      1,2,0,1,
                      1,0,1,0,
                      0,0,1,1});

    int r2 = LinAlg::rank(x2);
    std::cout<<"rank:"<<r2<<std::endl;
    return;
}

void test_lu()
{
    Tensor x({3, 3}, { 1, 1, 1,
                       0, 0.5, -2,
                       0, 1, 1});
    Tensor xi;
    LinAlg::LU::inv(x, xi);
    xi.printValue();
    std::cout<<"test inverse:"<<std::endl;
    Tensor I(x.shape);
    Tensor::MM::ikkj(I, x, xi);
    I.printValue();
    return;
}

void test_cholesky()
{
    Tensor x({3, 3}, {8, 0, 0,
                      0, 2, 6,
                      1, 1, 3 });
    Tensor l;
    int ret = LinAlg::Cholesky::solve(x, l);
    if (ret == -1) {
        std::cout<<"x is not a square matrix."<<std::endl;
    } else if (ret == -2) {
        std::cout<<"x is not a positive matrix."<<std::endl;
    } else if (ret == 0) {
        l.printValue();
    }
    return;
}

void test_gaussSeidel()
{
    Tensor a({3, 3}, { 20, 2, 3,
                       1,  8, 1,
                       2, -3, 15 });
    Tensor b({3, 1}, {24.0, 12.0, 30.0});
    Tensor x(3, 1);
    int ret = LinAlg::gaussSeidel(a, b, x, 1000, 1e-3);
    if (ret == -2) {
        std::cout<<"a is not a positive definite matrix"<<std::endl;
        return;
    }
    x.printValue();
    return;
}

void test_gaussianElimination()
{
    Tensor a({3, 4}, { 3, -4,  6,  24,
                       5,  2, -8, -24,
                      -1,  1,  2,  11 });
    Tensor u;
    LinAlg::GaussianElimination::solve(a, u);
    Tensor x(3, 1);
    LinAlg::GaussianElimination::evaluate(u, x);
    x.printValue();
    return;
}

void test_det()
{
    float value;
    Tensor x1({3, 3}, {1, 1, 1,
                       1, 2, 3,
                       1, 5, 1});
    value = LinAlg::det(x1);
    std::cout<<"det:"<<value<<std::endl;

    Tensor x2({4, 4}, {1, 1, 1, 2,
                       1, 2, 3, 0,
                       0, 5, 1, -1,
                       1, 0, -3, 1});
    value = LinAlg::det(x2);
    std::cout<<"det:"<<value<<std::endl;
    return;
}

void test_qr()
{
    Tensor x({3, 3}, {1, 1, 1,
                      1, 2, 3,
                      1, 5, 1});
    Tensor q;
    Tensor r;
    LinAlg::QR::solve(x, q, r);
    std::cout<<"Q:"<<std::endl;
    q.printValue();
    std::cout<<"R:"<<std::endl;
    r.printValue();
    return;
}

void test_svd()
{
    Tensor x1({3, 3}, {1, 1, 1,
                       1, 2, 3,
                       1, 5, 1});

    Tensor x({10, 5}, {0.0162, 0.878, 0.263, 0.0955, 0.359,
                       0.329, 0.326, 0.757, 0.165, 0.728,
                       0.609, 0.515, 0.385, 0.908, 0.89,
                       0.91, 0.06, 0.43, 0.691, 0.96,
                       0.476, 0.0498, 0.65, 0.378, 0.672,
                       0.914, 0.788, 0.285, 0.447, 0.0846,
                       0.495, 0.463, 0.962, 0.758, 0.558,
                       0.321, 0.0872, 0.884, 0.0788, 0.252,
                       0.612, 0.688, 0.767, 0.997, 0.597,
                       0.657, 0.907, 0.657, 0.0873, 0.598
          });
    Tensor u;
    Tensor v;
    Tensor s;
    LinAlg::SVD::solve(x, u, s, v, 1e-7, 1000);
    std::cout<<"U:"<<std::endl;
    u.printValue2D();
    std::cout<<"S:"<<std::endl;
    s.printValue2D();
    std::cout<<"V:"<<std::endl;
    v.printValue2D();
    /* x = U*S*V^T */
    Tensor y1(x.shape);
    Tensor::MM::ikkj(y1, u, s);
    Tensor y2(x.shape);
    Tensor::MM::ikjk(y2, y1, v);
    std::cout<<"x:"<<std::endl;
    x.printValue2D();
    std::cout<<"y:"<<std::endl;
    y2.printValue2D();
    return;
}

void test_transpose_multiply()
{
    Mat x1(2, 3, {1, 2, 3,
                  4, 5, 6});
    Mat x2(3, 2, {1, 2,
                  3, 4,
                  5, 6});
    Mat x3(2, 3, {1, 0, 3,
                  4, 2, 0});
    Mat x4(3, 2, {0, 2,
                  3, 4,
                  1, 0});
    std::cout<<"ikkj:"<<std::endl;
    Mat x5(2, 2);
    /* (2, 3) * (3, 2) = (2, 2) */
    Mat::Multi::ikkj(x5, x1, x2);
    /* [[22, 28], [49, 64]] */
    x5.show();
    std::cout<<"kikj:"<<std::endl;
    Mat x6(2, 2);
    /* (3, 2) * (3, 2) = (2, 2) */
    Mat::Multi::kikj(x6, x4, x2);
    /*
        0, 3, 1     1, 2
        2, 4, 0     3, 4
                    5, 6
        [[14, 18], [14, 20]]
    */
    x6.show();
    std::cout<<"ikjk:"<<std::endl;
    Mat x7(2, 2);
    /* (2, 3) * (2, 3) = (2, 2) */
    Mat::Multi::ikjk(x7, x1, x3);
    /*
        1, 2, 3   1, 4
        4, 5, 6   0, 2
                  3, 0

        [[10, 8], [22, 26]]
    */
    x7.show();
    std::cout<<"kijk:"<<std::endl;
    Mat x8(2, 2);
    /* (3, 2) * (2, 3) = (2, 2) */
    Mat::Multi::kijk(x8, x4, x3);
    /*
        0, 2      1, 0, 3
        3, 4      4, 2, 0
        1, 0

        0, 3, 1     1, 4
        2, 4, 0     0, 2
                    3, 0

        [[3, 6], [2, 16]]
    */
    x8.show();
    /* rand */
    Mat x9(3, 4);
    Mat x10(3, 5);
    Mat x11(4, 5);
    LinAlg::uniform(x9, 0, 9);
    LinAlg::uniform(x10, 0, 9);
    Mat::Multi::kikj(x11, x9, x10);
    std::cout<<"x9:"<<std::endl;
    x9.show();
    std::cout<<"x10:"<<std::endl;
    x10.show();
    std::cout<<"x11:"<<std::endl;
    x11.show();
    return;
}

void test_static_matrix()
{
    Mats<3, 3> x1 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Mats<3, 3> x2 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    /* case 1 */
    std::cout<<"-------case1:-------"<<std::endl;
    x1 /= 2;
    x1.show();
    x1 += 2;
    x1.show();
    /* case 2 */
    std::cout<<"-------case2:-------"<<std::endl;
    Mats<3, 3> x3 = x1 + x2;
    x3.show();
    /* case 3 */
    std::cout<<"-------case3:-------"<<std::endl;
    x1 += x2;
    x1.show();
    /* case 4 */
    std::cout<<"-------case4:-------"<<std::endl;
    float p = expt::dot(x1, x2);
    std::cout<<"p:"<<p<<std::endl;
    /* case 5 */
    std::cout<<"-------case5:-------"<<std::endl;

    {
        Mats<4, 3> x1 = {1, 2, 3,
                         4, 5, 6,
                         7, 8, 9,
                         1, 2, 1};
        Mats<3, 4> x2 = {1, 1, 1, 1,
                         1, 2, 1, 2,
                         0, 1, 0, 1};
        auto x3 = MatsFunc::mul(x1, x2);
        x3.show();

        CVector<4> v1 = {1, 2, 3, 4};
        RVector<3> v2 = {1, 2, 3};
        auto x4 = MatsFunc::mul(v1, v2);
        x4.show();
    }
    /* case 6 */
    std::cout<<"-------case6:-------"<<std::endl;
    /* msvc config: /bigobj */
#if 0 // compile cost too much time
    Mats<200, 200> x4(4);
    Mats<200, 200> x5(5);
    p = expt::dot(x4, x5);
    std::cout<<"p:"<<p<<std::endl;
#endif
    return;
}

void test_tensor()
{
    /* get sub space */
    std::cout<<"get sub space:"<<std::endl;
    {
        Tensori x(2, 3, 3);
        for (std::size_t i = 0; i < x.sizes.size(); i++) {
            std::cout<<x.sizes[i]<<",";
        }
        std::cout<<std::endl;
        for (std::size_t i = 0; i < x.shape.size(); i++) {
            std::cout<<x.shape[i]<<",";
        }
        std::cout<<std::endl;

        x.val = std::vector<int>{
                                    1, 2, 3,
                                    4, 5, 6,
                                    7, 8, 9,

                                    11, 12, 13,
                                    14, 15, 16,
                                    17, 18, 19
                                 };

        std::cout<<x(1, 1, 2)<<std::endl;
        x.printShape();

        Tensori x1 = x.sub(0);
        x1.printValue();

        x.reshape(3, 3, 2);
        Tensori x2 = x.sub(0, 1);
        x2.printValue();
    }
    /* assign sub space */
    std::cout<<"assign sub space:"<<std::endl;
    {
        Tensor x({3, 3, 3}, {1, 2, 3,
                             4, 5, 6,
                             7, 8, 9,

                             1, 1, 1,
                             1, 1, 1,
                             1, 1, 1,

                             2, 2, 2,
                             2, 2, 2,
                             2, 2, 2
                        });
        x.embedding({1, 1}, Tensor({1, 3}, {0, 9, 0}));
        x.embedding({2, 1}, Tensor({1, 3}, {2, 1, 0}));
        x.printValue();
    }
    /* reshpae */
    std::cout<<"reshpae:"<<std::endl;
    {
        Tensor x({2, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 0, 1, 1, 1, 1, 1, 1, 1});
        Tensor x1 = x.sub(0, 1).reshape(3, 1);
        x1.printShape();
        x1.printValue();
    }
    /* sum of sub space */
    std::cout<<"sum of sub space:"<<std::endl;
    {
        Tensor x(2, 3, 3);
        x = 7;
        x.printValue();
        std::cout<<"size = "<<x.size(1, 2)<<std::endl;
        std::cout<<"sum(1, 2) = "<<x.at(1, 2).sum()<<std::endl;
        std::cout<<"sum(1) = "<<x.at(1).sum()<<std::endl;
        std::cout<<"sum = "<<x.sum()<<std::endl;
    }
    /* max */
    std::cout<<"max:"<<std::endl;
    {
        Tensor x({3, 3, 3}, {1, 2, 3,
                             4, 5, 6,
                             7, 8, 9,

                             1, 1, 1,
                             1, 18, 1,
                             1, 1, 1,

                             2, 2, 2,
                             2, 19, 2,
                             2, 2, 2
                        });
        std::cout<<"max 0 = "<<x.at(0).max()<<std::endl;
        std::cout<<"max 1 = "<<x.at(1).max()<<std::endl;
        std::cout<<"max 2 = "<<x.at(2).max()<<std::endl;
        x.printValue(2, 1);
        x.sub(1, 1).printValue();
        std::cout<<"(1, 1, 1) = "<<x(1, 1, 1)<<std::endl;
    }
    /* block */
    std::cout<<"block:"<<std::endl;
    {
        Tensor x({6, 6}, {1, 2, 3, 4, 5, 6,
                          7, 8, 9, 6, 1, 2,
                          2, 6, 4, 5, 9, 3,
                          4, 1, 5, 7, 8, 4,
                          3, 7, 2, 8, 4, 9,
                          9, 0, 1, 3, 2, 5});
        Tensor y = x.block({2, 2}, {3, 3});
        y.printValue();
    }
    /* to string */
    std::cout<<"to string:"<<std::endl;
    {
        Tensor x1({3, 3}, {1.1, 0.02, 3.14, 4, 5.0, 6, 7, 8, 9.12});
        std::string s = x1.toString();
        std::cout<<s<<std::endl;
        Tensor x2 = Tensor::fromString(s);
        x2.printShape();
        x2.printValue();
    }
    return;
}

void test_product()
{
    Tensor x1({2, 2}, {1, 2, 3, 4});
    Tensor x2({2, 2}, {9, 9, 9, 9});
    Tensor y = Tensor::product2D(x1, x2);
    y.printValue();
    return;
}

void test_permute()
{
    Tensor x({2, 4, 3}, {
                 1, 2, 3,
                 1, 4, 5,
                 6, 0, 7,
                 8, 9, 2,

                 10, 11, 12,
                 13, 14, 15,
                 16, 17, 18,
                 19, 20, 21
             });
    /*
       permute:(0, 1, 2) -> (2, 0, 1)
       shape:  (2, 4, 3) -> (3, 2, 4)
       sizes:  (12, 3, 1) -> (8, 4, 1)

       value:
                 1, 1, 6, 8,
                 10, 13, 16, 19,

                 2, 4, 0, 9,
                 11, 14, 17, 20,

                 3, 5, 7, 2,
                 12, 15, 18, 21

    */

    Tensor y = x.permute(2, 0, 1);
    y.printShape();
    y.printValue();

    /* validate */
    Tensor y_(3, 2, 4);
    for (int i = 0; i < x.shape[0]; i++) {
        for (int j = 0; j < x.shape[1]; j++) {
            for (int k = 0; k < x.shape[2]; k++) {
                y_(k, i, j) = x(i, j, k);
            }
        }
    }
    y_.printShape();
    y_.printValue();

    /* matrix transpose */
    Tensor x1({2, 3}, {1, 2, 3,
                       4, 5, 6});
    Tensor y1 = x1.permute(1, 0);
    y1.printShape();
    y1.printValue();
    return;
}

void test_dft1d()
{
    int N = 8;
    /* signal: sin(x) */
    CTensor x(N);
    float value = 0;
    for (int i = 0; i < N; i++) {
        x[i].re = std::sin(value);
        std::cout<<x[i].re<<",";
        value += 1;
    }
    /* DFT1D */
    std::cout<<std::endl;
    std::cout<<"DFT:"<<std::endl;
    CTensor X = DFT::transform1D(x);
    for (int i = 0; i < N; i++) {
        X[i].print();
    }
    /* iDFT1D */
    std::cout<<"iDFT:"<<std::endl;
    CTensor xr = DFT::inverse1D(X);
    for (int i = 0; i < N; i++) {
        xr[i].print();
    }
    return;
}

void test_dft2d()
{
    /* signal */
    CTensor x({3, 3}, {Complex(1), Complex(-2), Complex(1),
                       Complex(-2), Complex(0), Complex(2),
                       Complex(1), Complex(2), Complex(1)
              });
    /* DFT2D */
    std::cout<<"DFT2D:"<<std::endl;
    CTensor X = DFT::transform2D(x);
    for (std::size_t i = 0; i < X.totalSize; i++) {
        X[i].print();
    }
    /* iDFT2D */
    std::cout<<"iDFT2D:"<<std::endl;
    CTensor xr = DFT::inverse2D(X);
    for (std::size_t i = 0; i < X.totalSize; i++) {
        xr[i].print();
    }
    return;
}

void test_fft1d()
{
    int N = 8;
    /* signal: sin(x) */
    CTensor x(N);
    float value = 0;
    for (int i = 0; i < N; i++) {
        x[i].re = std::sin(value);
        std::cout<<x[i].re<<",";
        value += 1;
    }
    /* FFT1D */
    std::cout<<std::endl;
    std::cout<<"FFT:"<<std::endl;
    CTensor X(N);
    FFT::transform1D(X, x);
    for (int i = 0; i < N; i++) {
        X[i].print();
    }
    return;
}

void test_quaternion()
{
    Quaternion q1, q2;
    Quaternion q = Quaternion::slerp(q1, q2, 0.8);
    return;
}

void test_inv()
{
    Tensor x({3, 3}, {8, 0, 1,
                      0, 2, 6,
                      1, 1, 3 });
    Tensor ix = LinAlg::inv(x);
    ix.printValue();
    Tensor I(3, 3);
    Tensor::MM::ikkj(I, x, ix);
    I.printValue();
    Tensor adjugate = ix*LinAlg::det(x);
    adjugate.printValue();
    return;
}

void test_eigen()
{
    Tensor x1({3, 3}, {8, 0, 1,
                       0, 2, 6,
                       1, 1, 3 });
    Tensor x2({3, 3}, {1, 1, 1,
                       1, 2, 10,
                       1, 10, 100 });

    Tensor vec(3, 3);
    Tensor value(3, 1);
    LinAlg::eigen(x2, vec, value, 2000);
    std::cout<<"eigen value:";
    value.printValue();
    std::cout<<"eigen vector:"<<std::endl;
    vec.printValue2D();
    return;
}

void test_transpose()
{
    Tensor x({3, 4}, {8, 0, 1, 9,
                      0, 2, 6, 7,
                      1, 1, 3, 0});
    x.printValue2D();
    Tensor xt = x.tr();
    xt.printValue2D();
    std::cout<<"x(1, 2)="<<x(1, 2)<<std::endl;
    std::cout<<"x(2, 1)="<<x(2, 1)<<std::endl;
    std::cout<<"xt(1, 2)="<<xt(1, 2)<<std::endl;
    return;
}

void test_BFGS1()
{
    class FnExp
    {
    public:
        FnExp(){}
        Tensor operator()(const Tensor &w, const Tensor &x)
        {
            /*
                y = exp(-(a*x1^2 + b*(x1 - x2) + c*x1*x2 + d*x2^2))
                dy/da = -x1^2*y
                dy/db = -(x1 - x2)*y
                dy/dc = -x1*x2*y
                dy/dd = -x2^2*y

                w0 = a
                w1 = b
                w2 = c
                w3 = d
            */
            Tensor y(1, 1);
            float s = w[0]*x[0]*x[0] +
                    w[1]*(x[0] - x[1]) +
                    w[2]*x[1]*x[1] +
                    w[3];
            y[0] = std::exp(-s);
            return y;
        }

        Tensor df(const Tensor &w, const Tensor &x)
        {
            Tensor dw(4, 1);
            float s = w[0]*x[0]*x[0] +
                    w[1]*(x[0] - x[1]) +
                    w[2]*x[1]*x[1] +
                    w[3];
            float y = std::exp(-s);
            dw[0] = -x[0]*x[0]*y;
            dw[1] = -(x[0] - x[1])*y;
            dw[2] = -x[0]*x[1]*y;
            dw[3] = -x[1]*x[1]*y;
            return dw;
        }
    };
    /* objective function:
       error = (f(w, x) - f(wt, x))^2
    */
    class FnObjective
    {
    public:
       FnExp f;
       Tensor wt;
    public:
        FnObjective(){}
        FnObjective(FnExp &f_, const Tensor &w)
            :f(f_),wt(w){}
        Tensor operator()(const Tensor &w, const Tensor &x)
        {
            Tensor delta = f(w, x) - f(wt, x);
            return delta*delta/2;
        }
        Tensor df(const Tensor &w, const Tensor &x)
        {
            Tensor delta = f(w, x) - f(wt, x);
            return f.df(w, x)*delta[0];
        }
    };

    Tensor w(4, 1);
    LinAlg::uniform(w, -1, 1);
    Tensor wt({4, 1}, {-2, 8, 6, -12});
    FnExp f;
    /* sample data */
    std::size_t N = 10;
    std::vector<Tensor> x(N, Tensor(2, 1));
    std::uniform_real_distribution<float> uniform(-10, 10);
    for (std::size_t i = 0; i < N; i++) {
        x[i][0] = uniform(LinAlg::Random::engine);
        x[i][1] = uniform(LinAlg::Random::engine);
    }
    /* objective */
    FnObjective objective(f, wt);
    /* optimizer */
    LinAlg::Armijo armijo(0.6, 0.4, 100);
    LinAlg::BFGS<LinAlg::Armijo> optimize(armijo, 10, 1e-3);
    optimize(objective, w, x);
    w.printValue();
    return;
}

class FnPolyomial
{
public:
    FnPolyomial(){}
    Tensor operator()(const Tensor &x)
    {
        Tensor y(1, 1);
        y[0] = x[0]*x[0]*x[0] + 2*x[0]*x[1] - x[1]*x[2] + x[2]*x[2]/2 - x[3]*x[0] + x[3]*x[3] +
                -x[0] + x[1] - x[2] + 1;
        return y;
    }
    Tensor df(const Tensor &x)
    {
        Tensor dy(4, 1);
        dy[0] = 3*x[0]*x[0] + 2*x[1] - x[3] - 1;
        dy[1] = 2*x[1] - x[2] + 1;
        dy[2] = -x[1] + x[2] - 1;
        dy[3] = -x[0] + 2*x[3];
        /*
            while dy/dx0=0, dy/dx1=0,dy/dx2=0,dy/dx3=0:
            x0 = 2/3, x0 = -0.5
            x1 = 0
            x2 = 1
            x3 = 1/3, x3 = -0.25
        */
        return dy;
    }
};

void test_BFGS2()
{
    std::cout<<"BFGS:"<<std::endl;
    LinAlg::Armijo armijo(0.6, 1, 30);
    LinAlg::Wolfe wolfe;
    LinAlg::BFGS<LinAlg::Wolfe> optimize(wolfe, 10000, 1e-8);
    Tensor x0(4, 1);
    LinAlg::uniform(x0, -1, 1);
    std::cout<<"x0:";
    x0.printValue();
    FnPolyomial fn;
    Tensor x = optimize.solve(fn, x0);
    std::cout<<"x:";
    x.printValue();
    std::cout<<"y="<<fn(x)[0]<<std::endl;;
    return;
}

void test_conjugate_gradient()
{
    std::cout<<"ConjugateGradient:"<<std::endl;
    LinAlg::Armijo armijo(0.6, 0.4, 30);
    LinAlg::Wolfe wolfe(0.1, 0.5, 100);
    LinAlg::ConjugateGradient<LinAlg::Armijo> optimize(armijo, 10000, 1e-8);
    Tensor x0(4, 1);
    LinAlg::uniform(x0, -1, 1);
    std::cout<<"x0:";
    x0.printValue();
    FnPolyomial fn;
    Tensor x = optimize(fn, x0);
    std::cout<<"x:";
    x.printValue();
    std::cout<<"y="<<fn(x)[0]<<std::endl;
    return;
}

void test_DFP()
{
    std::cout<<"DFP:"<<std::endl;
    LinAlg::DFP<LinAlg::Armijo> optimize(1000, 1e-3);
    Tensor x0(4, 1);
    LinAlg::uniform(x0, -1, 1);
    FnPolyomial fn;
    Tensor x = optimize(fn, x0);
    x.printValue();
    std::cout<<"y="<<fn(x)[0]<<std::endl;
    return;
}
int main()
{
#if 0
    test_static_matrix();
    test_lu();
    test_qr();
    test_det();
    test_svd();
    test_conv();
    test_permute();
    test_transpose();
#endif
    //test_tensor();
    //getchar();
    //std::cout<<"size of tensor = "<<sizeof (Tensor)<<std::endl;

    //test_dft1d();
    //test_fft1d();
    //test_dft2d();

    //test_gaussianElimination();
    //test_cholesky();
    //test_gaussSeidel();

    //test_inv();
    //test_eigen();
    //test_svd();
    //test_rank();
    //test_BFGS1();
    test_BFGS2();
    test_conjugate_gradient();
    return 0;
}



