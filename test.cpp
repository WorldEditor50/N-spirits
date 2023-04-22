#include <iostream>
#include <tuple>
#include "basic/mat.h"
#include "basic/mats.hpp"
#include "basic/linearalgebra.h"
#include "basic/tensor.hpp"
#include "basic/complexnumber.h"
#include "basic/statistics.h"
#include "basic/fft.h"
#include "utils/csv.h"
#include "utils/dataset.h"
#include "utils/clock.hpp"

void test_lu()
{
    Mat x(3, 3, { 1, 1, 1,
                  0, 0.5, -2,
                  0, 1, 1});
    Mat xi;
    LinearAlgebra::LU::inv(x, xi);
    xi.show();
    std::cout<<"test inverse:"<<std::endl;
    Mat I(x.rows, x.cols);
    Mat::Multiply::ikkj(I, x, xi);
    I.show();
    return;
}

void test_det()
{
    float value;
    Mat x1(3, 3, {1, 1, 1,
                  1, 2, 3,
                  1, 5, 1});
    LinearAlgebra::det(x1, value);
    std::cout<<"det:"<<value<<std::endl;



    Mat x2(4, 4, {1, 1, 1, 2,
                  1, 2, 3, 0,
                  0, 5, 1, -1,
                  1, 0, -3, 1});
    LinearAlgebra::det(x2, value);
    std::cout<<"det:"<<value<<std::endl;
    return;
}

void test_qr()
{
    Mat x(3, 3, {1, 1, 1,
                 1, 2, 3,
                 1, 5, 1});
    Mat q;
    Mat r;
    LinearAlgebra::QR::solve(x, q, r);
    std::cout<<"Q:"<<std::endl;
    q.show();
    std::cout<<"R:"<<std::endl;
    r.show();
    return;
}

void test_svd()
{
    Mat x1(3, 3, {1, 1, 1,
                 1, 2, 3,
                 1, 5, 1});

    Mat x(10, 5, {0.0162, 0.878, 0.263, 0.0955, 0.359,
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
    Mat u;
    Mat v;
    Mat s;
    LinearAlgebra::SVD::solve(x, u, s, v);
    std::cout<<"U:"<<std::endl;
    u.show();
    std::cout<<"S:"<<std::endl;
    s.show();
    std::cout<<"V:"<<std::endl;
    v.show();
    /* x = U*S*V^T */
    Mat y = u*s*v.tr();
    std::cout<<"x:"<<std::endl;
    x.show();
    std::cout<<"y:"<<std::endl;
    y.show();
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
    Mat::Multiply::ikkj(x5, x1, x2);
    /* [[22, 28], [49, 64]] */
    x5.show();
    std::cout<<"kikj:"<<std::endl;
    Mat x6(2, 2);
    /* (3, 2) * (3, 2) = (2, 2) */
    Mat::Multiply::kikj(x6, x4, x2);
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
    Mat::Multiply::ikjk(x7, x1, x3);
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
    Mat::Multiply::kijk(x8, x4, x3);
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
    Statistics::uniform(x9, 0, 9);
    Statistics::uniform(x10, 0, 9);
    Mat::Multiply::kikj(x11, x9, x10);
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
        x.at(1, 1) = Tensor({1, 3}, {0, 9, 0});
        x.at(2, 1) = Tensor({1, 3}, {2, 1, 0});
        x.printValue();
    }
    /* reshpae */
    {
        Tensor x({2, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 0, 1, 1, 1, 1, 1, 1, 1});
        Tensor x1 = x.sub(0, 1).reshape(3, 1);
        x1.printShape();
        x1.printValue();
    }
    /* sum of sub space */
    {
        Tensor x(2, 3, 3);
        x = 7;
        x.printValue();
        std::cout<<"size = "<<x.size(1, 2)<<std::endl;
        std::cout<<"sum(1, 2) = "<<x.sum(1, 2)<<std::endl;
        std::cout<<"sum(1) = "<<x.sum(1)<<std::endl;
        std::cout<<"sum = "<<x.sum()<<std::endl;
    }
    /* max */
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
        std::cout<<"max 0 = "<<x.max(0)<<std::endl;
        std::cout<<"max 1 = "<<x.max(1)<<std::endl;
        std::cout<<"max 2 = "<<x.max(2)<<std::endl;
        x.printValue(2, 1);
        x.sub(1, 1).printValue();
        std::cout<<"(1, 1, 1) = "<<x(1, 1, 1)<<std::endl;
    }
    /* row and column */
    {
        Tensor x({3, 3}, {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9});
        std::cout<<"row:";
        Tensor r = x.row(1);
        r.printValue();
        std::cout<<"col:";
        Tensor c = x.column(1);
        c.printValue();
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
#endif
    //test_tensor();
    //std::cout<<"size of tensor = "<<sizeof (Tensor)<<std::endl;
    //test_dft2d();
    test_dft1d();
    test_fft1d();
    return 0;
}



