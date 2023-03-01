#include <iostream>
#include <tuple>
#include "basic/mat.h"
#include "basic/mats.hpp"
#include "basic/linearalgebra.h"
#include "basic/tensor.hpp"
#include "basic/complexnumber.h"
#include "basic/utils.h"
#include "utils/csv.h"
#include "utils/dataset.h"
#include "clock.hpp"
#include "kmeans.h"
#include "svm.h"
#include "gmm.h"
#include "./basic/simd.hpp"
#include "net/net.h"
#include "./net/optimizer.h"
#include "./net/layer.h"
#include "./net/loss.h"
#include "./net/conv.h"

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
    Utils::uniform(x9, 0, 9);
    Utils::uniform(x10, 0, 9);
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
void test_kmeans()
{
    /* load data */
    NumericDB db("D:/home/dataset/wine-clustering.csv");
    std::vector<Mat> x;
    db.load(x);
    /* clustering */
    KMeans model(3);
    model.cluster(x, 1000);
    /* predict */
    std::size_t label = model(x[0]);
    std::cout<<"label:"<<label<<std::endl;
    /* project to 2d-plane */
    LinearAlgebra::PCA pca;
    Mat x1;
    Mat::fromArray(x, x1);
    pca.fit(x1);
    Mat y;
    pca.project(x[0], 2, y);
    y.show();
    return;
}


void test_tensor()
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
    return;
}


void conv(Tensori &o, const Tensori &kernels, const Tensori &x, int stride=1, int padding=1)
{
    for (int n = 0; n < o.shape[0]; n++) {
        for (int i = 0; i < o.shape[1]; i++) {
            for (int j = 0; j < o.shape[2]; j++) {
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
                            //std::cout<<"(h, k)=("<<h<<","<<k<<"), "<<"(row, col)=("<<row<<","<<col<<")"<<std::endl;
                            /* sum up all input channel's convolution result */
                            o(n, i, j) += kernels(n, c, h, k)*x(c, row, col);
                        }
                    }
                }
                //std::cout<<"-----------------------------------"<<std::endl;
            }
        }
    }
    return;
}

void test_conv()
{

    /*
            0   0   0   0   0

            0   1   2   3   0        1     0     1           4     8     2

            0   4   5   6   0   *    0    -1     0      =    6     15    4

            0   7   8   9   0        1     0     1          -2     2    -4

            0   0   0   0   0



            0   0   0   0   0

            0   1   1   1   0        1     1     1           4     6     4

            0   1   1   1   0   *    1     1     1      =    6     9     6

            0   1   1   1   0        1     1     1           4     6     4

            0   0   0   0   0



            1   1   1        1     1           4     4

            1   1   1   *    1     1      =    4     4

            1   1   1


            0   0   0   0   0   0   0

            0   0   0   0   0   0   0                                 1    2     3     2    1

            0   0   1   1   1   0   0           1     1     1         2    4     6     4    2

            0   0   1   1   1   0   0      *    1     1     1      =  3    6     9     6    3

            0   0   1   1   1   0   0           1     1     1         2    4     6     4    2

            0   0   0   0   0   0   0                                 1    2     3     2    1

            0   0   0   0   0   0   0


    */

    Tensori o(1, 3, 3);
    Tensori kernels({1, 1, 3, 3}, {1, 0, 1, 0, -1, 0, 1, 0, 1});
    Tensori x({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    /* case 1: */
    conv(o, kernels, x);
    o.printValue();
    /* case 2: */
    o.zero();
    kernels.zero();
    x.zero();

    kernels.assign(1, 1, 1, 1, 1, 1, 1, 1, 1);
    x.assign(1, 1, 1, 1, 1, 1, 1, 1, 1);

    conv(o, kernels, x);
    o.printValue();
    /* case 3: */
    Tensori y1(1, 2, 2);
    Tensori x1({1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Tensori k1({1, 1, 2, 2}, {1, 1, 1, 1});
    conv(y1, k1, x1, 1, 0);
    y1.printValue();
    /* case 4: */
    Tensori y2(1, 5, 5);
    Tensori x2({1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Tensori k2({1, 1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    conv(y2, k2, x2, 1, 2);
    y2.printValue();
    /* case 5: */
    Tensori y3(1, 3, 3);
    conv(y3, k2, x2, 2, 2);
    y3.printValue();
    return;
}

void test_tensor_func()
{
    Tensor x({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor y = Tensor::func(x, [&](float xi)->float{ return xi*xi;});
    y.printValue();
    y = Tensor::func(x, std::exp);
    y.printValue();
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

void test_bpnn()
{
    using BPNN = Net<FcLayer, LayerNorm, FcLayer, LayerNorm, FcLayer>;
    BPNN bp(FcLayer(2, 128, true, ACTIVE_TANH),
            LayerNorm(128, 128, true, ACTIVE_SIGMOID),
            FcLayer(128, 128, true, ACTIVE_TANH),
            LayerNorm(128, 128, true, ACTIVE_SIGMOID),
            FcLayer(128, 1, true, ACTIVE_SIGMOID));
    Optimizer<BPNN, Optimize::RMSProp> optimizer(bp, 1e-3);
    /* train xor */
    std::vector<Tensor> x = {Tensor({2, 1}, {1, 1}),
                             Tensor({2, 1}, {1, 0}),
                             Tensor({2, 1}, {0, 1}),
                             Tensor({2, 1}, {0, 0})};
    std::vector<Tensor> yt = {Tensor({1, 1}, {0}),
                              Tensor({1, 1}, {1}),
                              Tensor({1, 1}, {1}),
                              Tensor({1, 1}, {0})};
    std::uniform_int_distribution<int> distribution(0, 3);
    auto t1 = Clock::tiktok();
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 4; j++) {
            int k = distribution(Utils::engine);
            /* forward */
            Tensor& y = bp(x[k]);
            /* loss */
            Tensor loss = Loss::MSE(y, yt[k]);
            /* backward */
            optimizer.backward(loss, x[k]);
        }
        /* update */
        optimizer.update();
    }
    auto t2 = Clock::tiktok();
    std::cout<<"train cost time:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    /* predict */
    for (int i = 0; i < 4; i++) {
        Tensor& y = bp(x[i]);
        y.printValue();
    }
    return;
}

void test_lenet5()
{
    /*
        (3,32,32)
        (6,28,28)
        (6,14,14)
        (16,10,10)
        (16,5,5)
        (120,1)
        (84,1)
        (10,1)
    */
    using LeNet5 = Net<Conv2d, MaxPooling2d, Conv2d, MaxPooling2d, FcLayer, FcLayer, SoftmaxLayer>;

    LeNet5 lenet5(Conv2d(3, 32, 32, 6, 5, 1, 0),
                  MaxPooling2d(6, 28, 28, 2, 2),
                  Conv2d(6, 14, 14, 16, 5, 1, 0),
                  MaxPooling2d(16, 10, 10, 2, 2),
                  FcLayer(16*5*5, 120, true, ACTIVE_SIGMOID),
                  FcLayer(120, 84, true, ACTIVE_SIGMOID),
                  SoftmaxLayer(84, 10));
    Optimizer<LeNet5, Optimize::RMSProp> optimizer(lenet5, 1e-3);
    Tensor x(3, 32, 32);
    Tensor yt(10, 1);
    for (std::size_t i = 0; i < 100; i++) {
        Utils::uniform(x, 0, 1);
        /* forward */
        Tensor& y = lenet5(x);
        y.printValue();
        /* loss */
        Utils::uniform(yt, 0, 1);
        Tensor loss = Loss::CROSS_EMTROPY(y, yt);
        /* backward */
        optimizer.backward(loss, x, yt);
    }
    return;
#if 0
    Conv2d& conv1 = std::get<0>(lenet5.layers);
    MaxPooling2d& pool1 = std::get<1>(lenet5.layers);
    Conv2d& conv2 = std::get<2>(lenet5.layers);
    MaxPooling2d& pool2 = std::get<3>(lenet5.layers);
    FcLayer& fc1 = std::get<4>(lenet5.layers);
    FcLayer& fc2 = std::get<5>(lenet5.layers);
    SoftmaxLayer& softmax = std::get<6>(lenet5.layers);

    Tensor& c1 = conv1.forward(x);
    c1.printShape();
    Tensor& p1 = pool1.forward(c1);
    p1.printShape();
    Tensor& c2 = conv2.forward(p1);
    c2.printShape();
    Tensor& p2 = pool2.forward(c2);
    p2.printShape();
    Tensor& f1 = fc1.forward(Tensor({int(p2.totalSize), 1}, p2.val));
    f1.printShape();
    Tensor& f2 = fc2.forward(f1);
    f2.printShape();
    Tensor& f3 = softmax.forward(f2);
    f3.printShape();
#endif
    return;
}

void test_mnist()
{
    using LeNet5 = Net<Conv2d, MaxPooling2d, Conv2d, MaxPooling2d, FcLayer, FcLayer, FcLayer>;
    LeNet5 lenet5(Conv2d(1, 28, 28, 6, 5, 1, 0, false, ACTIVE_LEAKRELU),
                  MaxPooling2d(6, 24, 24, 2, 2),
                  Conv2d(6, 12, 12, 16, 5, 1, 0, false, ACTIVE_LEAKRELU),
                  MaxPooling2d(16, 8, 8, 2, 2),
                  FcLayer(16*4*4, 120, true, ACTIVE_TANH),
                  LayerNorm(120, 84, true, ACTIVE_SIGMOID),
                  FcLayer(84, 10, true, ACTIVE_SIGMOID));

    /* load data */
    std::unique_ptr<uint8_t> datas = BinaryLoader::load("./dataset/train-images.idx3-ubyte");
    if (datas == nullptr) {
        std::cout<<"load training data failed."<<std::endl;
        return;
    }
    std::size_t N = BinaryLoader::byteswap(*(uint32_t*)(datas.get() + 4));
    std::vector<Tensor> x(N, Tensor(1, 28, 28));
    for (std::size_t n = 0; n < N; n++ ) {
        uint8_t* img = datas.get() + 16 + n * (28*28);
        for (int i = 0; i < 28; i++ ) {
            for (int j = 0; j < 28; j++ ) {
                x[n](0, i, j) = img[i + j*28] / 255.f;
            }
        }

    }
    /* load label */
    std::unique_ptr<uint8_t> labels = BinaryLoader::load("./dataset/train-labels.idx1-ubyte");
    if (labels == nullptr) {
        std::cout<<"load training label failed."<<std::endl;
        return;
    }
    std::vector<Tensor> yt(N, Tensor(10, 1));
    for (std::size_t n = 0; n < N; n++ ) {
        uint8_t* label = labels.get() + 8 + n;
        for (int i = 0; i < 10; i++ ) {
            if (*label == i) {
                yt[n](i, 0) = 1.0f;
            } else {
                yt[n](i, 0) = 0.0f;
            }
        }
        //yt[n].printValue();
    }
    /* train: max epoch = 1000, batch size = 100, learning rate = 1e-3 */
    Optimizer<LeNet5, Optimize::RMSProp> optimizer(lenet5, 1e-3);

    std::uniform_int_distribution<int> distribution(0, N - 1);
    for (std::size_t epoch = 0; epoch < 1000; epoch++) {
        for (std::size_t i = 0; i < 100; i++) {
            /* forward */
            int k = distribution(Utils::engine);
            Tensor& y = lenet5(x[k]);
            /* loss */
            Tensor loss = Loss::MSE(y, yt[k]);
            /* optimize */
            optimizer.backward(loss, x[k]);
        }
        optimizer.update();
    }

    /* predict */
    for (std::size_t i = 0; i < 10; i++) {
        /* forward */
        int k = distribution(Utils::engine);
        Tensor& y = lenet5(x[k]);
        int p = y.argmax();
        int t = yt[k].argmax();
        std::cout<<" target number:"<<t<<", predict number:"<<p<<std::endl;
    }
    return;
}

void test_simd_matmul()
{
#if defined(__AVX2__)
    std::cout<<"__m128/float:"<<sizeof (__m128)/sizeof (float)<<std::endl;
    std::cout<<"__m128d/double:"<<sizeof (__m128d)/sizeof (double)<<std::endl;
    std::cout<<"__m256/float:"<<sizeof (__m256)/sizeof (float)<<std::endl;
    std::cout<<"__m256d/double:"<<sizeof (__m256d)/sizeof (double)<<std::endl;

    /*

        N:       128         256         512        1024        2048
        cost:    0.000206s   0.002147s   0.014766s  0.211389s   2.58015s

    */
    int N = 2048;
    Tensor x(N, N);
    Tensor x1(N, N);
    Tensor x2(N, N);
    Utils::uniform(x1, -9, 9);
    Utils::uniform(x2, -9, 9);
    /* simd matmul */
    float* xPtr = x.ptr();
    float* x1Ptr = x1.ptr();
    float* x2Ptr = x2.ptr();
    /* 8-channel */
    {
        auto t1 = Clock::tiktok();
        simd::AVX2::matMul(xPtr, x.shape[0], x.shape[1],
                           x1Ptr, x1.shape[0], x1.shape[1],
                           x2Ptr, x2.shape[0], x2.shape[1]);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"simd matmul8 cost:"<<cost<<"s"<<std::endl;
        //x.printValue();
    }

    /* 64 channel */
    {
        x.zero();
        auto t1 = Clock::tiktok();
        simd::AVX2::matMul64(xPtr, x.shape[0], x.shape[1],
                             x1Ptr, x1.shape[0], x1.shape[1],
                             x2Ptr, x2.shape[0], x2.shape[1]);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"simd matmul64 cost:"<<cost<<"s"<<std::endl;
        //x.printValue();
    }

    /* 8-channel wrapper */
    {
        x.zero();
        auto t1 = Clock::tiktok();
        simd::wrap<float, simd::AVX>::matMul(xPtr, x.shape[0], x.shape[1],
                                             x1Ptr, x1.shape[0], x1.shape[1],
                                             x2Ptr, x2.shape[0], x2.shape[1]);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"avx2 wrapper matmul cost:"<<cost<<"s"<<std::endl;
        //x.printValue();
    }
    /* trivial matmul */
    if (true)
    {
        x.zero();
        auto t1 = Clock::tiktok();
        Tensor::Mul::ikkj(x, x1, x2);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"trivial matmul cost:"<<cost<<"s"<<std::endl;
        //x.printValue();
    }
#endif
    return;
}
void test_simd()
{
#if defined(__AVX2__)
    /* horizontal sum */
    {
        __m256 x1 = _mm256_setr_ps(1, 1, 1, 1, 1, 1, 1, 1);
        float s1 = simd::AVX2::reduce(x1);
        std::cout<<"horizontal sum float = "<<s1<<std::endl;
        __m256d x2 = _mm256_setr_pd(1, 2, 3, 4);
        double s2 = simd::AVX2::reduce(x2);
        std::cout<<"horizontal sum double = "<<s2<<std::endl;
    }
    /* max */
    Tensor x(100);
    Utils::uniform(x, 0, 100);
    x[50] = 101;
    x.printValue();
    std::cout<<"max:"<<simd::AVX2::max(x.ptr(), x.totalSize)<<std::endl;
    /* sum */
    Tensor x1(100, 100);
    x1.fill(1.0);
    std::cout<<"sum:"<<simd::AVX2::sum(x1.ptr(), x1.totalSize)<<std::endl;
    /* mean, variance */
    {
        Tensorsi_<float, simd::AVX2> x1(512, 512);
        Utils::uniform(x1, -1, 1);
        float u = x1.mean();
        auto t1 = Clock::tiktok();
        float sigma = x1.variance(u);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"cost time:"<<cost<<std::endl;
        std::cout<<"u:"<<u<<" sigma:"<<sigma<<std::endl;
    }
    /* exp */
    {
        Tensor y1(100);
        x.fill(3);
        simd::AVX2::exp(y1.ptr(), x.ptr(), x.totalSize);
        std::cout<<"simd exp:"<<std::endl;
        y1.printValue();
        std::cout<<"cmath exp:"<<std::endl;
        Tensor y2(100);
        Utils::exp(x, y2);
        y2.printValue();
    }
    /* sqrt */
    {
        Tensor x1 = Tensor::ones(16, 16, 16);
        Tensor x2 = Tensor::ones(16, 16, 16);
        x1 += x2;
        Tensor x3 = Tensor::ones(16, 16, 16);
        simd::AVX2::sqrt(x3.ptr(), x1.ptr(), x1.totalSize);
        x3.printValue();
    }
#endif
    return;
}

void test_simd_transpose()
{
#if defined(__AVX2__)
    int N = 4096;
    {
        Tensor_<double> x(N, N);
        Tensor_<double> y(N, N);
        Utils::uniform(x, 0, 9);
        auto t1 = Clock::tiktok();
        simd::AVX2::transpose(y.ptr(), N, N,
                              x.ptr(), N, N);
        auto t2 = Clock::tiktok();
        std::cout<<"avx2 transpose cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
        t1 = Clock::tiktok();
        Tensor_<double> x1 = x.tr();
        t2 = Clock::tiktok();
        std::cout<<"trivial transpose cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    } 
    if (0)
    {
        Tensor x(16, 16);
        Tensor y(16, 16);
        Utils::uniform(x, 0, 9);
        simd::AVX2::transpose(y.ptr(), 16, 16,
                              x.ptr(), 16, 16);
        Tensor::Mat::print(x);
        std::cout<<"simd transpose:"<<std::endl;
        Tensor::Mat::print(y);
    }
    return;
#endif
}

int main()
{
#if 0
    test_static_matrix();
    test_lu();
    test_qr();
    test_det();
    test_svd();
    test_kmeans();
    test_conv();
    test_permute();
    test_bpnn();
    test_lenet5();
    test_simd();
    test_simd_matmul();
#endif
    //test_simd_matmul();
    //test_bpnn();
    //test_lenet5();
    test_mnist();
    return 0;
}



