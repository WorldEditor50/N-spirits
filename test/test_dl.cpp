#include <iostream>
#include <thread>
#include "../dl/net.hpp"
#include "../dl/optimizer.hpp"
#include "../dl/layer.hpp"
#include "../dl/loss.hpp"
#include "../dl/conv2d.hpp"
#include "../dl/lstm.hpp"
#include "../utils/clock.hpp"
#include "../utils/dataset.h"
#include "../dl/vae.hpp"
#include "../dl/transformer.hpp"

void convi(Tensori &o, const Tensori &kernels, const Tensori &x, int stride=1, int padding=1)
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
    convi(o, kernels, x);
    o.printValue();
    /* case 2: */
    o.zero();
    kernels.zero();
    x.zero();

    kernels.fill(1);
    x.fill(1);

    convi(o, kernels, x);
    o.printValue();
    /* case 3: */
    Tensori y1(1, 2, 2);
    Tensori x1({1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Tensori k1({1, 1, 2, 2}, {1, 1, 1, 1});
    convi(y1, k1, x1, 1, 0);
    y1.printValue();
    /* case 4: */
    Tensori y2(1, 5, 5);
    Tensori x2({1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Tensori k2({1, 1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    convi(y2, k2, x2, 1, 2);
    y2.printValue();
    /* case 5: */
    Tensori y3(1, 3, 3);
    convi(y3, k2, x2, 2, 2);
    y3.printValue();
    return;
}


void test_bpnn()
{
    using BPNN = Net<FcLayer, LayerNorm, FcLayer, LayerNorm, FcLayer>;
    BPNN bp(FcLayer(2, 32, true, Active_Tanh),
            LayerNorm(32, 32, true, Active_Sigmoid),
            FcLayer(32, 32, true, Active_Tanh),
            LayerNorm(32, 32, true, Active_Sigmoid),
            FcLayer(32, 1, true, Active_Sigmoid));
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
            int k = distribution(LinAlg::Random::engine);
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
                  FcLayer(16*5*5, 120, true, Active_Sigmoid),
                  FcLayer(120, 84, true, Active_Sigmoid),
                  SoftmaxLayer(84, 10, true));
    Optimizer<LeNet5, Optimize::RMSProp> optimizer(lenet5, 1e-3);
    Tensor x(3, 32, 32);
    Tensor yt(10, 1);
    auto t1 = Clock::tiktok();
    for (std::size_t i = 0; i < 1; i++) {
        LinAlg::uniform(x, 0, 1);
        /* forward */
        Tensor& y = lenet5(x);
        y.printValue();
        /* loss */
        LinAlg::uniform(yt, 0, 1);
        Tensor loss = Loss::CrossEntropy(y, yt);
        /* backward */
        optimizer.backward(loss, x, yt);
    }
    auto t2 = Clock::tiktok();
    std::cout<<"lenet5 train const:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
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
    using LeNet5 = Net<Conv2d, NMS, MaxPooling2d,
                       Conv2d, NMS, MaxPooling2d,
                       FcLayer, LayerNorm, FcLayer>;
    LeNet5 lenet5(Conv2d(1, 28, 28, 6, 5, 1, 0, false, Active_LeakyRelu),
                  NMS(6, 24, 24),
                  MaxPooling2d(6, 24, 24, 2, 2),
                  Conv2d(6, 12, 12, 16, 5, 1, 0, false, Active_LeakyRelu),
                  NMS(16, 8, 8),
                  MaxPooling2d(16, 8, 8, 2, 2),
                  FcLayer(16*4*4, 120, true, Active_Tanh),
                  LayerNorm(120, 84, true, Active_Sigmoid),
                  FcLayer(84, 10, true, Active_Sigmoid));
    /* load data */
    MnistLoader mnist("./dataset/train-images.idx3-ubyte",
                      "./dataset/train-labels.idx1-ubyte");
    int ret = mnist.load();
    if (ret < 0) {
        return;
    }
    std::vector<Tensor> &x = mnist.x;
    std::vector<Tensor> &yt = mnist.yt;
    std::size_t N = mnist.N;
    /* train: max epoch = 1000, batch size = 100, learning rate = 1e-3 */
    Optimizer<LeNet5, Optimize::RMSProp> optimizer(lenet5, 1e-3);
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    auto t1 = Clock::tiktok();
    for (std::size_t epoch = 0; epoch < 1000; epoch++) {
        for (std::size_t i = 0; i < 100; i++) {
            /* forward */
            int k = distribution(engine);
            Tensor& y = lenet5(x[k]);
            /* loss */
            Tensor loss = Loss::MSE(y, yt[k]);
            /* optimize */
            optimizer.backward(loss, x[k]);
        }
        /* update */
        optimizer.update();
        std::cout<<"progress:---"<<epoch<<"---"<<std::endl;
    }
    auto t2 = Clock::tiktok();
    std::cout<<"lenet5 training cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    /* predict */
    float count = 0;
    std::size_t Nt = 100;
    for (std::size_t i = 0; i < Nt; i++) {
        /* forward */
        int k = distribution(engine);
        Tensor& y = lenet5(x[k]);
        int p = y.argmax();
        int t = yt[k].argmax();
        std::cout<<" target number:"<<t<<", predict number:"<<p<<std::endl;
        if (p == t) {
            count++;
        }
    }
    std::cout<<"accuracy:"<<count/float(Nt)<<std::endl;
    return;
}


void test_lstm()
{
    using LSTMNET = Net<LSTM,
                        FcLayer, FcLayer>;
    LSTMNET lstm(LSTM(1, 16, 16),
                 FcLayer(16, 16, true, Active_Tanh),
                 FcLayer(16, 1, true, Active_Linear));

    Optimizer<LSTMNET, Optimize::RMSProp> optimizer(lstm, 1e-4);
    /* data */
    std::size_t N = 10000;
    std::vector<Tensor> x(N, Tensor(1, 1));
    float value = -500;
    for (std::size_t i = 0; i < N; i++) {
        x[i][0] = std::sin(value);
        value += 0.1;
    }
    /* train */
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_int_distribution<int> distribution(0, N - 9);
    for (std::size_t epoch = 0; epoch < 10000; epoch++) {
        int k = distribution(engine);
        std::get<0>(lstm.layers).reset();
        for (std::size_t i = 0; i < 8; i++) {
            /* forward */
            Tensor& y = lstm(x[k + i]);
            /* loss */
            Tensor loss = Loss::MSE(y, x[k + 8]);
            /* optimize */
            optimizer.backward(loss, x[k + i]);
        }
        /* update */
        optimizer.update();
    }
    /* predict */
    for (std::size_t i = 0; i < 4; i++) {
        std::get<0>(lstm.layers).reset();
        int k = distribution(engine);
        for (std::size_t j = 0; j < 8; j++) {
            Tensor& y = lstm(x[k + i]);
            float error = LinAlg::normL2(y, x[k + 8]);
            std::cout<<"target="<<x[k + 8][0]<<", predict="<<y[0]<<", error="<<error<<std::endl;
        }
    }
    return;
}

void test_alexnet()
{
    using AlexNet = Net<Conv2d, MaxPooling2d,
                        Conv2d, MaxPooling2d,
                        Conv2d, Conv2d, Conv2d, MaxPooling2d,
                        FcLayer, FcLayer, FcLayer>;
    AlexNet alexnet(Conv2d(3, 227, 227, 48, 11, 4, 2, false, Active_LeakyRelu),
                    MaxPooling2d(48, 56, 56, 3, 2),

                    Conv2d(48, 27, 27, 128, 5, 1, 2, false, Active_LeakyRelu),
                    MaxPooling2d(128, 27, 27, 3, 2),

                    Conv2d(128, 13, 13, 192, 3, 1, 1, false, Active_LeakyRelu),
                    Conv2d(192, 13, 13, 192, 3, 1, 1, false, Active_LeakyRelu),
                    Conv2d(192, 13, 13, 128, 3, 1, 1, false, Active_LeakyRelu),
                    MaxPooling2d(128, 13, 13, 3, 2),

                    FcLayer(128*6*6, 2048, true, Active_Tanh),
                    FcLayer(2048, 2048, true, Active_Sigmoid),
                    FcLayer(2048, 1000, true, Active_Sigmoid));

    Tensor x(3, 227, 227);
    /* alexnet forward cost:3.24633s */
    auto t1 = Clock::tiktok();
    alexnet(x);
    auto t2 = Clock::tiktok();
    std::cout<<"alexnet forward cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    Optimizer<AlexNet, Optimize::RMSProp> optimizer(alexnet, 1e-4);
    return;
}

void test_vgg16()
{
    using VGG16 = Net<Conv2d, Conv2d, MaxPooling2d,
                      Conv2d, Conv2d, MaxPooling2d,
                      Conv2d, Conv2d, Conv2d, MaxPooling2d,
                      Conv2d, Conv2d, Conv2d, MaxPooling2d,
                      Conv2d, Conv2d, Conv2d, MaxPooling2d,
                      FcLayer, FcLayer, FcLayer>;
    /* 16 conv2d */
    VGG16 vgg16(Conv2d(3,  224, 224, 64, 3, 1, 1, false, Active_LeakyRelu),
                Conv2d(64, 224, 224, 64, 3, 1, 1, false, Active_LeakyRelu),
                MaxPooling2d(64, 224, 224, 2, 2),

                Conv2d(64,  112, 112, 128, 3, 1, 1, false, Active_LeakyRelu),
                Conv2d(128, 112, 112, 128, 3, 1, 1, false, Active_LeakyRelu),
                MaxPooling2d(128, 112, 112, 2, 2),

                Conv2d(128, 56, 56, 256, 3, 1, 1, false, Active_LeakyRelu),
                Conv2d(256, 56, 56, 256, 3, 1, 1, false, Active_LeakyRelu),
                Conv2d(256, 56, 56, 256, 3, 1, 1, false, Active_LeakyRelu),
                MaxPooling2d(256, 56, 56, 2, 2),

                Conv2d(256, 28, 28, 512, 3, 1, 1, false, Active_LeakyRelu),
                Conv2d(512, 28, 28, 512, 3, 1, 1, false, Active_LeakyRelu),
                Conv2d(512, 28, 28, 512, 3, 1, 1, false, Active_LeakyRelu),
                MaxPooling2d(512, 28, 28, 2, 2),

                Conv2d(512, 14, 14, 512, 3, 1, 1, false, Active_LeakyRelu),
                Conv2d(512, 14, 14, 512, 3, 1, 1, false, Active_LeakyRelu),
                Conv2d(512, 14, 14, 512, 3, 1, 1, false, Active_LeakyRelu),
                MaxPooling2d(512, 7, 7, 2, 2),
                FcLayer(512*7*7, 4096, true, Active_Tanh),
                FcLayer(4096, 4096, true, Active_Sigmoid),
                FcLayer(4096, 1000, true, Active_Sigmoid));

    Optimizer<VGG16, Optimize::RMSProp> optimizer(vgg16, 1e-3);

    Tensor x(3, 224, 224);
    /*
          vgg16 forward cost:
                          @ naive conv: 194.117s
                          @ conv with im2col:  75.4293s
    */
    auto t1 = Clock::tiktok();
    vgg16(x);
    auto t2 = Clock::tiktok();
    std::cout<<"vgg16 forward cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    return;
}

void test_conv1d()
{
    /*
        x:
        +--+--+--+--+--+--+--+--+--+
        |  |  |  |  |  |  |  |  |  |
        +--+--+--+--+--+--+--+--+--+
        kernel:
        +--+--+--+
        |  |  |  |
        +--+--+--+
    */
    Tensor x = Tensor::ones(9);
    Tensor kernel = Tensor::ones(1, 3);
    int stride = 1;
    int padding = 1;
    int n = conv::out(9, 3, stride, padding);
    Tensor y(1, n);
    conv::conv1d(y, kernel, x, stride, padding);
    y.printValue();
    return;
}

int main()
{
#if 0
    test_conv1d();
    test_bpnn();
    test_lenet5();
    test_lstm();
    test_mnist();
    test_alexnet();
    test_vgg16();
#endif
    test_bpnn();
	return 0;
}
