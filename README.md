# N-spirits
fun with cpp

## Features

- kmeans
- gmm
- svm
- lr
- BPNET
- LeNet5
- LSTM
- expression template
- matrix
- tensor
- basic linear algerbra (LU, QR, SVD, PCA, det)
- complex number
- simd (support SSE2, AVX2)
- LBM (D2Q9)

## Requirements

- libjpeg

  https://github.com/stohrendorf/libjpeg-cmake.git

## Examples

### 1. LeNet5 (MNIST)

```c++
    using LeNet5 = Net<Conv2d, MaxPooling2d,
                       Conv2d, MaxPooling2d,
                       FcLayer, LayerNorm, FcLayer>;
    LeNet5 lenet5(Conv2d(1, 28, 28, 6, 5, 1, 0, false, ACTIVE_LEAKRELU),
                  MaxPooling2d(6, 24, 24, 2, 2),
                  Conv2d(6, 12, 12, 16, 5, 1, 0, false, ACTIVE_LEAKRELU),
                  MaxPooling2d(16, 8, 8, 2, 2),
                  FcLayer(16*4*4, 120, true, ACTIVE_TANH),
                  LayerNorm(120, 84, true, ACTIVE_SIGMOID),
                  FcLayer(84, 10, true, ACTIVE_SIGMOID));
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
    std::cout<<"correct/total:"<<count/float(Nt)<<std::endl;
```

### 2.  von karman vortex street

```c++
    Cylinder cylinder(400/4, 400/2, 12);
    LBM2d<Cylinder> lbm(400, 400, // shape
                 cylinder,
                 0.005, // niu
                 Tensord({4}, {0, 0, 1, 0}),//boundary type
                 Tensord({4, 2}, {0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})// boundary value
                      );


    std::shared_ptr<uint8_t[]> raw = nullptr;
    lbm.solve(20000,
              {0.8, 0.1, 0.1}, // color scaler
              [&](std::size_t i, Tensor &img){

        if (i % 20 == 0) {
            std::string fileName = "./cylinder2/cylinder_" + std::to_string(i/20) + ".jpg";
            improcess::fromTensor(img, raw);

            improcess::Jpeg::save(fileName.c_str(),
                                  raw.get(),
                                  img.shape[0], img.shape[1], img.shape[2]);
            std::cout<<"progress:"<<i<<" / "<<20000<<std::endl;
        }
    });
```

#### 2.1 jpeg to video

```sh
ffmpeg -i ./cylinder1/cylinder_%d.jpg -vcodec libx264 cylinder1.avi
```

#### 2.2 reference:

- http://yangwc.com/2020/06/24/LBM/
- https://forum.taichi-lang.cn/t/homework0/506

## Dataset

- https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering
- https://www.kaggle.com/datasets/hojjatk/mnist-dataset

