# N-spirits




## Features

- SIMD (support SSE2, AVX2)

- Tensor

- Linear Algerbra

   invert, eigen, LU, QR, SVD,  DET, 

- Machine Learning

  kmeans , gmm, svm, BPNET, LeNet5, LSTM

- Image Process

  - spatial filter

    average blur, gaussian blur,  median blur, curvature filter, bilateral filter, sobel, laplace, canny

  - frequency filter

    FFT, HarrWavelet, LPF, gaussianHPF

  - geometry tranform

    move, flip, rotate, affine, nearest interpolation, bilinear interpolation, cubic interpolation

  - segmentation

    threshold, region grow, hough line, kmeans, gmm

  - morphology

    erode, dilate

  - feature

    histogram, LBP, SVD, HOG

- LBM (D2Q9)

## Requirements

- libjpeg

  https://github.com/stohrendorf/libjpeg-cmake.git

## Examples

### 1. image process 

#### 1.1 pixel cluster

```c++
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* pixel cluster */
    int h = img.shape[ns::HWC_H];
    int w = img.shape[ns::HWC_W];
    Tensor x = img;
    x.reshape(h*w, 3, 1);
    std::vector<Tensor> xi;
    x.toVector(xi);
    Kmeans model(16, 3, LinAlg::normL2);
    model.cluster(xi, 200, 0, 1e-6);
    /* classify */
    Tensor result(h, w, 3, 1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            Tensor p = img.sub(i, j);
            int k = model(p);
            Tensor &c = model.centers[k];
            result.at(i, j) = c;
        }
    }
    result.reshape(h, w, 3);
    Tensor dst = Tensor::concat(1, img, result);
    ns::show(dst);
```

![cluster](https://github.com/WorldEditor50/N-spirits/raw/main/images/cluster.bmp)

#### 1.2 template match

```c++
Tensor img = ns::load("D:/home/picture/dota2.bmp");
if (img.empty()) {
    std::cout<<"failed to load image."<<std::endl;
    return;
}
Tensor temp = ns::load("D:/home/picture/CrystalMaiden.bmp");
if (temp.empty()) {
    std::cout<<"failed to load temp image."<<std::endl;
    return;
}
/* resize */
Tensor dota2;
ns::resize(dota2, img, imp::imageSize(img)/4);
Tensor crystalMaiden;
ns::resize(crystalMaiden, temp, imp::imageSize(temp)/4);
auto t1 = Clock::tiktok();
/* template match */
ns::Rect rect;
ns::templateMatch(dota2, crystalMaiden, rect);
auto t2 = Clock::tiktok();
rect *= 4;
std::cout<<"templateMatch cost time:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
std::cout<<"x:"<<rect.x<<",y:"<<rect.y
    <<", width:"<<rect.width<<",height:"<<rect.height<<std::endl;
Tensor target;
ns::copy(target, img, rect);
ns::save(target, "data2_CrystalMaiden.bmp");
```

#### 1.3 sobel3x3

```c++
Tensor img = ns::load("D:/home/picture/dota2.bmp");
if (img.empty()) {
    std::cout<<"failed to load image."<<std::endl;
    return;
}
Tensor dst;
ns::sobel3x3(dst, img);
ns::save(dst, "sobel3x3.bmp");
```



### 2.  von karman vortex street

```c++
    int W = 320;
    int H = 240;
    int R = 12;
    Cylinder cylinder(H/2, W/5, R);
    Square square(H/2, W/5, R);
    Cross cr(H/2, W/5, R);
    ICylinder icylinder(H/2, W/5, R);
    LBM2d<Cylinder> lbm(H, W, // shape
                        cylinder,
                        1e-3, // relaxtion
                        /* boundary type : in coming direction (top, right, bottom, left) */
                        Tensor({4}, {0, 1, 0, 0}),
                        /* boundary value : wave reflection (ny, nx) */
                        Tensor({4, 2}, {0.0, 0.0,
                                        0.0, 0.0,
                                        0.0, 0.0,
                                        0.0, 0.1}));

    std::shared_ptr<uint8_t[]> rgb = nullptr;
    std::size_t totalsize = ns::BMP::size(H, W, 3);
    std::shared_ptr<uint8_t[]> bmp(new uint8_t[totalsize]);
    std::size_t N = 20000;
    lbm.solve(N, {0.8, 0.1, 0.1}, // color scaler
              [&](std::size_t i, Tensor &img){

        if (i % 20 == 0) {
            std::string fileName = "./cylinder/cylinder_" + std::to_string(i/20) + ".bmp";
            ns::fromTensor(img, rgb);
            ns::BMP::save(fileName, bmp, totalsize, rgb, H, W);
            std::cout<<"progress:"<<i<<"-->"<<N<<std::endl;
        }

    });
```

![cylinder_468](https://github.com/WorldEditor50/N-spirits/raw/main/images/cylinder_468.bmp)![cylinder_68](https://github.com/WorldEditor50/N-spirits/raw/main/images/cylinder_68.bmp)

#### 2.1 image to video

```sh
ffmpeg -i ./cylinder1/cylinder_%d.jpg -vcodec libx264 cylinder.avi
ffmpeg -i ./cylinder_%d.bmp -vcodec libx264 cylinder.avi
```

#### 2.2 reference:

- http://yangwc.com/2020/06/24/LBM/
- https://forum.taichi-lang.cn/t/homework0/506



### 3. LeNet5 (MNIST)

```c++
   using LeNet5 = Net<Conv2d, NMS, MaxPooling2d,
                       Conv2d, NMS, MaxPooling2d,
                       FcLayer, FcLayer, FcLayer>;
    LeNet5 lenet5(Conv2d(1, 28, 28, 6, 5, 1, 0, false, Fn_LeakyRelu),
                  NMS(6, 24, 24),
                  MaxPooling2d(6, 24, 24, 2, 2),
                  Conv2d(6, 12, 12, 16, 5, 1, 0, false, Fn_LeakyRelu),
                  NMS(16, 8, 8),
                  MaxPooling2d(16, 8, 8, 2, 2),
                  FcLayer(16*4*4, 120, true, Fn_Sigmoid),
                  FcLayer(120, 84, true, Fn_Tanh),
                  FcLayer(84, 10, true, Fn_Sigmoid));
    /* load data */
    MnistLoader mnist("./dataset/mnist/train-images.idx3-ubyte",
                      "./dataset/mnist/train-labels.idx1-ubyte");
    int ret = mnist.load();
    if (ret < 0) {
        return;
    }
    std::vector<Tensor> &x = mnist.x;
    std::vector<Tensor> &yt = mnist.yt;
    std::size_t N = mnist.N;
    /* train: max epoch = 1000, batch size = 100, learning rate = 1e-3 */
    Optimizer<LeNet5, Optimize::RMSProp> optimizer(lenet5, 1e-3, 1e-5, true);
    std::uniform_int_distribution<int> uniform(0, N - 1);
    auto t1 = Clock::tiktok();
    std::size_t maxEpoch = 2000;
    std::size_t batchSize = 256;
    Tensor totalLoss(10, 1);
    for (std::size_t epoch = 0; epoch < maxEpoch; epoch++) {
        totalLoss.zero();
        for (std::size_t i = 0; i < batchSize; i++) {
            /* forward */
            int k = uniform(LinAlg::Random::engine);
            Tensor& y = lenet5(x[k]);
            /* loss */
            Tensor loss = Loss::MSE(y, yt[k]);
            totalLoss += loss;
            /* optimize */
            optimizer.backward(loss, x[k]);
        }
        /* update */
        optimizer.update();
        totalLoss /= batchSize;
        std::cout<<"progress:---loss="<<totalLoss.norm2()<<", "
                <<epoch<<"/"<<maxEpoch<<"---"<<std::endl;
    }
    auto t2 = Clock::tiktok();
    std::cout<<"lenet5 training cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    /* predict */
    float count = 0;
    std::size_t Nt = 100;
    for (std::size_t i = 0; i < Nt; i++) {
        /* forward */
        int k = uniform(LinAlg::Random::engine);
        Tensor& y = lenet5(x[k]);
        int p = y.argmax();
        int t = yt[k].argmax();
        std::cout<<" target number:"<<t<<", predict number:"<<p<<std::endl;
        if (p == t) {
            count++;
        }
    }
    std::cout<<"accuracy:"<<count/float(Nt)<<std::endl;
```



## Dataset

- https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering
- https://www.kaggle.com/datasets/hojjatk/mnist-dataset

