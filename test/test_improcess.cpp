#include <iostream>
#include <cmath>
#include <numeric>
#include <fstream>
#include <algorithm>
#include "../basic/linalg.h"
#include "../basic/tensor.hpp"
#include "../utils/clock.hpp"
#include "../improcess/improcess_def.h"
#include "../improcess/improcess.h"
#include "../improcess/bmp.hpp"
#include "../ml/kmeans.h"
#include "../ml/gmm.h"
#include "../ml/svm.h"

#ifdef ENABLE_JPEG
#include "../improcess/jpegwrap/jpegwrap.h"

void test_jpeg()
{
    std::shared_ptr<uint8_t[]> data = nullptr;
    int w = 0;
    int h = 0;
    int c = 0;
    int ret = imp::Jpeg::load("./images/dota-2-official.jpg", data, h, w, c);
    if (ret < 0) {
        std::cout<<"load jpeg failed."<<std::endl;
        return;
    }

    ret = imp::Jpeg::save("data2.jpg", data.get(), h, w, c);
    if (ret < 0) {
        std::cout<<"save jpeg failed."<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

void test_jpeg_to_tensor()
{
    /* load img(h, w, c) */
    Tensor img = imp::load("./images/dota-2-official.jpg");
    if (img.empty() == true) {
        std::cout<<"load jpeg failed."<<std::endl;
        return;
    }
    /* take red channel */
    Tensor red(img);
    for (int i = 0; i < img.shape[0]; i++) {
        for (int j = 0; j < img.shape[1]; j++) {
            red(i, j, 1) = 0;
            red(i, j, 2) = 0;
        }
    }
    /* save img */
    int ret = imp::save(red, "data2_red.jpg");
    if (ret < 0) {
        std::cout<<"save jpeg failed., ret = "<<ret<<std::endl;
        return;
    }

    /* take green channel */
    Tensor green(img);
    for (int i = 0; i < img.shape[0]; i++) {
        for (int j = 0; j < img.shape[1]; j++) {
            green(i, j, 0) = 0;
            green(i, j, 2) = 0;
        }
    }
    /* save img */
    ret = imp::save(green, "data2_green.jpg");
    if (ret < 0) {
        std::cout<<"save jpeg failed., ret = "<<ret<<std::endl;
        return;
    }

    /* take blue channel */
    Tensor blue(img);
    for (int i = 0; i < img.shape[0]; i++) {
        for (int j = 0; j < img.shape[1]; j++) {
            blue(i, j, 0) = 0;
            blue(i, j, 1) = 0;
        }
    }
    /* save img */
    ret = imp::save(blue, "data2_blue.jpg");
    if (ret < 0) {
        std::cout<<"save jpeg failed., ret = "<<ret<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}
#endif

void test_convert2gray()
{
    /* load img(h, w, c) */
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"load bmp failed."<<std::endl;
        return;
    }
    /* convert to gray image */
    Tensor gray;
    int ret = imp::rgb2gray(gray, img);
    Tensor rgb;
    imp::gray2rgb(rgb, gray);
    /* save img */
    ret = imp::save(rgb, "data2_gray.bmp");
    if (ret < 0) {
        std::cout<<"save bmp failed., ret = "<<ret<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

void test_bmp()
{
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"load bmp failed."<<std::endl;
        return;
    }
    int ret = imp::save(img, "dota2_write.bmp");
    if (ret != 0) {
        std::cout<<"save error = "<<ret<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

void test_write_ppm()
{
    std::shared_ptr<uint8_t[]> data = nullptr;
    int w = 0;
    int h = 0;
    int ret = imp::BMP::load("./images/dota-2-official.bmp", data, h, w);
    if (ret < 0) {
        std::cout<<"load bmp failed."<<std::endl;
        return;
    }
    /* write ppm */
    ret = imp::PPM::save("dota2.ppm", data, h, w);
    if (ret != 0) {
        std::cout<<"save ppm error = "<<ret<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

void test_read_ppm()
{
    std::shared_ptr<uint8_t[]> data = nullptr;
    int w = 0;
    int h = 0;
    /* read ppm */
    int ret = imp::PPM::load("dota2.ppm", data, h, w);
    if (ret != 0) {
        std::cout<<"save ppm error = "<<ret<<std::endl;
        return;
    }
    ret = imp::BMP::save("dota2_read_ppm.bmp", data, h, w);
    if (ret < 0) {
        std::cout<<"save bmp failed."<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

void test_line()
{
    Tensor img(480, 640, 3);
    imp::line(img, {100, 100}, {320, 240}, imp::Color3(0, 255, 0), 1);

    imp::line(img, {320, 0}, {0, 240}, imp::Color3(255, 0, 0), 1);
    imp::save(img, "line.bmp");
    return;
}

void test_polygon()
{
    Tensor img(480, 640, 3);
    imp::polygon(img, {{200, 100}, {400, 100}, {450, 200}, {400, 300}, {200, 300}, {150, 200}},
                 imp::Color3(0, 255, 0), 1);
    imp::save(img, "polygon.bmp");
    return;
}

void test_circle()
{
    Tensor img(480, 640, 3);
    imp::circle(img, {320, 240}, 200, imp::Color3(0, 255, 0), 10);
    imp::save(img, "circle.bmp");
    return;
}

void test_rect()
{
    Tensor img(480, 640, 3);
    imp::rectangle(img, {100, 100}, {300, 200}, imp::Color3(0, 255, 0), 10);
    imp::save(img, "rect.bmp");
    return;
}

void test_conv()
{
    /* load image */
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* conv2d */
    Tensor y;

    Tensor kernel1({3, 3}, {1, 0, -1,
                            0, 1,  0,
                           -1, 0,  1});


    Tensor kernel2({3, 3}, {-1,  0, 1,
                             2,  0, -2,
                            -1,  0, 1});

    Tensor kernel3({3, 3}, {-1, 0, -1,
                             0, 4,  0,
                            -1, 0, -1});
    imp::conv2d(y, kernel3, img, 1);
    Tensor out = LinAlg::abs(y);
    imp::clamp(out, 0, 255);
    imp::save(out, "conv.bmp");
    return;
}

void test_averageBlur()
{
    /* load image */
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* average blur */
    Tensor dst;
    imp::averageFilter(dst, img, imp::Size(3, 3));
    /* save */
    imp::save(dst, "averageblur.bmp");
    return;
}

void test_medianBlur()
{
    /* load image */
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* median blur */
    Tensor dst;
    imp::medianFilter(dst, img, imp::Size(3, 3));
    /* save */
    imp::save(dst, "median.bmp");
    return;
}

void test_sobel()
{
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor dst;
    imp::sobel3x3(dst, img);
    //imp::save(dst, "sobel3x3.bmp");
    imp::show(dst);
    return;
}

void test_laplacian()
{
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor blur;
    imp::gaussianFilter3x3(blur, img);
    Tensor dst;
    imp::laplacian3x3(dst, blur);
    imp::save(LinAlg::abs(dst), "laplacian3x3.bmp");
    return;
}

void test_prewitt()
{
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor dst;
    imp::prewitt3x3(dst, img);
    imp::save(dst, "prewitt3x3.bmp");
    return;
}

void test_rotate()
{
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor dst;
    imp::rotate(dst, img, 45);
    imp::save(dst, "rotate_45.bmp");
    return;
}

void test_nearest_interpolation()
{
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    imp::Size size(img.shape[imp::HWC_H], img.shape[imp::HWC_W]);
    Tensor dst;
    imp::nearestInterpolate(dst, img, size*2);
    imp::save(dst, "nearestInterpolate_x2.bmp");
    return;
}

void test_bilinear_interpolation()
{
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    imp::Size size(img.shape[imp::HWC_H], img.shape[imp::HWC_W]);
    Tensor dst;
    imp::bilinearInterpolate(dst, img, size*4);
    imp::save(dst, "bilinear-interpolate.bmp");
    return;
}

void noise_img()
{
    Tensor img = imp::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor epsilon(img.shape);
    //Statistics::gaussian(epsilon, 0, 1);
    LinAlg::uniform(epsilon, -1, 1);
    Tensor dst = img + epsilon;
    imp::save(dst, "dota2_noise.bmp");
    return;
}

void test_make_border()
{
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor xo;
    imp::copyMakeBorder(xo, img, 2);
    imp::save(xo, "dota2_padding2.bmp");
    return;
}

void test_cut()
{
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor xo;
    imp::Rect rect(100, 100, 200, 200);
    imp::copy(xo, img, rect);
    imp::save(xo, "dota2_cut.bmp");
    return;
}

void test_autoThreshold()
{
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    imp::rgb2gray(gray, img);
    Tensor xo;
    imp::autoThreshold(xo, gray, 0, 255);
    if (xo.empty()) {
        std::cout<<"empty"<<std::endl;
        return;
    }
    Tensor rgb;
    imp::gray2rgb(rgb, xo);
    imp::save(rgb, "dota2_autoThreshold.bmp");
    return;
}

void test_otsuThreshold()
{
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    imp::rgb2gray(gray, img);
    Tensor xo;
    imp::otsuThreshold(xo, gray, 0, 255);
    if (xo.empty()) {
        std::cout<<"empty"<<std::endl;
        return;
    }
    Tensor rgb;
    imp::gray2rgb(rgb, xo);
    imp::save(rgb, "dota2_otsu.bmp");
    return;
}

void test_entropyThreshold()
{
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    imp::rgb2gray(gray, img);
    Tensor xo;
    imp::entropyThreshold(xo, gray, 0, 255);
    if (xo.empty()) {
        std::cout<<"empty"<<std::endl;
        return;
    }
    Tensor rgb;
    imp::gray2rgb(rgb, xo);
    imp::save(rgb, "dota2_entropy.bmp");
    imp::show("dota2_entropy.bmp");
    return;
}
void test_templateMatch()
{
    /*
        data2: (1920, 1080, 3)
        CrystalMaiden: (329, 315, 3)
        cost time: 165.36s
        resize:
            data2: (480, 270, 3)
            CrystalMaiden: (82, 79, 3)
            cost time: 0.657489s
    */
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor temp = imp::load("./images/crystalmaiden.bmp");
    if (temp.empty()) {
        std::cout<<"failed to load temp image."<<std::endl;
        return;
    }
    /* resize */
    Tensor dota2;
    imp::resize(dota2, img, imp::imageSize(img)/4);
    Tensor crystalMaiden;
    imp::resize(crystalMaiden, temp, imp::imageSize(temp)/4);
    auto t1 = Clock::tiktok();
    /* template match */
    imp::Rect rect;
    imp::templateMatch(dota2, crystalMaiden, rect);
    auto t2 = Clock::tiktok();
    rect *= 4;
    std::cout<<"templateMatch cost time:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    std::cout<<"x:"<<rect.x<<",y:"<<rect.y
            <<", width:"<<rect.width<<",height:"<<rect.height<<std::endl;

    imp::rectangle(img, Point2i(rect.y, rect.x), Point2i(rect.height, rect.width));
    imp::show(img);
    //Tensor target;
    //imp::copy(target, img, rect);
    //
    //imp::save(target, "data2_crystalmaiden.bmp");
    return;
}

void test_barycenter()
{
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    imp::rgb2gray(gray, img);
    Point2i center;
    imp::barycenter(gray, center);
    center = center.yx();
    std::cout<<"x:"<<center.x<<", y:"<<center.y<<std::endl;
    imp::circle(img, center, 8, imp::Color3(0, 255, 0));
    imp::save(img, "data2_barycenter.bmp");
    return;
}

void test_show()
{
    Tensor img = imp::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    imp::show(img);
    return;
}

void test_concat()
{
    Tensor img = imp::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load temp image."<<std::endl;
        return;
    }
    Tensor x1 = Tensor::concat(imp::HWC_H, img, img, img);
    imp::show(x1);
    Tensor x2 = Tensor::concat(imp::HWC_W, img, img, img);
    imp::show(x2);
    return;
}

void test_erode()
{
    Tensor img = imp::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor out;
    Tensor k33({3, 3}, { -1, 1,  -1,
                          1, 1,  1,
                         -1, 1,  -1 });
    imp::erode(out, img, k33);
    imp::show(out);
    return;
}

void test_histogram()
{
    Tensor img = imp::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    imp::showHistogram(img);
    return;
}

void test_kmeansPixelCluster()
{
    Tensor img = imp::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* pixel cluster */
    int h = img.shape[imp::HWC_H];
    int w = img.shape[imp::HWC_W];
    Tensor x = img;
    x.reshape(h*w, 3, 1);
    std::vector<Tensor> xi;
    x.toVector(xi);
#if 0
    Kmeans model(16, 3, LinAlg::normL2);
    Kmeans model(16, 3, LinAlg::cosine);
    Kmeans model(16, 3, LinAlg::Kernel::rbf);
#endif
    Kmeans model(16, 3, [](const Tensor &x1, const Tensor &x2)->float{
        return LinAlg::Kernel::laplace(x1, x2, 1.0);
    });
    model.cluster(xi, 1000, 0, 1e-6);
    /* centers */
    for (int i = 0; i < 16; i++) {
        model.centers[i].printValue();
    }
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
    /* pixels */
    Tensor pixels[16];
    for (int k = 0; k < 16; k++) {
        pixels[k] = Tensor(80, 80, 3);
        for (int i = 0; i < 80; i++) {
            for (int j = 0; j < 80; j++) {
                pixels[k].at(i, j) = model.centers[k];
            }
        }
    }
    Tensor pixelRow1 = Tensor::concat(1, pixels[0], pixels[1], pixels[2], pixels[3]);
    Tensor pixelRow2 = Tensor::concat(1, pixels[4], pixels[5], pixels[6], pixels[7]);
    Tensor pixelRow3 = Tensor::concat(1, pixels[8], pixels[9], pixels[10], pixels[11]);
    Tensor pixelRow4 = Tensor::concat(1, pixels[12], pixels[13], pixels[14], pixels[15]);

    Tensor pixelTable = Tensor::concat(0, pixelRow1, pixelRow2, pixelRow3, pixelRow4);
    Tensor dst = Tensor::concat(1, img, result, pixelTable);
    imp::save(dst, "kmeans_pixels_cluster.bmp");
    imp::show(dst);
    return;
}

void test_gmmPixelCluster()
{
    Tensor img = imp::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* pixel cluster */
    int h = img.shape[imp::HWC_H];
    int w = img.shape[imp::HWC_W];
    Tensor x = img;
    x.reshape(h*w, 3, 1);
    std::vector<Tensor> xi;
    x.toVector(xi);
    GMM model(16, 3);
    model.cluster(xi, 2000, 1e-6);
    for (int i = 0; i < 16; i++) {
        model.u[i].printValue();
    }
    /* classify */
    x.reshape(h, w, 3, 1);
    Tensor result(h, w, 3, 1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            Tensor p = x.sub(i, j);
            int k = model(p);
            result.at(i, j) = model.u[k];
        }
    }
    result.reshape(h, w, 3);
    /* pixels */
    Tensor pixels[16];
    for (int k = 0; k < 16; k++) {
        pixels[k] = Tensor(80, 80, 3);
        for (int i = 0; i < 80; i++) {
            for (int j = 0; j < 80; j++) {
                pixels[k].at(i, j) = model.u[k];
            }
        }
    }
    Tensor pixelRow1 = Tensor::concat(1, pixels[0], pixels[1], pixels[2], pixels[3]);
    Tensor pixelRow2 = Tensor::concat(1, pixels[4], pixels[5], pixels[6], pixels[7]);
    Tensor pixelRow3 = Tensor::concat(1, pixels[8], pixels[9], pixels[10], pixels[11]);
    Tensor pixelRow4 = Tensor::concat(1, pixels[12], pixels[13], pixels[14], pixels[15]);

    Tensor pixelTable = Tensor::concat(0, pixelRow1, pixelRow2, pixelRow3, pixelRow4);
    Tensor dst = Tensor::concat(1, img, result, pixelTable);
    imp::save(dst, "gmm_pixel_cluster.bmp");
    imp::show(dst);
    return;
}

void test_svmSegmentation()
{
    Tensor img = imp::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    int h = img.shape[imp::HWC_H];
    int w = img.shape[imp::HWC_W];
    Tensor x({32, 3, 1}, {192.809,102.132,60.0544,
                          137.285,115.915,119.617,
                          55.2037,31.1758,20.0327,
                          30.3464,18.6829,22.2349,
                          154.731,80.5239,48.0519,
                          232.267,237.611,236.204,
                          232.292,139.758,80.9688,
                          73.3562,46.9485,36.1936,
                          62.8855,124.351,170.202,
                          111.413,59.3942,34.6949,
                          25.2693,38.096,45.0215,
                          177.55,182.591,185.713,
                          14.2886,10.1763,11.6879,
                          38.6168,61.2919,75.1922,
                          120.659,79.6996,65.7762,
                          84.55,77.4764,80.6302,

                          137.159,116.533,120.635,
                          171.312,182.065,188.31,
                          33.5789,23.4374,25.038,
                          227.257,131.947,76.4435,
                          194.274,103.003,60.3912,
                          238.604,150.567,88.0304,
                          67.3103,41.4888,30.2706,
                          134.466,72.6122,48.2335,
                          16.0664,10.9622,12.9307,
                          232.469,236.613,234.68,
                          62.0919,123.628,169.949,
                          116.126,82.1803,70.5809,
                          66.7037,71.3558,85.0584,
                          162.679,84.56,49.9659,
                          30.9546,51.1428,59.5971,
                          99.9761,57.4912,36.9727

//                          230.556,238.295,237.882,
//                          154.325,80.3832,48.1483,
//                          31.2625,18.9064,21.894,
//                          71.2946,141.31,184.135,
//                          138.148,116.637,120.032,
//                          191.235,181.286,178.447,
//                          119.182,79.7098,66.0631,
//                          79.3307,80.369,94.2726,
//                          37.9445,60.8348,73.7385,
//                          14.4002,10.2842,11.967,
//                          74.0245,49.3418,39.3594,
//                          57.7311,32.6839,20.9997,
//                          232.279,139.495,80.554,
//                          192.738,102.061,59.9891,
//                          111.175,59.0094,34.305,
//                          24.9936,37.6517,44.5468

             });
    Tensor y({32, 1}, {-1, -1, -1, 1,
                       -1,  1, -1, 1,
                       -1, -1,  1, 1,
                        1, -1,  1, 1,

                       -1,  1,  1, -1,
                       -1, -1,  1, -1,
                        1,  1,  1, -1,
                       -1,  1,  1, -1

//                        1, -1,  1, -1,
//                       -1,  -1, 1, 1,
//                        1,  1,  1, -1,
//                       -1, -1, -1, 1
             });
    std::vector<Tensor> xi;
    x.toVector(xi);
    /* classifier */
    SVM svm([](const Tensor &x1, const Tensor &x2)->float{
        return LinAlg::Kernel::laplace(x1, x2, 1.0);
    }, 1e-4, 1);
    svm.fit(xi, y, 6000);
    /* segment */
    Tensor result(h, w, 3, 1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            Tensor p = img.sub(i, j);
            float s = svm(p);
            if (s > 0) {
                result.at(i, j) = p;
            }
        }
    }
    result.reshape(h, w, 3);
    /* display */
    Tensor dst = Tensor::concat(1, img, result);
    imp::save(dst, "svm_segmentation.bmp");
    imp::show(dst);
    return;
}

int main()
{
#ifdef ENABLE_JPEG
    test_jpeg_to_tensor();
#endif
#if 0
    test_bmp();
    noise_img();
    test_line();
    test_write_ppm();
    test_read_ppm();
    test_circle();
    test_polygon();
    test_rect();
    test_conv();
    test_averageBlur();
    test_sobel();
    test_laplacian();
    test_medianBlur();
    test_prewitt();
    test_nearest_interpolation();
    test_bilinear_interpolation();
    //test_rotate();
    test_make_border();
    test_cut();
    test_autoThreshold();
    test_templateMatch();
    test_barycenter();

    test_autoThreshold();
    test_otsuThreshold();
    test_entropyThreshold();
    test_show();
#endif
    //test_sobel();
    //test_erode();
    //test_histogram();
    //test_svmSegmentation();
    //test_kmeansPixelCluster();
    test_gmmPixelCluster();
	return 0;
}
