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
#include "../ml/hopfieldnet.hpp"

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
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"load bmp failed."<<std::endl;
        return;
    }
    /* convert to gray image */
    Tensor gray;
    int ret = ns::rgb2gray(gray, img);
    Tensor rgb;
    ns::gray2rgb(rgb, gray);
    /* save img */
    ret = ns::save(rgb, "data2_gray.bmp");
    if (ret < 0) {
        std::cout<<"save bmp failed., ret = "<<ret<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

void test_bmp()
{
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"load bmp failed."<<std::endl;
        return;
    }
    int ret = ns::save(img, "dota2_write.bmp");
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
    int ret = ns::BMP::load("./images/dota-2-official.bmp", data, h, w);
    if (ret < 0) {
        std::cout<<"load bmp failed."<<std::endl;
        return;
    }
    /* write ppm */
    ret = ns::PPM::save("dota2.ppm", data, h, w);
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
    int ret = ns::PPM::load("dota2.ppm", data, h, w);
    if (ret != 0) {
        std::cout<<"save ppm error = "<<ret<<std::endl;
        return;
    }
    ret = ns::BMP::save("dota2_read_ppm.bmp", data, h, w);
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
    ns::line(img, {100, 100}, {320, 240}, ns::Color3(0, 255, 0), 1);

    ns::line(img, {320, 0}, {0, 240}, ns::Color3(255, 0, 0), 1);
    ns::save(img, "line.bmp");
    return;
}

void test_polygon()
{
    Tensor img(480, 640, 3);
    ns::polygon(img, {{200, 100}, {400, 100}, {450, 200}, {400, 300}, {200, 300}, {150, 200}},
                 ns::Color3(0, 255, 0), 1);
    ns::save(img, "polygon.bmp");
    return;
}

void test_circle()
{
    Tensor img(480, 640, 3);
    ns::circle(img, {320, 240}, 200, ns::Color3(0, 255, 0), 10);
    ns::save(img, "circle.bmp");
    return;
}

void test_rect()
{
    Tensor img(480, 640, 3);
    ns::rectangle(img, {100, 100}, {300, 200}, ns::Color3(0, 255, 0), 10);
    ns::save(img, "rect.bmp");
    return;
}

void test_conv()
{
    /* load image */
    Tensor img = ns::load("./images/dota-2-official.bmp");
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
    ns::conv2d(y, kernel3, img, 1);
    Tensor out = LinAlg::abs(y);
    ns::clamp(out, 0, 255);
    ns::save(out, "conv.bmp");
    return;
}

void test_averageBlur()
{
    /* load image */
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* average blur */
    Tensor dst;
    ns::averageBlur(dst, img, ns::Size(3, 3));
    /* save */
    ns::save(dst, "averageblur.bmp");
    return;
}

void test_medianBlur()
{
    /* load image */
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* median blur */
    Tensor dst;
    Tensor e = ns::Noise::saltPepper(img.shape[0],img.shape[1], img.shape[2], 0.01);
    Tensor noisedImg = img*e;
    ns::medianBlur(dst, noisedImg, ns::Size(3, 3));
    Tensor result = Tensor::concat(1, noisedImg, dst, img);
    ns::show(result);
    return;
}

void test_sobel()
{
    Tensor img = ns::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor dst;
    ns::sobel3x3(dst, img);
    //imp::save(dst, "sobel3x3.bmp");
    ns::show(dst);
    return;
}

void test_laplacian()
{
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor blur;
    ns::gaussianBlur3x3(blur, img);
    Tensor dst;
    ns::laplacian3x3(dst, blur);
    //imp::save(LinAlg::abs(dst), "laplacian3x3.bmp");
    ns::show(LinAlg::abs(dst));
    return;
}

void test_prewitt()
{
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor dst;
    ns::prewitt3x3(dst, img);
    ns::save(dst, "prewitt3x3.bmp");
    ns::show(dst);
    return;
}

void test_rotate()
{
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor dst;
    ns::rotate(dst, img, 45);
    ns::save(dst, "rotate_45.bmp");
    ns::show(dst);
    return;
}

void test_nearest_interpolation()
{
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    ns::Size size(img.shape[ns::HWC_H], img.shape[ns::HWC_W]);
    Tensor dst;
    ns::nearestInterpolate(dst, img, size*2);
    ns::save(dst, "nearestInterpolate_x2.bmp");
    return;
}

void test_bilinearInterpolate()
{
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    ns::Size size(img.shape[ns::HWC_H], img.shape[ns::HWC_W]);
    Tensor dst;
    ns::bilinearInterpolate(dst, img, size*2);
    ns::show(dst);
    return;
}

void noise_img()
{
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor epsilon(img.shape);
    //Statistics::gaussian(epsilon, 0, 1);
    LinAlg::uniform(epsilon, -1, 1);
    Tensor dst = img + epsilon;
    ns::save(dst, "dota2_noise.bmp");
    return;
}

void test_make_border()
{
    Tensor img = ns::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor xo;
    ns::copyMakeBorder(xo, img, 2);
    ns::save(xo, "dota2_padding2.bmp");
    return;
}

void test_cut()
{
    Tensor img = ns::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor xo;
    ns::Rect rect(100, 100, 200, 200);
    ns::copy(xo, img, rect);
    ns::save(xo, "dota2_cut.bmp");
    return;
}

void test_autoThreshold()
{
    Tensor img = ns::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::rgb2gray(gray, img);
    Tensor xo;
    ns::autoThreshold(xo, gray, 0, 255);
    if (xo.empty()) {
        std::cout<<"empty"<<std::endl;
        return;
    }
    Tensor rgb;
    ns::gray2rgb(rgb, xo);
    ns::save(rgb, "dota2_autoThreshold.bmp");
    return;
}

void test_otsuThreshold()
{
    Tensor img = ns::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::rgb2gray(gray, img);
    Tensor xo;
    ns::otsuThreshold(xo, gray, 0, 255);
    if (xo.empty()) {
        std::cout<<"empty"<<std::endl;
        return;
    }
    Tensor rgb;
    ns::gray2rgb(rgb, xo);
    ns::save(rgb, "dota2_otsu.bmp");
    return;
}

void test_entropyThreshold()
{
    Tensor img = ns::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::rgb2gray(gray, img);
    Tensor xo;
    ns::entropyThreshold(xo, gray, 0, 255);
    if (xo.empty()) {
        std::cout<<"empty"<<std::endl;
        return;
    }
    Tensor rgb;
    ns::gray2rgb(rgb, xo);
    ns::save(rgb, "dota2_entropy.bmp");
    ns::show("dota2_entropy.bmp");
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
    Tensor img = ns::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor temp = ns::load("./images/crystalmaiden.bmp");
    if (temp.empty()) {
        std::cout<<"failed to load temp image."<<std::endl;
        return;
    }
    /* resize */
    Tensor dota2;
    ns::resize(dota2, img, ns::imageSize(img)/4);
    Tensor crystalMaiden;
    ns::resize(crystalMaiden, temp, ns::imageSize(temp)/4);
    auto t1 = Clock::tiktok();
    /* template match */
    ns::Rect rect;
    ns::templateMatch(dota2, crystalMaiden, rect);
    auto t2 = Clock::tiktok();
    rect *= 4;
    std::cout<<"templateMatch cost time:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    std::cout<<"x:"<<rect.x<<",y:"<<rect.y
            <<", width:"<<rect.width<<",height:"<<rect.height<<std::endl;

    ns::rectangle(img, Point2i(rect.y, rect.x), Point2i(rect.height, rect.width));
    ns::show(img);
    //Tensor target;
    //imp::copy(target, img, rect);
    //
    //imp::save(target, "data2_crystalmaiden.bmp");
    return;
}

void test_barycenter()
{
    Tensor img = ns::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::rgb2gray(gray, img);
    Point2i center;
    ns::barycenter(gray, center);
    center = center.yx();
    std::cout<<"x:"<<center.x<<", y:"<<center.y<<std::endl;
    ns::circle(img, center, 8, ns::Color3(0, 255, 0));
    ns::save(img, "data2_barycenter.bmp");
    return;
}

void test_show()
{
    Tensor img = ns::load("./images/dota2.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    ns::show(img);
    return;
}

void test_concat()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load temp image."<<std::endl;
        return;
    }
    Tensor x1 = Tensor::concat(ns::HWC_H, img, img, img);
    ns::show(x1);
    Tensor x2 = Tensor::concat(ns::HWC_W, img, img, img);
    ns::show(x2);
    return;
}

void test_erode()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor k33({3, 3}, {  0, 1, 0,
                          1, 1, 1,
                          0, 1, 0 });
    Tensor k55({5, 5}, {  0, 0, 1, 0, 0,
                          0, 1, 1, 1, 0,
                          1, 1, 1, 1, 1,
                          0, 1, 1, 1, 0,
                          0, 0, 1, 0, 0});
    Tensor out;
    ns::erode(out, img, k33, 6);
    ns::save(out, "erode.bmp");
    ns::show(out);
    return;
}

void test_dilate()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor out;
    Tensor k33({3, 3}, {  0, 1, 0,
                          1, 1, 1,
                          0, 1, 0 });
    Tensor k55({5, 5}, {  0, 0, 1, 0, 0,
                          0, 1, 1, 1, 0,
                          1, 1, 1, 1, 1,
                          0, 1, 1, 1, 0,
                          0, 0, 1, 0, 0});
    ns::dilate(out, img, k33, 6);
    ns::save(out, "dilate.bmp");
    ns::show(out);
    return;
}

void test_histogram()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    ns::showHistogram(img);
    return;
}

void test_kmeansPixelCluster()
{
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
    Kmeans model(16, 3, [](const Tensor &x1, const Tensor &x2)->float{
        return LinAlg::Kernel::laplace(x1, x2, 1.0);
    });
    model.cluster(xi, 200, 0, 1e-6);
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
    Tensor dst = Tensor::concat(1, pixelTable, result, img);
    ns::save(dst, "./kmeans_pixels_cluster.bmp");
    ns::show(dst);
    return;
}

void test_gmmPixelCluster()
{
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
    GMM model(16, 3);
    model.cluster(xi, 1000, 1e-6);
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
    Tensor dst = Tensor::concat(1, pixelTable, result, img);
    ns::save(dst, "gmm_pixel_cluster.bmp");
    ns::show(dst);
    return;
}

void test_svmSegmentation()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    int h = img.shape[ns::HWC_H];
    int w = img.shape[ns::HWC_W];
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
    ns::save(dst, "svm_segmentation.bmp");
    ns::show(dst);
    return;
}

void test_houghLine()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor xo;
    int thres = 200;
    Tensor xs;
    ns::sobel3x3(xs, img);
    int ret = ns::houghLine(xo, xs, thres, 12, ns::Color3(0, 255, 0));
    if (ret != 0) {
        std::cout<<"no lines"<<std::endl;
        return;
    }
    Tensor result = Tensor::concat(ns::HWC_W, img, xo);
    ns::show(result);
    return;
}

void test_regionGrow()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    int h = img.shape[ns::HWC_H];
    int w = img.shape[ns::HWC_W];
    Tensor gray;
    ns::minGray(gray, img);
    Tensor mask(h, w);
    Point2i seed;
    ns::barycenter(gray, seed);
    uint8_t thres = 60;
    ns::otsu(gray, thres);
    std::cout<<"otsu thres:"<<(int)thres<<",seed:"<<seed.x<<","<<seed.y<<std::endl;
    ns::regionGrow(mask, gray, seed, {25, thres, 255});
    Tensor xo(h, w, 3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (mask(i, j) == 1) {
                xo.at(i, j) = {255, 0, 0};
            } else if (mask(i, j) == 2) {
                xo.at(i, j) = {0, 255, 0};
            } else if (mask(i, j) == 3) {
                xo.at(i, j) = {0, 0, 255};
            }
        }
    }
    Tensor result = Tensor::concat(1, xo, img);
    ns::show(result);
    return;
}

void test_LBP()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::rgb2gray(gray, img);
    /* LBP */
    Tensor lbp1;
    ns::LBP(lbp1, gray);
    /* circle LBP */
    Tensor lbp2;
    ns::circleLBP(lbp2, gray, 3, 8, false);
    /* multi scale block LBP */
    Tensor lbp3;
    ns::multiScaleBlockLBP(lbp3, gray, 9);
    Tensor result = Tensor::concat(1, lbp1, lbp2, lbp3, gray);
    ns::save(result, "lbp.bmp");
    ns::show(result);
    return;
}

void test_SVD()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::rgb2gray(gray, img);
    Tensor u;
    Tensor s;
    Tensor v;
    LinAlg::SVD::solve(gray, u, s, v, 1e-8, 1000);
    int n = s.argmax();
    s[n] *= 0.6;
    int N = std::min(s.shape[0], s.shape[1]);
    Tensor eigen(N, 1);
    for (int i = 0; i < N; i++) {
        eigen[i] = s(i, i);
    }
    /* img = u*s*v^T */
    Tensor r1(gray.shape);
    Tensor::MM::ikkj(r1, u, s);
    Tensor r2(gray.shape);
    Tensor::MM::ikjk(r2, r1, v);
    Tensor result = Tensor::concat(1, gray, r2);
    ns::save(result, "svd.bmp");
    ns::show(result);
    return;
}

void test_fft()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::maxGray(gray, img);
    /* FFT */
    Tensor spectrum;
    CTensor xf;
    ns::FFT2D(spectrum, xf, gray);
    /* filter */
    Tensor filter = ns::gaussHPF(xf.shape[0], xf.shape[1], 1);
    xf *= filter;
    /* iFFT */
    Tensor out;
    ns::iFFT2D(out, xf);
    Tensor result = Tensor::concat(1, spectrum, out, gray);
    ns::show(result);
    return;
}

void test_CMY()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor cmy;
    ns::RGB2CMY(cmy, img);
    ns::show(cmy);
    return;
}

void test_HSI()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor hsi;
    ns::RGB2HSI(hsi, img);
    ns::show(hsi);
    return;
}


void test_HSV()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor hsv;
    ns::RGB2HSV(hsv, img);
    ns::show(hsv);
    return;
}

void test_canny()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::meanGray(gray, img);
    Tensor x1;
    ns::canny(x1, gray, 30, 80);
    Tensor result = Tensor::concat(1, gray, x1);
    ns::show(result);
    return;
}

void test_HOG()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }

    Tensor gray;
    ns::meanGray(gray, img);
    Tensor hog;
    Tensor hist;
    ns::HOG(hog, hist, gray, 8, 8, 2);
    Tensor result = Tensor::concat(1, img, hog);
    ns::save(hog, "./hog.bmp");
    ns::save(result, "./hog_result.bmp");
    ns::show(result);
    return;
}

void test_affine()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    int h = img.shape[0];
    int w = img.shape[1];
    Tensor imgAffine;
    /* no change */
    Tensor op1({3, 3}, {1, 0, 0,
                        0, 1, 0,
                        0, 0, 1});
    /* translate */
    Tensor translateOp = ns::AffineOperator::translate(160, 80);
    /* scale */
    Tensor scaleOp = ns::AffineOperator::scale(0.5, 0.5);
    /* rotate */
    Tensor rotateOp = ns::AffineOperator::rotate(30);
    /* shear in x direction */
    Tensor shearXOp = ns::AffineOperator::shearX(0.5);
    /* shear in y direction */
    Tensor shearYOp = ns::AffineOperator::shearY(0.5);
    /* reflect about x */
    Tensor reflectXOp = ns::AffineOperator::flipX();
    /* reflect about y */
    Tensor reflectYOp = ns::AffineOperator::flipY();
    /* affine = Translate(Scale((Rotate)img))
              = (Translate*Scale*Rotate)img
    */
    Tensor affineOp = translateOp%scaleOp%rotateOp;
    affineOp.printValue2D();
    /* operation center */
    Tensor originCenter = ns::AffineOperator::center(0.5f*h, -0.5f*w);
    Tensor newCenter = ns::AffineOperator::center(0.5f*h, 0.5f*w);
    Tensor op = originCenter%affineOp%newCenter;
    ns::affine(imgAffine, img, op);
    ns::show(imgAffine);
    return;
}

void test_cubicInterpolate()
{
    Tensor img = ns::load("./images/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    ns::Size size(img.shape[ns::HWC_H], img.shape[ns::HWC_W]);
    Tensor dst;
    ns::cubicInterpolate(dst, img, size*2, ns::cubic::bspLine);
    ns::show(dst);
    return;
}

void test_scharr()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::maxGray(gray, img);
    Tensor dst;
    ns::scharr3x3(dst, gray);
    ns::show(dst);
    return;
}

void test_HarrWavelet()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::maxGray(gray, img);
    Tensor wavelet;
    /* wavelet transform */
    ns::HarrWavelet2D(wavelet, img, 4);
    /* filter */
    for (std::size_t i = 0; i < wavelet.totalSize; i++) {
        if (std::abs(wavelet[i]) < 10) {
            wavelet[i] = 0;
        }
    }
    /* invert */
    Tensor dst;
    ns::iHarrWavelet2D(dst, wavelet, 4);

    Tensor result = Tensor::concat(1, wavelet, dst, img);
    ns::show(result);
    return;
}

void test_eigen()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::maxGray(gray, img);
    Tensor img1;
    ns::copy(img1, gray, ns::Rect(0, 0, 300, 300));
    Tensor e;
    Tensor v;
    LinAlg::eigen(img1, e, v, 2000, 1e-4);
    Tensor g = LinAlg::diag(v);
    Tensor r1(img1.shape);
    Tensor::MM::ikkj(r1, e, g);
    Tensor r2(img1.shape);
    Tensor::MM::ikjk(r2, r1, e);
    Tensor result = Tensor::concat(1, r2, img1);
    ns::show(result);
    return;
}

void test_projectToSphere()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    int h = img.shape[0];
    Tensor sphere;
    ns::planeToSphere(sphere, img, h/2);
    ns::show(sphere);
    return;
}

void test_curvatrueBlur()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor xo;
    Tensor e = ns::Noise::saltPepper(img.shape[0],img.shape[1], img.shape[2], 0.01);
    Tensor noisedImg = img*e;
    ns::curvatureBlur3x3(xo, noisedImg);
    Tensor result = Tensor::concat(1, noisedImg, xo, img);
    ns::show(result);
    return;
}

void test_bilateralBlur()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor xo;
    Tensor e = ns::Noise::saltPepper(img.shape[0],img.shape[1], img.shape[2], 0.01);
    Tensor noisedImg = img*e;
    ns::bilateralBlur(xo, noisedImg, ns::Size(7, 7), 16, 16);
    Tensor result = Tensor::concat(1, noisedImg, xo, img);
    ns::show(result);
    return;
}

void test_harrisCorner()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::meanGray(gray, img);
    Tensor xo;
    ns::harrisCorner(xo, gray, 0.03);
    float thres = xo.max()*0.001;
    for (std::size_t i = 0; i < xo.totalSize; i++) {
        if (xo[i] < thres) {
            xo[i] = 0;
        }
    }
    int h = img.shape[0];
    int w = img.shape[1];
    Tensor mask(h, w, 3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (xo(i, j, 0) != 0) {
                mask(i, j, 1) = 255;
                img(i, j, 1) = 255;
            }
        }
    }
    Tensor result = Tensor::concat(1, mask, img);
    ns::show(result);
    return;
}

void test_hopfieldNet()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* resize */
    Tensor xr;
    ns::resize(xr, img, ns::Size(128, 128));
    Tensor gray;
    ns::meanGray(gray, xr);
    uint8_t thres = -1;
    ns::otsu(gray, thres);
    /* shift image */
    Tensor x(gray.totalSize, 1);
    for (std::size_t i = 0; i < gray.totalSize; i++) {
        x[i] = gray[i] >= thres ? 1 : -1;
    }
    /* model */
    HopfieldNet model(x.totalSize, 0.01);
    /* fit data */
    std::vector<Tensor> data;
    data.push_back(x);
    model.fit(data);
    /* noised data */
    Tensor xn(x.shape);
    std::uniform_real_distribution<float> uniform(0, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        float p = uniform(LinAlg::Random::generator);
        if (p > 0.8) {
            /* flip */
            xn[i] = x[i] > 0 ? -1 : 1;
        }
    }
    /* predict */
    Tensor xp = model(xn, 20);
    /* convert to image */
    Tensor result(gray.shape);
    for (std::size_t i = 0; i < result.totalSize; i++) {
        result[i] = xp[i] > 0 ? 255 : 0;
    }
    ns::show(result);
    return;
}

void test_normColor()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor normRgb;
    ns::normColor(normRgb, img);
    Tensor gray1;
    ns::minGray(gray1, normRgb);
    Tensor gray2;
    ns::gray2rgb(gray2, gray1);
    Tensor result = Tensor::concat(1, img, normRgb, gray2);
    ns::show(result);
    return;
}

void test_histogramStandardize()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::meanGray(gray, img);
    Tensor hist;
    ns::histogramStandardize(hist, gray);
    Tensor result = Tensor::concat(1, gray, hist);
    ns::show(result);
    return;
}

void test_histogramEqualize()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor gray;
    ns::meanGray(gray, img);
    Tensor hist;
    ns::histogramEqualize(hist, gray);
    Tensor result = Tensor::concat(1, gray, hist);
    ns::show(result);
    return;
}

void test_gammaTransform()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor g1;
    ns::gammaTransform(g1, img, 0.1, 0.1);
    Tensor g2;
    ns::gammaTransform(g2, img, 0.1, 0.2);
    Tensor g3;
    ns::gammaTransform(g3, img, 0.1, 0.3);
    Tensor result1 = Tensor::concat(1, img, g1, g2, g3);
    Tensor g4;
    ns::gammaTransform(g4, img, 0.1, 1);
    Tensor g5;
    ns::gammaTransform(g5, img, 0.1, 2);
    Tensor g6;
    ns::gammaTransform(g6, img, 0.1, 3);
    Tensor result2 = Tensor::concat(1, img, g4, g5, g6);
    Tensor result = Tensor::concat(0, result1, result2);
    ns::show(result);
    return;
}

void test_hsvHistogramEqualize()
{
    Tensor img = ns::load("./images/crystalmaiden.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor heq;
    ns::hsvHistogramEqualize(heq, img);
    Tensor result = Tensor::concat(1, img, heq);
    ns::show(result);
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
    //test_CMY();
    //test_HSI();
    //test_HSV();
    //test_rotate();
    //test_sobel();
    //test_laplacian();
    //test_histogram();
    //test_svmSegmentation();
    //test_kmeansPixelCluster();
    //test_gmmPixelCluster();
    //test_houghLine();
    //test_regionGrow();
    //test_LBP();
    //test_SVD();
    //test_erode();
    //test_dilate();
    //test_fft();
    //test_canny();
    //test_HOG();
    //test_affine();
    //test_cubicInterpolate();
    //test_HarrWavelet();
    //test_eigen();
    //test_projectToSphere();
    //test_curvatrueBlur();
    //test_bilateralBlur();
    //test_harrisCorner();
    //test_hopfieldNet();
    //test_normColor();
    //test_histogramStandardize();
    //test_histogramEqualize();
    //test_gammaTransform();
    test_hsvHistogramEqualize();
    return 0;
}
