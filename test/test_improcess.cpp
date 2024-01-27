#include <iostream>
#include <cmath>
#include <numeric>
#include <fstream>
#include <algorithm>
#include "../basic/util.hpp"
#include "../basic/tensor.hpp"
#include "../utils/clock.hpp"
#include "../improcess/image.hpp"
#include "../improcess/improcess.h"
#include "../improcess/bmp.hpp"
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
    imp::polygon(img, {{200, 100}, {400, 100}, {450, 200},
                                      {400, 300}, {200, 300}, {150, 200}},
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
    Tensor out = util::abs(y);
    imp::bound(out, 0, 255);
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
    imp::save(dst, "sobel3x3.bmp");
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
    imp::save(util::abs(dst), "laplacian3x3.bmp");
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
    util::uniform(epsilon, -1, 1);
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
    Tensor target;

    imp::copy(target, img, rect);

    imp::save(target, "data2_crystalmaiden.bmp");
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
#endif
    test_show();
	return 0;
}
