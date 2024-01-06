#include <iostream>
#include <cmath>
#include <numeric>
#include <fstream>
#include <algorithm>
#include "../basic/util.hpp"
#include "../basic/tensor.hpp"
#include "../improcess/image.hpp"
#include "../improcess/jpegwrap.h"
#include "../improcess/improcess.h"
#include "../improcess/bmp.hpp"

void test_jpeg()
{
    std::shared_ptr<uint8_t[]> data = nullptr;
    int w = 0;
    int h = 0;
    int c = 0;
    int ret = imp::Jpeg::load("D:/home/picture/dota-2-official.jpg", data, h, w, c);
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
    Tensor img = imp::load("D:/home/picture/dota-2-official.jpg");
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

void test_convert2gray()
{
    /* load img(h, w, c) */
    Tensor img = imp::load("D:/home/picture/dota-2-official.jpg");
    if (img.empty() == true) {
        std::cout<<"load jpeg failed."<<std::endl;
        return;
    }
    /* convert to gray image */
    Tensor gray;
    int ret = imp::rgb2gray(gray, img);
    /* save img */
    ret = imp::save(gray, "data2_gray.jpg");
    if (ret < 0) {
        std::cout<<"save jpeg failed., ret = "<<ret<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

void test_bmp()
{
    std::shared_ptr<uint8_t[]> data = nullptr;
    int w = 0;
    int h = 0;
    int ret = imp::BMP::load("D:/home/picture/dota-2-official.bmp", data, h, w);
    if (ret != 0) {
        std::cout<<"load error = "<<ret<<std::endl;
        return;
    }

    ret = imp::BMP::save("dota2_write.bmp", data, h, w);
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
    int c = 0;
    int ret = imp::Jpeg::load("D:/home/picture/dota-2-official.jpg", data, h, w, c);
    if (ret < 0) {
        std::cout<<"load jpeg failed."<<std::endl;
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
    imp::graphic2D::line(img, {100, 100}, {320, 240}, imp::Color3(0, 255, 0), 1);

    imp::graphic2D::line(img, {320, 0}, {0, 240}, imp::Color3(255, 0, 0), 1);
    imp::save(img, "line.bmp");
    return;
}

void test_polygon()
{
    Tensor img(480, 640, 3);
    imp::graphic2D::polygon(img, {{200, 100}, {400, 100}, {450, 200},
                                      {400, 300}, {200, 300}, {150, 200}},
                                imp::Color3(0, 255, 0), 1);
    imp::save(img, "polygon.bmp");
    return;
}

void test_circle()
{
    Tensor img(480, 640, 3);
    imp::graphic2D::circle(img, {320, 240}, 200, imp::Color3(0, 255, 0), 10);
    imp::save(img, "circle.bmp");
    return;
}

void test_rect()
{
    Tensor img(480, 640, 3);
    imp::graphic2D::rectangle(img, {100, 100}, {300, 200}, imp::Color3(0, 255, 0), 10);
    imp::save(img, "rect.bmp");
    return;
}

void test_conv()
{
    /* load image */
    Tensor img = imp::load("D:/home/picture/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    /* conv2d */
    Tensor y;
#if 0
    Tensor kernel({3, 3}, {1, 0, -1,
                           0, 1,  0,
                          -1, 0,  1});

    Tensor kernel({3, 3}, {-1, 0, -1,
                            0, 4,  0,
                           -1, 0, -1});
#endif
    Tensor kernel({3, 3}, {-3,  0, 3,
                           -10, 0, 10,
                           -3,  0, 3});
    imp::conv2d(y, kernel, img, 1);
    imp::save(y, "filter.bmp");
    return;
}

void test_averageBlur()
{
    /* load image */
    Tensor img = imp::load("D:/home/picture/dota-2-official.bmp");
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
    Tensor img = imp::load("D:/home/picture/dota-2-official.bmp");
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
    Tensor img = imp::load("D:/home/picture/dota-2-official.bmp");
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
    Tensor img = imp::load("D:/home/picture/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    Tensor blur;
    imp::gaussianFilter3x3(blur, img);
    Tensor dst;
    imp::laplacian3x3(dst, blur);
    imp::save(dst, "laplacian3x3.bmp");
    return;
}

void test_prewitt()
{
    Tensor img = imp::load("D:/home/picture/dota-2-official.bmp");
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
    Tensor img = imp::load("D:/home/picture/dota-2-official.bmp");
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
    Tensor img = imp::load("D:/home/picture/dota-2-official.bmp");
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
    Tensor img = imp::load("D:/home/picture/dota-2-official.bmp");
    if (img.empty()) {
        std::cout<<"failed to load image."<<std::endl;
        return;
    }
    imp::Size size(img.shape[imp::HWC_H], img.shape[imp::HWC_W]);
    Tensor dst;
    imp::bilinearInterpolate(dst, img, size/2);
    imp::save(dst, "bilinear-interpolate.bmp");
    return;
}

void test_dilate()
{

}

void noise_img()
{
    Tensor img = imp::load("dota-2-official.bmp");
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
int main()
{
    noise_img();
    test_line();
    test_jpeg_to_tensor();
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
    test_rotate();
	return 0;
}
