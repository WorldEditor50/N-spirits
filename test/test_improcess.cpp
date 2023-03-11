#include <iostream>
#include <cmath>
#include <numeric>
#include <fstream>
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
    int ret = improcess::Jpeg::load("D:/home/picture/dota-2-official.jpg", data, h, w, c);
    if (ret < 0) {
        std::cout<<"load jpeg failed."<<std::endl;
        return;
    }

    ret = improcess::Jpeg::save("data2.jpg", data.get(), h, w, c);
    if (ret < 0) {
        std::cout<<"save jpeg failed."<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

void test_jpeg_to_tensor()
{
    Tensor img;
    /* load img(h, w, c) */
    int ret = improcess::load("D:/home/picture/dota-2-official.jpg", img);
    if (ret < 0) {
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
    ret = improcess::save("data2_red.jpg", red);
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
    ret = improcess::save("data2_green.jpg", green);
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
    ret = improcess::save("data2_blue.jpg", blue);
    if (ret < 0) {
        std::cout<<"save jpeg failed., ret = "<<ret<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

void test_convert2gray()
{
    Tensor img;
    /* load img(h, w, c) */
    int ret = improcess::load("D:/home/picture/dota-2-official.jpg", img);
    if (ret < 0) {
        std::cout<<"load jpeg failed."<<std::endl;
        return;
    }
    /* convert to gray image */
    Tensor gray;
    ret = improcess::rgb2gray(img, gray);
    /* save img */
    ret = improcess::save("data2_gray.jpg", gray);
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
    int ret = improcess::BMP::load("D:/home/picture/dota2.bmp", data, h, w);
    if (ret != 0) {
        std::cout<<"load error = "<<ret<<std::endl;
        return;
    }

    ret = improcess::BMP::save("dota2_write.bmp", data, h, w);
    if (ret != 0) {
        std::cout<<"save error = "<<ret<<std::endl;
        return;
    }
    std::cout<<"finished."<<std::endl;
    return;
}

int main()
{
    //test_jpeg_to_tensor();
    test_bmp();

	return 0;
}
