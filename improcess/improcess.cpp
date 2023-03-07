#include "improcess.h"

int improcess::load(const std::string &fileName, Tensor &img)
{
    if (fileName.empty()) {
        return -1;
    }
    std::shared_ptr<uint8_t[]> data = nullptr;
    int h = 0;
    int w = 0;
    int c = 0;
    int ret = improcess::Jpeg::load(fileName.c_str(), data, h, w, c);
    if (ret < 0) {
        return -2;
    }
    img = Tensor(h, w, c);
    for (std::size_t i = 0; i < img.totalSize; i++) {
        img.val[i] = data[i];
    }
    return 0;
}

int improcess::save(const std::string &fileName, const Tensor &img)
{
    if (fileName.empty()) {
        return -1;
    }
    int h = img.shape[HWC_H];
    int w = img.shape[HWC_W];
    int c = img.shape[HWC_C];
    /* rgb image */
    if (c != 3) {
        std::cout<<"c="<<c<<std::endl;
        return -2;
    }
    /* clamp */
    std::unique_ptr<uint8_t[]> data = fromTensor(img);
    /* compress to jpeg */
    int ret = improcess::Jpeg::save(fileName.c_str(), data.get(), h, w, c);
    if (ret < 0) {
        std::cout<<"Jpeg::save failed"<<std::endl;
        return -3;
    }
    return 0;
}

int improcess::fromTensor(const Tensor &x, std::shared_ptr<improcess::uint8_t[]> &img)
{
    if (img == nullptr) {
        img = std::shared_ptr<uint8_t[]>(new uint8_t[x.totalSize]);
    }
    for (std::size_t i = 0; i < x.totalSize; i++) {
        if (x.val[i] > 255) {
            img[i] = 255;
        } else if (x.val[i] < 0) {
            img[i] = 0;
        } else {
            img[i] = x.val[i];
        }
    }
    return 0;
}


std::unique_ptr<improcess::uint8_t[]> improcess::fromTensor(const Tensor &x)
{
    std::unique_ptr<uint8_t[]> img(new uint8_t[x.totalSize]);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        if (x.val[i] > 255) {
            img[i] = 255;
        } else if (x.val[i] < 0) {
            img[i] = 0;
        } else {
            img[i] = x.val[i];
        }
    }
    return img;
}


int improcess::rgb2gray(const Tensor &rgb, Tensor &gray)
{
    if (isRGB(rgb) == false) {
        return -1;
    }
    gray = Tensor(rgb.shape);
    for (int i = 0; i < rgb.shape[0]; i++) {
        for (int j = 0; j < rgb.shape[1]; j++) {
            float avg = (rgb(i, j, 1) + rgb(i, j, 1) + rgb(i, j, 2))/3;
            gray(i, j, 0) = avg;
            gray(i, j, 1) = avg;
            gray(i, j, 2) = avg;
        }
    }
    return 0;
}

int improcess::gray2rgb(const Tensor &gray, Tensor &rgb)
{
    if (isGray(gray) == false) {
        return -1;
    }
    rgb = Tensor(gray.shape[HWC_H], gray.shape[HWC_W], 3);
    for (int i = 0; i < rgb.shape[0]; i++) {
        for (int j = 0; j < rgb.shape[1]; j++) {
            float pixel = gray(i, j, 0);
            rgb(i, j, 0) = pixel;
            rgb(i, j, 1) = pixel;
            rgb(i, j, 2) = pixel;
        }
    }
    return 0;
}

