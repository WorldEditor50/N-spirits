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
    /* gray or rgb image */
    if (c != 1 && c != 3) {
        std::cout<<"c="<<c<<std::endl;
        return -2;
    }
    /* clamp */
    std::unique_ptr<uint8_t[]> data(new uint8_t[img.totalSize]);
    for (std::size_t i = 0; i < img.totalSize; i++) {
        if (img.val[i] > 255) {
            data[i] = 255;
        } else if (img.val[i] < 0) {
            data[i] = 0;
        } else {
            data[i] = img.val[i];
        }
    }
    /* compress to jpeg */
    int ret = improcess::Jpeg::save(fileName.c_str(), data.get(), h, w, c);
    if (ret < 0) {
        return -3;
    }
    return 0;
}

int improcess::convert2Gray(const Tensor &rgb, Tensor &gray)
{
    if (rgb.shape[HWC_C] != 3) {
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
