#ifndef IMPROCESS_H
#define IMPROCESS_H
#include <string>
#include <memory>
#include <functional>
#include "jpegwrap.h"
#include "image.hpp"


namespace improcess {

    /* tensor shape: (h, w, c) */
    int load(const std::string &fileName, Tensor &img);
    int save(const std::string &fileName, const Tensor &img);
    int fromTensor(const Tensor &x, std::shared_ptr<uint8_t[]> &img);
    std::unique_ptr<uint8_t[]> fromTensor(const Tensor &x);
    /* gray */
    int rgb2gray(const Tensor& rgb, Tensor &gray);
    int gray2rgb(const Tensor& gray, Tensor &rgb);

}


#endif // IMPROCESS_H

