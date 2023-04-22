#ifndef IMPROCESS_H
#define IMPROCESS_H
#include <string>
#include <memory>
#include <functional>
#include "jpegwrap.h"
#include "bmp.hpp"
#include "ppm.hpp"
#include "image.hpp"
#include "graphic2d.h"
#include "geometrytransform.h"
#include "filter.h"

namespace imp {

    /* tensor shape: (h, w, c) */
    Tensor load(const std::string &fileName);
    int save(const Tensor &img, const std::string &fileName);
    Tensor toTensor(int h, int w, int c, std::shared_ptr<uint8_t[]> &img);
    int fromTensor(const Tensor &x, std::shared_ptr<uint8_t[]> &img);
    std::unique_ptr<uint8_t[]> fromTensor(const Tensor &x);
    std::shared_ptr<imp::uint8_t[]> tensor2Rgb(const Tensor &x);
    /* gray */
    int rgb2gray(const Tensor& rgb, Tensor &gray);
    int gray2rgb(const Tensor& gray, Tensor &rgb);
    /* resize */
    int resize(Tensor &dst, Tensor &src, const Size &size);
}


#endif // IMPROCESS_H

