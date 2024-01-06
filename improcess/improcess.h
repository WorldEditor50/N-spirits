#ifndef IMPROCESS_H
#define IMPROCESS_H
#include <string>
#include <memory>
#include <functional>
#include "improcess_def.h"
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
    int save(InTensor img, const std::string &fileName);
    Tensor toTensor(int h, int w, int c, std::shared_ptr<uint8_t[]> &img);
    int fromTensor(InTensor x, std::shared_ptr<uint8_t[]> &img);
    std::unique_ptr<uint8_t[]> fromTensor(InTensor x);
    std::shared_ptr<uint8_t[]> tensor2Rgb(InTensor x);
    /* gray */
    int rgb2gray(OutTensor gray, InTensor rgb);
    int gray2rgb(OutTensor rgb, InTensor gray);
    /* rgb <--> rgba */
    int rgb2rgba(OutTensor rgba, InTensor rgb, int alpha=120);
    int rgba2rgb(OutTensor rgba, InTensor rgb);
    int transparent(OutTensor rgba, InTensor rgb, int alpha=120);
    /* resize */
    int resize(OutTensor xo, InTensor xi, const Size &size, int type=INTERPOLATE_NEAREST);
    /* erode */
    int erode(OutTensor xo, InTensor xi, InTensor kernel);
    /* dilate */
    int dilate(OutTensor xo, InTensor xi, InTensor kernel);
    /* trace boundary */
    int traceBoundary(OutTensor xo, InTensor xi, std::vector<Point2i> &boundary);
    /* connected region */
    int findConnectedRegion(InTensor x, OutTensor mask, int &labelCount);
}


#endif // IMPROCESS_H

