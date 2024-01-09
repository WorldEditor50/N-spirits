#include "improcess.h"

int imp::fromTensor(InTensor x, std::shared_ptr<uint8_t[]> &img)
{
    if (img == nullptr) {
        img = std::shared_ptr<uint8_t[]>(new uint8_t[x.totalSize]);
    }
    for (std::size_t i = 0; i < x.totalSize; i++) {
        img[i] = imp::bound(x.val[i], 0, 255);
    }
    return 0;
}

std::unique_ptr<uint8_t[]> imp::fromTensor(InTensor x)
{
    std::unique_ptr<uint8_t[]> img(new uint8_t[x.totalSize]);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        img[i] = imp::bound(x.val[i], 0, 255);
    }
    return img;
}

std::shared_ptr<uint8_t[]> imp::tensor2Rgb(InTensor x)
{
    std::shared_ptr<uint8_t[]> img(new uint8_t[x.totalSize]);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        img[i] = imp::bound(x.val[i], 0, 255);
    }
    return img;
}


int imp::copyMakeBorder(OutTensor xo, InTensor xi, int padding)
{
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    xo = Tensor(h + 2*padding, w + 2*padding, c);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < c; k++) {
                xo(i + padding, j + padding, k) = xi(i, j, k);
            }
        }
    }
    return 0;
}

int imp::rgb2gray(OutTensor gray, InTensor rgb)
{
    if (rgb.shape[HWC_C] != 3) {
        return -1;
    }
    gray = Tensor(rgb.shape);
    for (int i = 0; i < rgb.shape[0]; i++) {
        for (int j = 0; j < rgb.shape[1]; j++) {
            float avg = (rgb(i, j, 0) + rgb(i, j, 1) + rgb(i, j, 2))/3;
            gray(i, j, 0) = avg;
            gray(i, j, 1) = avg;
            gray(i, j, 2) = avg;
        }
    }
    return 0;
}

int imp::gray2rgb(OutTensor rgb, InTensor gray)
{
    if (gray.shape[HWC_C] != 1) {
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


int imp::rgb2rgba(OutTensor rgba, InTensor rgb, int alpha)
{
    if (rgb.shape[HWC_C] != 3) {
        return -1;
    }
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    rgba = Tensor(h, w, 4);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            rgba(i, j, 0) = rgb(i, j, 0);
            rgba(i, j, 1) = rgb(i, j, 1);
            rgba(i, j, 2) = rgb(i, j, 2);
            rgba(i, j, 3) = alpha%256;
        }
    }
    return 0;
}

int imp::rgba2rgb(OutTensor rgb, InTensor rgba)
{
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    rgb = Tensor(h, w, 3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            rgb(i, j, 0) = rgba(i, j, 0);
            rgb(i, j, 1) = rgba(i, j, 1);
            rgb(i, j, 2) = rgba(i, j, 2);
        }
    }
    return 0;
}


int imp::transparent(OutTensor rgba, InTensor rgb, int alpha)
{
    rgb2rgba(rgba, rgb, alpha);
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (rgb(i, j, 0) == 255 &&
                    rgb(i, j, 1) == 255 &&
                    rgb(i, j, 2) == 255) {
                rgba(i, j, 3) = 0;
            }

        }
    }
    return 0;
}

Tensor imp::toTensor(int h, int w, int c, std::shared_ptr<uint8_t[]> &img)
{
    Tensor x(h, w, c);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x[i] = img[i];
    }
    return x;
}

Tensor imp::load(const std::string &fileName)
{
    Tensor img;
    if (fileName.empty()) {
        return img;
    }
    std::shared_ptr<uint8_t[]> data = nullptr;
    int h = 0;
    int w = 0;
    int c = 0;
    if (fileName.find(".jpg") != std::string::npos) {
        int ret = imp::Jpeg::load(fileName.c_str(), data, h, w, c);
        if (ret < 0) {
            return img;
        }
    } else if (fileName.find(".bmp") != std::string::npos) {
        c = 3;
        int ret = imp::BMP::load(fileName.c_str(), data, h, w);
        if (ret < 0) {
            return img;
        }
    } else if (fileName.find(".ppm") != std::string::npos) {
        c = 3;
        int ret = imp::PPM::load(fileName.c_str(), data, h, w);
        if (ret < 0) {
            return img;
        }
    } else {
        return img;
    }
    img = Tensor(h, w, c);
    for (std::size_t i = 0; i < img.totalSize; i++) {
        img.val[i] = data[i];
    }
    return img;
}

int imp::save(InTensor img, const std::string &fileName)
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
    std::shared_ptr<uint8_t[]> data = tensor2Rgb(img);
    /* save */
    if (fileName.find(".jpg") != std::string::npos) {
        int ret = imp::Jpeg::save(fileName.c_str(), data.get(), h, w, c);
        if (ret < 0) {
            return -3;
        }
    } else if (fileName.find(".bmp") != std::string::npos) {
        int ret = imp::BMP::save(fileName, data, h, w);
        if (ret < 0) {
            return -3;
        }
    } else if (fileName.find(".ppm") != std::string::npos) {
        int ret = imp::PPM::save(fileName, data, h, w);
        if (ret < 0) {
            return -3;
        }
    } else {
        int ret = imp::BMP::save(fileName, data, h, w);
        if (ret < 0) {
            return -3;
        }
    }
    return 0;
}

int imp::resize(OutTensor xo, InTensor xi, const imp::Size &size, int type)
{
    switch (type) {
    case INTERPOLATE_NEAREST:
        imp::nearestInterpolate(xo, xi, size);
        break;
    case INTERPOLATE_BILINEAR:
        imp::bilinearInterpolate(xo, xi, size);
        break;
    case INTERPOLATE_CUBIC:
        imp::cubicInterpolate(xo, xi, size);
        break;
    default:
        imp::nearestInterpolate(xo, xi, size);
        break;
    }
    return 0;
}
/*
    kernel: -1 --> ignore,
             0 --> background,
             1 --> roi
*/
int imp::erode(OutTensor xo, InTensor xi, InTensor kernel)
{
    int width = xi.shape[HWC_W];
    int height = xi.shape[HWC_H];
    int kernelSize = kernel.shape[HWC_H];
    xo = Tensor(xi.shape);
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            bool matched = true;
            for (int h = 0; h < kernelSize; h++) {
                for (int k = 0; k < kernelSize; k++) {
                    if (kernel(h, k) == -1) {
                        continue;
                    }
                    if (kernel(h, k) == 1) {
                        if (xi(i - 1 + h, j - 1 + k) != 0) {
                            matched = false;
                            break;
                        }
                    } else if (kernel(h, k) == 0) {
                        if (xi(i - 1 + h, j - 1 + k) != 255) {
                            matched = false;
                            break;
                        }
                    }
                }
            }
            xo(i, j) = matched ? 0 : 255;
        }
    }
    return 0;
}
/*
    kernel: -1 --> ignore,
             1 --> roi
*/
int imp::dilate(OutTensor xo, InTensor xi, InTensor kernel)
{
    int width = xi.shape[HWC_W];
    int height = xi.shape[HWC_H];
    int kernelSize = kernel.shape[HWC_H];
    xo = Tensor(xi.shape);
    xo.fill(255);
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            for (int h = 0; h < kernelSize; h++) {
                for (int k = 0; k < kernelSize; k++) {
                    if (kernel(h, k) == -1) {
                        continue;
                    }
                    if (kernel(h, k) == 1) {
                        if (xi(i - 1 + h, j - 1 + k) == 0) {
                            xo(i, j) = 0;
                            break;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int imp::traceBoundary(OutTensor xo, InTensor xi, std::vector<Point2i> &boundary)
{
    Tensor gray;
    rgb2gray(gray, xi);
    xo = Tensor(xi);
    int width = gray.shape[HWC_W];
    int height = gray.shape[HWC_H];
    /* boundary */
    for (int i = 0; i < height; i++) {
        gray(i, 0) = 255;
        gray(i, width - 1) = 255;
    }
    for (int i = 0; i < width; i++) {
        gray(0, i) = 255;
        gray(height - 1, i) = 255;;
    }

    Point2i startPoint;
    Point2i currentPoint;
    bool isAtStartPoint = true;
    int k = 0;
    Point2i directs[8] = {{-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}};
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (gray(i, j) != 0) {
                continue;
            }
            startPoint = Point2i(i, j);
            currentPoint = startPoint;
            isAtStartPoint = true;
            while ((startPoint.x != currentPoint.x || startPoint.y != currentPoint.y) ||
                   isAtStartPoint) {
                isAtStartPoint = false;
                Point2i pos = currentPoint + directs[k];
                int searchTime = 1;
                while (gray(pos.x, pos.y) == 255) {
                    k = (k + 1)%8;
                    pos = currentPoint + directs[k];
                    if (++searchTime >= 8) {
                        pos = currentPoint;
                        break;
                    }
                }
                currentPoint = pos;
                boundary.push_back(pos);
                xo(currentPoint.x, currentPoint.y, 0) = 0;
                xo(currentPoint.x, currentPoint.y, 1) = 0;
                xo(currentPoint.x, currentPoint.y, 2) = 0;

                k -= 2;
                k = k < 0 ? (k + 8) : k;
            }
            break;
        }
    }
    return 0;
}


int imp::findConnectedRegion(InTensor x, OutTensor mask, int &labelCount)
{
    int width = x.shape[HWC_W];
    int height = x.shape[HWC_H];


    return 0;
}

