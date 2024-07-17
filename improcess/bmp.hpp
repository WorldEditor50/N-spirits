#ifndef BMP_HPP
#define BMP_HPP
#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <string.h>

namespace imp {
#pragma pack(push, 1)
struct BmpHead {
    uint16_t type;
    uint32_t filesize;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
};
#pragma pack(pop)

#pragma pack(push 1)
struct BmpInformation {
    uint32_t infosize;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t depth;
    uint32_t compression;
    uint32_t imagesize;
    uint32_t x;
    uint32_t y;
    uint32_t colorused;
    uint32_t colorimportant;
};
#pragma pack(pop)

class BMP
{
public:
    enum Offset {
        OFFSET_HEAD_TYPE = 0,
        OFFSET_HEAD_FILEISZE = 2,
        OFFSET_HEAD_RESERVED = 6,
        OFFSET_HEAD_IMAGEOFFSET = 10,
        OFFSET_INFO_INFOSIZE = 14,
        OFFSET_INFO_WIDTH = 18,
        OFFSET_INFO_HEIGHT = 22,
        OFFSET_INFO_PLANES = 26,
        OFFSET_INFO_DEPTH = 28,
        OFFSET_INFO_COMPRESS = 30,
        OFFSET_INFO_IMAGESIZE = 34,
        OFFSET_INFO_X = 38,
        OFFSET_INFO_Y = 42,
        OFFSET_INFO_COLORUSED = 46,
        OFFSET_INFO_COLORIMP = 50,
    };
    constexpr static int IMAGE_OFFSET = 54;
public:
    inline static int align4(int width, int channel) {return  (width*channel+3)/4*4;}

    inline static uint8_t byteOf(uint32_t x, uint8_t pos) { return ((uint8_t*)&x)[pos]; }

    inline static uint32_t fromByte(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3)
    {
        uint32_t x = 0;
        uint8_t *p = (uint8_t*)&x;
        p[0] = x0;
        p[1] = x1;
        p[2] = x2;
        p[3] = x3;
        return x;
    }

    inline static uint16_t fromByte(uint8_t x0, uint8_t x1)
    {
        uint16_t x = 0;
        uint8_t *p = (uint8_t*)&x;
        p[0] = x0;
        p[1] = x1;
        return x;
    }

    static void writeHeader(std::shared_ptr<uint8_t[]> bmp, const BmpHead &head)
    {
        /* type */
        bmp[OFFSET_HEAD_TYPE]     = byteOf(head.type, 0);
        bmp[OFFSET_HEAD_TYPE + 1] = byteOf(head.type, 1);
        /* file size */
        bmp[OFFSET_HEAD_FILEISZE + 0] = byteOf(head.filesize, 0);
        bmp[OFFSET_HEAD_FILEISZE + 1] = byteOf(head.filesize, 1);
        bmp[OFFSET_HEAD_FILEISZE + 2] = byteOf(head.filesize, 2);
        bmp[OFFSET_HEAD_FILEISZE + 3] = byteOf(head.filesize, 3);
        /* reserved */
        bmp[OFFSET_HEAD_RESERVED + 0] = 0;
        bmp[OFFSET_HEAD_RESERVED + 1] = 0;
        bmp[OFFSET_HEAD_RESERVED + 2] = 0;
        bmp[OFFSET_HEAD_RESERVED + 3] = 0;
        /* offset */
        bmp[OFFSET_HEAD_IMAGEOFFSET]     = byteOf(head.offset, 0);
        bmp[OFFSET_HEAD_IMAGEOFFSET + 1] = byteOf(head.offset, 1);
        bmp[OFFSET_HEAD_IMAGEOFFSET + 2] = byteOf(head.offset, 2);
        bmp[OFFSET_HEAD_IMAGEOFFSET + 3] = byteOf(head.offset, 3);
        return;
    }

    static void writeInfo(std::shared_ptr<uint8_t[]> bmp, const BmpInformation &info)
    {
        /* info size */
        bmp[OFFSET_INFO_INFOSIZE + 0] = byteOf(info.infosize, 0);
        bmp[OFFSET_INFO_INFOSIZE + 1] = byteOf(info.infosize, 1);
        bmp[OFFSET_INFO_INFOSIZE + 2] = byteOf(info.infosize, 2);
        bmp[OFFSET_INFO_INFOSIZE + 3] = byteOf(info.infosize, 3);
        /* height */
        bmp[OFFSET_INFO_HEIGHT + 0] = byteOf(info.height, 0);
        bmp[OFFSET_INFO_HEIGHT + 1] = byteOf(info.height, 1);
        bmp[OFFSET_INFO_HEIGHT + 2] = byteOf(info.height, 2);
        bmp[OFFSET_INFO_HEIGHT + 3] = byteOf(info.height, 3);
        /* width */
        bmp[OFFSET_INFO_WIDTH + 0] = byteOf(info.width, 0);
        bmp[OFFSET_INFO_WIDTH + 1] = byteOf(info.width, 1);
        bmp[OFFSET_INFO_WIDTH + 2] = byteOf(info.width, 2);
        bmp[OFFSET_INFO_WIDTH + 3] = byteOf(info.width, 3);
        /* plane */
        bmp[OFFSET_INFO_PLANES + 0] = byteOf(info.planes, 0);
        bmp[OFFSET_INFO_PLANES + 1] = byteOf(info.planes, 1);
        /* depth */
        bmp[OFFSET_INFO_DEPTH + 0] = byteOf(info.depth, 0);
        bmp[OFFSET_INFO_DEPTH + 1] = byteOf(info.depth, 1);
        /* compress */
        bmp[OFFSET_INFO_COMPRESS + 0] = byteOf(info.compression, 0);
        bmp[OFFSET_INFO_COMPRESS + 1] = byteOf(info.compression, 1);
        bmp[OFFSET_INFO_COMPRESS + 2] = byteOf(info.compression, 2);
        bmp[OFFSET_INFO_COMPRESS + 3] = byteOf(info.compression, 3);
        /* image size */
        bmp[OFFSET_INFO_IMAGESIZE + 0] = byteOf(info.imagesize, 0);
        bmp[OFFSET_INFO_IMAGESIZE + 1] = byteOf(info.imagesize, 1);
        bmp[OFFSET_INFO_IMAGESIZE + 2] = byteOf(info.imagesize, 2);
        bmp[OFFSET_INFO_IMAGESIZE + 3] = byteOf(info.imagesize, 3);
        /* x */
        bmp[OFFSET_INFO_X + 0] = byteOf(info.x, 0);
        bmp[OFFSET_INFO_X + 1] = byteOf(info.x, 1);
        bmp[OFFSET_INFO_X + 2] = byteOf(info.x, 2);
        bmp[OFFSET_INFO_X + 3] = byteOf(info.x, 3);
        /* y */
        bmp[OFFSET_INFO_Y + 0] = byteOf(info.y, 0);
        bmp[OFFSET_INFO_Y + 1] = byteOf(info.y, 1);
        bmp[OFFSET_INFO_Y + 2] = byteOf(info.y, 2);
        bmp[OFFSET_INFO_Y + 3] = byteOf(info.y, 3);
        /* color used */
        bmp[OFFSET_INFO_COLORUSED + 0] = byteOf(info.colorused, 0);
        bmp[OFFSET_INFO_COLORUSED + 1] = byteOf(info.colorused, 1);
        bmp[OFFSET_INFO_COLORUSED + 2] = byteOf(info.colorused, 2);
        bmp[OFFSET_INFO_COLORUSED + 3] = byteOf(info.colorused, 3);
        /* color important */
        bmp[OFFSET_INFO_COLORIMP + 0] = byteOf(info.colorimportant, 0);
        bmp[OFFSET_INFO_COLORIMP + 1] = byteOf(info.colorimportant, 1);
        bmp[OFFSET_INFO_COLORIMP + 2] = byteOf(info.colorimportant, 2);
        bmp[OFFSET_INFO_COLORIMP + 3] = byteOf(info.colorimportant, 3);
        return;
    }

    static uint32_t size(int h, int w, int c) {return 54 + h * align4(w, 3);}

    static int rgb24ToBmp(std::shared_ptr<uint8_t[]> rgb, int h, int w,
                         std::shared_ptr<uint8_t[]> &bmp, uint32_t &totalsize)
    {
        if (rgb == nullptr) {
            return -1;
        }
        /* allocate */
        int alignstep = align4(w, 3);
        totalsize = 54 + h*alignstep;
        bmp = std::shared_ptr<uint8_t[]>(new uint8_t[totalsize]);
        memset(bmp.get(), 0, totalsize);
        return BMP::rgb24ToBmp(rgb, h, w, bmp);
    }

    static int rgb24ToBmp(std::shared_ptr<uint8_t[]> rgb, int h, int w,
                         std::shared_ptr<uint8_t[]> &bmp)
    {
        if (rgb == nullptr || bmp == nullptr) {
            return -1;
        }
        int alignstep = align4(w, 3);
        /* write header */
        BmpHead head;
        head.type = 0x4d42;
        head.filesize = 54 + h*alignstep;
        head.reserved1 = 0;
        head.reserved2 = 0;
        head.offset = 54;
        memcpy(bmp.get(), &head, sizeof(BmpHead));
        /* write info */
        BmpInformation info;
        info.infosize = sizeof(BmpInformation);
        info.height = h;
        info.width  = w;
        info.planes = 1;
        info.depth  = 24;
        info.compression = 0;
        info.imagesize   = h*alignstep;
        info.x = 5000;
        info.y = 5000;
        info.colorused      = 0;
        info.colorimportant = 0;
        memcpy(bmp.get() + sizeof(BmpHead), &info, sizeof(BmpInformation));

        uint8_t *img = bmp.get() + sizeof(BmpHead) + sizeof(BmpInformation);
        int c = 3;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                img[i*alignstep + j*c]     = rgb[(h - 1 - i)*w*c + j*c + 2];
                img[i*alignstep + j*c + 1] = rgb[(h - 1 - i)*w*c + j*c + 1];
                img[i*alignstep + j*c + 2] = rgb[(h - 1 - i)*w*c + j*c];
            }
        }
        return 0;
    }

    static int save(const std::string& fileName, std::shared_ptr<uint8_t[]> rgb, int h, int w)
    {
        if (fileName.empty()) {
            return -1;
        }
        if (rgb == nullptr) {
            return -2;
        }
        std::fstream file(fileName, std::ios::binary|std::ios::out);
        if (file.is_open() == false) {
            return -3;
        }
        std::shared_ptr<uint8_t[]> bmp;
        uint32_t totalsize = 0;
        rgb24ToBmp(rgb, h, w, bmp, totalsize);
        file.write((char*)bmp.get(), totalsize);
        file.close();
        return 0;
    }

    static int save(const std::string& fileName,
                    std::shared_ptr<uint8_t[]> bmp, uint32_t totalsize,
                    std::shared_ptr<uint8_t[]> rgb, int h, int w)
    {
        if (fileName.empty()) {
            return -1;
        }
        if (bmp == nullptr) {
            return -2;
        }
        std::ofstream file(fileName, std::ios::binary|std::ios::out);
        if (file.is_open() == false) {
            return -3;
        }
        rgb24ToBmp(rgb, h, w, bmp);
        file.write((char*)bmp.get(), totalsize);
        file.close();
        return 0;
    }

    static int load(const std::string& fileName, std::shared_ptr<uint8_t[]> &rgb, int &h, int &w)
    {
        if (fileName.empty()) {
            return -1;
        }
        std::fstream file(fileName, std::ios::in|std::ios::binary);
        if (file.is_open() == false) {
            return -2;
        }
        /* read header */
        BmpHead header;
        file.read((char*)&header, sizeof (BmpHead));
        /* check format */
        if (header.type != 0x4d42) {
            std::cout<<"invalid format"<<std::endl;
            return -3;
        }
        /* check depth */
        BmpInformation info;
        file.read((char*)&info, sizeof (BmpInformation));
        if (info.depth != 24) {
            std::cout<<"invalid depth"<<std::endl;
            return -4;
        }
        /* width */
        w = info.width;
        /* height */
        h = info.height;
        /* totalsize */
        uint32_t totalsize = info.imagesize;
        std::shared_ptr<uint8_t[]> img = std::shared_ptr<uint8_t[]>(new uint8_t[totalsize]);
        /* read img */
        file.read((char*)img.get(), totalsize);
        file.close();
        /* reverse: B1G1R1B2G2R2 -> R2G2B2R1G1B1 */
        int c = 3;
        int alignstep = align4(w, c);
        rgb = std::shared_ptr<uint8_t[]>(new uint8_t[h*w*c]);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                rgb[(h - 1 - i)*w*c + j*c + 2] = img[i*alignstep + j*c];
                rgb[(h - 1 - i)*w*c + j*c + 1] = img[i*alignstep + j*c + 1];
                rgb[(h - 1 - i)*w*c + j*c]     = img[i*alignstep + j*c + 2];
            }
        }
        return 0;
    }
};

}
#endif // BMP_HPP
