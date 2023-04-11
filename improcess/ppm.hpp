#ifndef PPM_HPP
#define PPM_HPP
#include <string>
#include <fstream>
#include <memory>
#include <sstream>
#include <iostream>
namespace imp {

/*
    +---------------
    | P6
    +---------------
    | width height
    +---------------
    | maxcol
    +---------------
    | rgb
    +---------------
*/
class PPM
{
public:
    static int save(const std::string &fileName, std::shared_ptr<uint8_t[]> rgb, int h, int w)
    {
        if (fileName.empty()) {
            return -1;
        }
        if (rgb == nullptr) {
            return -2;
        }
        std::ofstream file(fileName, std::ios::binary|std::ios::out);
        if (file.is_open() == false) {
            return -3;
        }
        /* write head */
        std::string head = "P6 \n";
        /* width height */
        head += std::to_string(w) + " " + std::to_string(h) + "\n";
        /* max col value */
        head += "255\n";
        file.write(head.c_str(), head.size());
        /* write rgb */
        file.write((char*)rgb.get(), h*w*3);
        file.close();
        return 0;
    }
    static int load(const std::string &fileName, std::shared_ptr<uint8_t[]> &rgb, int &h, int &w)
    {
        if (fileName.empty()) {
            return -1;
        }
        std::ifstream file(fileName, std::ifstream::binary);
        if (file.is_open() == false) {
            return -2;
        }
        /* check format: P6 */
        std::string line;
        std::getline(file, line);
        line = line.substr(0, 2);
        if (line != "P6") {
            std::cout<<"format:"<<line<<std::endl;
            return -3;
        }
        /* read width, height */
        line.clear();
        std::getline(file, line);
        std::istringstream stream(line);
        std::string width;
        std::getline(stream, width, ' ');
        w = std::atoi(width.c_str());
        std::string height;
        std::getline(stream, height, ' ');
        h = std::atoi(height.c_str());
        //std::cout<<"width="<<width<<", height="<<height<<std::endl;
        /* maxcol */
        std::getline(file, line);
        /* read rgb */
        std::size_t totalsize = h*w*3;
        rgb = std::shared_ptr<uint8_t[]>(new uint8_t[totalsize]);
        file.read((char*)rgb.get(), totalsize);
        file.close();
        return 0;
    }
};

}

#endif // PPM_HPP
