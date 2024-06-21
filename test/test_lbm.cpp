#include <iostream>
#include <string>
#include "../fluid/lbm.h"
#include "../fluid/eulerian.hpp"
#include "../improcess/improcess.h"

int main()
{
    int W = 320;
    int H = 240;
    int R = 12;
    Cylinder cylinder(H/2, W/5, R);
    Square square(H/2, W/5, R);
    Cross cr(H/2, W/5, R);
    ICylinder icylinder(H/2, W/5, R);
    LBM2d<Cylinder> lbm(H, W, // shape
                        cylinder,
                        1e-2, // relaxtion
                        /* boundary type : in coming direction (top, right, bottom, left) */
                        Tensor({4}, {0, 1, 0, 0}),
                        /* boundary value : wave reflection (ny, nx) */
                        Tensor({4, 2}, {0.0, 0.0,
                                         0.0, 0.0,
                                         0.0, 0.0,
                                         0.0, 0.1}));

    std::shared_ptr<uint8_t[]> rgb = nullptr;
    std::size_t totalsize = imp::BMP::size(H, W, 3);
    std::shared_ptr<uint8_t[]> bmp(new uint8_t[totalsize]);
    std::size_t N = 20000;
    lbm.solve(N, {0.8, 0.1, 0.1}, // color scaler
              [&](std::size_t i, Tensor &img){

        if (i % 20 == 0) {
            std::string fileName = "./cylinder/cylinder_" + std::to_string(i/20) + ".bmp";
            imp::fromTensor(img, rgb);
#if 0
            improcess::Jpeg::save(fileName.c_str(),
                                  rgb.get(),
                                  img.shape[0], img.shape[1], img.shape[2]);
#else
            imp::BMP::save(fileName, bmp, totalsize, rgb, H, W);
#endif
            std::cout<<"progress:"<<i<<"-->"<<N<<std::endl;
        }

    });
	return 0;
}
