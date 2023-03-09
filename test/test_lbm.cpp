#include <iostream>
#include <string>
#include "../LBM/lbm.h"
#include "../improcess/improcess.h"

int main()
{
    Cylinder cylinder(100/2, 400/5, 12);
    Square square(100/2, 400/5, 12);
    Cross cr(200/2, 400/5, 12);
    ICylinder icylinder(200/2, 400/5, 12);
    LBM2d<ICylinder> lbm(200, 400, // shape
                         icylinder,
                         3e-3, // relaxtion
                         /* boundary type : in coming direction (top, right, bottom, left) */
                         Tensord({4}, {0, 1, 0, 0}),
                         /* boundary value : wave reflection (ny, nx) */
                         Tensord({4, 2}, {0.0, 0.0,
                                          0.0, 0.0,
                                          0.0, 0.0,
                                          0.0, 0.1}));

    std::shared_ptr<uint8_t[]> raw = nullptr;
    std::size_t N = 20000;
    lbm.solve(N, {0.8, 0.1, 0.1}, // color scaler
              [&](std::size_t i, Tensor &img){

        if (i % 20 == 0) {
            std::string fileName = "./cylinder2/cylinder_" + std::to_string(i/20) + ".jpg";
            improcess::fromTensor(img, raw);

            improcess::Jpeg::save(fileName.c_str(),
                                  raw.get(),
                                  img.shape[0], img.shape[1], img.shape[2]);
            std::cout<<"progress:"<<i<<"-->"<<N<<std::endl;
        }
    });
	return 0;
}
