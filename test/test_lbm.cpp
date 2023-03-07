#include <iostream>
#include <string>
#include "../LBM/lbm.h"
#include "../improcess/improcess.h"

int main()
{
    Cylinder cylinder(400/4, 400/2, 12);
    LBM2d<Cylinder> lbm(400, 400, // shape
                      cylinder,
                      0.005, // niu
                      Tensord({4}, {0, 0, 1, 0}),//boundary type
                      Tensord({4, 2}, {0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}) // boundary value
                      );


    std::shared_ptr<uint8_t[]> raw = nullptr;
    lbm.solve(20000,
              {0.8, 0.1, 0.1}, // color scaler
              [&](std::size_t i, Tensor &img){

        if (i % 20 == 0) {
            std::string fileName = "./cylinder2/cylinder_" + std::to_string(i/20) + ".jpg";
            improcess::fromTensor(img, raw);

            improcess::Jpeg::save(fileName.c_str(),
                                  raw.get(),
                                  img.shape[0], img.shape[1], img.shape[2]);
            std::cout<<"progress:"<<i<<" / "<<20000<<std::endl;
        }
    });
	return 0;
}
