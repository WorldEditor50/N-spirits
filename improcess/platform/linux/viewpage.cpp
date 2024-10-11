#include "viewpage.h"
#include <xcb/xcb.h>
#include <memory>
#include <iostream>

void ns::View2D::onPaint(NaivePage::Event *e)
{
    /* copy pixmap to window */
    xcb_copy_area(connection, pixmap, window, gc,
                  0, 0, 0, 0, width, height);
    xcb_flush(connection);
    return;
}

ns::View2D::View2D():
    pixmap(0)
{

}

ns::View2D::~View2D()
{
    if (pixmap) {
        xcb_free_pixmap(connection, pixmap);
    }
}

void ns::View2D::display(int h, int w, int c, uint8_t *data)
{
    /* create window */
    int ret = createWindow(0, 0, w, h);
    if (ret != 0) {
        return;
    }

    /* create pixmap */
    pixmap = xcb_generate_id(connection);
    xcb_create_pixmap_checked(connection,
                              screen->root_depth,
                              pixmap, window, w, h);
    //std::cout<<"depth:"<<(int)screen->root_depth<<std::endl;
    /* align image */
    std::unique_ptr<uint8_t[]> image(new uint8_t[h*w*4]);
    uint8_t* img = image.get();
    if (c == 1) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                uint8_t pixel =data[i*w*c + j*c];
                img[i*w*4 + j*4]     = pixel;
                img[i*w*4 + j*4 + 1] = pixel;
                img[i*w*4 + j*4 + 2] = pixel;
                img[i*w*4 + j*4 + 3] = 0;
            }
        }
    } else if (c == 3) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                img[i*w*4 + j*4]     = data[i*w*c + j*c + 2];
                img[i*w*4 + j*4 + 1] = data[i*w*c + j*c + 1];
                img[i*w*4 + j*4 + 2] = data[i*w*c + j*c];
                img[i*w*4 + j*4 + 3] = 0;
            }
        }
    } else if (c == 4) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                img[i*w*4 + j*4]     = data[i*w*c + j*c + 2];
                img[i*w*4 + j*4 + 1] = data[i*w*c + j*c + 1];
                img[i*w*4 + j*4 + 2] = data[i*w*c + j*c];
                img[i*w*4 + j*4 + 3] = data[i*w*c + j*c + 3];
            }
        }
    }
    /* draw image on pixmap */
    xcb_put_image_checked(connection,
                          XCB_IMAGE_FORMAT_Z_PIXMAP,
                          pixmap,
                          gc,
                          w, h, 0, 0, 0,
                          screen->root_depth,
                          h*w*4,
                          img);
    xcb_flush(connection);
    /* polling event */
    handleEvent();
    return;
}
