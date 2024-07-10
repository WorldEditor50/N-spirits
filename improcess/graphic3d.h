#ifndef GRAPHIC3D_H
#define GRAPHIC3D_H


namespace imp {

class Camera
{
public:
    int width;
    int height;
    int *bitmap;
public:
    explicit Camera(int w, int h)
        :width(w),height(h)
    {
        bitmap = new int[h*w*4];
    }
    ~Camera()
    {
        if (bitmap != nullptr) {
            delete [] bitmap;
        }
    }

}; // camera

}

#endif // GRAPHIC3D_H
