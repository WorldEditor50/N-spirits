#ifndef GRAPHIC3D_H
#define GRAPHIC3D_H


namespace graphic3D {

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
        bitmap = new int[h*w];
    }
    ~Camera()
    {
        if (bitmap != nullptr) {
            delete [] bitmap;
        }
    }

}; // camera

};// graphic3d

#endif // GRAPHIC3D_H
