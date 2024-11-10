#ifndef VIEWPAGE_H
#define VIEWPAGE_H
#include "naivepage.h"
namespace ns {

class View2D : public NaivePage
{
protected:
    HBITMAP canvasBitmap;
    double scale;
protected:
    virtual LRESULT onResize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    virtual LRESULT onPaint(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    virtual LRESULT onClose(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
public:
    View2D();
    virtual ~View2D();
    void updateImage(int h, int w, int c, uint8_t* data);
    void display(int h, int w, int c, uint8_t* data);
};

}
#endif // VIEWPAGE_H
