#ifndef VIEWPAGE_H
#define VIEWPAGE_H
#include "naivepage.h"
namespace imp {

class View2D : public NaivePage
{
protected:
    xcb_pixmap_t pixmap;
protected:
    virtual void onPaint(Event *e) override;

public:
    View2D();
    virtual ~View2D();
    void display(int h, int w, int c, uint8_t* data);
};

}
#endif // VIEWPAGE_H
