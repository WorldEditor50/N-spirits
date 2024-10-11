#ifndef NAIVEPAGE_H
#define NAIVEPAGE_H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <xcb/xcb.h>
namespace ns {

class NaivePage
{
public:
    using Event = xcb_generic_event_t;
protected:
    int width;
    int height;
    xcb_connection_t *connection;
    xcb_screen_t *screen;
    xcb_window_t window;
    xcb_gcontext_t gc;
protected:
    virtual void handleEvent();
    virtual void onPaint(Event *e);
    virtual void onMousePress(Event *e);
    virtual void onMouseMove(Event *e);
    virtual void onMouseRelease(Event *e);
    virtual void onKeyPress(Event *e);
    virtual void onKeyRelease(Event *e);
    virtual void onCreate(Event *e);
    virtual void onDestroy(Event *e);
    virtual void onResize(Event *e);
    static xcb_gcontext_t createGraphicsContext(xcb_connection_t *connection,
                                                xcb_screen_t *screen,
                                                xcb_window_t window);
public:
    NaivePage();
    virtual ~NaivePage();
    virtual int createWindow(int x, int y, int w, int h);
};

}
#endif // NAIVEPAGE_H
