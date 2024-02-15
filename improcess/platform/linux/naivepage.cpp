#include "naivepage.h"

void imp::NaivePage::handleEvent()
{
    while (1) {
        Event *e = xcb_poll_for_event(connection);
        if (e == nullptr) {
            continue;
        }
        switch (e->response_type & ~0x80) {
        case XCB_EXPOSE:
            onPaint(e);
            break;
        case XCB_BUTTON_PRESS:
            onMousePress(e);
            break;
        case XCB_MOTION_NOTIFY:
            onMouseMove(e);
        case XCB_BUTTON_RELEASE:
            onMouseRelease(e);
            break;
        case XCB_KEY_PRESS:
            onKeyPress(e);
        case XCB_KEY_RELEASE:
            onKeyRelease(e);
            break;
        case XCB_CREATE_NOTIFY:
            onCreate(e);
            break;
        case XCB_DESTROY_NOTIFY:
            onDestroy(e);
            break;
        case XCB_RESIZE_REQUEST:
            onResize(e);
            break;
        default:
            break;
        }
        free (e);
    }
    xcb_disconnect(connection);
    return;
}

void imp::NaivePage::onKeyRelease(NaivePage::Event *e)
{
    xcb_key_release_event_t *ev = (xcb_key_release_event_t *)e;
    switch (ev->detail) {
    case 9:
        break;
    default:
        break;
    }
    return;
}

void imp::NaivePage::onPaint(NaivePage::Event *e)
{

    return;
}

void imp::NaivePage::onMousePress(NaivePage::Event *e)
{
    xcb_button_press_event_t *ev = (xcb_button_press_event_t*)e;
    int x = ev->event_x;
    int y = ev->event_y;

    return;
}

void imp::NaivePage::onMouseMove(NaivePage::Event *e)
{

    return;
}

void imp::NaivePage::onMouseRelease(NaivePage::Event *e)
{
    xcb_button_release_event_t *ev = (xcb_button_release_event_t*)e;
    int x = ev->event_x;
    int y = ev->event_y;
    return;
}

void imp::NaivePage::onKeyPress(NaivePage::Event *e)
{

}

void imp::NaivePage::onCreate(NaivePage::Event *e)
{

}

void imp::NaivePage::onDestroy(NaivePage::Event *e)
{

}

void imp::NaivePage::onResize(NaivePage::Event *e)
{

}

xcb_gcontext_t imp::NaivePage::createGraphicsContext(xcb_connection_t *connection,
                                                xcb_screen_t *screen,
                                                xcb_window_t window)
{
    xcb_gcontext_t gc;
    uint32_t mask;
    uint32_t values[3];
    mask = XCB_GC_FOREGROUND|XCB_GC_BACKGROUND;
    values[0] = screen->black_pixel;
    values[1] = screen->white_pixel;
    values[2] = 0;
    gc = xcb_generate_id(connection);
    xcb_create_gc_checked(connection, gc, window, mask, values);
    return gc;
}

imp::NaivePage::NaivePage():
      width(0),height(0),
      connection(nullptr),screen(nullptr),
      window(0),gc(0)
{

}

imp::NaivePage::~NaivePage()
{

}

int imp::NaivePage::createWindow(int x, int y, int w, int h)
{
    width = w;
    height = h;
    xcb_generic_error_t *error;
    xcb_void_cookie_t cookieWindow;
    xcb_void_cookie_t cookieMap;
    uint32_t mask;
    uint32_t values[2];
    int screenNumber;
    /* getting the connection */
    connection = xcb_connect (NULL, &screenNumber);
    if (!connection) {
        fprintf (stderr, "ERROR: can't connect to an X server\n");
        return -1;
    }
    /* getting the current screen */
    const xcb_setup_t *setup = xcb_get_setup(connection);
    xcb_screen_iterator_t screenIter = xcb_setup_roots_iterator(setup);
    for (; screenIter.rem != 0; --screenNumber, xcb_screen_next(&screenIter))
        if (screenNumber == 0) {
            screen = screenIter.data;
            break;
        }
    if (!screen) {
        fprintf (stderr, "ERROR: can't get the current screen\n");
        xcb_disconnect(connection);
        return -2;
    }

    /* creating the window */
    window = xcb_generate_id(connection);
    mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
    values[0] = screen->white_pixel;
    values[1] = XCB_EVENT_MASK_EXPOSURE |
            XCB_EVENT_MASK_KEY_PRESS |
            XCB_EVENT_MASK_KEY_RELEASE |
            XCB_EVENT_MASK_BUTTON_PRESS |
            XCB_EVENT_MASK_POINTER_MOTION;
    cookieWindow = xcb_create_window_checked(connection,
                                              screen->root_depth,
                                              window, screen->root,
                                              x, y, width, height,
                                              0, XCB_WINDOW_CLASS_INPUT_OUTPUT,
                                              screen->root_visual,
                                              mask, values);
    cookieMap = xcb_map_window_checked(connection, window);
    /* error managing */
    error = xcb_request_check(connection, cookieWindow);
    if (error) {
        fprintf (stderr, "ERROR: can't create window : %d\n", error->error_code);
        xcb_disconnect(connection);
        return -3;
    }
    error = xcb_request_check(connection, cookieMap);
    if (error) {
        fprintf (stderr, "ERROR: can't map window : %d\n", error->error_code);
        xcb_disconnect(connection);
        return -4;
    }
    /* create graphics context */
    gc = createGraphicsContext(connection, screen, window);
    return 0;
}
