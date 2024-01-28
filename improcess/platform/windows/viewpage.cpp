#include "viewpage.h"
#include <assert.h>
#include <iostream>


LRESULT imp::View2D::onResize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout<<"onResize"<<std::endl;
    int w = (SHORT)LOWORD(lParam);
    int h = (SHORT)HIWORD(lParam);
    std::cout<<"w:"<<w<<",width:"<<width<<",h:"<<h<<",height:"<<height<<std::endl;
    if (w == width && h == height) {
        return 0;
    }
    width = w;
    height = h;
    /* resize page bitmap */
    if (pageBitmap != NULL) {
        DeleteObject(pageBitmap);
    }
    HDC hdc = GetDC(hWnd);
    pageBitmap = CreateCompatibleBitmap(hdc, width, height);
    /* resize canvas bitmap */
    //if (canvasBitmap != NULL) {
    //    DeleteObject(canvasBitmap);
    //}
    //canvasBitmap = createBitmap32(width, height);
    //assert(canvasBitmap != NULL);
    //BITMAP bitmap;
    //GetObject(canvasBitmap, sizeof(BITMAP), &bitmap);
    //memset(bitmap.bmBits, 0, bitmap.bmWidthBytes*bitmap.bmHeight);
    return 0;
}

LRESULT imp::View2D::onPaint(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout<<"onPaint"<<std::endl;
    BITMAP pageBmp;
    GetObject(pageBitmap, sizeof(BITMAP), &pageBmp);
    RECT pageRect = {0, 0, pageBmp.bmWidth, pageBmp.bmHeight};

    HDC hdc = GetDC(hWnd);
    HDC pageMem = CreateCompatibleDC(hdc);
    HBITMAP oldPageBitmap = (HBITMAP)SelectObject(pageMem, pageBitmap);
    HRGN pageRgn = CreateRectRgnIndirect(&pageRect);
    SelectClipRgn(pageMem, pageRgn);
    SetBkMode(pageMem, TRANSPARENT);
    ExtTextOutW(pageMem, 0, 0, ETO_OPAQUE, &pageRect, NULL, 0, NULL);
    /* paint image */
    BITMAP canvas;
    GetObject(canvasBitmap, sizeof(BITMAP), &canvas);
    HDC canvasMem = CreateCompatibleDC(pageMem);
    HBITMAP oldCanvasBitmap = (HBITMAP)SelectObject(canvasMem, canvasBitmap);
    /* display */
    BitBlt(hdc, 0, 0, pageBmp.bmWidth, pageBmp.bmHeight,
           canvasMem, 0, 0, SRCCOPY);
    /* delete canvas */
    SelectObject(canvasMem, oldCanvasBitmap);
    DeleteDC(canvasMem);
    /* delete page rgn */
    SelectClipRgn(pageMem, NULL);
    DeleteObject(pageRgn);
    /* delete page */
    SelectObject(pageMem, oldPageBitmap);
    DeleteDC(pageMem);
    ReleaseDC(hWnd, hdc);
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::View2D::onClose(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (canvasBitmap != NULL) {
        DeleteObject(canvasBitmap);
        canvasBitmap = NULL;
    }
    return NaivePage::onClose(hWnd, msg, wParam, lParam);
}

imp::View2D::View2D()
    :canvasBitmap(NULL)
{

}

imp::View2D::~View2D()
{

}

void imp::View2D::updateImage(int h, int w, int c, uint8_t *data)
{
    BITMAP canvas;
    GetObject(canvasBitmap, sizeof(BITMAP), &canvas);
    uint8_t* img = (uint8_t*)canvas.bmBits;
    if (c == 1) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                uint8_t pixel =data[(h - 1 - i)*w*c + j*c];
                img[i*w*4 + j*4]     = pixel;
                img[i*w*4 + j*4 + 1] = pixel;
                img[i*w*4 + j*4 + 2] = pixel;
                img[i*w*4 + j*4 + 3] = 0;
            }
        }
    } else if (c == 3) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                img[i*w*4 + j*4]     = data[(h - 1 - i)*w*c + j*c + 2];
                img[i*w*4 + j*4 + 1] = data[(h - 1 - i)*w*c + j*c + 1];
                img[i*w*4 + j*4 + 2] = data[(h - 1 - i)*w*c + j*c];
                img[i*w*4 + j*4 + 3] = 0;
            }
        }
    } else if (c == 4) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                img[i*w*4 + j*4]     = data[(h - 1 - i)*w*c + j*c + 2];
                img[i*w*4 + j*4 + 1] = data[(h - 1 - i)*w*c + j*c + 1];
                img[i*w*4 + j*4 + 2] = data[(h - 1 - i)*w*c + j*c];
                img[i*w*4 + j*4 + 3] = data[(h - 1 - i)*w*c + j*c + 3];
            }
        }
    }
    //HDC hdc = GetDC(pageHWND);
    //BITMAP pageBmp;
    //GetObject(pageBitmap, sizeof(BITMAP), &pageBmp);
    //HDC canvasMem = CreateCompatibleDC(hdc);
    //HBITMAP oldCanvasBitmap = (HBITMAP)SelectObject(canvasMem, canvasBitmap);
    //BitBlt(hdc, 0, 0, pageBmp.bmWidth, pageBmp.bmHeight,
    //       canvasMem, 0, 0, SRCCOPY);
    //SelectObject(canvasMem, oldCanvasBitmap);
    //DeleteDC(canvasMem);
    ::PostMessageW(pageHWND, WM_PAINT, 0, 0);
    return;
}

void imp::View2D::display(int h, int w, int c, uint8_t *data)
{
    /* create window */
    int x = (GetSystemMetrics(SM_CXSCREEN) - w)/2;
    int y = (GetSystemMetrics(SM_CYSCREEN) - h)/2;
    NaivePage::create(x, y, w, h, L"ViewPageClass", L"View Image", NULL);
    /* canvas */
    canvasBitmap = createBitmap32(w, h);
    /* show */
    NaivePage::show();
    updateImage(h, w, c, data);
    MSG msg = {0};
    while (GetMessage(&msg, NULL, 0, 0)) {
        DispatchMessage(&msg);
    }
    return;
}
