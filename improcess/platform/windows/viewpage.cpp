#include "viewpage.h"
#include <assert.h>
#include <iostream>

#pragma comment(lib, "Msimg32.lib")

LRESULT imp::ViewPage::onResize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout<<"onResize"<<std::endl;
    width = (SHORT)LOWORD(lParam);
    height = (SHORT)HIWORD(lParam);
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

LRESULT imp::ViewPage::onPaint(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
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
    //SetBkColor(pageMem, RGB(255, 0, 0));
    ExtTextOutW(pageMem, 0, 0, ETO_OPAQUE, &pageRect, NULL, 0, NULL);

    /* paint image */
    BITMAP canvas;
    GetObject(canvasBitmap, sizeof(BITMAP), &canvas);
    HDC canvasMem = CreateCompatibleDC(hdc);
    HBITMAP oldCanvasBitmap = (HBITMAP)SelectObject(canvasMem, canvasBitmap);
    BLENDFUNCTION blendFunc;
    blendFunc.BlendOp = AC_SRC_OVER;
    blendFunc.BlendFlags = 0;
    blendFunc.SourceConstantAlpha = 255;
    blendFunc.AlphaFormat = AC_SRC_ALPHA;
    AlphaBlend(pageMem, 0, 0, pageBmp.bmWidth, pageBmp.bmHeight,
               canvasMem, 0, 0, canvas.bmWidth, canvas.bmHeight,
               blendFunc);
    SelectObject(canvasMem, oldCanvasBitmap);
    DeleteDC(canvasMem);
    /* display */
    BitBlt(hdc, 0, 0, pageBmp.bmWidth, pageBmp.bmHeight,
           pageMem, 0, 0, SRCCOPY);
    SelectClipRgn(pageMem, NULL);
    DeleteObject(pageRgn);
    SelectObject(pageMem, oldPageBitmap);
    DeleteDC(pageMem);
    ReleaseDC(hWnd, hdc);
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::ViewPage::onClose(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (canvasBitmap != NULL) {
        DeleteObject(canvasBitmap);
        canvasBitmap = NULL;
    }
    return NaivePage::onClose(hWnd, msg, wParam, lParam);
}

imp::ViewPage::ViewPage()
    :canvasBitmap(NULL)
{

}

imp::ViewPage::~ViewPage()
{

}

void imp::ViewPage::updateImage(int h, int w, int c, uint8_t *data)
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
    HDC hdc = GetDC(pageHWND);
    BITMAP pageBmp;
    GetObject(pageBitmap, sizeof(BITMAP), &pageBmp);
    HDC canvasMem = CreateCompatibleDC(hdc);
    HBITMAP oldCanvasBitmap = (HBITMAP)SelectObject(canvasMem, canvasBitmap);
    BitBlt(hdc, 0, 0, pageBmp.bmWidth, pageBmp.bmHeight,
           canvasMem, 0, 0, SRCCOPY);
    SelectObject(canvasMem, oldCanvasBitmap);
    DeleteDC(canvasMem);
    return;
}

void imp::ViewPage::display(int h, int w, int c, uint8_t *data)
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
