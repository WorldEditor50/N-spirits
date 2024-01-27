#include "naivepage.h"
#include <commctrl.h>
#include <iostream>
#pragma comment(lib, "Comctl32.lib")

LRESULT imp::NaivePage::handleWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    NaivePage *page = (NaivePage*)GetWindowLongPtrW(hWnd, GWLP_USERDATA);
    if (page != nullptr) {
        return page->onWndProc(hWnd, msg, wParam, lParam);
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::subClassProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, UINT_PTR uIdSubClass, DWORD_PTR dwRefData)
{
    NaivePage* page = (NaivePage*)dwRefData;
    if (page == nullptr) {
        return ::DefSubclassProc(hWnd, uMsg, wParam, lParam);
    }
    switch (uMsg) {
    case WM_SIZE:
    {
        RECT rect;
        ::GetClientRect(hWnd, &rect);
        int width = rect.right - rect.left;
        int height = rect.bottom - rect.top;
        /* repaint */
        page->resize(rect.left, rect.top, width, height);
    }
        break;
    default:
        break;
    }
    return ::DefSubclassProc(hWnd, uMsg, wParam, lParam);
}

LRESULT imp::NaivePage::onWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CREATE:
        return onCreate(hWnd, msg, wParam, lParam);
    case WM_SIZE:
        return onResize(hWnd, msg, wParam, lParam);
    case WM_PAINT:
        return onPaint(hWnd, msg, wParam, lParam);
    case WM_RBUTTONDBLCLK:
    case WM_LBUTTONDBLCLK:
        return onMouseDoubleClick(hWnd, msg, wParam, lParam);
    case WM_RBUTTONDOWN:
    case WM_LBUTTONDOWN:
        return onMousePress(hWnd, msg, wParam, lParam);
    case WM_MOUSEMOVE:
        return onMouseMove(hWnd, msg, wParam, lParam);
    case WM_RBUTTONUP:
    case WM_LBUTTONUP:
        return onMouseRelease(hWnd, msg, wParam, lParam);
    case WM_MOUSEWHEEL:
        return onMouseWheel(hWnd, msg, wParam, lParam);
    case WM_KEYDOWN:
        return onKeyDown(hWnd, msg, wParam, lParam);
    case WM_KEYUP:
        return onKeyUp(hWnd, msg, wParam, lParam);   
    case WM_TIMER:
        return onTimer(hWnd, msg, wParam, lParam);
    case WM_CLOSE:
        return onClose(hWnd, msg, wParam, lParam);
    case WM_DESTROY:
        return onDestroy(hWnd, msg, wParam, lParam);
    default:
        return onCustomMessage(hWnd, msg, wParam, lParam);
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onCreate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onResize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    width = (SHORT)LOWORD(lParam);
    height = (SHORT)HIWORD(lParam);
    /* resize page bitmap */
    if (pageBitmap != NULL) {
        DeleteObject(pageBitmap);
    }
    HDC hdc = GetDC(hWnd);
    pageBitmap = CreateCompatibleBitmap(hdc, width, height);
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onPaint(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hWnd, &ps);
    HDC pageMem = CreateCompatibleDC(hdc);
    HBITMAP oldPageBitmap = (HBITMAP)SelectObject(pageMem, pageBitmap);
    SetBkColor(pageMem, RGB(255, 0, 0));
    int x = ps.rcPaint.left;
    int y = ps.rcPaint.top;
    int w = ps.rcPaint.right - ps.rcPaint.left;
    int h = ps.rcPaint.bottom - ps.rcPaint.top;
    BitBlt(hdc, x, y, w, h, pageMem, x, y, SRCCOPY);

    SelectObject(pageMem, oldPageBitmap);
    DeleteDC(pageMem);
    EndPaint(hWnd, &ps);
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onMouseDoubleClick(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onMousePress(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onMouseMove(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onMouseRelease(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onMouseWheel(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onKeyDown(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onKeyUp(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onTimer(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onClose(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout<<"onClose"<<std::endl;
    if (pageBitmap != NULL) {
        DeleteObject(pageBitmap);
        pageBitmap = NULL;
    }
    if (pageHWND != NULL) {
        DestroyWindow(pageHWND);
        pageHWND = NULL;
    }
    ReleaseCapture();
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onDestroy(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout<<"onDestroy"<<std::endl;

    PostQuitMessage(0);
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT imp::NaivePage::onCustomMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

HBITMAP imp::NaivePage::createBitmap32(int width, int height)
{
    BYTE bmi[sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD)] = { 0 };
    BITMAPINFOHEADER* pbmihdr = (BITMAPINFOHEADER*)bmi;
    pbmihdr->biSize = sizeof(BITMAPINFOHEADER);
    pbmihdr->biWidth = width;
    pbmihdr->biHeight = height;
    pbmihdr->biBitCount = 32;
    pbmihdr->biPlanes = 1;
    pbmihdr->biCompression = BI_RGB;

    void* pBits = NULL;
    HBITMAP hBmp = CreateDIBSection(NULL, (BITMAPINFO*)bmi,
                                    DIB_RGB_COLORS, &pBits, NULL, 0);
    return hBmp;
}

imp::NaivePage::NaivePage():
    width(0),
    height(0),
    pageHWND(NULL),
    pageBitmap(NULL)
{

}

imp::NaivePage::~NaivePage()
{

}

void imp::NaivePage::resize(int x, int y, int w, int h)
{
    if (pageHWND != nullptr) {
        width = w;
        height = h;
        ::MoveWindow(pageHWND, x, y, w, h, TRUE);
    }
    return;
}

int imp::NaivePage::create(int x, int y, int w, int h,
                 const std::wstring &className_,
                 const std::wstring &windowName_,
                 HWND hParent_)
{
    width = w;
    height = h;
    className = className_;
    windowName = windowName_;
    HINSTANCE   hInstance = GetModuleHandle(NULL);
    /* register window class */
    WNDCLASSEXW wcex;
    wcex.cbSize			= sizeof(WNDCLASSEXW);
    wcex.style			= CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
    wcex.lpfnWndProc	= handleWndProc;
    wcex.cbClsExtra		= 0;
    wcex.cbWndExtra		= 0;
    wcex.hInstance		= hInstance;
    wcex.hIcon			= LoadIcon(NULL, IDI_APPLICATION);
    wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground	= (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName	= NULL;
    wcex.lpszClassName	= className.c_str();
    wcex.hIconSm		= NULL;
    RegisterClassExW(&wcex);

    /* create window */

    HWND hWnd;
    if (hParent_ != NULL) {
        hWnd = CreateWindowExW(0, className.c_str(), windowName.c_str(),
                                 WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN,
                                 x, y, w, h,
                                 hParent_, NULL, hInstance, NULL);
        parentHWND = hParent_;
    } else {
        hWnd = CreateWindowExW(0, className.c_str(), windowName.c_str(),
                               WS_TILEDWINDOW,
                               x, y, w, h,
                               NULL, NULL, hInstance, NULL);

    }
    if (hWnd == NULL) {
        ::UnregisterClassW(className.c_str(), 0);
        return -1;
    }
    SetWindowLongPtrW(hWnd, GWLP_USERDATA, (LONG_PTR)this);

    if (parentHWND != NULL) {
        ::SetWindowSubclass(parentHWND, subClassProc, 0, (DWORD_PTR)this);
    }
    UpdateWindow(hWnd);
    /* create bitmap */
    HDC hdc = GetDC(hWnd);
    pageBitmap = CreateCompatibleBitmap(hdc, w, h);
    ReleaseDC(hWnd, hdc);
    pageHWND = hWnd;
    return 0;
}

void imp::NaivePage::show()
{
    ShowWindow(pageHWND, SW_SHOW);
    UpdateWindow(pageHWND);
    return;
}

void imp::NaivePage::showMaximized()
{
    ShowWindow(pageHWND, SW_SHOWMAXIMIZED);
    UpdateWindow(pageHWND);
    return;
}

void imp::NaivePage::showMinimized()
{
    ShowWindow(pageHWND, SW_SHOWMINIMIZED);
    UpdateWindow(pageHWND);
    return;
}

void imp::NaivePage::hide()
{
    ShowWindow(pageHWND, SW_HIDE);
    return;
}


