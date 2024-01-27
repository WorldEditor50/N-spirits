#ifndef NAIVEPAGE_H
#define NAIVEPAGE_H
#include <Windows.h>
#include <string>

namespace imp {

class NaivePage
{
protected:
    std::wstring className;
    std::wstring windowName;
    int width;
    int height;
    HWND parentHWND;
    HWND pageHWND;
    HBITMAP pageBitmap;
protected:
    static LRESULT CALLBACK handleWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    static LRESULT CALLBACK subClassProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, UINT_PTR uIdSubClass, DWORD_PTR dwRefData);
    virtual LRESULT onWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onCreate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onResize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onPaint(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onMouseDoubleClick(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onMousePress(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onMouseMove(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onMouseRelease(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onMouseWheel(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onKeyDown(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onKeyUp(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onTimer(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onClose(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onDestroy(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual LRESULT onCustomMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
public:
    static HBITMAP createBitmap32(int width, int height);
    NaivePage();
    virtual ~NaivePage();
    virtual void resize(int x, int y, int w, int h);
    virtual int create(int x, int y, int w, int h,
                       const std::wstring &className_,
                       const std::wstring &windowName_,
                       HWND hParent_);
    virtual void show();
    virtual void showMaximized();
    virtual void showMinimized();
    virtual void hide();

};

}
#endif // NAIVEPAGE_H
