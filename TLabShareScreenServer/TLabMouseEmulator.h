#pragma once

#include "TLabWindows.h"

namespace {
    //////////////////////////////////////////////////////
    // library emulate mouse event from program.
    //

    // target window handle.
    HWND _emulateTarget;

    void SetEmulateTarget(HWND target) {
        _emulateTarget = target;
    }

    void WindowSendMessage(float hitCoordX, float hitCoordY, int message, int wheelScroll) {
        int x;
        int y;
        RECT windowRect;

        GetWindowRect(_emulateTarget, &windowRect);

        if (windowRect.top == 0 && windowRect.bottom == 0) {
            x = (int)(1920 * hitCoordY);
            y = (int)(1080 * hitCoordX);
        }
        else {
            x = (int)((windowRect.right - windowRect.left) * hitCoordY);
            y = (int)((windowRect.bottom - windowRect.top) * hitCoordX);
        }

        SetForegroundWindow(_emulateTarget);
        SetCursorPos(x + windowRect.left, y + windowRect.top);
        mouse_event(message, 0, 0, wheelScroll, 0);
    }

}
