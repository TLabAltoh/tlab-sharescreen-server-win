#pragma once

#include "windows_common.h"

namespace {

    HWND _emulate_target;   // target window handle

    void SetEmulateTarget(HWND target) {
        _emulate_target = target;
    }

    void WindowSendMessage(float hit_coord_X, float hit_coord_Y, int message, int wheel_scroll) {
        int x;
        int y;
        RECT window_rect;

        GetWindowRect(_emulate_target, &window_rect);

        if (window_rect.top == 0 && window_rect.bottom == 0) {
            x = (int)(1920 * hit_coord_Y);
            y = (int)(1080 * hit_coord_X);
        }
        else {
            x = (int)((window_rect.right - window_rect.left) * hit_coord_Y);
            y = (int)((window_rect.bottom - window_rect.top) * hit_coord_X);
        }

        SetForegroundWindow(_emulate_target);
        SetCursorPos(x + window_rect.left, y + window_rect.top);
        mouse_event(message, 0, 0, wheel_scroll, 0);
    }

}
