#pragma once

#include "TLabWindows.h"
#include "TLabCppSocket.h"
#include "TLabSharedMemory.h"
#include "TLabMouseEmulator.h"
#include "TLabFrameEncoder.h"

#define RELEASE_CONFIG 1
#define ENABLE_SCREENSHARE 1
#define ENABLE_TOUCH_EMULATION 0
#define DEBUG_MODE 1
#define DEBUG_WINDOW 1
#define DEBUG_FPS 0
#define ENABLE_RESEND 1

#define BUFFERS_FRAME_COUNT 2

// 01 ---> 0 to 1
#define BUFFERS_FRAME_COUNT_LOOP_BIT 1

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::System;
using namespace winrt::Windows::Graphics;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

namespace TLabShareScreenServer
{
    struct TLabShareScreenServer
    {
    public:

        // public functions

        /* ----------------------------------------------------------- */

        // for capture

        TLabShareScreenServer();
        void SetOwnerHandle(HWND ownerHend);
        bool StartCaptureForDesiredWindow();
        bool StopCapture();
        bool IsCapturing() { return _framePool != nullptr; }
        bool IsConnecting() { return _isConnecting; }

        // for conenct client
        bool ConnectToClient(
            uint16_t serverPort,
            uint16_t clientPort,
            char* clientAddr
        );
        bool DisconnectClient();

        void Resize();

        /* ----------------------------------------------------------- */

        HWND _targetHwnd;

    private:

        // private functions

        /* -------------------------------------------------------------------------------------- */

        // for capture

        HRESULT CreateDevice();
        HRESULT CreateBufferTexture(Size const& itemSize);
        GraphicsCaptureItem CreateItemForWindow(HWND hWnd);
        GraphicsCaptureItem CreateItemForMonitor(HWND hWnd);
        void StartCapture(winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item);
        void OnFrameArrived(
            Direct3D11CaptureFramePool const& sender,
            winrt::Windows::Foundation::IInspectable const& args
        );

        // for send packet

        HRESULT CreatePacketBuffer();
        HRESULT CreateEncodeDevice(int width, int height);
        HRESULT CastScreen(
            winrt::com_ptr<ID3D11DeviceContext> context,
            winrt::Windows::Graphics::SizeInt32 contentSize,
            winrt::com_ptr<ID3D11Texture2D> frameSurface
        );
        HRESULT ShowResult(
            winrt::com_ptr<ID3D11DeviceContext> context,
            winrt::Windows::Graphics::SizeInt32 contentSize,
            winrt::com_ptr<ID3D11ShaderResourceView> frameSurfaceSRV
        );
        HRESULT CheckWinSize(Direct3D11CaptureFrame frame);

        /* -------------------------------------------------------------------------------------- */

        HWND _ownerHwnd;

        /* ---------------------------------------------------- */

        // Convert texture to byte array to send as a packet

        D3D11_TEXTURE2D_DESC _bufferTextureDesc;
        ID3D11Texture2D* _bufferTexture;
        D3D11_BOX _capWinSizeInTexture;

        int _encBufferSize;
        int _packetBufferSize;
        char _currentFrameOffset = 0;
        char _isItResendRequested = 0;
        char* _packetBuffer;
        char* _resendPacketHedder;
        char* _encBuffer;
        unsigned char* _decBuffer;
        unsigned short _lastFrameEndIdx = 0;

#if DEBUG_MODE && DEBUG_FPS
        DWORD _lastTime = 0;
#endif

        /* ---------------------------------------------------- */

        winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice _device{ nullptr };
        winrt::com_ptr<ID3D11Device> _d3dDevice;
        winrt::com_ptr<IDXGISwapChain1> _dxgiSwapChain;
        winrt::com_ptr<ID3D11RenderTargetView> _chainedBufferRTV;

        std::unique_ptr<::DirectX::SpriteBatch> _spriteBatch;

        winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool _framePool{ nullptr };
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem _captureItem{ nullptr };
        winrt::Windows::Graphics::Capture::GraphicsCaptureSession _captureSession{ nullptr };
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::FrameArrived_revoker _frameArrived;
    };
}
