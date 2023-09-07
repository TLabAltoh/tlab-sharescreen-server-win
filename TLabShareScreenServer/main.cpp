#pragma once

#include "pch.h"
#include "TLabWindows.h"
#include "TLabShareScreen.h"
#include "Direct3DHelper.h"

using namespace ::DirectX;
using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::System;
using namespace winrt::Windows::Graphics;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

// printf("---------------------------------------------------------------------\n");

TLabShareScreenServer::TLabShareScreenServer server = TLabShareScreenServer::TLabShareScreenServer();

static LRESULT CALLBACK mainWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CLOSE :
        if (server.IsConnecting() == true)server.DisconnectClient();
        if (server.IsCapturing() == true)server.StopCapture();
        DestroyWindow(hWnd);
        break;
    case WM_DESTROY :
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, uMsg, wParam, lParam);
    }
    return 0;
}

int main(int argc, char* argv[]) {
    init_apartment();

    printf("---------------------------------------------------------------------\n");
    printf("start TLabShareScreenSever\n");

    std::cout << "TLabShareScreenServer started successfully." << std::flush;

    printf("create window ...\n");

    TCHAR lpClassName[] = TEXT("TLabShareScreen");
    TCHAR lpTitleName[] = TEXT("Debug");

    // https://p-tal.hatenadiary.org/entry/20091006/1254790659#:~:text=initDialogue()%0A%7B%0A%20%20HINSTANCE%20hInst%20%3D-,GetModuleHandle,-(NULL)%3B%0A%0A%20%20m_hMenu
    HINSTANCE hInstance = GetModuleHandle(NULL);

    WNDCLASS wc;
    // set window class attributes.
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = mainWindowProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName = NULL;
    wc.lpszClassName = lpClassName;

    // register window class.
    RegisterClass(&wc);

    // https://blog.goo.ne.jp/masaki_goo_2006/e/d4ba04d58719e1f28b6e90c2e4d09a3d#:~:text=//%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%0A//-,%E3%82%A6%E3%82%A4%E3%83%B3%E3%83%89%E3%82%A6%E3%81%AE%E4%BD%9C%E6%88%90(OK),-//%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%0Astatic%20HWND
    HWND ownerHwnd = CreateWindow(
        lpClassName,
        lpTitleName,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        800,
        450,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    // https://www.tokovalue.jp/function/ShowWindow.htm#:~:text=%E3%81%95%E3%82%8C%E3%81%AA%E3%81%84%E3%80%82-,SW_SHOWNA,-%E3%82%A6%E3%82%A3%E3%83%B3%E3%83%89%E3%82%A6%E3%82%92%E7%8F%BE%E5%9C%A8
    ShowWindow(ownerHwnd, SW_SHOWNORMAL);
    UpdateWindow(ownerHwnd);

    printf("window created ...\n");

    if (OpenSharedMemoryMappingObiect(
        L"TLabShareScreenServer",
        L"Shareing") != 1)
    {
        printf("an error has occurred.\n");
        printf("press enter to finish.");
        while (1) if ('\r' == getch()) break;
        return 0;
    }

    server.SetOwnerHandle(ownerHwnd);
    server.ConnectToClient(
        strtol(argv[1], NULL, 10),
        strtol(argv[2], NULL, 10),
        argv[3]
    );
    server.StartCaptureForDesiredWindow();

    WaitForSingleObject(_keepAliveMutexHandle, INFINITE);
    *_mappingObject = 1;
    ReleaseMutex(_keepAliveMutexHandle);

    MSG Msg;
    while (GetMessage(&Msg, NULL, 0, 0) > 0) {
        printf("ui roop ...\n");
        WaitForSingleObject(_keepAliveMutexHandle, INFINITE);
        if (*_mappingObject == 0) {
            ReleaseMutex(_keepAliveMutexHandle);
            break;
        }
        ReleaseMutex(_keepAliveMutexHandle);

        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    FreeUp();

    return 0;
}

namespace {

    // printf("---------------------------------------------------------------------\n");

    auto FitInBox(Size const& source, Size const& destination) {

        // Compute a rectangle that fits in the box
        // while preserving the aspect ratio.

        Rect box;

        box.Width = destination.Width;
        box.Height = destination.Height;

        // ---------------------------------
        // |          |         |          |
        // |          |         |          |
        // |          |         |          |
        // |          |         |          |
        // |          |         |          |
        // |          |         |          |
        // ---------------------------------
        //
        // --------------
        // |            |
        // |            |
        // |            |
        // |------------|
        // |            |
        // |            |
        // |------------|
        // |            |
        // |            |
        // |            |
        // --------------

        float aspect = source.Width / source.Height;
        if (box.Width >= box.Height * aspect) {
            box.Width = box.Height * aspect;
        }

        aspect = source.Height / source.Width;
        if (box.Height >= box.Width * aspect) {
            box.Height = box.Width * aspect;
        }

        box.X = (destination.Width - box.Width) * 0.5f;
        box.Y = (destination.Height - box.Height) * 0.5f;

        return CRect(
            static_cast<int>(box.X),
            static_cast<int>(box.Y),
            static_cast<int>(box.X + box.Width),
            static_cast<int>(box.Y + box.Height)
        );
    }
}

namespace TLabShareScreenServer
{
    TLabShareScreenServer::TLabShareScreenServer() { };

    void TLabShareScreenServer::SetOwnerHandle(HWND ownerHwnd) {
        _ownerHwnd = ownerHwnd;

        this->CreateDevice();
    }

    bool TLabShareScreenServer::ConnectToClient(
        uint16_t serverPort,
        uint16_t clientPort,
        char* clientAddr
    ) {

#if ENABLE_TOUCH_EMULATION
        SetEmulateTarget(_targetHwnd);

        SetReceiveCallbackHandler(
            [](char* receivePacket) {
                WindowSendMessage(
                    (float)receivePacket[0 * sizeof(float)],
                    (float)receivePacket[1 * sizeof(float)],
                    (float)receivePacket[2 * sizeof(float)],
                    (float)receivePacket[3 * sizeof(float)]
                );
            },
            4
        );
#endif

        printf("---------------------------------------------------------------------\n");
        printf("start create socket\n");
        int successful = CreateSocket(serverPort, clientPort, clientAddr, 2);
        printf("---------------------------------------------------------------------\n");

#if ENABLE_SCREENSHARE && ENABLE_RESEND
        // frame index(1byte)
        // packet index(1byte)
        int receiveBufferSize = 3;

        // Don't forget to put this,
        // which is the reference destination of the variable,
        // in the capture list [inside this].
        SetCallback(
            [this](char* receivePacket) {
                WaitForSingleObject(_socketMutexHandle, INFINITE);
                _resendPacketHedder =
                    _packetBuffer +
                    (char)receivePacket[2] * _packetBufferSize +
                    (((unsigned short)receivePacket[0] << 8) + receivePacket[1]) * (PACKET_HEDDER_SIZE + DG_BUFFER_SIZE);

                if (_resendPacketHedder < _packetBuffer ||
                    _resendPacketHedder > _packetBuffer + 2 * _packetBufferSize)
                {
                    printf("problem is occured\n");
                    ReleaseMutex(_socketMutexHandle);
                    return;
                }

                _isItResendRequested = 1;
                ReleaseMutex(_socketMutexHandle);
            }, receiveBufferSize
        );

        StartReceiveAsync(0, 1);
#endif

        return (successful == 0);
    }

    bool TLabShareScreenServer::DisconnectClient() {
        CloseSocket();
        return true;
    }

    HRESULT TLabShareScreenServer::CastScreen(
        winrt::com_ptr<ID3D11DeviceContext> context,
        winrt::Windows::Graphics::SizeInt32 contentSize,
        winrt::com_ptr<ID3D11Texture2D> frameSurface
    ) {

#if RELEASE_CONFIG
        ///////////////////////////////////////
        // Check client is connecting.
        //
        if (IsConnecting() == false) {
            return S_FALSE;
        }
#endif
        //////////////////////////////////////////////////////
        // Copy captured frame texture, from GPU to CPU
        // with translating byte array.
        //

        // Copy captured frame texture to sub resource.
        _capWinSizeInTexture.right = contentSize.Width;
        _capWinSizeInTexture.bottom = contentSize.Height;
        context->CopySubresourceRegion(
            _bufferTexture,
            0,
            0,
            0,
            0,
            frameSurface.get(),
            0,
            &_capWinSizeInTexture
        );

        // map texture sub resource to access from CPU
        // (with unsigned char array).
        D3D11_MAPPED_SUBRESOURCE mapd;
        context->Map(
            _bufferTexture,
            0,
            D3D11_MAP_READ,
            0,
            &mapd
        );

        // this data can access from unsigned char array.
        unsigned char* source = static_cast<unsigned char*>(mapd.pData);

        //////////////////////////////
        // Encode frame.
        //

        EncodeFrame(source);

        /////////////////////////////////////////////////////////////
        // Send encoded buffer while dividing.
        //

        unsigned short packetIdx = 0;

        char* packetBufferHedderPt = _packetBuffer + _packetBufferSize * _currentFrameOffset;
        char* packetBufferNextHedderPt = packetBufferHedderPt + PACKET_HEDDER_SIZE + DG_BUFFER_SIZE;

        char* packetBufferDCTBlockPt = packetBufferHedderPt + PACKET_HEDDER_SIZE;

        char* encBufferPt = _encBuffer;
        char* encBufferEndPoint = _encBuffer + _encBufferSize;

        // Encoded size of each channel.
        // Since the char type can only
        // represent positive values ​​up to 127,
        // we use unsigned char
        // so that 64*2=128 can be represented.
        unsigned char tmpBlockBufferSizeY;
        unsigned char tmpBlockBufferSizeCr;
        unsigned char tmpBlockBufferSizeCb;

        do {
            /**********************/
            /* Set Packet hedder. */
            /**********************/

            //////////////////////////////////////////////////////////////
            // Packet headers:
            // Index of this packet (upper 8bit).
            // Index of this packet (lower 8bit).
            // Last packet index of previous frame (upper 8bit).
            // Last packet index of previous frame (lower 8 bits).
            // Frame offset idx (frame offset)
            // Is this packet the last frame (flag)
            //

            packetBufferHedderPt[PACKET_IDX_UPPER_IDX] = (char)((unsigned short)packetIdx >> 8);
            packetBufferHedderPt[PACKET_IDX_LOWER_IDX] = (char)packetIdx;
            packetBufferHedderPt[LAST_FRAME_END_IDX_UPPER_IDX] = (char)((unsigned short)_lastFrameEndIdx >> 8);
            packetBufferHedderPt[LAST_FRAME_END_IDX_LOWER_IDX] = (char)_lastFrameEndIdx;
            packetBufferHedderPt[FRAME_OFFSET_IDX] = (char)_currentFrameOffset;
            packetBufferHedderPt[IS_THIS_PACKETEND_IDX] = (char)THIS_PACEKT_IS_NOT_FRAMES_LAST;
            packetBufferHedderPt[IS_THIS_FIX_PACKET_IDX] = (char)THIS_PACKET_IS_NOT_FOR_FIX;

            /*********************/
            /* Copy Block hedder */
            /*********************/

            /////////////////////////////////////////////////////
            // Block header:
            // Block index (2 bytes)
            // Block size after encoding (1 byte)
            //

            while (1) {

                // Get encoded block's YCrCb channel poitner.
                tmpBlockBufferSizeY = (unsigned char)(encBufferPt[Y_BIT_SIZE_IDX]) << ENDIAN_SIZE_LOG2;
                tmpBlockBufferSizeCr = (unsigned char)(encBufferPt[Cr_BIT_SIZE_IDX]) << ENDIAN_SIZE_LOG2;
                tmpBlockBufferSizeCb = (unsigned char)(encBufferPt[Cb_BIT_SIZE_IDX]) << ENDIAN_SIZE_LOG2;

                // Since we want to add a flag
                // (both block index big endian and little endian are 255
                // to the packet to notify that it is the end of valid data,
                // be sure to allocate a 2-byte buffer.

                if (packetBufferDCTBlockPt +
                    BLOCK_HEDDER_SIZE +
                    tmpBlockBufferSizeY +
                    tmpBlockBufferSizeCr +
                    tmpBlockBufferSizeCb >
                    packetBufferNextHedderPt - 2)
                {
                    packetBufferDCTBlockPt[BLOCK_IDX_UPPER_IDX] = (char)255;
                    packetBufferDCTBlockPt[BLOCK_IDX_LOWER_IDX] = (char)255;

                    break;
                }

                //////////////////////////////////////////////////////////////////////////////////////////
                // Set block hedder.
                //

                packetBufferDCTBlockPt[BLOCK_IDX_UPPER_IDX] = encBufferPt[BLOCK_IDX_UPPER_IDX];
                packetBufferDCTBlockPt[BLOCK_IDX_LOWER_IDX] = encBufferPt[BLOCK_IDX_LOWER_IDX];
                packetBufferDCTBlockPt[Y_BIT_SIZE_IDX] = encBufferPt[Y_BIT_SIZE_IDX];
                packetBufferDCTBlockPt[Cr_BIT_SIZE_IDX] = encBufferPt[Cr_BIT_SIZE_IDX];
                packetBufferDCTBlockPt[Cb_BIT_SIZE_IDX] = encBufferPt[Cb_BIT_SIZE_IDX];

                // Increment packet buffer dct block pointer to YCrCb data field.
                packetBufferDCTBlockPt += (tmpBlockBufferSizeY != NO_NEED_TO_ENCODE) * BLOCK_HEDDER_SIZE;

                // Increment encoded buffer pointer to YCrCb data field.
                encBufferPt += BLOCK_HEDDER_SIZE;

                //////////////////////////////////////////////////////////////////////////////////////////
                // Copy Y Cr Cb values
                //

                // Copy Y values.
                memcpy((void*)packetBufferDCTBlockPt, (void*)encBufferPt, tmpBlockBufferSizeY);
                packetBufferDCTBlockPt += tmpBlockBufferSizeY;
                encBufferPt += BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE * ENDIAN_SIZE;

                // Copy Cr value.
                memcpy((void*)packetBufferDCTBlockPt, (void*)encBufferPt, tmpBlockBufferSizeCr);
                packetBufferDCTBlockPt += tmpBlockBufferSizeCr;
                encBufferPt += BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE * ENDIAN_SIZE;

                // Copy Cb value.
                memcpy((void*)packetBufferDCTBlockPt, (void*)encBufferPt, tmpBlockBufferSizeCb);
                packetBufferDCTBlockPt += tmpBlockBufferSizeCb;
                encBufferPt += BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE * ENDIAN_SIZE;

                // Now encBufferPt's pointer is next block's hedder.
            }

            WaitForSingleObject(_socketMutexHandle, INFINITE);
            if (_isItResendRequested == 1) {
                _isItResendRequested = 0;
                _resendPacketHedder[IS_THIS_FIX_PACKET_IDX] = THIS_PACKET_IS_FOR_FIX;
                ResendFrame(_resendPacketHedder, PACKET_HEDDER_SIZE + DG_BUFFER_SIZE);

                auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(150);
                while (std::chrono::steady_clock::now() < end);
            }
            ReleaseMutex(_socketMutexHandle);

            // Send frame data.
            SendFrame(packetBufferHedderPt, PACKET_HEDDER_SIZE + DG_BUFFER_SIZE);

            // Increment packet index.
            packetIdx++;

            // Increment packet buffer's hedder pointer to the next packet.
            packetBufferHedderPt = packetBufferNextHedderPt;
            packetBufferNextHedderPt = packetBufferHedderPt + PACKET_HEDDER_SIZE + DG_BUFFER_SIZE;

            packetBufferDCTBlockPt = packetBufferHedderPt + PACKET_HEDDER_SIZE;

            // auto end = std::chrono::steady_clock::now() + std::chrono::nanoseconds(50000);
            auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(150);
            while (std::chrono::steady_clock::now() < end);

        } while (encBufferPt < encBufferEndPoint);

        // Create frame data send finished notice packet.
        packetBufferHedderPt[PACKET_IDX_UPPER_IDX] = (char)((unsigned short)packetIdx >> 8);
        packetBufferHedderPt[PACKET_IDX_LOWER_IDX] = (char)packetIdx;
        packetBufferHedderPt[LAST_FRAME_END_IDX_UPPER_IDX] = (char)((unsigned short)_lastFrameEndIdx >> 8);
        packetBufferHedderPt[LAST_FRAME_END_IDX_LOWER_IDX] = (char)_lastFrameEndIdx;
        packetBufferHedderPt[FRAME_OFFSET_IDX] = (char)_currentFrameOffset;
        packetBufferHedderPt[IS_THIS_PACKETEND_IDX] = (char)THIS_PACEKT_IS_FRAMES_LAST;
        packetBufferHedderPt[IS_THIS_FIX_PACKET_IDX] = (char)THIS_PACKET_IS_NOT_FOR_FIX;

        WaitForSingleObject(_socketMutexHandle, INFINITE);
        if (_isItResendRequested == 1) {
            _isItResendRequested = 0;
            _resendPacketHedder[IS_THIS_FIX_PACKET_IDX] = THIS_PACKET_IS_FOR_FIX;
            ResendFrame(_resendPacketHedder, PACKET_HEDDER_SIZE + DG_BUFFER_SIZE);

            auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(150);
            while (std::chrono::steady_clock::now() < end);
        }
        ReleaseMutex(_socketMutexHandle);

        // Notification of transmission complete packet
        SendFrame(packetBufferHedderPt, PACKET_HEDDER_SIZE + DG_BUFFER_SIZE);

        // Record last index of packet split.
        _lastFrameEndIdx = packetIdx;

        // Loop index.
        _currentFrameOffset = (_currentFrameOffset + 1) & BUFFERS_FRAME_COUNT_LOOP_BIT;

        context->Unmap(_bufferTexture, 0);

        return S_OK;
    }

    HRESULT TLabShareScreenServer::ShowResult(
        winrt::com_ptr<ID3D11DeviceContext> context,
        winrt::Windows::Graphics::SizeInt32 contentSize,
        winrt::com_ptr<ID3D11ShaderResourceView> frameSurfaceSRV
    ) {
        // RenderTargetViewをセット
        ID3D11RenderTargetView* pRTVs[1];
        pRTVs[0] = _chainedBufferRTV.get();
        context->OMSetRenderTargets(1, pRTVs, nullptr);

        // RenderTargetの中に ViewPortを指定のオフセットとサイズでセット
        D3D11_VIEWPORT vp = { 0 };
        DXGI_SWAP_CHAIN_DESC1 scd;
        _dxgiSwapChain->GetDesc1(&scd);
        vp.Width = static_cast<float>(scd.Width);
        vp.Height = static_cast<float>(scd.Height);
        context->RSSetViewports(1, &vp);

        // レンダーターゲットの色を初期化(背景色)
        auto clearColor = D2D1::ColorF(D2D1::ColorF::CornflowerBlue);
        context->ClearRenderTargetView(_chainedBufferRTV.get(), &clearColor.r);

        // テクスチャを切り取ってスプライト画像としてディスプレイに表示する処理

        _spriteBatch->Begin();

        CRect sourceRect, destinationRect;

        sourceRect.left = 0;
        sourceRect.top = 0;
        sourceRect.right = contentSize.Width;
        sourceRect.bottom = contentSize.Height;

        destinationRect = FitInBox(
            {
                static_cast<float>(contentSize.Width),
                static_cast<float>(contentSize.Height)
            },
            {
                static_cast<float>(scd.Width),
                static_cast<float>(scd.Height)
            }
        );

        // キャプチャしたフレームをスプライトとしてビューポートにレンダリング
        _spriteBatch->Draw(
            frameSurfaceSRV.get(),
            destinationRect,
            &sourceRect
        );

        _spriteBatch->End();

        DXGI_PRESENT_PARAMETERS pp = { 0 };
        _dxgiSwapChain->Present1(1, 0, &pp);

        return S_OK;
    }

    HRESULT TLabShareScreenServer::CheckWinSize(Direct3D11CaptureFrame frame) {

        auto surfaceDesc = frame.Surface().Description();
        auto itemSize = _captureItem.Size();

        if (itemSize.Width != surfaceDesc.Width ||
            itemSize.Height != surfaceDesc.Height)
        {
            int width = itemSize.Width;
            int height = itemSize.Height;

            SizeInt32 size;
            size.Width = std::max(width, 1);
            size.Height = std::max(height, 1);

            _framePool.Recreate(
                _device,
                DirectXPixelFormat::B8G8R8A8UIntNormalized,
                2,
                size
            );
        }

        return S_OK;
    }

    HRESULT TLabShareScreenServer::CreateDevice() {
        // Create swap chain to rendering capture result.

        // check error.
        WINRT_VERIFY(IsWindow(_ownerHwnd));

        com_ptr<ID3D11Device> d3dDevice = nullptr;
        _device = CreateDirect3DDevice();
        _d3dDevice = GetDXGIInterfaceFromObject<ID3D11Device>(_device);

        auto dxgiDevice = _d3dDevice.as<IDXGIDevice2>();
        com_ptr<IDXGIAdapter> dxgiAdapter;
        check_hresult(dxgiDevice->GetParent(
            guid_of<IDXGIAdapter>(),
            dxgiAdapter.put_void()
        ));
        com_ptr<IDXGIFactory2> dxgiFactory;
        check_hresult(dxgiAdapter->GetParent(
            guid_of<IDXGIFactory2>(),
            dxgiFactory.put_void()
        ));

        // Get pictureBox's Rect transform
        CRect clientRect;
        GetWindowRect(_ownerHwnd, clientRect);

        int width = clientRect.Width();
        int height = clientRect.Height();

        DXGI_SWAP_CHAIN_DESC1 scd = {};
        scd.Width = width;
        scd.Height = height;
        scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.BufferCount = 2;
        scd.SampleDesc.Count = 1;
        scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        scd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

        // set picutreBox's window handle to swap chain.
        check_hresult(
            dxgiFactory->CreateSwapChainForHwnd(
                _d3dDevice.get(),
                _ownerHwnd, // *this,
                &scd,
                nullptr,
                nullptr,
                _dxgiSwapChain.put()
            )
        );

        com_ptr<ID3D11Texture2D> chainedBuffer;
        check_hresult(_dxgiSwapChain->GetBuffer(
            0,
            guid_of<ID3D11Texture2D>(),
            chainedBuffer.put_void()
        ));
        check_hresult(_d3dDevice->CreateRenderTargetView(
            chainedBuffer.get(),
            nullptr,
            _chainedBufferRTV.put()
        ));

        com_ptr<ID3D11DeviceContext> context;
        _d3dDevice->GetImmediateContext(context.put());
        _spriteBatch = std::make_unique<SpriteBatch>(context.get());

        return S_OK;
    }

    HRESULT TLabShareScreenServer::CreateBufferTexture(Size const& itemSize) {
        // context ... com_ptr<ID3D11DeviceContext>
        com_ptr<ID3D11DeviceContext> context;
        _d3dDevice->GetImmediateContext(context.put());

        // _bufferTextureDesc ... D3D11_TEXTURE2D_DESC
        _bufferTextureDesc.Width = (int)itemSize.Width;
        _bufferTextureDesc.Height = (int)itemSize.Height;
        _bufferTextureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        _bufferTextureDesc.ArraySize = 1;
        _bufferTextureDesc.BindFlags = 0;
        _bufferTextureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        _bufferTextureDesc.MipLevels = 1;
        _bufferTextureDesc.MiscFlags = 0;
        _bufferTextureDesc.SampleDesc.Count = 1;
        _bufferTextureDesc.SampleDesc.Quality = 0;
        _bufferTextureDesc.Usage = D3D11_USAGE_STAGING;

        // _bufferTexture...ID3D11Texture2D型
        _d3dDevice->CreateTexture2D(&_bufferTextureDesc, 0, &_bufferTexture);

        _capWinSizeInTexture.left = 0;
        _capWinSizeInTexture.right = 1;
        _capWinSizeInTexture.top = 0;
        _capWinSizeInTexture.bottom = 1;
        _capWinSizeInTexture.front = 0;
        _capWinSizeInTexture.back = 1;

        return S_OK;
    }

    HRESULT TLabShareScreenServer::CreatePacketBuffer() {

        printf("---------------------------------------------------------------------\n");
        printf("start create pacekt buffer\n");

        // Calc frame's block count. and frame's max packet num.
        int maxPacketDivNum = (int)((float)_encBufferSize / (float)DG_BUFFER_SIZE) + 1;

        // Add for end notice packet.
        maxPacketDivNum++;

        printf("maxPacketDivNum: %d\n", maxPacketDivNum);

        // Surplus buffer per packet.
        int surplusPerPacket =
            BLOCK_HEDDER_SIZE +
            BLOCK_AXIS_SIZE *
            BLOCK_AXIS_SIZE *
            ENDIAN_SIZE *
            DST_COLOR_SIZE;

        int surplusDataGramSize = maxPacketDivNum * surplusPerPacket;
        int surplusPacketCount = (int)((float)surplusDataGramSize / (float)DG_BUFFER_SIZE) + 1;
        int surplusTotalBufferSize = surplusDataGramSize + surplusPacketCount * PACKET_HEDDER_SIZE;

        // MSS: Maximum segment size.
        int MSS = PACKET_HEDDER_SIZE + DG_BUFFER_SIZE;

        // Add more buffers assuming that there will be uncopied areas in the packet.
        // One packet has space for block unit size at maximum.
        _packetBufferSize = maxPacketDivNum * MSS + surplusTotalBufferSize;

        _packetBuffer = new char[_packetBufferSize * BUFFERS_FRAME_COUNT];
        memset(_packetBuffer, 0, _packetBufferSize * BUFFERS_FRAME_COUNT);

        printf("_pacektBufferSize: %d\n", _packetBufferSize);
        printf("---------------------------------------------------------------------\n");

        return S_OK;
    }

    HRESULT TLabShareScreenServer::CreateEncodeDevice(int width, int height) {

        printf("---------------------------------------------------------------------\n");
        printf("start create encode device\n");

        // Calclate encoded buffer size.
        _encBufferSize = CalcEncBufferSize(width, height);

        int blockWidth = width / BLOCK_AXIS_SIZE;
        int blockHeight = height / BLOCK_AXIS_SIZE;

        int blockUnitSize =
            BLOCK_HEDDER_SIZE +
            BLOCK_AXIS_SIZE *
            BLOCK_AXIS_SIZE *
            ENDIAN_SIZE *
            DST_COLOR_SIZE;

        int blockSize =
            BLOCK_AXIS_SIZE *
            BLOCK_AXIS_SIZE *
            ENDIAN_SIZE *
            DST_COLOR_SIZE;

        ////////////////////////////////////////////////////////////////////////
        // Create validity with effective and invalid buffer.
        //

        // create validity block buffer.
        int validityBlockNum = blockWidth * blockHeight;

        // Add invalid block buffer.
        int invalidBlockNum = (int)((float)DG_BUFFER_SIZE / (float)blockUnitSize) + 1;

        _encBuffer = new char[_encBufferSize + invalidBlockNum * blockUnitSize];

        ////////////////////////////////////////////////////////////////////////
        // Create decBuffer
        //

        _decBuffer = new unsigned char[width * height * SRC_COLOR_SIZE];

        ////////////////////////////////////////////////////////////////////////
        // Set validity buffer area's packet value.

        char* validityDCTBlockHedderPt = (char*)_encBuffer;

        for (int i = 0; i < validityBlockNum; i++) {
            validityDCTBlockHedderPt[BLOCK_IDX_UPPER_IDX] = (char)((unsigned short)i >> 8);
            validityDCTBlockHedderPt[BLOCK_IDX_LOWER_IDX] = i;
            validityDCTBlockHedderPt[Y_BIT_SIZE_IDX] = NO_NEED_TO_ENCODE;
            validityDCTBlockHedderPt[Cr_BIT_SIZE_IDX] = NO_NEED_TO_ENCODE;
            validityDCTBlockHedderPt[Cb_BIT_SIZE_IDX] = NO_NEED_TO_ENCODE;

            validityDCTBlockHedderPt += blockUnitSize;
        }

        ////////////////////////////////////////////////////////////////////////
        // Set invalid buffer area's packet value.
        //

        char* invalidDCTBlockHedderPt = _encBuffer + _encBufferSize;
        char* invalidDCTBlockYCrCbPt;

        // Set invalid block hedder's value.
        for (int i = 0; i < invalidBlockNum; i++) {
            invalidDCTBlockYCrCbPt = invalidDCTBlockHedderPt + BLOCK_HEDDER_SIZE;

            invalidDCTBlockHedderPt[BLOCK_IDX_UPPER_IDX] = (char)255;
            invalidDCTBlockHedderPt[BLOCK_IDX_LOWER_IDX] = (char)255;
            invalidDCTBlockHedderPt[Y_BIT_SIZE_IDX] = BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE;
            invalidDCTBlockHedderPt[Cr_BIT_SIZE_IDX] = BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE;
            invalidDCTBlockHedderPt[Cb_BIT_SIZE_IDX] = BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE;

            for (int j = 0; j < blockSize; j++) *(invalidDCTBlockYCrCbPt + j) = 0;

            invalidDCTBlockHedderPt += blockUnitSize;
        }

        int result = DecoderInitialize(
            width,
            height,
            _encBuffer,
            _decBuffer
        );

        printf("---------------------------------------------------------------------\n");

        if (result == 0)
            return S_OK;
        else
            return S_FALSE;
    }

    bool TLabShareScreenServer::StopCapture() {
        // if this soft already capturing. stop capture.
        if (IsCapturing() == true) {
            _frameArrived.revoke();

            _captureSession = nullptr;

            _framePool.Close();
            _framePool = nullptr;

            _captureItem = nullptr;
        }

        return true;
    }

    void TLabShareScreenServer::StartCapture(
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item)
    {
        // start screen capture.

        StopCapture();

        _captureItem = item;

        // adjust window size.
        int width = _captureItem.Size().Width;
        int height = _captureItem.Size().Height;

        // Initalize encoder device.
        check_hresult(CreateEncodeDevice(width, height));

        // Create packet buffer.
        check_hresult(CreatePacketBuffer());

        // create texture buffer
        check_hresult(CreateBufferTexture(
            {
                static_cast<float>(width),
                static_cast<float>(height)
            }
        ));

        // create frame pool(texture : rgba32)
        _framePool = Direct3D11CaptureFramePool::Create(
            _device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            SizeInt32{ width, height }
        );

        // set call back function when capture result arrived.
        _frameArrived = _framePool.FrameArrived(
            auto_revoke, { this, &TLabShareScreenServer::OnFrameArrived }
        );

        // regist this capture session.
        _captureSession = _framePool.CreateCaptureSession(item);

        // start capture.
        _captureSession.StartCapture();

        printf("capture started\n");
    }

    GraphicsCaptureItem TLabShareScreenServer::CreateItemForWindow(HWND hWnd) {
        // Create capture item for graphics capture from window handle.

        auto interopFactory = get_activation_factory<
            GraphicsCaptureItem,
            IGraphicsCaptureItemInterop
        >();

        GraphicsCaptureItem item{ nullptr };

        check_hresult(
            interopFactory->CreateForWindow(
                hWnd,
                winrt::guid_of<GraphicsCaptureItem>(),
                winrt::put_abi(item)
            )
        );

        return item;
    }
   
    GraphicsCaptureItem TLabShareScreenServer::CreateItemForMonitor(HWND hWnd) {
        // return monitor handle most closed to specified window handle.

        auto interopFactory = get_activation_factory<
            GraphicsCaptureItem,
            IGraphicsCaptureItemInterop
        >();

        GraphicsCaptureItem item{ nullptr };

        check_hresult(
            interopFactory->CreateForMonitor(
                MonitorFromWindow(hWnd, MONITOR_DEFAULTTOPRIMARY),
                winrt::guid_of<GraphicsCaptureItem>(),
                winrt::put_abi(item)
            )
        );

        return item;
    }

    bool TLabShareScreenServer::StartCaptureForDesiredWindow() {
        // get desktop window handle.
        _targetHwnd = GetDesktopWindow();

        // get monitor's capture item for graphics capture.
        GraphicsCaptureItem item = CreateItemForMonitor(_targetHwnd);

        bool successful = (item != nullptr);

        if (successful == true) {
            StartCapture(item);
        }

        return successful;
    }

    void TLabShareScreenServer::Resize() {
        if (_dxgiSwapChain == nullptr) {
            return;
        }

        CRect clientRect;
        GetWindowRect(_ownerHwnd, clientRect);

        if (IsIconic(_ownerHwnd) == false &&
            clientRect.Width() > 0 &&
            clientRect.Height() > 0)
        {
            _chainedBufferRTV = nullptr;

            _dxgiSwapChain->ResizeBuffers(
                2,
                clientRect.Width(),
                clientRect.Height(),
                DXGI_FORMAT_B8G8R8A8_UNORM,
                0
            );

            com_ptr<ID3D11Texture2D> chainedBuffer;

            check_hresult(
                _dxgiSwapChain->GetBuffer(
                    0,
                    guid_of<ID3D11Texture2D>(),
                    chainedBuffer.put_void()
                )
            );

            check_hresult(
                _d3dDevice->CreateRenderTargetView(
                    chainedBuffer.get(),
                    nullptr,
                    _chainedBufferRTV.put()
                )
            );

            InvalidateRect(_ownerHwnd, nullptr, true);
        }
    }

    void TLabShareScreenServer::OnFrameArrived(
        Direct3D11CaptureFramePool const& sender,
        winrt::Windows::Foundation::IInspectable const& args)
    {
#if DEBUG_MODE && DEBUG_FPS
        DWORD current = GetTickCount();
        DWORD delta = current - _lastTime;
        printf("Current: %d, Delta: %d, FPS: %f\n", current, delta, 1 / ((float)delta) * 1000);
        _lastTime = current;
#endif

        // On caputred frame arrived.
        
        // this function maybe running in main thread.
        // printf("Frame arrived: %d\n");

        Direct3D11CaptureFrame frame = sender.TryGetNextFrame();

        com_ptr<ID3D11Texture2D> frameSurface =
            GetDXGIInterfaceFromObject<ID3D11Texture2D>(frame.Surface());
        SizeInt32 contentSize = frame.ContentSize();

        com_ptr<ID3D11DeviceContext> context;
        _d3dDevice->GetImmediateContext(context.put());

        com_ptr<ID3D11ShaderResourceView> frameSurfaceSRV;
        check_hresult(_d3dDevice->CreateShaderResourceView(
            frameSurface.get(),
            nullptr,
            frameSurfaceSRV.put()
        ));

#if ENABLE_SCREENSHARE
        // send capture result to connected client via network.
        CastScreen(context, contentSize, frameSurface);
#endif

#if DEBUG_MODE && DEBUG_WINDOW
        // debug captured result on desktop window.
        ShowResult(context, contentSize, frameSurfaceSRV);
        CheckWinSize(frame);
#endif
    }
}
