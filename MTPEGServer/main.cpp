#pragma once

#include "pch.h"
#include "tpeg_common.h"
#include "windows_common.h"
#include "motion_tpeg.h"
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

TLab::MTPEGServer mtpeg_server = TLab::MTPEGServer();

static LRESULT CALLBACK mainWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CLOSE:
        if (mtpeg_server.IsConnecting()) {
            mtpeg_server.DisconnectClient();
        }
        if (mtpeg_server.IsCapturing()) {
            mtpeg_server.StopCapture();
        }
        DestroyWindow(hWnd);
        break;
    case WM_DESTROY:
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
    printf("start Motion TPEG Server\n");

    std::cout << "Motion TPEG Server started successfully." << std::flush;

    printf("create window ...\n");

    TCHAR lp_class_name[] = TEXT("Motion TPEG Server");
    TCHAR lp_title_name[] = TEXT("Capture Result Debug Window");

    // https://p-tal.hatenadiary.org/entry/20091006/1254790659#:~:text=initDialogue()%0A%7B%0A%20%20HINSTANCE%20hInst%20%3D-,GetModuleHandle,-(NULL)%3B%0A%0A%20%20m_hMenu
    HINSTANCE h_instance = GetModuleHandle(NULL);

    WNDCLASS wc;    // set window class attributes
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = mainWindowProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = h_instance;
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName = NULL;
    wc.lpszClassName = lp_class_name;

    RegisterClass(&wc); // register window class

    // https://blog.goo.ne.jp/masaki_goo_2006/e/d4ba04d58719e1f28b6e90c2e4d09a3d#:~:text=//%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%0A//-,%E3%82%A6%E3%82%A4%E3%83%B3%E3%83%89%E3%82%A6%E3%81%AE%E4%BD%9C%E6%88%90(OK),-//%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%2D%0Astatic%20HWND
    HWND owner_hwnd = CreateWindow(
        lp_class_name,
        lp_title_name,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        800,
        450,
        NULL,
        NULL,
        h_instance,
        NULL
    );

#if DEBUG_MODE && DEBUG_WINDOW
    // https://www.tokovalue.jp/function/ShowWindow.htm#:~:text=%E3%81%95%E3%82%8C%E3%81%AA%E3%81%84%E3%80%82-,SW_SHOWNA,-%E3%82%A6%E3%82%A3%E3%83%B3%E3%83%89%E3%82%A6%E3%82%92%E7%8F%BE%E5%9C%A8
    ShowWindow(owner_hwnd, SW_SHOWNORMAL);
    UpdateWindow(owner_hwnd);
#endif

    if (OpenSharedMemoryMappingObiect(L"MTPEGServer", L"Shareing") != 1)
    {
        printf("error while running mtpeg server, press enter to finish");

        while (1) {
            if ('\r' == getch()) {
                break;
            }
        }

        return 0;
    }

    mtpeg_server.SetOwnerHandle(owner_hwnd);
    mtpeg_server.ConnectToClient(
        strtol(argv[1], NULL, 10),
        strtol(argv[2], NULL, 10),
        argv[3]
    );
    mtpeg_server.StartCaptureForDesiredWindow();

    WaitForSingleObject(_keep_alive_mutex_handle, INFINITE);
    *_mapping_object = 1;
    ReleaseMutex(_keep_alive_mutex_handle);

    MSG Msg;
    while (GetMessage(&Msg, NULL, 0, 0) > 0) {
        WaitForSingleObject(_keep_alive_mutex_handle, INFINITE);
        if (*_mapping_object == 0) {
            ReleaseMutex(_keep_alive_mutex_handle);

            Msg.hwnd = owner_hwnd;
            Msg.message = WM_CLOSE;
            TranslateMessage(&Msg);
            DispatchMessage(&Msg);

            Msg.message = WM_DESTROY;
            TranslateMessage(&Msg);
            DispatchMessage(&Msg);

            break;
        }
        else {
            ReleaseMutex(_keep_alive_mutex_handle);

            TranslateMessage(&Msg);
            DispatchMessage(&Msg);
        }
    }

    FreeUp();

    return Msg.wParam;
}

namespace {

    auto FitInBox(Size const& source, Size const& destination) {

        // Compute a rectangle that fits in the box while preserving the aspect ratio

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

namespace TLab
{
    void MTPEGServer::SetOwnerHandle(HWND owner_hwnd) {
        _owner_hwnd = owner_hwnd;

        this->CreateDevice();
    }

    bool MTPEGServer::ConnectToClient(uint16_t server_port, uint16_t client_port, char* client_addr)
    {

#if TOUCH_EMULATION
        SetEmulateTarget(_target_hwnd);

        SetReceiveCallbackHandler(
            [](char* recv_packet) {
                WindowSendMessage(
                    (float)recv_packet[0 * sizeof(float)],
                    (float)recv_packet[1 * sizeof(float)],
                    (float)recv_packet[2 * sizeof(float)],
                    (float)recv_packet[3 * sizeof(float)]
                );
            },
            4
                );
#endif

        printf("---------------------------------------------------------------------\n");
        printf("start create socket\n");
        int successful = CreateSocket(server_port, client_port, client_addr, 2);
        printf("---------------------------------------------------------------------\n");

#if CAST_FRAME && ENABLE_RESEND
        /**
        * frame index(1byte)
        * packet index(1byte)
        */
        int recv_buffer_size = 3;

        /**
        * Don't forget to put this, which is the reference destination
        * of the variable, in the capture list [inside this].
        */
        SetCallback(
            [this](char* recv_packet) {
                WaitForSingleObject(_socket_mutex_handle, INFINITE);
                _resend_packet_hedder =
                    _packet_buffer +
                    recv_packet[2] * _packet_buffer_size +
                    (((unsigned short)recv_packet[0] << 8) + recv_packet[1]) * (PACKET_HEDDER_SIZE + DGRAM_BUFFER_SIZE);

                if (_resend_packet_hedder < _packet_buffer ||
                    _resend_packet_hedder > _packet_buffer + 2 * _packet_buffer_size)
                {
                    printf("[packet resend request] problem is occured\n");
                    ReleaseMutex(_socket_mutex_handle);
                    return;
                }

                _resend_requested = 1;
                ReleaseMutex(_socket_mutex_handle);
            }, recv_buffer_size
        );

        StartReceiveAsync(0, 1);
#endif

        return (successful == 0);
    }

    bool MTPEGServer::DisconnectClient() {
        CloseSocket();
        return true;
    }

    HRESULT MTPEGServer::CastScreen(
        winrt::com_ptr<ID3D11DeviceContext> context,
        winrt::Windows::Graphics::SizeInt32 content_size,
        winrt::com_ptr<ID3D11Texture2D> frame_surface)
    {

#if RELEASE_CONFIG
        if (!IsConnecting()) {
            return S_FALSE;
        }
#endif
        _cap_win_size_in_texture.right = content_size.Width;    // Copy captured frame texture to sub resource.
        _cap_win_size_in_texture.bottom = content_size.Height;
        context->CopySubresourceRegion(
            _buffer_texture,
            0,
            0,
            0,
            0,
            frame_surface.get(),
            0,
            &_cap_win_size_in_texture
        );

        D3D11_MAPPED_SUBRESOURCE mapd;  // map texture sub resource to access from CPU (with unsigned char array)
        context->Map(
            _buffer_texture,
            0,
            D3D11_MAP_READ,
            0,
            &mapd
        );

        // this data can access from unsigned char array.
        unsigned char* source = static_cast<unsigned char*>(mapd.pData);

        EncodeFrame(source);

        unsigned short packet_idx = 0;

        char* packet_buffer_ptr = _packet_buffer + _packet_buffer_size * _current_frame_offset;
        char* packet_buffer_next_ptr = packet_buffer_ptr + PACKET_HEDDER_SIZE + DGRAM_BUFFER_SIZE;

        char* packet_buffer_data_ptr = packet_buffer_ptr + PACKET_HEDDER_SIZE;

        char* encoded_frame_buffer_ptr = _encoded_frame_buffer;
        char* encoded_frame_buffer_end_point = _encoded_frame_buffer + _encoded_frame_buffer_size;

        /**
        * since the char type can only represent positive values ​​up to 127,
        * we use unsigned char so that 64*2=128 can be represented.
        */
        unsigned char block_buffer_size_y, block_buffer_size_cr, block_buffer_size_cb;

        do {
            packet_buffer_ptr[PACKET_INDEX_BE] = (char)((unsigned short)packet_idx >> 8);
            packet_buffer_ptr[PACKET_INDEX_LE] = (char)packet_idx;
            packet_buffer_ptr[LAST_FRAME_FINAL_INDEX_BE] = (char)((unsigned short)_last_frame_final_idx >> 8);
            packet_buffer_ptr[LAST_FRAME_FINAL_INDEX_LE] = (char)_last_frame_final_idx;
            packet_buffer_ptr[FRAME_OFFSET_INDEX] = (char)_current_frame_offset;
            packet_buffer_ptr[IS_THIS_PACKETEND] = (char)FALSE;
            packet_buffer_ptr[IS_THIS_FIX_PACKET] = (char)FALSE;

            while (1) {

                block_buffer_size_y = (unsigned char)(encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_B]) << ENDIAN_SIZE_LOG2;
                block_buffer_size_cr = (unsigned char)(encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_G]) << ENDIAN_SIZE_LOG2;
                block_buffer_size_cb = (unsigned char)(encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_R]) << ENDIAN_SIZE_LOG2;

                /**
                * add flag both block index big endian and little endian are 255
                * to the packet to notify that it is the end of valid data.
                */
                if (packet_buffer_next_ptr - 2 <
                    packet_buffer_data_ptr +
                    BLOCK_HEDDER_SIZE +
                    block_buffer_size_y + block_buffer_size_cr + block_buffer_size_cb)
                {
                    packet_buffer_data_ptr[BLOCK_INDEX_BE] = (char)255;
                    packet_buffer_data_ptr[BLOCK_INDEX_LE] = (char)255;

                    break;
                }

                packet_buffer_data_ptr[BLOCK_INDEX_BE] = encoded_frame_buffer_ptr[BLOCK_INDEX_BE];
                packet_buffer_data_ptr[BLOCK_INDEX_LE] = encoded_frame_buffer_ptr[BLOCK_INDEX_LE];
                packet_buffer_data_ptr[BLOCK_BIT_SIZE_B] = encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_B];
                packet_buffer_data_ptr[BLOCK_BIT_SIZE_G] = encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_G];
                packet_buffer_data_ptr[BLOCK_BIT_SIZE_R] = encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_R];

                packet_buffer_data_ptr += (block_buffer_size_y != NO_NEED_TO_ENCODE) * BLOCK_HEDDER_SIZE;

                encoded_frame_buffer_ptr += BLOCK_HEDDER_SIZE;

                // Copy Y values.
                memcpy((void*)packet_buffer_data_ptr, (void*)encoded_frame_buffer_ptr, block_buffer_size_y);
                packet_buffer_data_ptr += block_buffer_size_y;
                encoded_frame_buffer_ptr += BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE * ENDIAN_SIZE;

                // Copy Cr value.
                memcpy((void*)packet_buffer_data_ptr, (void*)encoded_frame_buffer_ptr, block_buffer_size_cr);
                packet_buffer_data_ptr += block_buffer_size_cr;
                encoded_frame_buffer_ptr += BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE * ENDIAN_SIZE;

                // Copy Cb value.
                memcpy((void*)packet_buffer_data_ptr, (void*)encoded_frame_buffer_ptr, block_buffer_size_cb);
                packet_buffer_data_ptr += block_buffer_size_cb;
                encoded_frame_buffer_ptr += BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE * ENDIAN_SIZE;

                // Now encoded_frame_buffer_ptr's pointer is next block's hedder.
            }

            WaitForSingleObject(_socket_mutex_handle, INFINITE);
            if (_resend_requested == 1) {
                _resend_requested = 0;
                _resend_packet_hedder[IS_THIS_FIX_PACKET] = TRUE;
                ResendFrame(_resend_packet_hedder, PACKET_HEDDER_SIZE + DGRAM_BUFFER_SIZE);

#if CONGESTION_CONTROL
                auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(150);
                while (std::chrono::steady_clock::now() < end);
#endif
            }
            ReleaseMutex(_socket_mutex_handle);

            SendFrame(packet_buffer_ptr, PACKET_HEDDER_SIZE + DGRAM_BUFFER_SIZE);   // Send frame data

            packet_idx++;

            packet_buffer_ptr = packet_buffer_next_ptr;
            packet_buffer_next_ptr = packet_buffer_ptr + PACKET_HEDDER_SIZE + DGRAM_BUFFER_SIZE;

            packet_buffer_data_ptr = packet_buffer_ptr + PACKET_HEDDER_SIZE;

#if CONGESTION_CONTROL
            auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(150);
            while (std::chrono::steady_clock::now() < end);
#endif

        } while (encoded_frame_buffer_ptr < encoded_frame_buffer_end_point);

        packet_buffer_ptr[PACKET_INDEX_BE] = (char)((unsigned short)packet_idx >> 8);
        packet_buffer_ptr[PACKET_INDEX_LE] = (char)packet_idx;
        packet_buffer_ptr[LAST_FRAME_FINAL_INDEX_BE] = (char)((unsigned short)_last_frame_final_idx >> 8);
        packet_buffer_ptr[LAST_FRAME_FINAL_INDEX_LE] = (char)_last_frame_final_idx;
        packet_buffer_ptr[FRAME_OFFSET_INDEX] = (char)_current_frame_offset;
        packet_buffer_ptr[IS_THIS_PACKETEND] = (char)TRUE;
        packet_buffer_ptr[IS_THIS_FIX_PACKET] = (char)FALSE;

        WaitForSingleObject(_socket_mutex_handle, INFINITE);
        if (_resend_requested == 1) {
            _resend_requested = 0;
            _resend_packet_hedder[IS_THIS_FIX_PACKET] = TRUE;
            ResendFrame(_resend_packet_hedder, PACKET_HEDDER_SIZE + DGRAM_BUFFER_SIZE);

#if CONGESTION_CONTROL
            auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(150);
            while (std::chrono::steady_clock::now() < end);
#endif
        }
        ReleaseMutex(_socket_mutex_handle);

        SendFrame(packet_buffer_ptr, PACKET_HEDDER_SIZE + DGRAM_BUFFER_SIZE);

        _last_frame_final_idx = packet_idx;

        _current_frame_offset = (_current_frame_offset + 1) & FRAME_LOOP_NUM;

        context->Unmap(_buffer_texture, 0);

        return S_OK;
    }

    HRESULT MTPEGServer::ShowResult(
        winrt::com_ptr<ID3D11DeviceContext> context,
        winrt::Windows::Graphics::SizeInt32 content_size,
        winrt::com_ptr<ID3D11ShaderResourceView> frame_surface_srv)
    {
        ID3D11RenderTargetView* pRTVs[1];
        pRTVs[0] = _chained_buffer_rtv.get();
        context->OMSetRenderTargets(1, pRTVs, nullptr);

        D3D11_VIEWPORT vp = { 0 };
        DXGI_SWAP_CHAIN_DESC1 scd;
        _dxgi_swap_chain->GetDesc1(&scd);
        vp.Width = static_cast<float>(scd.Width);
        vp.Height = static_cast<float>(scd.Height);
        context->RSSetViewports(1, &vp);

        auto clearColor = D2D1::ColorF(D2D1::ColorF::CornflowerBlue);
        context->ClearRenderTargetView(_chained_buffer_rtv.get(), &clearColor.r);

        _sprite_batch->Begin();

        CRect src_rect, dst_rect;
        src_rect.left = 0;
        src_rect.top = 0;
        src_rect.right = content_size.Width;
        src_rect.bottom = content_size.Height;

        dst_rect = FitInBox(
            { static_cast<float>(content_size.Width), static_cast<float>(content_size.Height) },
            { static_cast<float>(scd.Width), static_cast<float>(scd.Height) });

        _sprite_batch->Draw(frame_surface_srv.get(), dst_rect, &src_rect);

        _sprite_batch->End();

        DXGI_PRESENT_PARAMETERS pp = { 0 };
        _dxgi_swap_chain->Present1(1, 0, &pp);

        return S_OK;
    }

    HRESULT MTPEGServer::CheckWinSize(Direct3D11CaptureFrame frame) {

        auto surfaceDesc = frame.Surface().Description();
        auto item_size = _capture_item.Size();

        if (item_size.Width != surfaceDesc.Width ||
            item_size.Height != surfaceDesc.Height)
        {
            int width = item_size.Width;
            int height = item_size.Height;

            SizeInt32 size;
            size.Width = std::max(width, 1);
            size.Height = std::max(height, 1);

            _frame_pool.Recreate(
                _device,
                DirectXPixelFormat::B8G8R8A8UIntNormalized,
                2,
                size
            );
        }

        return S_OK;
    }

    HRESULT MTPEGServer::CreateDevice() {

        WINRT_VERIFY(IsWindow(_owner_hwnd));

        com_ptr<ID3D11Device> d3dDevice = nullptr;
        _device = CreateDirect3DDevice();
        _d3d_device = GetDXGIInterfaceFromObject<ID3D11Device>(_device);

        auto dxgiDevice = _d3d_device.as<IDXGIDevice2>();
        com_ptr<IDXGIAdapter> dxgiAdapter;
        check_hresult(dxgiDevice->GetParent(
            guid_of<IDXGIAdapter>(),
            dxgiAdapter.put_void()));

        com_ptr<IDXGIFactory2> dxgiFactory;
        check_hresult(dxgiAdapter->GetParent(
            guid_of<IDXGIFactory2>(),
            dxgiFactory.put_void()));

        CRect client_rect;
        GetWindowRect(_owner_hwnd, client_rect);

        int width = client_rect.Width();
        int height = client_rect.Height();

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
                _d3d_device.get(),
                _owner_hwnd, // *this,
                &scd,
                nullptr,
                nullptr,
                _dxgi_swap_chain.put()
            )
        );

        com_ptr<ID3D11Texture2D> chainedBuffer;
        check_hresult(_dxgi_swap_chain->GetBuffer(
            0,
            guid_of<ID3D11Texture2D>(),
            chainedBuffer.put_void()
        ));
        check_hresult(_d3d_device->CreateRenderTargetView(
            chainedBuffer.get(),
            nullptr,
            _chained_buffer_rtv.put()
        ));

        com_ptr<ID3D11DeviceContext> context;
        _d3d_device->GetImmediateContext(context.put());
        _sprite_batch = std::make_unique<SpriteBatch>(context.get());

        return S_OK;
    }

    HRESULT MTPEGServer::CreateBufferTexture(Size const& item_size) {
        // context ... com_ptr<ID3D11DeviceContext>
        com_ptr<ID3D11DeviceContext> context;
        _d3d_device->GetImmediateContext(context.put());

        // _buffer_texture_desc ... D3D11_TEXTURE2D_DESC
        _buffer_texture_desc.Width = (int)item_size.Width;
        _buffer_texture_desc.Height = (int)item_size.Height;
        _buffer_texture_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        _buffer_texture_desc.ArraySize = 1;
        _buffer_texture_desc.BindFlags = 0;
        _buffer_texture_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        _buffer_texture_desc.MipLevels = 1;
        _buffer_texture_desc.MiscFlags = 0;
        _buffer_texture_desc.SampleDesc.Count = 1;
        _buffer_texture_desc.SampleDesc.Quality = 0;
        _buffer_texture_desc.Usage = D3D11_USAGE_STAGING;

        // _buffer_texture...ID3D11Texture2D型
        _d3d_device->CreateTexture2D(&_buffer_texture_desc, 0, &_buffer_texture);

        _cap_win_size_in_texture.left = 0;
        _cap_win_size_in_texture.right = 1;
        _cap_win_size_in_texture.top = 0;
        _cap_win_size_in_texture.bottom = 1;
        _cap_win_size_in_texture.front = 0;
        _cap_win_size_in_texture.back = 1;

        return S_OK;
    }

    HRESULT MTPEGServer::CreatePacketBuffer() {

        printf("---------------------------------------------------------------------\n");
        printf("start create packet buffer\n");

        // Calc frame's block count. and frame's max packet num.
        int packet_div_num_limmit = (int)((float)_encoded_frame_buffer_size / DGRAM_BUFFER_SIZE) + 1;

        packet_div_num_limmit++;    // add for end notice packet.

        printf("packet_div_num_limmit: %d\n", packet_div_num_limmit);

        int surplus_datagram_size = packet_div_num_limmit * BLOCK_BUFFER_SIZE;
        int surplus_packet_count = (int)((float)surplus_datagram_size / DGRAM_BUFFER_SIZE) + 1;
        int surplus_total_buffer_size = surplus_datagram_size + surplus_packet_count * PACKET_HEDDER_SIZE;

        // add more buffers assuming that there will be uncopied areas in the packet.
        // one packet has space for block unit size at maximum.
        _packet_buffer_size = packet_div_num_limmit * MSS + surplus_total_buffer_size;

        _packet_buffer = new char[_packet_buffer_size * BUFFERS_FRAME_COUNT];
        memset(_packet_buffer, 0, _packet_buffer_size * BUFFERS_FRAME_COUNT);

        printf("_packet_buffer_size: %d\n", _packet_buffer_size);
        printf("---------------------------------------------------------------------\n");

        return S_OK;
    }

    HRESULT MTPEGServer::CreateEncodeDevice(int width, int height) {

        printf("---------------------------------------------------------------------\n");
        printf("start create encode device\n");

        _encoded_frame_buffer_size = EncodedFrameBufferSize(width, height);

        int block_width = width / BLOCK_AXIS_SIZE;
        int block_height = height / BLOCK_AXIS_SIZE;

        int validity_block_num = block_width * block_height;

        int invalid_block_num = (int)((float)DGRAM_BUFFER_SIZE / BLOCK_BUFFER_SIZE) + 1;

        _encoded_frame_buffer = new char[_encoded_frame_buffer_size + invalid_block_num * BLOCK_BUFFER_SIZE];

        _decoded_frame_buffer = new unsigned char[width * height * SRC_COLOR_SIZE];

        char* validity_block_buffer_ptr = (char*)_encoded_frame_buffer;
        for (int i = 0; i < validity_block_num; i++) {
            validity_block_buffer_ptr[BLOCK_INDEX_BE] = (char)((unsigned short)i >> 8);
            validity_block_buffer_ptr[BLOCK_INDEX_LE] = i;
            validity_block_buffer_ptr[BLOCK_BIT_SIZE_B] = NO_NEED_TO_ENCODE;
            validity_block_buffer_ptr[BLOCK_BIT_SIZE_G] = NO_NEED_TO_ENCODE;
            validity_block_buffer_ptr[BLOCK_BIT_SIZE_R] = NO_NEED_TO_ENCODE;

            validity_block_buffer_ptr += BLOCK_BUFFER_SIZE;
        }

        char* invalid_block_buffer_ptr = _encoded_frame_buffer + _encoded_frame_buffer_size;
        for (int i = 0; i < invalid_block_num; i++) { // Set invalid block hedder's value
            invalid_block_buffer_ptr[BLOCK_INDEX_BE] = (char)255;
            invalid_block_buffer_ptr[BLOCK_INDEX_LE] = (char)255;
            invalid_block_buffer_ptr[BLOCK_BIT_SIZE_B] = BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE;
            invalid_block_buffer_ptr[BLOCK_BIT_SIZE_G] = BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE;
            invalid_block_buffer_ptr[BLOCK_BIT_SIZE_R] = BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE;

            for (int j = 0; j < BLOCK_BUFFER_SIZE_WITHOUT_HEDDER; j++) {
                *(invalid_block_buffer_ptr + BLOCK_HEDDER_SIZE + j) = 0;
            }

            invalid_block_buffer_ptr += BLOCK_BUFFER_SIZE;
        }

        int result = DecoderInitialize(
            width, height,
            _encoded_frame_buffer,
            _decoded_frame_buffer);

        printf("---------------------------------------------------------------------\n");

        return (result == 0) ? S_OK : S_FALSE;
    }

    bool MTPEGServer::StopCapture() {

        if (IsCapturing()) {    // if this soft already capturing. stop capture
            _frame_arrived.revoke();

            _capture_session = nullptr;

            _frame_pool.Close();
            _frame_pool = nullptr;

            _capture_item = nullptr;
        }

        return true;
    }

    void MTPEGServer::StartCapture(
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item)
    {

        StopCapture();  // start screen capture

        _capture_item = item;

        int width = _capture_item.Size().Width; // adjust window size
        int height = _capture_item.Size().Height;

        check_hresult(CreateEncodeDevice(width, height));   // initalize encoder device

        check_hresult(CreatePacketBuffer());    // create packet buffer

        check_hresult(CreateBufferTexture(  // create texture buffer
            { static_cast<float>(width), static_cast<float>(height) }));

        _frame_pool = Direct3D11CaptureFramePool::Create(   // create frame pool(texture : rgba32)
            _device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            SizeInt32{ width, height }
        );

        _frame_arrived = _frame_pool.FrameArrived(  // set call back function when capture result arrived.
            auto_revoke, { this, &MTPEGServer::OnFrameArrived }
        );

        _capture_session = _frame_pool.CreateCaptureSession(item);  // regist this capture session.

        _capture_session.StartCapture();    // start capture.

        printf("capture started\n");
    }

    GraphicsCaptureItem MTPEGServer::CreateItemForWindow(HWND hWnd) {
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

    GraphicsCaptureItem MTPEGServer::CreateItemForMonitor(HWND hWnd) {
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

    bool MTPEGServer::StartCaptureForDesiredWindow() {

        _target_hwnd = GetDesktopWindow();  // get desktop window handle.

        GraphicsCaptureItem item = CreateItemForMonitor(_target_hwnd);  // get monitor's capture item for graphics capture.

        bool successful = (item != nullptr);

        if (successful) {
            StartCapture(item);
        }

        return successful;
    }

    void MTPEGServer::Resize() {
        if (_dxgi_swap_chain == nullptr) {
            return;
        }

        CRect client_rect;
        GetWindowRect(_owner_hwnd, client_rect);

        if (!IsIconic(_owner_hwnd) && client_rect.Width() > 0 && client_rect.Height() > 0)
        {
            _chained_buffer_rtv = nullptr;

            _dxgi_swap_chain->ResizeBuffers(
                2,
                client_rect.Width(),
                client_rect.Height(),
                DXGI_FORMAT_B8G8R8A8_UNORM,
                0
            );

            com_ptr<ID3D11Texture2D> chainedBuffer;

            check_hresult(
                _dxgi_swap_chain->GetBuffer(
                    0,
                    guid_of<ID3D11Texture2D>(),
                    chainedBuffer.put_void()
                )
            );

            check_hresult(
                _d3d_device->CreateRenderTargetView(
                    chainedBuffer.get(),
                    nullptr,
                    _chained_buffer_rtv.put()
                )
            );

            InvalidateRect(_owner_hwnd, nullptr, true);
        }
    }

    void MTPEGServer::OnFrameArrived(
        Direct3D11CaptureFramePool const& sender,
        winrt::Windows::Foundation::IInspectable const& args)
    {
#if DEBUG_MODE && DEBUG_FPS
        DWORD current = GetTickCount();
        DWORD delta = current - _last_time;
        printf("Current: %d, Delta: %d, FPS: %f\n", current, delta, 1 / ((float)delta) * 1000);
        _last_time = current;
#endif

        Direct3D11CaptureFrame frame = sender.TryGetNextFrame();

        com_ptr<ID3D11Texture2D> frame_surface =
            GetDXGIInterfaceFromObject<ID3D11Texture2D>(frame.Surface());
        SizeInt32 content_size = frame.ContentSize();

        com_ptr<ID3D11DeviceContext> context;
        _d3d_device->GetImmediateContext(context.put());

        com_ptr<ID3D11ShaderResourceView> frame_surface_srv;
        check_hresult(_d3d_device->CreateShaderResourceView(
            frame_surface.get(),
            nullptr,
            frame_surface_srv.put()
        ));

#if CAST_FRAME
        /**
        * send capture result to connected client via network
        * separate threads can be run to adjust the bit rate.
        */
        std::thread cast_screen_task(&MTPEGServer::CastScreen, this, context, content_size, frame_surface);
        cast_screen_task.join();
#endif

#if DEBUG_MODE && DEBUG_WINDOW
        /**
        * debug captured result on desktop window
        */
        ShowResult(context, content_size, frame_surface_srv);
        CheckWinSize(frame);
#endif
    }
}
