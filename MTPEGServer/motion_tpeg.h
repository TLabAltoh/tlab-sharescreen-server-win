#pragma once

#include "motion_tpeg_util.h"
#include "windows_common.h"
#include "cpp_socket_util.h"
#include "shared_memory_util.h"
#include "mouse_emulator.h"

#define RELEASE_CONFIG 1
#define CONGESTION_CONTROL 1
#define CAST_FRAME 1
#define ENABLE_RESEND 1
#define TOUCH_EMULATION 0
#define DEBUG_MODE 1
#define DEBUG_WINDOW 0
#define DEBUG_FPS 0

#define BUFFERS_FRAME_COUNT 2
#define FRAME_LOOP_NUM 1

#define MSS (PACKET_HEDDER_SIZE + DGRAM_BUFFER_SIZE)

#define BLOCK_BUFFER_SIZE_WITHOUT_HEDDER (BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE * ENDIAN_SIZE * DST_COLOR_SIZE)
#define BLOCK_BUFFER_SIZE (BLOCK_HEDDER_SIZE + BLOCK_BUFFER_SIZE_WITHOUT_HEDDER)

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::System;
using namespace winrt::Windows::Graphics;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

namespace TLab
{
    struct MTPEGServer
    {
    public:

        /* ----------------------------------------------------------- */

        // for capture

        void SetOwnerHandle(HWND owner_hwnd);
        bool StartCaptureForDesiredWindow();
        bool StopCapture();
        bool IsCapturing() { return _frame_pool != nullptr; }
        bool IsConnecting() { return _is_connecting; }

        // for conenct client
        bool ConnectToClient(
            uint16_t server_port,
            uint16_t client_port,
            char* client_addr);
        bool DisconnectClient();

        void Resize();

        /* ----------------------------------------------------------- */

        HWND _target_hwnd;

    private:

        /* -------------------------------------------------------------------------------------- */

        // for capture

        HRESULT CreateDevice();
        HRESULT CreateBufferTexture(Size const& item_size);
        GraphicsCaptureItem CreateItemForWindow(HWND hWnd);
        GraphicsCaptureItem CreateItemForMonitor(HWND hWnd);
        void StartCapture(winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item);
        void OnFrameArrived(
            Direct3D11CaptureFramePool const& sender,
            winrt::Windows::Foundation::IInspectable const& args);

        // for send packet

        HRESULT CreatePacketBuffer();
        HRESULT CreateEncodeDevice(int width, int height);
        HRESULT CastScreen(
            winrt::com_ptr<ID3D11DeviceContext> context,
            winrt::Windows::Graphics::SizeInt32 content_size,
            winrt::com_ptr<ID3D11Texture2D> frame_surface);
        HRESULT ShowResult(
            winrt::com_ptr<ID3D11DeviceContext> context,
            winrt::Windows::Graphics::SizeInt32 content_size,
            winrt::com_ptr<ID3D11ShaderResourceView> frame_surface_srv);
        HRESULT CheckWinSize(Direct3D11CaptureFrame frame);

        /* -------------------------------------------------------------------------------------- */

        HWND _owner_hwnd;

        /* ---------------------------------------------------- */

        // Convert texture to byte array to send as a packet

        D3D11_TEXTURE2D_DESC _buffer_texture_desc;
        ID3D11Texture2D* _buffer_texture;
        D3D11_BOX _cap_win_size_in_texture;

        int _encoded_frame_buffer_size;
        int _packet_buffer_size;
        char _current_frame_offset = 0;
        char _resend_requested = 0;
        char* _packet_buffer;
        char* _resend_packet_hedder;
        char* _encoded_frame_buffer;
        unsigned char* _decoded_frame_buffer;
        unsigned short _last_frame_final_idx = 0;

#if DEBUG_MODE && DEBUG_FPS
        DWORD _last_time = 0;
#endif

        /* ---------------------------------------------------- */

        winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice _device{ nullptr };
        winrt::com_ptr<ID3D11Device> _d3d_device;
        winrt::com_ptr<IDXGISwapChain1> _dxgi_swap_chain;
        winrt::com_ptr<ID3D11RenderTargetView> _chained_buffer_rtv;

        std::unique_ptr<::DirectX::SpriteBatch> _sprite_batch;

        winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool _frame_pool{ nullptr };
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem _capture_item{ nullptr };
        winrt::Windows::Graphics::Capture::GraphicsCaptureSession _capture_session{ nullptr };
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::FrameArrived_revoker _frame_arrived;
    };
}
