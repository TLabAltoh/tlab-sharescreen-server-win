#pragma once

#include "windows_common.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <list>
#include <thread>
#include <functional>

#define TLAB_SHARESCREEN true
#define USE_TCP false
#define DGRAM_BUFFER_SIZE 1443
#define MAXIMUN_SOCKET_SIZE 4

/**
*  MTU document
*   - https://milestone-of-se.nesuke.com/nw-basic/as-nw-engineer/udp-mtu-mss/
*   - https://qwerty.work/blog/2019/12/mtu-optimize-command-windows10.php
*/

namespace {

    unsigned short _server_ports[MAXIMUN_SOCKET_SIZE];
    unsigned short _client_ports[MAXIMUN_SOCKET_SIZE];
    int _src_sockets[MAXIMUN_SOCKET_SIZE];
    int _dst_sockets[MAXIMUN_SOCKET_SIZE];
    struct sockaddr_in _src_addrs[MAXIMUN_SOCKET_SIZE];
    struct sockaddr_in _dst_addrs[MAXIMUN_SOCKET_SIZE];

#if TLAB_SHARESCREEN
    int _src_socket_0;
    int _src_socket_1;
    struct sockaddr_in _dst_addr_0;
    struct sockaddr_in _dst_addr_1;
#endif

    int _num_sockets = 0;

    bool _is_connecting = false;
    bool _keep_alive = false;
    bool _recv_running = false;

#if USE_TCP == false
    char* _client_addr;
#endif

    HANDLE _socket_mutex_handle = NULL; // ソケットの排他制御

    bool IsConnecting() { return _is_connecting; }

    using OnReceiveMessage = std::function<void(char* recv_buffer)>;
    std::list<std::tuple<OnReceiveMessage, int>> _callbacks;

    std::list<std::thread> _recv_threads;

    void SetCallback(OnReceiveMessage on_recv_message, int recv_buffer_size)
    {
        _callbacks.push_front(std::make_tuple(on_recv_message, recv_buffer_size));
    }

    void CloseSocket() {

        _is_connecting = false;
        _keep_alive = false;

        if (_recv_running) {

            _recv_running = false;

            for (int i = 0; i < _num_sockets; i++) {

#if USE_TCP
                closesocket(_dst_sockets[i]);
#else
                // Generate a termination interrupt on sockets waiting for reception.
                // (The return value of recv_length becomes -1).
                closesocket(_src_sockets[i]);
#endif
            }

            for (auto threadItr = _recv_threads.begin(); threadItr != _recv_threads.end(); threadItr++)
            {
                (*threadItr).join();    // Kill all running background processes in a for loop
            }
        }

        if (_socket_mutex_handle != NULL) {
            CloseHandle(_socket_mutex_handle);
        }

        WSACleanup();
    }

    void ReceiveAsync(int callback_id, int* socket) {

        printf("callback_id: %d\n", callback_id);
        printf("socket: %d\n", *socket);

        std::tuple<OnReceiveMessage, int> callbackTuple = *std::next(
            _callbacks.begin(),
            callback_id
        );

        int recv_buffer_size = std::get<1>(callbackTuple);
        char* recv_buffer = new char[recv_buffer_size];
        memset(recv_buffer, 0, recv_buffer_size);

        printf("start receive loop ...\n");

        int recv_length;

        while (_recv_running == true) {

            memset(recv_buffer, 0, recv_buffer_size);

            recv_length = recv(
                *socket,
                recv_buffer,
                recv_buffer_size,
                0
            );

            if (recv_length == -1) {    // socket was shut down
                break;
            }

#if USE_TCP
            if (recv_length == 0) { // if the other party closes the socket
                printf("client disconnected ...\n");
                break;
            }
#endif

            (std::get<0>(callbackTuple))(recv_buffer);  // execute callback
        }

        printf("end receive loop ...\n");
    }

    int StartReceiveAsync(int callback_id, int socket_id) {

        // Stored in a list to properly terminate background processes
        // when the application terminates.

        _recv_threads.push_front(std::thread(
            ReceiveAsync,
            callback_id,
            &_src_sockets[socket_id]
        ));

        _recv_running = true;

        printf("receive thread started ...\n");

        return 0;
    }

    int CreateSocket(
        unsigned short server_port,
        unsigned short client_port,
        char* client_addr, int num_sockets
    ) {
        printf("start set up server\n\n");

        _num_sockets = num_sockets;

        printf("client addr: %s\n", client_addr);

        for (int i = 0; i < _num_sockets; i++) {
            _server_ports[i] = server_port + (unsigned short)i;
            printf("server port [%d]: %d\n", i, _server_ports[i]);
        }

#if !USE_TCP
        _client_addr = client_addr;

        for (int i = 0; i < _num_sockets; i++) {
            _client_ports[i] = client_port + (unsigned short)i;
            printf("client port [%d]: %d\n", i, _client_ports[i]);
        }
#endif

        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 0), &data) != 0) {
            printf("TLabCppSocket.WSAStartup(): error\n");
            return 1;
        }

#if USE_TCP
        int socket_type = SOCK_STREAM;
#else
        int socket_type = SOCK_DGRAM;
#endif

        for (int i = 0; i < _num_sockets; i++) {

            memset(&_src_addrs[i], 0, sizeof(_src_addrs[i]));
            _src_addrs[i].sin_port = htons(_server_ports[i]);
            _src_addrs[i].sin_family = AF_INET;
            _src_addrs[i].sin_addr.s_addr = htonl(INADDR_ANY);

            _src_sockets[i] = (int)socket(
                AF_INET,
                socket_type,
                0
            );

            if (_src_sockets[i] == INVALID_SOCKET) {
                printf("_src_sockets[ %d ]: Create socket error\n", i);
                return 1;
            }

            printf("_src_sockets[ %d ]: %d\n", i, _src_sockets[i]);

            int bind_result = bind(
                _src_sockets[i],
                (struct sockaddr*)&_src_addrs[i],
                sizeof(_src_addrs[i])
            );

            if (bind_result == SOCKET_ERROR) {
                printf("_src_sockets [ %d ]: Bind: error\n", i);
                return 1;
            }
        }

#if USE_TCP
        printf(
            "waits for a connection from the client\n"
            "because it uses the TCP protocol.\n"
            "(this waiting process should be executed in separate threads,\n"
            "so it will be fixed in the future).\n"
        );

        for (int i = 0; i < _num_sockets; i++) {
            printf("_src_socket_[ %d ]: allow socket to listen for connections\n", i);
            if (listen(_src_sockets[i], 1) == SOCKET_ERROR) {
                printf("socket[ %d ] :listen error\n", i);
                return 1;
            }

            printf("_src_socket_[ %d ]: Waiting for connection ... \n", i);
            int dstAddrSize = sizeof(_dst_addrs[i]);
            _dst_sockets[i] = (int)accept(
                _src_sockets[i],
                (struct sockaddr*)&_dst_addrs[i],
                &dstAddrSize
            );

            char buffer[sizeof(_dst_addrs[i].sin_addr)];
            printf(
                "_src_socket_[ %d ]: connected from %s\n",
                i,
                inet_ntop(
                    AF_INET,
                    &_dst_addrs[i].sin_addr,
                    buffer,
                    sizeof(_dst_addrs[i].sin_addr)
                )
            );
        }
#else

        for (int i = 0; i < _num_sockets; i++) {

            printf("since UDP is connectionless, set the destination address manually\n\n");

            memset(&_dst_addrs[i], 0, sizeof(_dst_addrs[i]));
            _dst_addrs[i].sin_family = AF_INET;
            _dst_addrs[i].sin_port = htons(_client_ports[i]);

            int pton_result = inet_pton( // inet_addr is deprecated
                _dst_addrs[i].sin_family,
                _client_addr,
                &_dst_addrs[i].sin_addr.S_un.S_addr
            );

            if (pton_result == 0) {
                printf("_dst_addrs [ %d ]: inet_pton error\n", i);
                return 0;
            }
        }
#endif

#if TLAB_SHARESCREEN
        printf("optimize sockets ...\n");

        _src_socket_0 = _src_sockets[0];
        _src_socket_1 = _src_sockets[1];
        _dst_addr_0 = _dst_addrs[0];
        _dst_addr_1 = _dst_addrs[1];

        printf("socket optimization complited\n\n");
#endif

        printf("flagged as successful connection with client\n");
        _is_connecting = true;
        _keep_alive = true;

        _socket_mutex_handle = CreateMutex(
            NULL,
            FALSE,
            L"TLabSocket"
        );

        printf("server set up succeed\n\n");
        return 0;
    }

    void SendPacket(char* packet, int size, int id) {
        sendto(
            _src_sockets[id],
            packet,
            size,
            0,
            (struct sockaddr*)&_dst_addrs[id],
            sizeof(_dst_addrs[id])
        );
    }

#if TLAB_SHARESCREEN
    void SendFrame(char* packet, int size) {
        sendto(
            _src_socket_0,
            packet,
            size,
            0,
            (struct sockaddr*)&_dst_addr_0,
            sizeof(_dst_addr_0)
        );
    }

    void ResendFrame(char* packet, int size) {
        sendto(
            _src_socket_0,
            packet,
            size,
            0,
            (struct sockaddr*)&_dst_addr_0,
            sizeof(_dst_addr_0)
        );
    }
#endif
}