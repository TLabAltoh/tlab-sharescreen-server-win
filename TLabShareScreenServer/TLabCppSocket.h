#pragma once

#include "TLabWindows.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <list>
#include <thread>
#include <functional>

#define ISTHISTCP false
#define TLAB_SCREENSHARE true
#define DG_BUFFER_SIZE 1443
#define MAXIMUN_SOCKET_SIZE 4

#pragma region MTU
// https://milestone-of-se.nesuke.com/nw-basic/as-nw-engineer/udp-mtu-mss/
// https://qwerty.work/blog/2019/12/mtu-optimize-command-windows10.php
#pragma endregion

namespace {

    // ポート番号、ソケット
    unsigned short _serverPorts[MAXIMUN_SOCKET_SIZE];
    unsigned short _clientPorts[MAXIMUN_SOCKET_SIZE];
    int _srcSockets[MAXIMUN_SOCKET_SIZE];
    int _dstSockets[MAXIMUN_SOCKET_SIZE];
    struct sockaddr_in _srcAddrs[MAXIMUN_SOCKET_SIZE];
    struct sockaddr_in _dstAddrs[MAXIMUN_SOCKET_SIZE];

    /* ------------------------------------------- */

    // 高速化のための変数のキャッシュ

#if TLAB_SCREENSHARE
    int _srcSocket0;
    int _srcSocket1;
    struct sockaddr_in _dstAddr0;
    struct sockaddr_in _dstAddr1;
#endif

    /* ------------------------------------------- */

    // 使用するソケットの数
    int _numSockets = 0;

    // ループフラグ
    bool _isConnecting = false;
    bool _keepAlive = false;
    bool _recvRunning = false;

#if ISTHISTCP == false
    char* _clientAddr;
#endif

    // ソケットの排他制御
    HANDLE _socketMutexHandle = NULL;

    bool IsConnecting() { return _isConnecting; }

    // データ送受信のコールバック
    using TLabReceiveMessageFunc = std::function<void(char* receivePacket)>;
    std::list<std::tuple<TLabReceiveMessageFunc, int>> _callbacks;

    // 受信スレッド
    std::list<std::thread> _recvThreads;

    void SetCallback(
        TLabReceiveMessageFunc onReceiveMessage,
        int receiveBufferSize)
    {
        // データ受信時のコールバックを登録
        _callbacks.push_front(std::make_tuple(
            onReceiveMessage,
            receiveBufferSize
        ));
    }

    void CloseSocket() {

        _isConnecting = false;
        _keepAlive = false;

        if (_recvRunning == true) {
            
            _recvRunning = false;

            for (int i = 0; i < _numSockets; i++) {

#if ISTHISTCP
                closesocket(_dstSockets[i]);
#else
                // Generate a termination interrupt on sockets waiting for reception.
                // (The return value of recvLength becomes -1).
                closesocket(_srcSockets[i]);
#endif
            }

            for (
                auto threadItr = _recvThreads.begin();
                threadItr != _recvThreads.end();
                threadItr++)
            {
                // Kill all running background processes in a for loop.
                (*threadItr).join();
            }
        }

        if (_socketMutexHandle != NULL) CloseHandle(_socketMutexHandle);

        WSACleanup();
    }

    void ReceiveAsync(int callbackIndex, int* socket) {

        // 非同期でクライアントからのメッセージを受信 / 処理する
        // 別スレッドで実行することで受信待ちによるプロセスのブロッキングを回避する

        printf("callbackIndex: %d\n", callbackIndex);
        printf("socket: %d\n", *socket);

        std::tuple<TLabReceiveMessageFunc, int> callbackTuple = *std::next(
            _callbacks.begin(),
            callbackIndex
        );

        int recvBufferSize = std::get<1>(callbackTuple);
        char* recvBuffer = new char[recvBufferSize];
        memset(recvBuffer, 0, recvBufferSize);

        /* ------------------------------------------------------------------------------ */

        // 受信ループ開始

        printf("start receive loop ...\n");

        int recvLength;

        while (_recvRunning == true) {

            // バッファクリア
            memset(recvBuffer, 0, recvBufferSize);

            // データ受信待ち
            recvLength = recv(
                *socket,
                recvBuffer,
                recvBufferSize,
                0
            );

            // printf("data received. %d\n", recvLength);

            // socket was shut down.
            if (recvLength == -1) break;

#if ISTHISTCP == true

            // 相手が閉じたら
            if (recvLength == 0) {
                printf("client disconnected ...\n");
                break;
            }
#endif

            // 受信時のコールバックを実行
            (std::get<0>(callbackTuple))(recvBuffer);
        }

        printf("end receive loop.\n");

        /* ------------------------------------------------------------------------------ */
    }

    int StartReceiveAsync(int callbackIndex, int socketIndex) {

        // Stored in a list to properly terminate background processes
        // when the application terminates.

        _recvThreads.push_front(std::thread(
            ReceiveAsync,
            callbackIndex,
            &_srcSockets[socketIndex]
        ));

        _recvRunning = true;

        printf("receive thread started.\n");

        return 0;
    }

    int CreateSocket(
        unsigned short serverPort,
        unsigned short clientPort,
        char* clientAddr,
        int numSockets
    ) {
        printf("start set up server\n\n");

        _numSockets = numSockets;

        printf("client addr: %s\n", clientAddr);

        for (int i = 0; i < _numSockets; i++) {
            _serverPorts[i] = serverPort + (unsigned short)i;
            printf("server port [%d]: %d\n", i, _serverPorts[i]);
        }

#if ISTHISTCP == false
        _clientAddr = clientAddr;

        for (int i = 0; i < _numSockets; i++) {
            _clientPorts[i] = clientPort + (unsigned short)i;
            printf("client port [%d]: %d\n", i, _clientPorts[i]);
        }
#endif

        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 0), &data) != 0) {
            printf("TLabCppSocket.WSAStartup(): error\n");
            return 1;
        }

#if ISTHISTCP
        int socketType = SOCK_STREAM;
#else
        int socketType = SOCK_DGRAM;
#endif

        for (int i = 0; i < _numSockets; i++) {

            memset(&_srcAddrs[i], 0, sizeof(_srcAddrs[i]));
            _srcAddrs[i].sin_port = htons(_serverPorts[i]);
            _srcAddrs[i].sin_family = AF_INET;
            _srcAddrs[i].sin_addr.s_addr = htonl(INADDR_ANY);

            _srcSockets[i] = (int)socket(
                AF_INET,
                socketType,
                0
            );

            if (_srcSockets[i] == INVALID_SOCKET) {
                printf("_srcSockets[ %d ]: Create socket error\n", i);
                return 1;
            }

            printf("_srcSockets[ %d ]: %d\n", i, _srcSockets[i]);

            int bindResult = bind(
                _srcSockets[i],
                (struct sockaddr*)&_srcAddrs[i],
                sizeof(_srcAddrs[i])
            );

            if (bindResult == SOCKET_ERROR) {
                printf("_srcSockets [ %d ]: Bind: error\n", i);
                return 1;
            }
        }

#if ISTHISTCP

        printf(
            "waits for a connection from the client\n"
            "because it uses the TCP protocol.\n"
            "(this waiting process should be executed in separate threads,\n"
            "so it will be fixed in the future).\n"
        );

        for (int i = 0; i < _numSockets; i++) {
            printf("_srcSocket[ %d ]: allow socket to listen for connections\n", i);
            if (listen(_srcSockets[i], 1) == SOCKET_ERROR) {
                printf("socket[ %d ] :listen error\n", i);
                return 1;
            }

            printf("_srcSocket[ %d ]: Waiting for connection ... \n", i);
            int dstAddrSize = sizeof(_dstAddrs[i]);
            _dstSockets[i] = (int)accept(
                _srcSockets[i],
                (struct sockaddr*)&_dstAddrs[i],
                &dstAddrSize
            );

            char buffer[sizeof(_dstAddrs[i].sin_addr)];
            printf(
                "_srcSocket[ %d ]: connected from %s\n",
                i,
                inet_ntop(
                    AF_INET,
                    &_dstAddrs[i].sin_addr,
                    buffer,
                    sizeof(_dstAddrs[i].sin_addr)
                )
            );
        }
#else

        for (int i = 0; i < _numSockets; i++) {

            printf("since UDP is connectionless, set the destination address manually\n\n");

            memset(&_dstAddrs[i], 0, sizeof(_dstAddrs[i]));
            _dstAddrs[i].sin_family = AF_INET;
            _dstAddrs[i].sin_port = htons(_clientPorts[i]);

            // inet_addrは非推奨
            int ptonResult = inet_pton(
                _dstAddrs[i].sin_family,
                _clientAddr,
                &_dstAddrs[i].sin_addr.S_un.S_addr
            );

            if (ptonResult == 0) {
                printf("_dstAddrs [ %d ]: inet_pton error\n", i);
                return 0;
            }
        }
#endif

#if TLAB_SCREENSHARE

        printf("optimize sockets ...\n");

        _srcSocket0 = _srcSockets[0];
        _srcSocket1 = _srcSockets[1];
        _dstAddr0 = _dstAddrs[0];
        _dstAddr1 = _dstAddrs[1];

        printf("socket optimization complited\n\n");
#endif

        printf("flagged as successful connection with client\n");
        _isConnecting = true;
        _keepAlive = true;

        _socketMutexHandle = CreateMutex(
            NULL,
            FALSE,
            L"TLabSocket"
        );

        printf("server set up succeed\n\n");
        return 0;
    }

    void SendPacket(char* packet, int size, int socketIndex) {
        sendto(
            _srcSockets[socketIndex],
            packet,
            size,
            0,
            (struct sockaddr*)&_dstAddrs[socketIndex],
            sizeof(_dstAddrs[socketIndex])
        );
    }

#if TLAB_SCREENSHARE
    void SendFrame(char* packet, int size) {
        sendto(
            _srcSocket0,
            packet,
            size,
            0,
            (struct sockaddr*)&_dstAddr0,
            sizeof(_dstAddr0)
        );
    }

    void ResendFrame(char* packet, int size) {
        sendto(
            _srcSocket0,
            packet,
            size,
            0,
            (struct sockaddr*)&_dstAddr0,
            sizeof(_dstAddr0)
        );
    }
#endif
}